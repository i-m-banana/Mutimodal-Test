"""数据库访问服务层，负责所有测试数据的数据库操作。

从 TestPage 中提取，保持原有逻辑不变。
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal

try:
    from .backend_client import get_backend_client
except ImportError:  # pragma: no cover
    from ui.services.backend_client import get_backend_client  # type: ignore


_logger = logging.getLogger("ui.database_service")


class DatabaseService(QObject):
    """统一的数据库访问服务，处理测试记录的创建和更新。
    
    负责：
    - 创建测试记录（insert）
    - 更新测试记录（update）
    - 错误处理与重试逻辑
    - 待处理更新的队列管理
    """
    
    # 信号：用于从后台线程安全调度 UI 回调
    _invoke_later_signal = pyqtSignal(object, int)  # (callback, delay_ms)
    
    def __init__(self, parent: Optional[QObject] = None, username: str = 'anonymous') -> None:
        """初始化数据库服务。
        
        Args:
            parent: Qt 父对象（可选）
            username: 当前用户名（用于自动创建记录）
        """
        super().__init__(parent)
        
        # 用户名（用于创建数据库记录）
        self._username = username or 'anonymous'
        
        # 数据库状态标志
        self._db_disabled = False  # 是否禁用数据库写入
        self._db_warning_logged = False  # 是否已记录警告
        
        # 记录状态
        self.row_id: Optional[int] = None  # 当前测试记录的数据库 ID
        self._row_id_future = None  # 创建记录的异步 future
        
        # 待处理更新队列（在记录创建前排队）
        self._pending_db_updates: List[Callable[[int], None]] = []
        
        # 连接内部信号
        self._invoke_later_signal.connect(self._handle_invoke_later_signal)
        
        _logger.debug("DatabaseService 已初始化")
    
    # =========================================================================
    # 公共接口方法
    # =========================================================================
    
    def create_test_record(self, username: Optional[str] = None) -> None:
        """创建新的测试记录（仅创建一次）。
        
        Args:
            username: 用户名（可选，未提供则使用初始化时的用户名）
        """
        if username:
            self._username = username
        self._ensure_db_row()
    
    def update_test_record(self, update_payload: Dict, context: str = "更新数据库记录") -> None:
        """更新测试记录（无回调）。
        
        Args:
            update_payload: 要更新的字段字典（不包含 row_id）
            context: 错误上下文描述
        """
        self._queue_db_update(update_payload, context)
    
    def update_test_record_with_callback(
        self, 
        update_payload: Dict, 
        on_success: Optional[Callable[[Dict], None]] = None,
        context: str = "更新数据库记录"
    ) -> None:
        """更新测试记录（带成功回调）。
        
        Args:
            update_payload: 要更新的字段字典（不包含 row_id）
            on_success: 成功回调函数，接收响应字典
            context: 错误上下文描述
        """
        self._queue_db_update_with_callback(update_payload, context, on_success)
    
    def is_disabled(self) -> bool:
        """检查数据库写入是否已禁用。
        
        Returns:
            True 如果数据库写入已禁用
        """
        return self._db_disabled
    
    def disable_writes(self, reason: str) -> None:
        """手动禁用数据库写入（例如用户设置了 SKIP_DATABASE）。
        
        Args:
            reason: 禁用原因
        """
        self._disable_db_writes(reason)
    
    def get_row_id(self) -> Optional[int]:
        """获取当前测试记录的数据库 ID。
        
        Returns:
            记录 ID，如果尚未创建则返回 None
        """
        return self.row_id
    
    # =========================================================================
    # 内部实现方法（保持原有逻辑不变）
    # =========================================================================
    
    def _disable_db_writes(self, reason: str) -> None:
        """禁用数据库写入并记录警告。
        
        Args:
            reason: 禁用原因
        """
        if not self._db_warning_logged:
            _logger.warning(reason)
            _logger.warning("后续数据库写入已禁用；请检查 MySQL 服务或设置 UI_SKIP_DATABASE=1 后重启应用。")
        self._db_warning_logged = True
        self._db_disabled = True
        self._row_id_future = None
        self._pending_db_updates.clear()
    
    def _handle_db_failure(self, error: Exception, context: str) -> None:
        """处理数据库错误。
        
        Args:
            error: 异常对象
            context: 错误上下文描述
        """
        _logger.error(f"{context}: {error}")
        message = str(error)
        lower = message.lower()
        
        # 检测连接错误
        if any(keyword in lower for keyword in [
            "10061", "2003", "connection refused", "econnrefused", "timeout"
        ]):
            self._disable_db_writes("检测到数据库连接被拒绝，已暂停后续数据库写入以避免界面卡顿。")
        elif "skip_database" in lower or "disabled" in lower:
            self._disable_db_writes(message or "数据库写入已禁用")
    
    def _send_db_command(
        self,
        action: str,
        payload: Dict,
        *,
        context: str,
        on_success: Optional[Callable[[Dict], None]] = None
    ):
        """发送数据库命令到后端。
        
        Args:
            action: 命令动作（如 "db.insert_test_record"）
            payload: 命令参数字典
            context: 错误上下文描述
            on_success: 成功回调函数
        
        Returns:
            Future 对象，如果数据库已禁用或失败则返回 None
        """
        if self._db_disabled:
            return None
        
        try:
            client = get_backend_client()
        except Exception as exc:
            self._handle_db_failure(exc, context)
            return None
        
        future = client.send_command_future(action, payload)
        
        def _dispatch_result(fut):
            try:
                result = fut.result()
            except Exception as exc:
                # 使用默认参数捕获 exc，避免闭包变量问题
                self._invoke_later(lambda error=exc, ctx=context: self._handle_db_failure(error, ctx))
                return
            if on_success:
                # 同样修复 result 的捕获
                self._invoke_later(lambda res=result: on_success(res or {}))
        
        future.add_done_callback(_dispatch_result)
        return future
    
    def _flush_pending_db_updates(self, row_id: int) -> None:
        """执行所有待处理的数据库更新。
        
        Args:
            row_id: 数据库记录 ID
        """
        if not self._pending_db_updates:
            return
        callbacks = list(self._pending_db_updates)
        self._pending_db_updates.clear()
        for callback in callbacks:
            try:
                callback(row_id)
            except Exception as exc:
                _logger.error(f"延迟数据库更新执行失败: {exc}")
    
    def _ensure_db_row(self) -> None:
        """确保数据库记录已创建（仅创建一次，后续使用更新）。"""
        if self._db_disabled or self.row_id:
            return
        if self._row_id_future:
            _logger.debug("数据库记录创建请求已在处理中，跳过重复创建")
            return
        
        # 只包含必填字段，其他数据通过后续更新添加
        payload = {
            "name": self._username,
        }
        
        def _on_created(result: Dict):
            row_id = result.get("row_id")
            if not row_id:
                _logger.warning("数据库返回的记录ID无效，后续更新将被忽略。")
                return
            self.row_id = row_id
            self._row_id_future = None
            _logger.info(f"✅ 数据库记录已创建，ID: {row_id}")
            # 执行所有待处理的更新
            self._flush_pending_db_updates(row_id)
        
        _logger.debug("📝 创建新的数据库记录...")
        self._row_id_future = self._send_db_command(
            "db.insert_test_record",
            payload,
            context="创建数据库记录失败",
            on_success=_on_created,
        )
    
    def _queue_db_update(self, update_payload: Dict, context: str) -> None:
        """排队数据库更新（无回调）。
        
        Args:
            update_payload: 要更新的字段字典（不包含 row_id）
            context: 错误上下文描述
        """
        if self._db_disabled:
            return
        
        def _dispatch(row_id: int) -> None:
            payload = dict(update_payload)
            payload["row_id"] = row_id
            self._send_db_command("db.update_test_record", payload, context=context)
        
        if self.row_id:
            _dispatch(self.row_id)
        else:
            self._pending_db_updates.append(_dispatch)
            # 自动触发记录创建（使用初始化时的用户名）
            self._ensure_db_row()
    
    def _queue_db_update_with_callback(
        self,
        update_payload: Dict,
        context: str,
        on_success: Optional[Callable[[Dict], None]] = None
    ) -> None:
        """排队数据库更新（带成功回调）。
        
        Args:
            update_payload: 要更新的字段字典（不包含 row_id）
            context: 错误上下文描述
            on_success: 成功回调函数
        """
        if self._db_disabled:
            return
        
        def _dispatch(row_id: int) -> None:
            payload = dict(update_payload)
            payload["row_id"] = row_id
            self._send_db_command("db.update_test_record", payload, context=context, on_success=on_success)
        
        if self.row_id:
            _dispatch(self.row_id)
        else:
            self._pending_db_updates.append(_dispatch)
            # 自动触发记录创建
            self._ensure_db_row()
    
    # =========================================================================
    # 线程安全的回调调度（保持原有逻辑）
    # =========================================================================
    
    def _invoke_later(self, callback: Callable[[], None], delay_ms: int = 0) -> None:
        """在 UI 线程上执行回调（从任何线程安全调用）。
        
        Args:
            callback: 要执行的回调函数
            delay_ms: 延迟毫秒数（0 表示立即）
        """
        self._invoke_later_signal.emit(callback, delay_ms)
    
    def _handle_invoke_later_signal(self, callback: Callable[[], None], delay_ms: int) -> None:
        """处理 _invoke_later_signal（在主线程中执行）。
        
        Args:
            callback: 要执行的回调函数
            delay_ms: 延迟毫秒数
        """
        from PyQt5.QtCore import QTimer
        
        timeout = max(0, int(delay_ms))
        if timeout == 0:
            # 立即执行
            try:
                callback()
            except Exception as e:
                _logger.error(f"执行立即回调时出错: {e}", exc_info=True)
        else:
            # 延迟执行
            QTimer.singleShot(timeout, lambda: self._safe_callback(callback))
    
    def _safe_callback(self, callback: Callable[[], None]) -> None:
        """执行回调并捕获异常。
        
        Args:
            callback: 要执行的回调函数
        """
        try:
            callback()
        except Exception as e:
            _logger.error(f"执行延迟回调时出错: {e}", exc_info=True)


__all__ = ["DatabaseService"]
