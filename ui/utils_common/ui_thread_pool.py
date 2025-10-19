"""UI层统一线程池管理器 (迁移自 ui.runtime.ui_thread_pool)."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional


class UIThreadPool:
    """UI层统一线程池管理器(单例)."""

    _instance: Optional[UIThreadPool] = None
    _lock = threading.Lock()

    def __new__(cls) -> "UIThreadPool":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.logger = logging.getLogger("ui.threadpool")

        # UI任务线程池 (轻量级代理任务、模拟数据生成)
        ui_workers = int(os.getenv("UI_THREAD_POOL_SIZE", "4"))
        self._ui_pool = ThreadPoolExecutor(
            max_workers=ui_workers,
            thread_name_prefix="ui-pool-"
        )

        # 托管长线程 (asyncio循环、模拟循环)
        self._managed_threads: dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()

        atexit.register(self.shutdown)
        self.logger.info("UI线程池初始化: 工作线程=%d", ui_workers)

    def submit_task(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Future[Any]:
        """提交UI任务 (后端命令转发、数据处理、模拟数据生成)."""
        return self._ui_pool.submit(fn, *args, **kwargs)

    def register_managed_thread(
        self,
        name: str,
        target: Callable[..., Any],
        *,
        daemon: bool = True,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None
    ) -> threading.Thread:
        """注册UI层托管线程."""
        with self._thread_lock:
            if name in self._managed_threads:
                existing = self._managed_threads[name]
                if existing.is_alive():
                    raise ValueError(f"UI托管线程 '{name}' 已存在且正在运行")
                # 清理已停止的线程
                del self._managed_threads[name]

            thread = threading.Thread(
                target=target,
                name=name,
                daemon=daemon,
                args=args,
                kwargs=kwargs or {}
            )
            self._managed_threads[name] = thread
            self.logger.debug("注册UI托管线程: %s (daemon=%s)", name, daemon)
            return thread

    def unregister_managed_thread(self, name: str, *, timeout: float = 2.0) -> None:
        """注销UI托管线程并等待其结束."""
        with self._thread_lock:
            thread = self._managed_threads.pop(name, None)
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                self.logger.warning("UI托管线程 '%s' 未在超时内停止", name)
            else:
                self.logger.debug("UI托管线程 '%s' 已停止", name)

    def get_managed_thread(self, name: str) -> Optional[threading.Thread]:
        """获取UI托管线程对象."""
        with self._thread_lock:
            return self._managed_threads.get(name)

    def shutdown(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """关闭UI线程池和所有托管线程."""
        self.logger.info("正在关闭UI线程池...")

        # 停止所有托管线程
        with self._thread_lock:
            managed = list(self._managed_threads.items())

        for name, thread in managed:
            if thread.is_alive():
                self.logger.debug("等待UI托管线程停止: %s", name)
                thread.join(timeout=timeout / len(managed) if managed else timeout)

        # 关闭线程池
        self._ui_pool.shutdown(wait=wait, cancel_futures=not wait)
        self.logger.info("UI线程池已关闭")

    def diagnostics(self) -> dict[str, Any]:
        """获取UI线程池诊断信息."""
        with self._thread_lock:
            managed_info = {
                name: {
                    "alive": thread.is_alive(),
                    "daemon": thread.daemon,
                    "ident": thread.ident
                }
                for name, thread in self._managed_threads.items()
            }

        return {
            "ui_pool": {
                "max_workers": self._ui_pool._max_workers,  # type: ignore[attr-defined]
                "_shutdown": self._ui_pool._shutdown  # type: ignore[attr-defined]
            },
            "managed_threads": managed_info
        }


# 全局单例实例
def get_ui_thread_pool() -> UIThreadPool:
    """获取UI层全局线程池单例."""
    return UIThreadPool()


__all__ = ["UIThreadPool", "get_ui_thread_pool"]
