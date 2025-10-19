"""Centralized thread pool manager for backend services.

优化策略:
1. 统一线程池管理 - 避免每个服务创建独立线程
2. IO密集型任务(WebSocket, 数据轮询) - 2个工作线程
3. CPU密集型任务(数据处理, 编码) - 根据CPU核心数配置
4. 长时间运行任务(硬件采集循环) - 独立托管线程
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional


class BackendThreadPool:
    """统一后端线程池管理器,避免线程风暴."""

    _instance: Optional[BackendThreadPool] = None
    _lock = threading.Lock()

    def __new__(cls) -> BackendThreadPool:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self.logger = logging.getLogger("backend.threadpool")

        # IO密集型线程池 (WebSocket, 事件分发, 轻量级任务)
        self._io_pool = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="backend-io-"
        )

        # CPU密集型线程池 (数据处理, 编码, 计算)
        cpu_count = os.cpu_count() or 4
        cpu_workers = min(cpu_count, 4)  # 最多4个CPU线程
        self._cpu_pool = ThreadPoolExecutor(
            max_workers=cpu_workers,
            thread_name_prefix="backend-cpu-"
        )

        # 长时间运行的托管线程 (硬件采集循环)
        self._managed_threads: dict[str, threading.Thread] = {}
        self._thread_lock = threading.Lock()

        atexit.register(self.shutdown)
        self.logger.info(
            "线程池初始化: IO工作线程=%d, CPU工作线程=%d",
            2,
            cpu_workers
        )

    def submit_io_task(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Future[Any]:
        """提交IO密集型任务 (WebSocket发送, 事件分发, 数据库查询)."""
        return self._io_pool.submit(fn, *args, **kwargs)

    def submit_cpu_task(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any
    ) -> Future[Any]:
        """提交CPU密集型任务 (数据编码, 图像处理, 疲劳评分计算)."""
        return self._cpu_pool.submit(fn, *args, **kwargs)

    def register_managed_thread(
        self,
        name: str,
        target: Callable[..., Any],
        *,
        daemon: bool = True,
        args: tuple[Any, ...] = (),
        kwargs: Optional[dict[str, Any]] = None
    ) -> threading.Thread:
        """注册长时间运行的托管线程 (硬件采集循环).

        Args:
            name: 线程唯一标识符
            target: 线程执行函数
            daemon: 是否为守护线程
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            创建的线程对象

        Raises:
            ValueError: 如果线程名称已存在
        """
        with self._thread_lock:
            if name in self._managed_threads:
                existing = self._managed_threads[name]
                if existing.is_alive():
                    raise ValueError(f"托管线程 '{name}' 已存在且正在运行")
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
            self.logger.debug("注册托管线程: %s (daemon=%s)", name, daemon)
            return thread

    def unregister_managed_thread(self, name: str, *, timeout: float = 2.0) -> None:
        """注销托管线程并等待其结束.

        Args:
            name: 线程标识符
            timeout: 等待超时时间(秒)
        """
        with self._thread_lock:
            thread = self._managed_threads.pop(name, None)
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                self.logger.warning("托管线程 '%s' 未在超时内停止", name)
            else:
                self.logger.debug("托管线程 '%s' 已停止", name)

    def get_managed_thread(self, name: str) -> Optional[threading.Thread]:
        """获取托管线程对象."""
        with self._thread_lock:
            return self._managed_threads.get(name)

    def shutdown(self, *, wait: bool = True, timeout: float = 5.0) -> None:
        """关闭所有线程池和托管线程."""
        self.logger.info("正在关闭线程池...")

        # 停止所有托管线程
        with self._thread_lock:
            managed = list(self._managed_threads.items())

        for name, thread in managed:
            if thread.is_alive():
                self.logger.debug("等待托管线程停止: %s", name)
                thread.join(timeout=timeout / len(managed) if managed else timeout)

        # 关闭线程池
        self._io_pool.shutdown(wait=wait, cancel_futures=not wait)
        self._cpu_pool.shutdown(wait=wait, cancel_futures=not wait)
        self.logger.info("线程池已关闭")

    def diagnostics(self) -> dict[str, Any]:
        """获取线程池诊断信息."""
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
            "io_pool": {
                "max_workers": 2,
                "_shutdown": self._io_pool._shutdown  # type: ignore[attr-defined]
            },
            "cpu_pool": {
                "max_workers": self._cpu_pool._max_workers,  # type: ignore[attr-defined]
                "_shutdown": self._cpu_pool._shutdown  # type: ignore[attr-defined]
            },
            "managed_threads": managed_info
        }


# 全局单例实例
def get_thread_pool() -> BackendThreadPool:
    """获取全局线程池单例."""
    return BackendThreadPool()


__all__ = ["BackendThreadPool", "get_thread_pool"]
