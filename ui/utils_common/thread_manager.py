import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any, Dict, Optional
import atexit
import time


class ThreadManager:
    """轻量线程管理器，为前端通用线程池提供服务。"""

    def __init__(self, max_workers: int = 8):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ui-tm-")
        self._tasks: Dict[str, Future] = {}
        self._seq = 0
        atexit.register(self.shutdown)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _next_task_id(self) -> str:
        self._seq += 1
        return f"TM_{int(time.time() * 1000)}_{self._seq}"

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        task_id = self._next_task_id()
        future = self._executor.submit(func, *args, **kwargs)
        self._tasks[task_id] = future
        return task_id

    def submit_io_task(self, func: Callable[..., Any], *args: Any, task_name: Optional[str] = None, **kwargs: Any) -> str:
        """兼容旧版接口，接收 task_name 但不做额外处理。"""
        return self.submit(func, *args, **kwargs)

    def submit_data_task(self, func: Callable[..., Any], *args: Any, task_name: Optional[str] = None, **kwargs: Any) -> str:
        """旧版接口别名，与 submit_io_task 行为一致。"""
        return self.submit(func, *args, **kwargs)

    def cancel_task(self, task_id: str) -> bool:
        future = self._tasks.get(task_id)
        if not future:
            return False
        try:
            return future.cancel()
        finally:
            self._tasks.pop(task_id, None)

    def shutdown(self, wait: bool = True) -> None:
        try:
            self._executor.shutdown(wait=wait)
        except Exception as exc:  # pragma: no cover - 清理阶段
            self._logger.error("线程管理器关闭失败: %s", exc)
        self._tasks.clear()

__all__ = ["ThreadManager"]
