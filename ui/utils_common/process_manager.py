import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from typing import Any, Callable, Dict, Optional
import atexit
import pickle
import time
from threading import Lock


class ProcessManager:
    """轻量进程管理器，兼容旧前端对异步推理的依赖。"""

    def __init__(self, max_workers: Optional[int] = None, start_method: str = "spawn"):
        self._executors: Dict[str, ProcessPoolExecutor] = {}
        self._default_key = "default"
        self._start_method = start_method
        self._tasks: Dict[str, Future] = {}
        self._tasks_lock = Lock()
        self._seq = 0
        try:
            mp.set_start_method(start_method, force=True)
        except RuntimeError:
            # 已经设置过，忽略
            pass
        self._executors[self._default_key] = ProcessPoolExecutor(max_workers=max_workers)
        self._thread_fallback = ThreadPoolExecutor(max_workers=max_workers or 4, thread_name_prefix="ui-pm-fallback-")
        atexit.register(self.shutdown)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _next_task_id(self) -> str:
        self._seq += 1
        return f"PM_{int(time.time() * 1000)}_{self._seq}"

    def get_executor(self, key: Optional[str] = None, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
        key = key or self._default_key
        if key not in self._executors:
            self._executors[key] = ProcessPoolExecutor(max_workers=max_workers)
        return self._executors[key]

    def submit(self, func: Callable[..., Any], *args: Any, key: Optional[str] = None, **kwargs: Any) -> str:
        try:
            pickle.dumps(func)
        except Exception as exc:
            raise TypeError(f"func not picklable: {exc}; submit a top-level function")
        try:
            pickle.dumps((args, kwargs))
        except Exception as exc:
            raise TypeError(f"args/kwargs not picklable: {exc}")
        task_id = self._next_task_id()
        exec_ = self.get_executor(key)
        future = exec_.submit(func, *args, **kwargs)
        with self._tasks_lock:
            self._tasks[task_id] = future
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        with self._tasks_lock:
            future = self._tasks.get(task_id)
        if not future:
            return False
        ok = future.cancel()
        with self._tasks_lock:
            self._tasks.pop(task_id, None)
        return ok

    # 兼容旧接口 ---------------------------------------------------------
    def submit_inference_task(self, func: Callable[..., Any], *args: Any,
                               task_name: Optional[str] = None, key: Optional[str] = None,
                               **kwargs: Any) -> str:
        """兼容旧版接口。

        优先尝试使用进程池；若函数/参数不可序列化，则回退到线程池执行。
        """
        try:
            return self.submit(func, *args, key=key, **kwargs)
        except TypeError as exc:
            self._logger.debug("submit_inference_task 使用线程池回退: %s", exc)
            task_id = self._next_task_id()
            future = self._thread_fallback.submit(func, *args, **kwargs)
            with self._tasks_lock:
                self._tasks[task_id] = future
            return task_id

    def get_process_info(self) -> Dict[str, Any]:
        with self._tasks_lock:
            pending = sum(1 for f in self._tasks.values() if not f.done())
            total = len(self._tasks)
        return {
            "executors": list(self._executors.keys()),
            "pending_tasks": pending,
            "total_tasks": total,
        }

    def get_process_pool_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {}
        for key, executor in self._executors.items():
            status[key] = {
                "max_workers": getattr(executor, "_max_workers", None),
            }
        status["thread_fallback"] = {
            "max_workers": getattr(self._thread_fallback, "_max_workers", None),
        }
        return status

    def shutdown(self, wait: bool = True) -> None:
        for key, exec_ in list(self._executors.items()):
            try:
                exec_.shutdown(wait=wait)
            except Exception as exc:  # pragma: no cover
                self._logger.error("进程管理器关闭失败: %s", exc)
        self._executors.clear()
        self._thread_fallback.shutdown(wait=wait)
        with self._tasks_lock:
            self._tasks.clear()

__all__ = ["ProcessManager"]
