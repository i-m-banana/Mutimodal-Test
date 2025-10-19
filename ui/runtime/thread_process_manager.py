"""Compatibility layer for legacy imports expecting thread_process_manager module."""

from __future__ import annotations

from .lifecycle_manager import (
    get_process_manager as _get_process_manager,
    get_thread_manager as _get_thread_manager,
    start as _start,
    shutdown as _shutdown,
)


class _LifecycleFacade:
    """Adapter providing the legacy lifecycle_manager API used by ui/main."""

    def __init__(self) -> None:
        self._initialized = False
        self.start_all()

    def get_all_status(self):
        return {"is_initialized": self._initialized}

    def start_all(self) -> None:
        _start()
        self._initialized = True

    def shutdown_all(self) -> None:
        _shutdown()
        self._initialized = False


def get_lifecycle_manager():
    return _LifecycleFacade()


def shutdown_all_managers() -> None:
    _shutdown()


__all__ = [
    "get_thread_manager",
    "get_process_manager",
    "get_lifecycle_manager",
    "shutdown_all_managers",
]

# 重导出旧函数名
def get_thread_manager():
    return _get_thread_manager()


def get_process_manager():
    return _get_process_manager()
