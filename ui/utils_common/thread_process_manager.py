"""Backward-compatible import shim for PT managers (utils_common first)."""

from __future__ import annotations

from .lifecycle_manager import (
    get_process_manager,
    get_thread_manager,
    start as _start,
    shutdown as _shutdown,
)


class _LifecycleFacade:
    def __init__(self) -> None:
        self._initialized = False

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
