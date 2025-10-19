"""Lifecycle helpers for lazily created thread/process managers."""

from __future__ import annotations

from typing import Optional

from .process_manager import ProcessManager
from .thread_manager import ThreadManager

_PM: Optional[ProcessManager] = None
_TM: Optional[ThreadManager] = None


def start(*, process_workers: Optional[int] = None, thread_workers: int = 8) -> None:
    global _PM, _TM
    if _PM is None:
        _PM = ProcessManager(max_workers=process_workers)
    if _TM is None:
        _TM = ThreadManager(max_workers=thread_workers)


def shutdown() -> None:
    global _PM, _TM
    if _TM is not None:
        _TM.shutdown()
        _TM = None
    if _PM is not None:
        _PM.shutdown()
        _PM = None


def get_process_manager() -> ProcessManager:
    global _PM
    if _PM is None:
        _PM = ProcessManager()
    return _PM


def get_thread_manager() -> ThreadManager:
    global _TM
    if _TM is None:
        _TM = ThreadManager()
    return _TM
