"""Backward-compatible import shim for legacy front-end modules."""

from __future__ import annotations

try:
    from .runtime.thread_process_manager import (
        get_lifecycle_manager,
        get_process_manager,
        get_thread_manager,
        shutdown_all_managers,
    )
except ImportError:  # pragma: no cover - fallback when imported as top-level module
    from ui.runtime.thread_process_manager import (  # type: ignore
        get_lifecycle_manager,
        get_process_manager,
        get_thread_manager,
        shutdown_all_managers,
    )

__all__ = [
    "get_thread_manager",
    "get_process_manager",
    "get_lifecycle_manager",
    "shutdown_all_managers",
]
