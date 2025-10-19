"""Wrapper around the Tobii eye-tracking SDK."""

from __future__ import annotations

from typing import Any

from .base import DeviceBase, DeviceException

try:  # Optional dependency provided by the Tobii SDK
    from tobii import TobiiEngine  # type: ignore
except Exception as exc:  # pragma: no cover - optional at runtime
    TobiiEngine = None  # type: ignore
    _IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - runtime branch
    _IMPORT_ERROR = None

HAS_TOBII = TobiiEngine is not None


class TobiiDevice(DeviceBase):
    """Thin adapter that exposes TobiiEngine via DeviceBase."""

    def __init__(self) -> None:
        super().__init__()
        if TobiiEngine is None:
            raise DeviceException("Tobii SDK is not installed") from _IMPORT_ERROR
        self._engine = TobiiEngine()

    def read(self) -> tuple[bool, Any]:
        data = self._engine.read()
        return True, data

    def on_start(self) -> None:
        self._engine.__enter__()

    def on_done(self) -> None:
        try:
            self._engine.__exit__(None, None, None)
        finally:
            self._engine = None  # type: ignore[assignment]


__all__ = ["TobiiDevice", "HAS_TOBII"]
