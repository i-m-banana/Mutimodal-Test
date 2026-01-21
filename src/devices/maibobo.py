"""Wrapper around the Maibobo blood-pressure monitor SDK."""

from __future__ import annotations

from typing import Any

from .base import DeviceBase

try:
    from maibobo import MaiboboEngine  # type: ignore
    HAS_MAIBOBO_DRIVER = True
except Exception:
    MaiboboEngine = None  # type: ignore
    HAS_MAIBOBO_DRIVER = False

try:
    from serial.tools import list_ports
    import serial  # type: ignore
except Exception:
    list_ports = None  # type: ignore
    serial = None  # type: ignore

DEFAULT_PORT = "COM4"


def detect_available_port(target_port: str | None = None) -> str | None:
    if list_ports is None:
        return None
    if not target_port:
        target_port = DEFAULT_PORT
    if not target_port:
        return None
    if serial is None:
        return target_port
    try:
        with serial.Serial(target_port, timeout=1):
            return target_port
    except Exception:
        return None


def parse_frame(frame: Any) -> dict[str, int] | None:
    try:
        if hasattr(frame, "systolic") and hasattr(frame, "diastolic") and hasattr(frame, "pulse"):
            return {
                "systolic": int(frame.systolic),
                "diastolic": int(frame.diastolic),
                "pulse": int(frame.pulse),
            }
        if isinstance(frame, (list, tuple)) and len(frame) >= 11:
            return {
                "systolic": int(frame[8]),
                "diastolic": int(frame[10]),
                "pulse": int(frame[2]),
            }
        if isinstance(frame, (list, tuple)) and len(frame) >= 3:
            return {
                "systolic": int(frame[0]),
                "diastolic": int(frame[1]),
                "pulse": int(frame[2]),
            }
    except Exception:
        pass
    return None


class MaiboboDevice(DeviceBase):
    """Thin adapter that exposes the MaiboboEngine through DeviceBase."""

    def __init__(self, port: str | None, *, timeout: int = 1) -> None:
        super().__init__()
        self._port = port
        self._timeout = timeout
        self._engine: Any = None

    def read(self) -> tuple[bool, object | None]:
        if self._engine is None:
            return False, None
        frame = self._engine.read()
        return frame is not None, frame

    def on_start(self) -> None:
        self._engine = MaiboboEngine(self._port, timeout=self._timeout)
        self._engine.connect()
        self._engine.start()

    def on_done(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine = None


__all__ = [
    "MaiboboDevice",
    "HAS_MAIBOBO_DRIVER",
    "DEFAULT_PORT",
    "detect_available_port",
    "parse_frame",
]
