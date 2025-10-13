"""Wrapper around the Maibobo blood-pressure monitor SDK."""

from __future__ import annotations

from .base import DeviceBase


class MaiboboDevice(DeviceBase):
    """Thin adapter that exposes the MaiboboEngine through DeviceBase."""

    def __init__(self, port: str | None, *, timeout: int = 1) -> None:
        from maibobo import MaiboboEngine  # type: ignore

        super().__init__()
        self.engine = MaiboboEngine(port, timeout=timeout)

    def read(self) -> tuple[bool, object | None]:
        frame = self.engine.read()
        return frame is not None, frame

    def on_start(self) -> None:
        self.engine.connect()
        self.engine.start()

    def on_done(self) -> None:
        self.engine.stop()
