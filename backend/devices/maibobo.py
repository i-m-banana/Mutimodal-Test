from __future__ import annotations

from .base import DeviceBase


class MaiboboDevice(DeviceBase):
    def __init__(self, port: str | None, *, timeout: int = 1):
        from maibobo import MaiboboEngine

        super().__init__()
        self.engine = MaiboboEngine(port, timeout=timeout)

    def read(self):
        frame = self.engine.read()
        return frame is not None, frame

    def on_start(self):
        self.engine.connect()
        self.engine.start()

    def on_done(self):
        self.engine.stop()
