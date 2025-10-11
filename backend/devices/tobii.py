from __future__ import annotations

from typing import Any

from .base import DeviceBase


class TobiiDevice(DeviceBase):
    def __init__(self) -> None:
        from tobii import TobiiEngine

        super().__init__()
        self.engine = TobiiEngine()

    def read(self):
        data: dict[str, Any] = self.engine.read()
        frame = data["gaze_point"] + data["head_pose"]
        return True, frame

    def on_start(self):
        self.engine.__enter__()

    def on_done(self):
        from tobii.engine import tobii_api_destroy

        tobii_api_destroy(self.engine.api)
        self.engine.api = None
        self.engine.devices = []
