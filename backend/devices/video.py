from __future__ import annotations

from typing import Any

from .base import DeviceBase, DeviceException


class VideoDevice(DeviceBase):
    def __init__(self, video: str | int | None = None) -> None:
        import cv2

        super().__init__()
        if video is None:
            for video_id in range(10):
                self.cap = cv2.VideoCapture(video_id, cv2.CAP_DSHOW)
                if self.cap.isOpened():
                    break
            else:
                raise DeviceException("No video device found")
        else:
            self.cap = cv2.VideoCapture(video, cv2.CAP_DSHOW)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> tuple[bool, Any]:
        ret, frame = self.cap.read()
        return ret, frame

    def on_start(self):
        pass

    def on_done(self):
        self.cap.release()
