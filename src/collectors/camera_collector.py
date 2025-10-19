"""Camera collector producing RGB frames either from hardware or a simulator."""

from __future__ import annotations

import base64
import random
import time
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import cv2
    import numpy as np
except Exception:  # pragma: no cover
    cv2 = None
    np = None

from ..constants import EventTopic
from ..core.event_bus import Event
from .base_collector import BaseCollector


class CameraCollector(BaseCollector):
    """Collect RGB frames and publish them onto the event bus."""

    def __init__(self, name: str, bus, *, options: Optional[Dict[str, Any]] = None, logger=None) -> None:
        super().__init__(name, bus, options=options, logger=logger)
        self.mode = (self.options.get("mode") or "simulator").lower()
        self.camera_index = int(self.options.get("camera_index", 0))
        self._cap = None
        self._frame_counter = 0

    def on_start(self) -> None:
        if self.mode == "hardware" and cv2 is not None:
            self.logger.info("Camera collector %s opening hardware index %s", self.name, self.camera_index)
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap or not self._cap.isOpened():
                self.logger.warning("Camera collector %s fallback to simulator (hardware unavailable)", self.name)
                self.mode = "simulator"
        else:
            if self.mode == "hardware":
                self.logger.warning("OpenCV not available, using simulator for %s", self.name)
            self.mode = "simulator"

    def run_once(self) -> None:
        frame_payload = self._capture_frame()
        event = Event(topic=EventTopic.CAMERA_FRAME, payload=frame_payload)
        self.bus.publish(event)

    def _capture_frame(self) -> Dict[str, Any]:
        self._frame_counter += 1
        if self.mode == "hardware" and self._cap is not None and cv2 is not None:
            ret, frame = self._cap.read()
            if not ret:
                self.logger.debug("Camera frame read failed, switching to simulator for this iteration")
                return self._simulate_frame()
            # Convert to JPEG bytes and then base64 encode to keep payload lightweight
            _, buffer = cv2.imencode(".jpg", frame)
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            return {
                "collector": self.name,
                "sequence": self._frame_counter,
                "timestamp": time.time(),
                "encoding": "image/jpeg",
                "data": encoded,
            }
        return self._simulate_frame()

    def _simulate_frame(self) -> Dict[str, Any]:
        width = int(self.options.get("width", 320))
        height = int(self.options.get("height", 240))
        seed = f"{self.name}:{self._frame_counter}".encode("utf-8")
        rng = random.Random(seed)
        pixel_intensity = rng.randint(0, 255)
        # Encode a very small synthetic image to keep payload deterministic
        intensity_hex = f"{pixel_intensity:02x}"
        data = {
            "collector": self.name,
            "sequence": self._frame_counter,
            "timestamp": time.time(),
            "encoding": "synthetic/plain",
            "metadata": {
                "width": width,
                "height": height,
                "intensity": pixel_intensity,
            },
            "data": intensity_hex,
        }
        return data

    def on_stop(self) -> None:
        if self._cap is not None and cv2 is not None:
            self._cap.release()
            self._cap = None


__all__ = ["CameraCollector"]
