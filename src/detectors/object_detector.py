"""Simple object detector consuming camera frames."""

from __future__ import annotations

from typing import Iterable

from ..constants import EventTopic
from ..core.event_bus import Event
from .base_detector import BaseDetector


class ObjectDetector(BaseDetector):
    def topics(self) -> Iterable[EventTopic]:
        return [EventTopic.CAMERA_FRAME]

    def handle_event(self, event: Event) -> None:
        model_name = self.options.get("model", "vision" )
        try:
            inference = self.model_manager.infer(model_name, {"data": event.payload})
        except KeyError:
            self.logger.debug("Model %s not available for detector %s", model_name, self.name)
            return
        payload = {
            "detector": self.name,
            "source_sequence": event.payload.get("sequence"),
            "confidence": inference.get("confidence", 0.0),
            "model": inference.get("model"),
        }
        self.bus.publish(Event(topic=EventTopic.DETECTION_RESULT, payload=payload))


__all__ = ["ObjectDetector"]
