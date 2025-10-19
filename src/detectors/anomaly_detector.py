"""Rule-based anomaly detector for physiological sensors."""

from __future__ import annotations

from typing import Iterable

from ..constants import EventTopic, Severity
from ..core.event_bus import Event
from .base_detector import BaseDetector


class AnomalyDetector(BaseDetector):
    def topics(self) -> Iterable[EventTopic]:
        return [EventTopic.SENSOR_PACKET]

    def handle_event(self, event: Event) -> None:
        packet = event.payload
        systolic = packet.get("blood_pressure", {}).get("systolic", 0)
        stress_index = packet.get("stress_index", 0.0)
        severity = Severity.OK
        if systolic >= self.options.get("systolic_error", 150) or stress_index >= self.options.get("stress_error", 0.85):
            severity = Severity.ERROR
        elif systolic >= self.options.get("systolic_warn", 135) or stress_index >= self.options.get("stress_warn", 0.7):
            severity = Severity.WARN
        payload = {
            "detector": self.name,
            "severity": severity.value,
            "metrics": {
                "systolic": systolic,
                "stress_index": stress_index,
            },
        }
        self.bus.publish(Event(topic=EventTopic.DETECTION_RESULT, payload=payload))


__all__ = ["AnomalyDetector"]
