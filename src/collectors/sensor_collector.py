"""Sensor collector simulating EEG/blood pressure packets."""

from __future__ import annotations

import random
import time
from typing import Any, Dict, Optional

from ..constants import EventTopic
from ..core.event_bus import Event
from .base_collector import BaseCollector


class SensorCollector(BaseCollector):
    """Emit synthetic sensor measurements representing EEG and vital signs."""

    def __init__(self, name: str, bus, *, options: Optional[Dict[str, Any]] = None, logger=None) -> None:
        super().__init__(name, bus, options=options, logger=logger)
        self._seed = int(self.options.get("seed", time.time()))
        self._rng = random.Random(self._seed)

    def run_once(self) -> None:
        packet = {
            "collector": self.name,
            "timestamp": time.time(),
            "blood_pressure": self._generate_bp(),
            "pulse": self._rng.randint(58, 95),
            "eeg_alpha": round(self._rng.uniform(0.1, 0.9), 3),
            "eeg_beta": round(self._rng.uniform(0.1, 0.9), 3),
            "stress_index": round(self._rng.uniform(0, 1), 3),
        }
        self.bus.publish(Event(topic=EventTopic.SENSOR_PACKET, payload=packet))

    def _generate_bp(self) -> dict[str, int]:
        systolic = self._rng.randint(105, 140)
        diastolic = self._rng.randint(65, 90)
        return {"systolic": systolic, "diastolic": diastolic}


__all__ = ["SensorCollector"]
