from __future__ import annotations

import time

from src.collectors.camera_collector import CameraCollector
from src.collectors.sensor_collector import SensorCollector
from src.constants import EventTopic
from src.core.event_bus import EventBus


def test_camera_collector_simulator_emits_frames():
    bus = EventBus()
    events = []
    bus.subscribe(EventTopic.CAMERA_FRAME, lambda event: events.append(event))

    collector = CameraCollector("camera", bus, options={"mode": "simulator", "interval": 0.01})
    collector.start()
    time.sleep(0.05)
    collector.stop()

    assert events, "Camera collector should emit at least one frame"
    sample = events[0]
    assert sample.payload["encoding"] == "synthetic/plain"
    assert sample.payload["collector"] == "camera"


def test_sensor_collector_emits_metrics():
    bus = EventBus()
    events = []
    bus.subscribe(EventTopic.SENSOR_PACKET, lambda event: events.append(event))

    collector = SensorCollector("biosensor", bus, options={"interval": 0.01, "seed": 123})
    collector.start()
    time.sleep(0.05)
    collector.stop()

    assert events, "Sensor collector should emit packets"
    packet = events[0].payload
    assert 50 <= packet["pulse"] <= 100
    assert "blood_pressure" in packet
