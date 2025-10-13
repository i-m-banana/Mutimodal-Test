from __future__ import annotations

from src.constants import EventTopic, Severity
from src.core.event_bus import Event, EventBus
from src.detectors.anomaly_detector import AnomalyDetector
from src.detectors.object_detector import ObjectDetector
from src.models.model_manager import ModelManager


def test_anomaly_detector_generates_result():
    bus = EventBus()
    manager = ModelManager([])
    detector = AnomalyDetector("health", bus, manager)
    results = []
    bus.subscribe(EventTopic.DETECTION_RESULT, lambda event: results.append(event))

    detector.start()
    bus.publish(Event(topic=EventTopic.SENSOR_PACKET, payload={
        "blood_pressure": {"systolic": 155, "diastolic": 90},
        "stress_index": 0.9,
    }))
    detector.stop()

    assert results, "Detector should publish anomaly event"
    payload = results[0].payload
    assert payload["detector"] == "health"
    assert payload["severity"] == Severity.ERROR.value


def test_object_detector_uses_stub_model():
    bus = EventBus()
    configs = [{
        "name": "vision",
        "class": "models.torch_model.TorchModel",
        "enabled": True,
        "options": {},
    }]
    manager = ModelManager(configs)
    manager.load_enabled()

    detector = ObjectDetector("vision_detector", bus, manager, options={"model": "vision"})
    results = []
    bus.subscribe(EventTopic.DETECTION_RESULT, lambda event: results.append(event))
    detector.start()

    bus.publish(Event(topic=EventTopic.CAMERA_FRAME, payload={
        "sequence": 1,
        "data": b"frame",
    }))

    detector.stop()
    manager.unload_all()

    assert results, "Object detector should produce detection event"
    payload = results[0].payload
    assert payload["detector"] == "vision_detector"
    assert payload["model"] == "vision"
