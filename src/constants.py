"""Centralised constants for event types, metadata keys, and status codes."""

from __future__ import annotations

from enum import Enum


class EventTopic(str, Enum):
    """Canonical topics available on the internal event bus."""

    CAMERA_FRAME = "camera.frame"
    MULTIMODAL_FRAME = "multimodal.frame"
    MULTIMODAL_SNAPSHOT = "multimodal.snapshot"
    AUDIO_LEVEL = "audio.level"
    SENSOR_PACKET = "sensor.packet"
    FILE_BATCH = "file.batch"
    MODEL_REQUEST = "model.request"
    MODEL_RESPONSE = "model.response"
    DETECTION_RESULT = "detector.result"
    EMOTION_REQUEST = "emotion.request"
    EEG_REQUEST = "eeg.request"
    SYSTEM_HEARTBEAT = "system.heartbeat"
    UI_COMMAND = "ui.command"
    UI_RESPONSE = "ui.response"


class ModuleState(str, Enum):
    """Lifecycle state for collectors, detectors, and models."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class Severity(str, Enum):
    """Severity levels for health checks and monitoring events."""

    OK = "ok"
    WARN = "warn"
    ERROR = "error"


DEFAULT_HEARTBEAT_INTERVAL = 5.0
DEFAULT_COMPONENT_TIMEOUT = 30.0
