"""System monitor emitting heartbeat metrics onto the event bus."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - psutil is optional
    import psutil
except Exception:  # pragma: no cover
    psutil = None

from ..constants import DEFAULT_HEARTBEAT_INTERVAL, EventTopic, Severity
from .event_bus import Event, EventBus


@dataclass
class SystemMetrics:
    cpu_percent: float
    memory_percent: float
    load_avg: Optional[float]
    platform: str

    def to_payload(self) -> Dict[str, Any]:
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "load_avg": self.load_avg,
            "platform": self.platform,
        }


class SystemMonitor:
    def __init__(self, bus: EventBus, *, interval: float = DEFAULT_HEARTBEAT_INTERVAL) -> None:
        self._bus = bus
        self._interval = interval
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _gather_metrics(self) -> SystemMetrics:
        if psutil:
            cpu = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory().percent
            try:
                load_avg = psutil.getloadavg()[0]
            except (AttributeError, OSError):  # Windows compatibility
                load_avg = None
        else:  # pragma: no cover - executed when psutil missing
            cpu = 0.0
            memory = 0.0
            load_avg = None
        return SystemMetrics(cpu_percent=cpu, memory_percent=memory, load_avg=load_avg, platform=platform.platform())

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="SystemMonitor", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            metrics = self._gather_metrics()
            severity = Severity.OK
            if metrics.cpu_percent > 90 or metrics.memory_percent > 90:
                severity = Severity.ERROR
            elif metrics.cpu_percent > 70 or metrics.memory_percent > 80:
                severity = Severity.WARN
            payload = metrics.to_payload() | {"severity": severity.value}
            self._bus.publish(Event(topic=EventTopic.SYSTEM_HEARTBEAT, payload=payload))
            time.sleep(self._interval)

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join(timeout=self._interval)
        self._thread = None


__all__ = ["SystemMonitor", "SystemMetrics"]
