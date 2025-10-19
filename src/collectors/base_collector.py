"""Base class for all data collectors."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..constants import ModuleState
from ..core.event_bus import EventBus


class BaseCollector(ABC):
    """Base class offering a cooperative-thread loop."""

    def __init__(self, name: str, bus: EventBus, *, options: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.bus = bus
        self.options = options or {}
        self.logger = logger or logging.getLogger("collector")
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.state = ModuleState.CREATED
        self._iteration_interval = float(self.options.get("interval", 0.5))

    @property
    def iteration_interval(self) -> float:
        return self._iteration_interval

    def start(self) -> None:
        if self.state == ModuleState.RUNNING:
            return
        self.state = ModuleState.STARTING
        self.logger.info("Starting collector %s", self.name)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_wrapper, name=f"Collector:{self.name}", daemon=True)
        self._thread.start()

    def _run_wrapper(self) -> None:
        try:
            self.state = ModuleState.RUNNING
            self.on_start()
            while not self._stop_event.is_set():
                start_ts = time.perf_counter()
                self.run_once()
                elapsed = time.perf_counter() - start_ts
                sleep_for = max(0.0, self.iteration_interval - elapsed)
                if sleep_for:
                    self._stop_event.wait(timeout=sleep_for)
        except Exception as exc:  # pragma: no cover - critical path
            self.state = ModuleState.FAILED
            self.logger.exception("Collector %s failed: %s", self.name, exc)
        finally:
            try:
                self.on_stop()
            finally:
                if self.state != ModuleState.FAILED:
                    self.state = ModuleState.STOPPED
                self.logger.info("Collector %s stopped", self.name)

    def stop(self, timeout: float = 5.0) -> None:
        if self.state not in {ModuleState.RUNNING, ModuleState.STARTING}:
            return
        self.state = ModuleState.STOPPING
        self.logger.info("Stopping collector %s", self.name)
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        if self._thread and self._thread.is_alive():
            self.logger.warning("Collector %s did not stop within timeout", self.name)
        else:
            self.state = ModuleState.STOPPED

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def on_config_update(self, options: Dict[str, Any]) -> None:
        self.options.update(options)
        self._iteration_interval = float(self.options.get("interval", self._iteration_interval))

    def on_start(self) -> None:
        """Hook executed in the worker thread right before the main loop."""

    def on_stop(self) -> None:
        """Hook executed in the worker thread after the loop exits."""

    @abstractmethod
    def run_once(self) -> None:
        """Single iteration of the collector loop."""


__all__ = ["BaseCollector"]
