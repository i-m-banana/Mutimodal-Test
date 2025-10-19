"""Base detector receiving events from the bus."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Optional

from ..constants import EventTopic, ModuleState
from ..core.event_bus import Event, EventBus
from ..models.model_manager import ModelManager


class BaseDetector(ABC):
    def __init__(self, name: str, bus: EventBus, model_manager: ModelManager, *, options: Optional[dict] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        self.name = name
        self.bus = bus
        self.model_manager = model_manager
        self.options = options or {}
        self.logger = logger or logging.getLogger("detector")
        self.state = ModuleState.CREATED
        self._subscriptions: list[tuple[EventTopic, Callable[[Event], None]]] = []

    def start(self) -> None:
        if self.state == ModuleState.RUNNING:
            return
        self.state = ModuleState.STARTING
        self.logger.info("Starting detector %s", self.name)
        for topic in self.topics():
            callback = self._create_callback()
            self.bus.subscribe(topic, callback)
            self._subscriptions.append((topic, callback))
        self.on_start()
        self.state = ModuleState.RUNNING

    def stop(self) -> None:
        if self.state not in {ModuleState.RUNNING, ModuleState.STARTING}:
            return
        self.state = ModuleState.STOPPING
        for topic, callback in self._subscriptions:
            self.bus.unsubscribe(topic, callback)
        self._subscriptions.clear()
        self.on_stop()
        self.state = ModuleState.STOPPED
        self.logger.info("Detector %s stopped", self.name)

    def _create_callback(self) -> Callable[[Event], None]:
        def _callback(event: Event) -> None:
            try:
                self.handle_event(event)
            except Exception as exc:  # pragma: no cover - runtime guard
                self.logger.exception("Detector %s failed processing event: %s", self.name, exc)
        return _callback

    @abstractmethod
    def topics(self) -> Iterable[EventTopic]:
        ...

    def on_start(self) -> None:
        """Optional start hook."""

    def on_stop(self) -> None:
        """Optional stop hook."""

    @abstractmethod
    def handle_event(self, event: Event) -> None:
        ...


__all__ = ["BaseDetector"]
