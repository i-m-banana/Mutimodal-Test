"""Base classes for external interfaces."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..core.event_bus import EventBus


class BaseInterface(ABC):
    """Common lifecycle contract for backend interfaces."""

    def __init__(self, name: str, bus: EventBus, *, options: Dict[str, Any] | None = None,
                 logger: logging.Logger | None = None) -> None:
        self.name = name
        self.bus = bus
        self.options = options or {}
        self.logger = logger or logging.getLogger(f"interface.{name}")
        self._running = False

    @abstractmethod
    def start(self) -> None:
        """Start the interface."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the interface."""

    @property
    def running(self) -> bool:
        return self._running
