"""Publish/subscribe event bus for decoupled communication between modules."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, DefaultDict, Dict, Iterable, List

from ..constants import EventTopic


@dataclass(frozen=True)
class Event:
    topic: EventTopic
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


Subscriber = Callable[[Event], None]


class EventBus:
    """Thread-safe event bus supporting publish/subscribe semantics."""

    def __init__(self) -> None:
        self._subscribers: DefaultDict[EventTopic, List[Subscriber]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: EventTopic, callback: Subscriber) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: EventTopic, callback: Subscriber) -> None:
        with self._lock:
            listeners = self._subscribers.get(topic, [])
            if callback in listeners:
                listeners.remove(callback)

    def publish(self, event: Event) -> None:
        with self._lock:
            listeners = list(self._subscribers.get(event.topic, []))
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Avoid a single listener breaking the bus. Real logging is handled upstream.
                continue

    def topics(self) -> Iterable[EventTopic]:
        with self._lock:
            return list(self._subscribers.keys())


__all__ = ["Event", "EventBus"]
