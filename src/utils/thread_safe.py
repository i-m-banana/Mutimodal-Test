"""Thread-safe utility classes used across the orchestration layer."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Generic, Iterable, Iterator, MutableMapping, Optional, TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeQueue(Generic[T]):
    """Small wrapper around :class:`queue.Queue` adding `__iter__` semantics."""

    def __init__(self, maxsize: int = 0):
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        self._queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        return self._queue.get(block=block, timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    def __len__(self) -> int:  # pragma: no cover - queue doesn't expose length reliably
        return self._queue.qsize()

    def __iter__(self) -> Iterator[T]:
        while True:
            yield self._queue.get()


@dataclass
class ThreadSafeCounter:
    value: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def increment(self, step: int = 1) -> int:
        with self._lock:
            self.value += step
            return self.value

    def reset(self, value: int = 0) -> None:
        with self._lock:
            self.value = value


class ThreadSafeDict(MutableMapping[K, V]):
    def __init__(self) -> None:
        self._data: dict[K, V] = {}
        self._lock = threading.RLock()

    def __getitem__(self, key: K) -> V:
        with self._lock:
            return self._data[key]

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = value

    def __delitem__(self, key: K) -> None:
        with self._lock:
            del self._data[key]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._data.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def items(self) -> Iterable[tuple[K, V]]:
        with self._lock:
            return list(self._data.items())


__all__ = ["ThreadSafeQueue", "ThreadSafeCounter", "ThreadSafeDict"]
