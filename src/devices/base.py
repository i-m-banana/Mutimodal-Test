"""Base classes for hardware devices."""

from __future__ import annotations

import time
from typing import Any


class DeviceException(Exception):
    """Raised when a device either fails to start or stops unexpectedly."""


class DeviceBase:
    """Simple synchronous wrapper around blocking hardware SDKs."""

    def __init__(self) -> None:
        self.running = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def __next__(self):
        if self.running:
            return self.read()
        raise StopIteration

    def __iter__(self):
        return self

    async def __anext__(self):
        if self.running:
            return self.read()
        raise StopAsyncIteration

    def __aiter__(self):
        return self

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.on_start()

    def stop(self) -> None:
        if self.running:
            self.running = False
            time.sleep(0.1)
            self.on_done()

    def read(self) -> tuple[bool, Any]:
        raise NotImplementedError

    def on_start(self) -> None:
        raise NotImplementedError

    def on_done(self) -> None:
        raise NotImplementedError
