"""Base classes for hardware devices."""

from __future__ import annotations

import time
from typing import Any, Generic, TypeVar


DeviceT_co = TypeVar("DeviceT_co", bound="DeviceBase", covariant=True)


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


class MultiDevice(DeviceBase, Generic[DeviceT_co]):
    """Container that aggregates multiple devices into one logical source."""

    def __init__(self, devices: list[DeviceT_co]) -> None:
        super().__init__()
        self.devices = devices

    def read(self) -> tuple[bool, Any]:
        ret, frame = True, []
        for device in self.devices:
            device_ok, device_frame = device.read()
            ret = ret and device_ok
            frame.append(device_frame)
        return ret, frame

    def start(self) -> None:
        self.running = True
        for device in self.devices:
            device.start()

    def stop(self) -> None:
        self.running = False
        for device in self.devices:
            device.stop()

    def on_start(self) -> None:
        for device in self.devices:
            device.on_start()

    def on_done(self) -> None:
        for device in self.devices:
            device.on_done()
