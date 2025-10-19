"""Backend EEG device abstractions built on top of the BLE SDK."""

from __future__ import annotations

from typing import Callable

from .base import DeviceException

try:  # Optional dependency installed with the EEG hardware SDK
    from bleak import BleakClient  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency is optional at import time
    BleakClient = None  # type: ignore
    _BLE_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - runtime branch when bleak is available
    _BLE_IMPORT_ERROR = None

# Public flag to let services know whether native hardware access is possible.
HAS_EEG_HARDWARE = BleakClient is not None

# Default BLE configuration for the current EEG headband model.
DEFAULT_DEVICE_ADDRESS = "F4:3C:7C:A6:29:E0"
TX_CHARACTERISTIC = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
RX_CHARACTERISTIC = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"


NotificationCallback = Callable[[int, bytearray], None]


class BleEEGDevice:
    """Thin asynchronous wrapper around the vendor BLE GATT interface."""

    def __init__(
        self,
        *,
        address: str = DEFAULT_DEVICE_ADDRESS,
        tx_characteristic: str = TX_CHARACTERISTIC,
        rx_characteristic: str = RX_CHARACTERISTIC,
    ) -> None:
        if BleakClient is None:
            raise DeviceException("Bleak (BLE SDK) is not available") from _BLE_IMPORT_ERROR
        self._client = BleakClient(address)
        self._address = address
        self._tx_char = tx_characteristic
        self._rx_char = rx_characteristic
        self._connected = False

    @property
    def address(self) -> str:
        """Return the BLE address currently targeted by this device."""
        return self._address

    async def connect(self, callback: NotificationCallback) -> None:
        """Connect to the device and start streaming notifications."""
        await self._client.connect()
        await self._client.start_notify(self._tx_char, callback)
        self._connected = True

    async def disconnect(self) -> None:
        """Gracefully disconnect from the device."""
        if not self._connected:
            return
        try:
            await self._client.stop_notify(self._tx_char)
        except Exception:
            pass
        try:
            await self._client.disconnect()
        finally:
            self._connected = False

    async def write(self, payload: bytes) -> None:
        """Send control payloads to the device if supported by the firmware."""
        await self._client.write_gatt_char(self._rx_char, payload, response=False)

    async def ensure_disconnected(self) -> None:
        """Best-effort cleanup for shutdown logic."""
        try:
            await self.disconnect()
        except Exception:
            pass


__all__ = [
    "BleEEGDevice",
    "HAS_EEG_HARDWARE",
    "DEFAULT_DEVICE_ADDRESS",
    "TX_CHARACTERISTIC",
    "RX_CHARACTERISTIC",
]
