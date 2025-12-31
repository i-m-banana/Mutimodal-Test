"""Hardware device wrappers used by backend services."""

from .base import DeviceBase, DeviceException
from .eeg import BleEEGDevice, HAS_EEG_HARDWARE
from .maibobo import (
    MaiboboDevice,
    HAS_MAIBOBO_DRIVER,
    DEFAULT_PORT,
    detect_available_port,
    parse_frame,
)
from .tobii import HAS_TOBII, TobiiDevice

__all__ = [
    "DeviceBase",
    "DeviceException",
    "BleEEGDevice",
    "HAS_EEG_HARDWARE",
    "MaiboboDevice",
    "HAS_MAIBOBO_DRIVER",
    "DEFAULT_PORT",
    "detect_available_port",
    "parse_frame",
    "TobiiDevice",
    "HAS_TOBII",
]
