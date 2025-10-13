"""External interfaces for orchestrator integrations."""

from .base import BaseInterface  # noqa: F401

try:  # pragma: no cover - optional interface dependency
	from .websocket_server import WebsocketPushInterface  # noqa: F401
except Exception:  # pragma: no cover - keep imports optional
	WebsocketPushInterface = None  # type: ignore

__all__ = ["BaseInterface", "WebsocketPushInterface"]
