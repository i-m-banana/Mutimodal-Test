"""Async WebSocket interface for pushing bus events to UI clients."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional, Set

if TYPE_CHECKING:  # pragma: no cover
    from websockets.server import WebSocketServerProtocol
else:  # pragma: no cover
    WebSocketServerProtocol = Any  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency at import time
    from websockets.server import serve  # type: ignore[import]
    from websockets.exceptions import ConnectionClosed  # type: ignore[import]
except ImportError:  # pragma: no cover
    serve = None  # type: ignore[assignment]
    ConnectionClosed = Exception

from ..constants import EventTopic
from ..core.event_bus import Event
from .base import BaseInterface


class WebsocketPushInterface(BaseInterface):
    """Broadcasts selected event topics to connected WebSocket clients."""

    def __init__(self, name: str, bus, *, options: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None) -> None:
        super().__init__(name, bus, options=options, logger=logger)
        cfg = self.options
        self.host: str = str(cfg.get("host", "127.0.0.1"))
        self.port: int = int(cfg.get("port", 8765))
        topics: Iterable[str] = cfg.get(
            "topics",
            [
                EventTopic.DETECTION_RESULT.value,
                EventTopic.SYSTEM_HEARTBEAT.value,
                EventTopic.MULTIMODAL_SNAPSHOT.value,
            ],
        )
        self.topics = tuple(EventTopic(topic) for topic in topics)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._clients: Set[WebSocketServerProtocol] = set()
        self._subscriptions: Dict[EventTopic, Any] = defaultdict(list)
        self._client_tokens: Dict[WebSocketServerProtocol, str] = {}
        self._cache_lock = threading.Lock()
        self._last_payloads: Dict[str, Dict[str, Any]] = {}

    def start(self) -> None:
        if self._running:
            return
        if serve is None:
            self.logger.error("websockets package not installed; interface disabled")
            return
        self.logger.info("Starting WebSocket interface on %s:%s", self.host, self.port)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name=f"ws-{self.name}", daemon=True)
        self._thread.start()
        start_future = asyncio.run_coroutine_threadsafe(self._start_server(), self._loop)
        start_future.result()
        for topic in self.topics:
            callback = self._make_event_handler(topic)
            self.bus.subscribe(topic, callback)
            self._subscriptions[topic].append(callback)
        response_callback = self._handle_ui_response
        self.bus.subscribe(EventTopic.UI_RESPONSE, response_callback)
        self._subscriptions[EventTopic.UI_RESPONSE].append(response_callback)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self.logger.info("Stopping WebSocket interface")
        for topic, callbacks in self._subscriptions.items():
            for callback in callbacks:
                self.bus.unsubscribe(topic, callback)
        self._subscriptions.clear()
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._stop_server(), self._loop).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._server = None
        self._clients.clear()
        self._client_tokens.clear()
        self._running = False

    def _run_loop(self) -> None:
        assert self._loop is not None
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _start_server(self) -> None:
        assert self._loop is not None
        self._server = await serve(self._client_handler, self.host, self.port)

    async def _stop_server(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _client_handler(self, websocket: WebSocketServerProtocol) -> None:
        self.logger.info("Client connected: %s", websocket.remote_address)
        self._clients.add(websocket)
        token = uuid.uuid4().hex
        self._client_tokens[websocket] = token
        if self._loop:
            await websocket.send(json.dumps({"type": "welcome", "client_token": token}))
            await self._send_cached_payloads(websocket)
        try:
            async for message in websocket:
                await self._handle_client_message(message, websocket)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            self._client_tokens.pop(websocket, None)
            self.logger.info("Client disconnected: %s", websocket.remote_address)

    async def _handle_client_message(self, message: str, websocket: WebSocketServerProtocol) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            self.logger.warning("Received invalid JSON from client: %s", message)
            return
        msg_type = data.get("type")
        if msg_type == "ping":
            await self._broadcast({"type": "pong"})
            return
        if msg_type == "command":
            action = data.get("action")
            request_id = data.get("id") or uuid.uuid4().hex
            body = data.get("payload") or {}
            self.logger.info("Received command from client: %s", action)
            self._emit_command(action, body, request_id, websocket)
            return
        self.logger.debug("Unhandled client message: %s", data)

    def _make_event_handler(self, topic: EventTopic):
        def handler(event: Event) -> None:
            payload = {
                "type": "event",
                "topic": event.topic.value,
                "timestamp": event.timestamp,
                "payload": event.payload,
            }
            with self._cache_lock:
                self._last_payloads[event.topic.value] = payload
            self._queue_broadcast(payload)
        return handler

    def _queue_broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self._loop)

    async def _broadcast(self, payload: Dict[str, Any]) -> None:
        if not self._clients:
            return
        message = json.dumps(payload, default=self._json_default)
        coros = [client.send(message) for client in list(self._clients) if not client.closed]
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    def _emit_command(self, action: str, body: Dict[str, Any], request_id: str,
                      websocket: WebSocketServerProtocol) -> None:
        client_token = self._client_tokens.get(websocket)
        payload = {
            "id": request_id,
            "action": action,
            "payload": body,
            "client_token": client_token,
        }
        self.bus.publish(Event(EventTopic.UI_COMMAND, payload))

    def _handle_ui_response(self, event: Event) -> None:
        payload = event.payload or {}
        message = {
            "type": "response",
            "id": payload.get("id"),
            "status": payload.get("status", "ok"),
            "result": payload.get("result"),
        }
        client_token = payload.get("client_token")
        if not self._loop:
            return
        targets = [client for client, token in self._client_tokens.items()
                   if client_token is None or token == client_token]
        if not targets:
            return
        asyncio.run_coroutine_threadsafe(self._send_to_targets(targets, message), self._loop)

    async def _send_to_targets(self, targets: Iterable[WebSocketServerProtocol], payload: Dict[str, Any]) -> None:
        serialized = json.dumps(payload, default=self._json_default)
        coros = [client.send(serialized) for client in targets if not client.closed]
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    async def _send_cached_payloads(self, websocket: WebSocketServerProtocol) -> None:
        with self._cache_lock:
            cached = list(self._last_payloads.values())
        if not cached:
            return
        for payload in cached:
            try:
                await websocket.send(json.dumps(payload, default=self._json_default))
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.debug("Failed to replay cached payload: %s", exc)
                break

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "__dict__"):
            return value.__dict__
        try:
            return asdict(value)  # type: ignore[arg-type]
        except Exception:
            return str(value)


__all__ = ["WebsocketPushInterface"]
