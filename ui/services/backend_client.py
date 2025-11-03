"""Client wrapper for bidirectional backend communication via WebSocket."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

try:  # pragma: no cover - optional dependency at import time
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

try:
    from ..utils_common.ui_thread_pool import get_ui_thread_pool
except ImportError:  # pragma: no cover - fallback
    from ui.utils_common.ui_thread_pool import get_ui_thread_pool  # type: ignore


@dataclass
class BackendEvent:
    topic: str
    timestamp: float
    payload: Dict[str, Any]


class BackendClient(QObject):
    detection_result = pyqtSignal(dict)
    system_heartbeat = pyqtSignal(dict)
    camera_frame = pyqtSignal(dict)
    multimodal_frame = pyqtSignal(dict)
    audio_level = pyqtSignal(dict)
    raw_event = pyqtSignal(object)
    connection_state_changed = pyqtSignal(bool)
    command_failed = pyqtSignal(str, str)

    def __init__(self, url: str = "ws://127.0.0.1:8765", parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.url = url
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread_pool = get_ui_thread_pool()
        self._thread_name = "ui-backend-ws-client"
        self._running = False
        self._socket = None
        self._client_token: Optional[str] = None
        self._pending_commands: Dict[str, Tuple[concurrent.futures.Future, str]] = {}
        self._logger = logging.getLogger("ui.backend_client")

    # Lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        if websockets is None:
            self._logger.error("websockets package not available; backend events disabled")
            return
        if self._running:
            return
        
        try:
            self._running = True
            self._loop = asyncio.new_event_loop()
            thread = self._thread_pool.register_managed_thread(
                self._thread_name,
                self._run_loop,
                daemon=True
            )
            thread.start()
            asyncio.run_coroutine_threadsafe(self._connect(), self._loop)
        except Exception as e:
            # 如果启动失败(例如线程名冲突),重置状态以便下次重试
            self._logger.error(f"启动后端客户端失败: {e}")
            self._running = False
            self._loop = None
            raise

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop:
            asyncio.run_coroutine_threadsafe(self._close(), self._loop)
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread_pool.unregister_managed_thread(self._thread_name, timeout=5.0)
        self._loop = None

    def ensure_started(self) -> None:
        """Ensure the backend client is started. 
        
        If the client failed to start previously (e.g. due to thread name conflict),
        this will attempt to clean up and restart.
        """
        if not self._running:
            # 清理可能存在的僵尸线程
            try:
                self._thread_pool.unregister_managed_thread(self._thread_name, timeout=0.1)
            except Exception:
                pass  # 忽略清理错误
            self.start()
    
    def is_connected(self) -> bool:
        """Check if client is connected to backend."""
        return self._socket is not None and self._client_token is not None
    
    def wait_for_connection(self, timeout: float = 10.0) -> bool:
        """Wait for backend connection to be established.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if connected within timeout, False otherwise
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected():
                return True
            time.sleep(0.1)
        return False

    # Internal async loop --------------------------------------------------------
    def _run_loop(self) -> None:
        assert self._loop
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _connect(self) -> None:
        retry_delay = 1.0
        while self._running and self._loop:
            try:
                async with websockets.connect(self.url) as ws:  # type: ignore[attr-defined]
                    self._socket = ws
                    self.connection_state_changed.emit(True)
                    retry_delay = 1.0
                    await self._listen(ws)
            except Exception as exc:  # pragma: no cover - network errors
                self._socket = None
                self.connection_state_changed.emit(False)
                self._logger.warning("Backend connection lost: %s", exc)
                self._fail_pending(exc)
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 30.0)
        self.connection_state_changed.emit(False)

    async def _listen(self, websocket) -> None:
        async for message in websocket:
            await self._handle_message(message)

    async def _handle_message(self, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            self._logger.warning("Invalid message from backend: %s", message)
            return

        msg_type = data.get("type")
        if msg_type == "welcome":
            self._client_token = data.get("client_token")
            self._logger.debug("Assigned client token %s", self._client_token)
            return
        if msg_type == "event":
            topic = data.get("topic", "")
            payload = data.get("payload", {})
            self.raw_event.emit(data)
            if topic == "detector.result":
                self.detection_result.emit(payload)
            elif topic == "system.heartbeat":
                self.system_heartbeat.emit(payload)
            elif topic == "camera.frame":
                self.camera_frame.emit(payload)
            elif topic == "audio.level":
                self.audio_level.emit(payload)
            elif topic == "multimodal.frame":
                self.multimodal_frame.emit(payload)
            return
        if msg_type == "response":
            request_id = data.get("id")
            entry = self._pending_commands.pop(request_id, None)
            if not entry:
                self._logger.debug("No pending future for response id %s", request_id)
                return
            future, action = entry
            status = data.get("status", "ok")
            action_name = data.get("action") or action
            result = data.get("result")
            if status == "ok":
                future.set_result(result or {})
            else:
                if isinstance(result, dict):
                    message = result.get("error") or str(result)
                elif result is None:
                    message = "command failed"
                else:
                    message = str(result)
                if action_name:
                    try:
                        self.command_failed.emit(action_name, message)
                    except Exception:
                        self._logger.exception("Failed to emit command_failed for action %s", action_name)
                future.set_exception(RuntimeError(message))
            return
        self._logger.debug("Unhandled message from backend: %s", data)

    async def _close(self) -> None:
        self._fail_pending(RuntimeError("backend connection closed"))
        if not self._loop:
            return
        for task in asyncio.all_tasks(loop=self._loop):
            task.cancel()
        await asyncio.sleep(0)

    # Command handling -----------------------------------------------------------
    def send_command_future(self, action: str, payload: Optional[Dict[str, Any]] = None) -> concurrent.futures.Future:
        payload = payload or {}
        future: concurrent.futures.Future = concurrent.futures.Future()
        if not self._loop or not self._running:
            future.set_exception(RuntimeError("Backend client not running"))
            return future
        if self._socket is None:
            future.set_exception(RuntimeError("Backend connection not established"))
            return future
        request_id = uuid.uuid4().hex
        self._pending_commands[request_id] = (future, action)
        coro = self._send_command_async(request_id, action, payload)
        send_future = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _handle_send_completion(send_future: "asyncio.Future") -> None:
            exc = send_future.exception()
            if exc is None:
                return
            entry = self._pending_commands.pop(request_id, None)
            if entry:
                pending, pending_action = entry
                if not pending.done():
                    pending.set_exception(exc)

        send_future.add_done_callback(_handle_send_completion)
        return future

    def send_command_sync(self, action: str, payload: Optional[Dict[str, Any]] = None,
                          timeout: float | None = 10.0) -> Dict[str, Any]:
        future = self.send_command_future(action, payload)
        result = future.result(timeout=timeout)
        return result or {}

    async def _send_command_async(self, request_id: str, action: str, payload: Dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("No active websocket connection")
        message = json.dumps({
            "type": "command",
            "id": request_id,
            "action": action,
            "payload": payload,
        })
        await self._socket.send(message)

    def _fail_pending(self, exc: Exception) -> None:
        for request_id, (future, action) in list(self._pending_commands.items()):
            if not future.done():
                future.set_exception(exc)
        self._pending_commands.clear()


# Singleton helper --------------------------------------------------------------
_shared_client: Optional[BackendClient] = None


def get_backend_client() -> BackendClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = BackendClient()
    _shared_client.ensure_started()
    return _shared_client


__all__ = ["BackendClient", "BackendEvent", "get_backend_client"]
