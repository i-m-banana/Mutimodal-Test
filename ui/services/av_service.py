"""UI-facing audio/video helper that proxies commands to the backend or simulates data."""

from __future__ import annotations

import base64
import logging
import math
import os
import random
import threading
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSlot

try:
    from .backend_client import BackendClient, get_backend_client
except ImportError:  # pragma: no cover - fallback when imported as top-level module
    from ui.services.backend_client import BackendClient, get_backend_client  # type: ignore

try:
    from ..utils_common.ui_thread_pool import get_ui_thread_pool
except ImportError:  # pragma: no cover - fallback
    from ui.utils_common.ui_thread_pool import get_ui_thread_pool  # type: ignore


_logger = logging.getLogger("ui.remote_av")
_SIM_FRAME_SIZE = (480, 640)


@dataclass
class SessionPaths:
    audio_paths: List[str]
    video_paths: List[str]


class _RemoteAVProxy(QObject):
    """Mirrors legacy ``AVCollector`` API while delegating work to backend services or simulating data."""

    def __init__(self) -> None:
        super().__init__()
        self._client: BackendClient = get_backend_client()
        self._thread_pool = get_ui_thread_pool()
        self._connected = False
        self._frame_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._audio_level: int = 0
        self._paths = SessionPaths(audio_paths=[], video_paths=[])
        self._ensure_signal_connections()

        self._force_simulation = os.getenv("UI_FORCE_SIMULATION", "").lower() in {"1", "true", "yes"}
        self._simulate = False
        self._simulate_thread_name = "ui-av-sim"
        self._simulate_stop = threading.Event()
        self._simulate_dir: Optional[Path] = None
        self._recording_active = False
        self._segment_index = 0
        self._current_audio_path: Optional[Path] = None
        self._current_video_path: Optional[Path] = None

    # ------------------------------------------------------------------
    def _ensure_signal_connections(self) -> None:
        if self._connected:
            return
        self._client.camera_frame.connect(self._handle_camera_frame)
        self._client.audio_level.connect(self._handle_audio_level)
        self._connected = True

    @pyqtSlot(dict)
    def _handle_camera_frame(self, payload: Dict) -> None:
        if self._simulate:
            return
        data = payload.get("data")
        encoding = payload.get("encoding")
        if not data:
            _logger.warning("收到空的摄像头帧数据")
            return
        try:
            raw = base64.b64decode(data)
        except Exception as exc:  # pragma: no cover - defensive
            _logger.debug("Failed to decode frame: %s", exc)
            return
        frame: Optional[np.ndarray] = None
        if encoding == "image/jpeg":
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif encoding == "raw":
            shape = payload.get("shape")
            if shape and len(shape) == 3:
                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
                except Exception:  # pragma: no cover - fallback
                    frame = None
        if frame is None:
            _logger.warning("解码摄像头帧失败，encoding: %s", encoding)
            return
        with self._frame_lock:
            self._latest_frame = frame
            if not hasattr(self, '_frame_received_logged'):
                self._frame_received_logged = True
                _logger.info("✅ 首个摄像头帧已接收，尺寸: %s", frame.shape)

    @pyqtSlot(dict)
    def _handle_audio_level(self, payload: Dict) -> None:
        if self._simulate:
            return
        level = payload.get("level")
        if level is None:
            return
        with self._audio_lock:
            self._audio_level = int(level)

    # ------------------------------------------------------------------
    def start_collection(self, save_dir: str, *, camera_index: int = 0, video_fps: float = 30.0,
                         audio_rate: int = 8000, input_device_index: Optional[int] = None) -> None:
        self._segment_index = 0
        self._paths = SessionPaths(audio_paths=[], video_paths=[])
        self._recording_active = False
        self._simulate_dir = Path(save_dir or "recordings")
        self._stop_simulation()
        self._force_simulation = os.getenv("UI_FORCE_SIMULATION", "").strip().lower() in {"1", "true", "yes", "on"}

        if self._force_simulation:
            _logger.info("UI_FORCE_SIMULATION enabled; using synthetic preview")
            self._enable_simulation()
            return

        requested_index = camera_index
        fallback_indices = [requested_index]
        if requested_index != 0:
            fallback_indices.append(0)

        last_error: Optional[Exception] = None
        for idx in fallback_indices:
            payload = {
                "session_dir": save_dir,
                "camera_index": idx,
                "video_fps": video_fps,
                "audio_rate": audio_rate,
                "input_device_index": input_device_index,
            }
            try:
                if idx == requested_index:
                    _logger.info("Requesting backend to start preview: %s", payload)
                else:
                    _logger.warning("Primary camera index %s failed, retrying with fallback index %s", requested_index, idx)
                self._client.send_command_sync("av.start_preview", payload)
                self._paths = self._fetch_paths()
                if idx != requested_index:
                    os.environ["UI_CAMERA_INDEX_RESOLVED"] = str(idx)
                return
            except Exception as exc:
                last_error = exc
                _logger.warning("Failed to start camera index %s: %s", idx, exc)

        debug_mode = os.getenv("UI_DEBUG_MODE", "0").strip().lower() in {"1", "true", "yes", "on"}
        if self._force_simulation or debug_mode:
            _logger.warning(
                "Backend preview unavailable (last error: %s); enabling simulated preview%s",
                last_error,
                " due to debug mode" if debug_mode and not self._force_simulation else "",
            )
            self._enable_simulation()
            return

        _logger.error("Backend preview unavailable and simulation is disabled: %s", last_error)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Backend preview unavailable")

    def stop_collection(self) -> None:
        if self._simulate:
            self._stop_simulated_recording()
            self._stop_simulation()
        else:
            _logger.info("Requesting backend to stop preview")
            try:
                result = self._client.send_command_sync("av.stop_preview")
                self._update_paths(result)
            except Exception as exc:
                _logger.warning("Failed to stop preview cleanly: %s", exc)
        with self._frame_lock:
            self._latest_frame = None

    def start_recording(self) -> None:
        if self._simulate:
            self._start_simulated_recording()
            return
        _logger.info("Requesting backend to start recording")
        self._client.send_command_sync("av.start_recording")

    def stop_recording(self) -> None:
        if self._simulate:
            self._stop_simulated_recording()
            return
        _logger.info("Requesting backend to stop recording")
        result = self._client.send_command_sync("av.stop_recording")
        self._update_paths(result)

    def get_current_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_audio_level(self) -> int:
        with self._audio_lock:
            level = self._audio_level
        if self._simulate:
            return level
        if level == 0:
            try:
                response = self._client.send_command_sync("av.audio_level")
                level = int(response.get("level", level))
                with self._audio_lock:
                    self._audio_level = level
            except Exception as exc:
                _logger.debug("Audio level request failed: %s", exc)
        return level

    def get_audio_paths(self) -> List[str]:
        return list(self._paths.audio_paths)

    def get_video_paths(self) -> List[str]:
        return list(self._paths.video_paths)

    # ------------------------------------------------------------------
    def _fetch_paths(self) -> SessionPaths:
        if self._simulate:
            return self._paths
        try:
            result = self._client.send_command_sync("av.list_paths")
        except Exception as exc:
            _logger.debug("Failed to fetch session paths: %s", exc)
            result = {"audio_paths": [], "video_paths": []}
        return SessionPaths(
            audio_paths=list(result.get("audio_paths", [])),
            video_paths=list(result.get("video_paths", [])),
        )

    def _update_paths(self, result: Optional[Dict]) -> None:
        if not result:
            return
        self._paths = SessionPaths(
            audio_paths=list(result.get("audio_paths", [])),
            video_paths=list(result.get("video_paths", [])),
        )

    # ------------------------------------------------------------------
    def _enable_simulation(self) -> None:
        if self._simulate:
            return
        self._simulate_dir = self._simulate_dir or Path("recordings/simulated")
        self._simulate_dir.mkdir(parents=True, exist_ok=True)
        self._simulate = True
        self._simulate_stop.clear()
        thread = self._thread_pool.register_managed_thread(
            self._simulate_thread_name,
            self._simulation_loop,
            daemon=True
        )
        thread.start()
        with self._audio_lock:
            self._audio_level = 45
        _logger.info("AV proxy running in simulation mode (dir=%s)", self._simulate_dir)

    def _stop_simulation(self) -> None:
        if not self._simulate:
            return
        self._simulate_stop.set()
        self._thread_pool.unregister_managed_thread(self._simulate_thread_name, timeout=1.0)
        self._simulate = False
        self._simulate_stop.clear()
        self._recording_active = False
        with self._audio_lock:
            self._audio_level = 0

    def _simulation_loop(self) -> None:
        phase = random.random() * math.pi
        while not self._simulate_stop.wait(1 / 30):
            frame = self._generate_simulated_frame(phase)
            with self._frame_lock:
                self._latest_frame = frame
            level = int((math.sin(phase) + 1) * 40 + random.randint(0, 10))
            with self._audio_lock:
                self._audio_level = max(0, min(100, level))
            phase += 0.2

    def _generate_simulated_frame(self, phase: float) -> np.ndarray:
        height, width = _SIM_FRAME_SIZE
        y = np.linspace(0, 1, height, dtype=np.float32)
        x = np.linspace(0, 1, width, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        base = (np.sin(phase + xv * math.pi) + 1) * 127
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(base, 0, 255)
        frame[..., 1] = np.clip(255 - base * yv, 0, 255)
        frame[..., 2] = np.clip((xv * 255 + phase * 10) % 255, 0, 255)
        return frame

    def _start_simulated_recording(self) -> None:
        if self._recording_active:
            return
        self._segment_index += 1
        base_dir = (self._simulate_dir or Path("recordings/simulated")) / "emotion"
        base_dir.mkdir(parents=True, exist_ok=True)
        stem = base_dir / f"segment_{self._segment_index:03d}"
        self._current_audio_path = stem.with_suffix(".wav")
        self._current_video_path = stem.with_suffix(".avi")
        self._recording_active = True
        _logger.info("Simulated recording started: %s", stem.name)

    def _stop_simulated_recording(self) -> None:
        if not self._recording_active:
            return
        try:
            self._write_silent_wav(self._current_audio_path)
            self._write_dummy_video(self._current_video_path)
        finally:
            self._recording_active = False
            if self._current_audio_path:
                self._paths.audio_paths.append(str(self._current_audio_path))
            if self._current_video_path:
                self._paths.video_paths.append(str(self._current_video_path))
            _logger.info("Simulated recording finished")
            self._current_audio_path = None
            self._current_video_path = None

    def _write_silent_wav(self, path: Optional[Path]) -> None:
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                frames = (b"\x00\x00" * 16000)  # 1s of silence
                wf.writeframes(frames)
        except Exception as exc:
            _logger.debug("Failed to write silent wav: %s", exc)
            path.touch()

    def _write_dummy_video(self, path: Optional[Path]) -> None:
        if not path:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(path), fourcc, 10.0, (_SIM_FRAME_SIZE[1], _SIM_FRAME_SIZE[0]))
            if writer.isOpened():
                frame = self._generate_simulated_frame(random.random() * math.pi)
                for _ in range(20):
                    writer.write(frame)
                writer.release()
            else:
                path.touch()
        except Exception as exc:
            _logger.debug("Failed to write dummy video: %s", exc)
            path.touch()


_proxy = _RemoteAVProxy()


def start_collection(save_dir: str, camera_index: int = 0, video_fps: float = 30.0,
                     audio_rate: int = 8000, input_device_index: Optional[int] = None):
    """Start remote or simulated preview."""
    _proxy.start_collection(
        save_dir,
        camera_index=camera_index,
        video_fps=video_fps,
        audio_rate=audio_rate,
        input_device_index=input_device_index,
    )
    return _proxy


def stop_collection() -> None:
    """Stop preview and release resources."""
    _proxy.stop_collection()


def start_recording() -> None:
    """Start audio/video recording."""
    _proxy.start_recording()


def stop_recording() -> None:
    """Stop recording and refresh cached media paths."""
    _proxy.stop_recording()


def get_current_frame() -> Optional[np.ndarray]:
    """Return the latest frame."""
    return _proxy.get_current_frame()


def get_audio_paths() -> List[str]:
    """Return audio file paths for current session."""
    return _proxy.get_audio_paths()


def get_video_paths() -> List[str]:
    """Return video file paths for current session."""
    return _proxy.get_video_paths()


def get_current_audio_level() -> int:
    """Return current audio level (simulated or backend)."""
    return _proxy.get_audio_level()


__all__ = [
    "start_collection",
    "stop_collection",
    "start_recording",
    "stop_recording",
    "get_current_frame",
    "get_audio_paths",
    "get_video_paths",
    "get_current_audio_level",
]
