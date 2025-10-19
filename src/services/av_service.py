"""Server-side audio/video manager invoked via UI command router."""

from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:  # pragma: no cover - optional dependency
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None

try:  # pragma: no cover - optional dependency
    import pyaudio  # type: ignore
except Exception:  # pragma: no cover
    pyaudio = None

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..core.thread_pool import get_thread_pool


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class AVSessionResult:
    audio_paths: list[str]
    video_paths: list[str]


class AVService:
    """Replicates legacy AVCollector behaviour but runs inside backend process."""

    def __init__(self, bus: EventBus, logger) -> None:
        self.bus = bus
        self.logger = logger
        self._thread_pool = get_thread_pool()
        self._preview_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._audio_lock = threading.Lock()
        self._audio_level_lock = threading.Lock()

        self._session_dir: Optional[str] = None
        self._camera_index = 0
        self._video_fps = 30.0
        self._audio_rate = 8000
        self._audio_device_index: Optional[int] = None

        self._cap = None
        self._latest_frame: Optional[Tuple[float, Any]] = None
        self._grab_thread_name = "av-grab"
        self._grab_running = False

        self._writer_thread_name = "av-writer"
        self._writer_running = False
        self._video_writer = None
        self._video_size: Optional[Tuple[int, int]] = None
        self._segment_index = 0
        self._is_recording = False
        self._audio_chunks: list[bytes] = []
        self._audio_paths: list[str] = []
        self._video_paths: list[str] = []
        self._video_filepath: Optional[str] = None
        self._audio_filepath: Optional[str] = None

        self._pa = pyaudio.PyAudio() if pyaudio else None
        self._audio_stream = None
        self._audio_level = 0

        self._stop_event = threading.Event()
        self._synthetic_mode = False
        self._allow_synthetic = _env_flag("BACKEND_ALLOW_SYNTHETIC_CAMERA", True) or _env_flag("UI_FORCE_SIMULATION", False)

    # ------------------------------------------------------------------
    def start_preview(self, *, session_dir: str, camera_index: int = 0,
                      video_fps: float = 30.0, audio_rate: int = 8000,
                      input_device_index: Optional[int] = None) -> None:
        with self._preview_lock:
            self._stop_event.clear()
            self._session_dir = session_dir
            os.makedirs(self._session_dir, exist_ok=True)
            self._camera_index = camera_index
            self._video_fps = float(video_fps) if video_fps > 0 else 30.0
            self._audio_rate = int(audio_rate) if audio_rate > 0 else 8000
            self._audio_device_index = input_device_index

            # 每次开启预览时重置录制状态，防止跨会话遗留的段落序号和路径
            self._segment_index = 0
            self._audio_paths = []
            self._video_paths = []
            self._audio_chunks = []
            self._audio_filepath = None
            self._video_filepath = None
            self._is_recording = False

            try:
                self._open_camera()
                self._synthetic_mode = False
            except RuntimeError as exc:
                if self._allow_synthetic:
                    self.logger.warning("Camera unavailable (%s)，启用模拟视频帧", exc)
                    self._enable_synthetic_camera()
                else:
                    raise
            self._start_grab_thread()
            self._start_realtime_audio()

    def stop_preview(self) -> None:
        with self._preview_lock:
            self._stop_recording_locked()
            self._stop_grab_thread()
            self._stop_realtime_audio()
            self._release_camera()
            self._session_dir = None

    def start_recording(self) -> Dict[str, Any]:
        with self._preview_lock:
            if self._is_recording:
                return {"status": "already-recording"}
            if self._cap is None:
                raise RuntimeError("Preview not started")
            if not self._session_dir:
                raise RuntimeError("Session directory missing")

            self._segment_index += 1
            base = os.path.join(self._session_dir, "emotion")
            os.makedirs(base, exist_ok=True)
            base = os.path.join(base, f"{self._segment_index}")
            self._audio_filepath = base + ".wav"
            self._video_filepath = base + ".avi"

            fourcc = cv2.VideoWriter_fourcc(*"XVID") if cv2 else None
            if fourcc is None:
                raise RuntimeError("OpenCV not available for recording")
            width, height = self._video_size or (640, 480)
            self._video_writer = cv2.VideoWriter(self._video_filepath, fourcc, self._video_fps, (width, height))
            if self._video_writer is None or not self._video_writer.isOpened():
                self._video_writer = None
                raise RuntimeError("Video writer failed to open")

            with self._audio_lock:
                self._audio_chunks = []

            if not self._writer_running:
                self._writer_running = True
                thread = self._thread_pool.register_managed_thread(
                    self._writer_thread_name,
                    self._writer_loop,
                    daemon=True
                )
                thread.start()

            self._is_recording = True
            return {"status": "recording"}

    def stop_recording(self) -> AVSessionResult:
        with self._preview_lock:
            return self._stop_recording_locked()

    def current_audio_level(self) -> int:
        with self._audio_level_lock:
            return self._audio_level

    def session_paths(self) -> AVSessionResult:
        return AVSessionResult(audio_paths=list(self._audio_paths), video_paths=list(self._video_paths))

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "opencv_available": cv2 is not None,
            "numpy_available": np is not None,
            "pyaudio_available": pyaudio is not None,
            "allow_synthetic": self._allow_synthetic,
            "synthetic_mode": self._synthetic_mode,
        }

    def shutdown(self) -> None:
        self._stop_event.set()
        self.stop_preview()
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:  # pragma: no cover - best effort cleanup
                pass

    # ------------------------------------------------------------------
    def _open_camera(self) -> None:
        if cv2 is None:
            self.logger.warning("OpenCV unavailable; AVService will emit synthetic frames")
            self._cap = None
            return
        if self._cap is None:
            self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("Cannot open camera index %s" % self._camera_index)
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self._video_size = (width, height)

    def _release_camera(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._synthetic_mode = False

    def _start_grab_thread(self) -> None:
        if self._grab_running:
            return
        self._grab_running = True
        thread = self._thread_pool.register_managed_thread(
            self._grab_thread_name,
            self._grab_loop,
            daemon=True
        )
        thread.start()

    def _stop_grab_thread(self) -> None:
        self._grab_running = False
        self._thread_pool.unregister_managed_thread(self._grab_thread_name, timeout=2.0)

    def _grab_loop(self) -> None:
        interval = 1.0 / max(5.0, min(self._video_fps, 30.0))
        while self._grab_running and not self._stop_event.is_set():
            frame = self._read_frame()
            if frame is not None:
                timestamp = time.time()
                with self._frame_lock:
                    self._latest_frame = (timestamp, frame)
                self._publish_frame(frame, timestamp)
            time.sleep(interval)

    def _read_frame(self) -> Optional[Any]:
        if cv2 is None:
            return self._generate_synthetic_frame()
        if self._cap is None:
            if self._synthetic_mode and self._allow_synthetic:
                return self._generate_synthetic_frame()
            return None
        ret, frame = self._cap.read()
        if not ret:
            if self._synthetic_mode and self._allow_synthetic:
                return self._generate_synthetic_frame()
            return None
        frame = cv2.flip(frame, 1)
        return frame

    def _enable_synthetic_camera(self) -> None:
        self._synthetic_mode = True
        self._cap = None
        if self._video_size is None:
            self._video_size = (320, 240)

    def _generate_synthetic_frame(self):  # pragma: no cover - fallback path
        width, height = self._video_size or (320, 240)
        import numpy as np_local  # lazy import to avoid dependency when unused

        idx = int(time.time() * 10) % 255
        frame = np_local.full((height, width, 3), idx, dtype=np_local.uint8)
        return frame

    def _publish_frame(self, frame, timestamp: float) -> None:
        if cv2 is None or np is None:
            encoded = base64.b64encode(frame.tobytes()).decode("ascii")
            payload = {
                "source": "av_service",
                "encoding": "raw",
                "timestamp": timestamp,
                "collector": "av_service",
                "data": encoded,
                "shape": frame.shape,
            }
        else:
            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                return
            encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
            payload = {
                "source": "av_service",
                "encoding": "image/jpeg",
                "timestamp": timestamp,
                "collector": "av_service",
                "data": encoded,
            }
        self.bus.publish(Event(EventTopic.CAMERA_FRAME, payload))

    def _start_realtime_audio(self) -> None:
        if self._audio_stream is not None or self._pa is None or pyaudio is None:
            return
        try:
            self._audio_stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._audio_rate,
                input=True,
                frames_per_buffer=1024,
                input_device_index=self._audio_device_index,
                stream_callback=self._audio_callback,
            )
            self._audio_stream.start_stream()
        except Exception as exc:  # pragma: no cover - hardware path
            self.logger.warning("Failed to start audio stream: %s", exc)
            self._audio_stream = None

    def _stop_realtime_audio(self) -> None:
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None

    def _audio_callback(self, in_data, frame_count, time_info, status):  # pragma: no cover - hardware path
        import struct

        count = len(in_data) // 2
        if count:
            shorts = struct.unpack(f"{count}h", in_data[:count * 2])
            rms = (sum(n ** 2 for n in shorts) / count) ** 0.5 if count > 0 else 0
            level = min(100, int(rms / 30))
        else:
            level = 0
        with self._audio_level_lock:
            self._audio_level = level
        self.bus.publish(Event(EventTopic.AUDIO_LEVEL, {
            "source": "av_service",
            "level": level,
            "timestamp": time.time(),
        }))
        if self._is_recording:
            with self._audio_lock:
                self._audio_chunks.append(in_data)
        return (None, pyaudio.paContinue if pyaudio else None)

    def _writer_loop(self) -> None:
        frame_interval = 1.0 / max(1.0, float(self._video_fps))
        next_ts = time.monotonic()
        while self._writer_running and not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_ts:
                time.sleep(min(frame_interval, next_ts - now))
                continue
            with self._frame_lock:
                frame = self._latest_frame[1] if self._latest_frame else None
            if frame is not None and self._video_writer is not None:
                try:
                    self._video_writer.write(frame)
                except Exception as exc:
                    self.logger.debug("Video writer error: %s", exc)
            next_ts += frame_interval
        self.logger.debug("Video writer thread exiting")

    def _stop_recording_locked(self) -> AVSessionResult:
        if not self._is_recording:
            return self.session_paths()

        self._writer_running = False
        self._thread_pool.unregister_managed_thread(self._writer_thread_name, timeout=2.0)

        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

        self._flush_wav()

        if self._audio_filepath and os.path.exists(self._audio_filepath):
            self._audio_paths.append(self._audio_filepath)
        if self._video_filepath and os.path.exists(self._video_filepath):
            self._video_paths.append(self._video_filepath)

        self._audio_filepath = None
        self._video_filepath = None
        self._is_recording = False
        return self.session_paths()

    def _flush_wav(self) -> None:
        if not self._audio_filepath or not self._audio_chunks:
            return
        try:
            import wave

            os.makedirs(os.path.dirname(self._audio_filepath), exist_ok=True)
            with wave.open(self._audio_filepath, "wb") as wf:
                wf.setnchannels(1)
                if self._pa and pyaudio:
                    wf.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
                else:
                    wf.setsampwidth(2)
                wf.setframerate(self._audio_rate)
                with self._audio_lock:
                    wf.writeframes(b"".join(self._audio_chunks))
        except Exception as exc:
            self.logger.warning("Failed to flush wav: %s", exc)


__all__ = ["AVService", "AVSessionResult"]
