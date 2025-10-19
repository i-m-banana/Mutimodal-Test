"""Backend-managed multimodal data collection service."""

from __future__ import annotations

import base64
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

try:  # Optional dependencies for real hardware
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    cv2 = None  # type: ignore

try:  # Optional dependency for Intel RealSense
    import pyrealsense2 as rs  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    rs = None  # type: ignore

try:  # Numpy is required for simulation fallbacks
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    np = None  # type: ignore

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..core.thread_pool import get_thread_pool
from ..devices import DeviceException, HAS_TOBII, TobiiDevice
# from .fatigue_estimator import estimate_fatigue_score  # 文件模式下不再使用模拟评分


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class CollectorSummary:
    total_samples: int
    queue_capacity: int
    fill_percentage: float
    is_running: bool
    save_directory: Optional[str]


class MultimodalStreamPublisher:
    """Encodes frames and publishes them to the event bus for UI streaming."""

    def __init__(self, bus: Optional[EventBus], *, logger: Optional[logging.Logger] = None) -> None:
        self.bus = bus
        self.logger = (logger or logging.getLogger("service.multimodal.stream")) if bus else logging.getLogger(
            "service.multimodal.stream.disabled"
        )
        self._queue: "queue.Queue[tuple[Any, str]]" = queue.Queue(maxsize=3)
        self._thread_name = "multimodal-stream"
        self._stop_event = threading.Event()
        self._enabled = bus is not None and cv2 is not None and np is not None
        self._thread_pool = get_thread_pool()

    def start(self) -> None:
        if not self._enabled:
            return
        existing = self._thread_pool.get_managed_thread(self._thread_name)
        if existing and existing.is_alive():
            return
        self._stop_event.clear()
        thread = self._thread_pool.register_managed_thread(
            self._thread_name,
            self._run,
            daemon=True
        )
        thread.start()

    def stop(self) -> None:
        thread = self._thread_pool.get_managed_thread(self._thread_name)
        if thread and thread.is_alive():
            self._stop_event.set()
            try:
                self._queue.put_nowait((None, ""))  # type: ignore[arg-type]
            except queue.Full:
                pass
            self._thread_pool.unregister_managed_thread(self._thread_name, timeout=2.0)
        self._drain_queue()

    def submit_frame(self, frame: Any, timestamp: str) -> None:
        if not self._enabled or self._stop_event.is_set():
            return
        try:
            while self._queue.full():
                self._queue.get_nowait()
            self._queue.put_nowait((frame, timestamp))
        except queue.Full:  # pragma: no cover - defensive
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("Failed to queue frame: %s", exc)

    def _drain_queue(self) -> None:
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if frame is None:
                continue
            encoded = self._encode_frame(frame)
            if not encoded or self.bus is None:
                continue
            payload = {
                "timestamp": timestamp,
                "jpeg": encoded,
            }
            try:
                self.bus.publish(Event(EventTopic.MULTIMODAL_FRAME, payload))
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Failed to publish streamed frame: %s", exc)

    def _encode_frame(self, frame: Any) -> Optional[str]:
        if cv2 is None or np is None:
            return None
        try:
            array = np.ascontiguousarray(frame)
            height, width = array.shape[:2]
            if max(height, width) > 640:
                scale = 640.0 / float(max(height, width))
                new_size = (int(width * scale), int(height * scale))
                array = cv2.resize(array, new_size, interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", array, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ok:
                return None
            return base64.b64encode(buffer).decode("ascii")
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("JPEG encode failed: %s", exc)
            return None


class MultiModalDataCollector:
    """Hardware-facing data collector running in the backend process."""

    def __init__(
        self,
        username: str = "anonymous",
        *,
        part: int = 1,
        queue_duration: float = 5.0,
        save_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        frame_callback: Optional[Callable[[Any, str], None]] = None,
    ) -> None:
        if np is None:  # pragma: no cover - defensive guard
            raise RuntimeError("numpy is required for multimodal collection")

        self.username = username
        self.part = part
        self.queue_duration = queue_duration
        self.save_dir = save_dir + '/fatigue'
        self.logger = logger or logging.getLogger("service.multimodal.collector")
        self._frame_callback = frame_callback
        self._thread_pool = get_thread_pool()
        self._thread_name = f"multimodal-collector-{username}-p{part}"

        # Sliding window queues
        self.sample_rate = 30
        self.rgb_depth_interval = 5
        self.queue_length = int(self.queue_duration * 15)
        self._rgb_queue: list[Any] = []
        self._depth_queue: list[Any] = []
        self._eyetrack_queue: list[Any] = []
        self._timestamp_queue: list[str] = []
        self._data_lock = threading.Lock()

        # Runtime state
        self._cycle_count = 0
        self._stop_event = threading.Event()
        self.running = False
        self._start_time = 0.0  # Track collection start time for fatigue scoring

        # Device handles
        self.rs_pipeline = None
        self.rs_align = None
        self.rs_config = None
        self.rs_profile = None
        self.depth_scale = None
        self.tobii_device: Optional[TobiiDevice] = None

        # Mode flags
        force_sim = _env_flag("UI_FORCE_SIMULATION") or _env_flag("UI_MULTIMODAL_SIMULATION")
        self.simulate_rgb_depth = force_sim or rs is None or cv2 is None
        self.simulate_eyetrack = force_sim or not HAS_TOBII
        self._sim_phase = 0.0

        # Output writers
        self.rgb_writer = None
        self.depth_writer = None

        # Resolutions (fallback friendly defaults)
        self.rgb_resolution = (1280, 720)
        self.depth_resolution = (1280, 720)

        self._rgb_path: Optional[str] = None
        self._depth_path: Optional[str] = None
        self._eyetrack_path: Optional[str] = None
        self._metadata_path: Optional[str] = None

        self.logger.debug(
            "Collector init: simulate_rgb_depth=%s simulate_eyetrack=%s",
            self.simulate_rgb_depth,
            self.simulate_eyetrack,
        )
        self._init_devices()

    # ------------------------------------------------------------------
    def _init_devices(self) -> None:
        if not self.simulate_rgb_depth and rs is not None and cv2 is not None:
            try:
                self._init_realsense()
            except Exception as exc:  # pragma: no cover - hardware failures
                self.logger.warning("RealSense init failed, falling back to simulation: %s", exc)
                self.simulate_rgb_depth = True
        if not self.simulate_eyetrack and HAS_TOBII:
            try:
                self._init_tobii()
            except DeviceException as exc:  # pragma: no cover - hardware failures
                self.logger.warning("Tobii init failed, falling back to simulation: %s", exc)
                self.simulate_eyetrack = True
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.warning("Unexpected Tobii error, falling back to simulation: %s", exc)
                self.simulate_eyetrack = True

    def _init_realsense(self) -> None:  # pragma: no cover - depends on hardware
        assert rs is not None and cv2 is not None
        ctx = rs.context()
        if not ctx.query_devices():
            raise RuntimeError("no RealSense device detected")
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        color_width, color_height = self.rgb_resolution
        depth_width, depth_height = 640, 480
        self.rs_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)
        self.rs_config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        depth_sensor = self.rs_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.rs_align = rs.align(rs.stream.color)
        self.logger.info("RealSense device ready")

    def _init_tobii(self) -> None:  # pragma: no cover - depends on hardware
        assert HAS_TOBII
        device = TobiiDevice()
        device.start()
        self.tobii_device = device
        self.logger.info("Tobii device ready")

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self.running:
            return
        if not self.save_dir:
            self._create_save_directory()
        else:
            self._prepare_output_dir(Path(self.save_dir))
        self.running = True
        self._start_time = time.time()  # Record start time for fatigue scoring
        self._stop_event.clear()
        thread = self._thread_pool.register_managed_thread(
            self._thread_name,
            self._collection_loop,
            daemon=True
        )
        thread.start()
        self.logger.info("Multimodal collector started")

    def stop(self, *, join_timeout: float = 2.0) -> None:
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        self._thread_pool.unregister_managed_thread(self._thread_name, timeout=join_timeout)
        self._save_remaining_data()
        self._cleanup_devices()
        self.logger.info("Multimodal collector stopped")

    # ------------------------------------------------------------------
    def _create_save_directory(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("./recordings") / self.username / timestamp / "fatigue"
        self._prepare_output_dir(base_dir)
        self.save_dir = str(base_dir)

    def _prepare_output_dir(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        self._create_data_files(base_dir)

    def _create_data_files(self, base_dir: Path) -> None:
        if cv2 is None:
            self.logger.debug("OpenCV unavailable, video writers disabled")
        part_suffix = f"{self.part}"
        # RGB video
        if cv2 is not None:
            rgb_path = base_dir / f"rgb{part_suffix}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.rgb_writer = cv2.VideoWriter(str(rgb_path), fourcc, 15.0, self.rgb_resolution)
            if not self.rgb_writer or not self.rgb_writer.isOpened():  # pragma: no cover - IO errors
                self.logger.warning("Failed to open RGB video writer at %s", rgb_path)
                self.rgb_writer = None
            self._rgb_path = str(rgb_path)
        # Depth video
        if cv2 is not None:
            depth_path = base_dir / f"depth{part_suffix}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.depth_writer = cv2.VideoWriter(str(depth_path), fourcc, 15.0, self.depth_resolution)
            if not self.depth_writer or not self.depth_writer.isOpened():  # pragma: no cover
                self.logger.warning("Failed to open depth video writer at %s", depth_path)
                self.depth_writer = None
            self._depth_path = str(depth_path)
        # Eye tracking log
        eyetrack_path = base_dir / f"eyetrack{part_suffix}.json"
        self._eyetrack_path = str(eyetrack_path)
        # Metadata file
        metadata = {
            "username": self.username,
            "start_time": datetime.now().isoformat(),
            "sample_rate": self.sample_rate,
            "queue_duration": self.queue_duration,
            "queue_length": self.queue_length,
            "simulate_rgb_depth": self.simulate_rgb_depth,
            "simulate_eyetrack": self.simulate_eyetrack,
        }
        metadata_path = base_dir / f"metadata{part_suffix}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._metadata_path = str(metadata_path)

    # ------------------------------------------------------------------
    def _collection_loop(self) -> None:
        interval = 1.0 / max(1, self.sample_rate * self.rgb_depth_interval * 2)
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            now = time.perf_counter()
            if now >= next_tick:
                try:
                    self._collect_sample()
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.debug("Collection iteration failed: %s", exc)
                next_tick = now + interval
            remaining = max(0.0, next_tick - time.perf_counter())
            if remaining:
                self._stop_event.wait(timeout=min(remaining, 0.01))

    def _collect_sample(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._cycle_count += 1
        eyetrack_raw = self._collect_eyetrack()
        rgb_data: Optional[Any] = None
        depth_data: Optional[Any] = None
        if self._cycle_count % self.rgb_depth_interval == 0:
            rgb_data, depth_data = self._collect_aligned_images()
        with self._data_lock:
            if eyetrack_raw is not None:
                self._eyetrack_queue.append(self._extract_eyetrack_features(eyetrack_raw))
                self._trim_queue(self._eyetrack_queue, self.queue_length * self.rgb_depth_interval)
            if rgb_data is not None:
                self._rgb_queue.append(rgb_data)
                self._trim_queue(self._rgb_queue, self.queue_length)
                self._timestamp_queue.append(timestamp)
                self._trim_queue(self._timestamp_queue, self.queue_length)
            if depth_data is not None:
                self._depth_queue.append(depth_data)
                self._trim_queue(self._depth_queue, self.queue_length)
        self._persist_sample(rgb_data, depth_data, eyetrack_raw, timestamp)
        if self._frame_callback and rgb_data is not None:
            try:
                self._frame_callback(rgb_data, timestamp)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.debug("Frame callback failed: %s", exc)

    def _collect_aligned_images(self) -> Tuple[Optional[Any], Optional[Any]]:
        if not self.simulate_rgb_depth and rs is not None and cv2 is not None and self.rs_pipeline is not None:
            try:  # pragma: no cover - depends on hardware
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=100)
                aligned = self.rs_align.process(frames)
                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                if color_frame and depth_frame:
                    rgb_image = np.asanyarray(color_frame.get_data())  # type: ignore[call-arg]
                    depth_image = np.asanyarray(depth_frame.get_data())  # type: ignore[call-arg]
                    return rgb_image.copy(), depth_image.copy()
            except Exception as exc:
                self.logger.debug("RealSense frame capture failed: %s", exc)
        return self._generate_mock_rgb(), self._generate_mock_depth()

    def _collect_eyetrack(self) -> Optional[Dict[str, Any]]:
        if not self.simulate_eyetrack and self.tobii_device is not None:
            try:  # pragma: no cover - depends on hardware
                ok, data = self.tobii_device.read()
            except Exception as exc:
                self.logger.debug("Tobii read failed: %s", exc)
            else:
                if ok:
                    return data
        return self._generate_mock_eyetrack()

    # ------------------------------------------------------------------
    def _persist_sample(
        self,
        rgb_data: Optional[Any],
        depth_data: Optional[Any],
        eyetrack_data: Optional[Dict[str, Any]],
        timestamp: str,
    ) -> None:
        try:
            if self.rgb_writer is not None and rgb_data is not None and cv2 is not None:
                self.rgb_writer.write(rgb_data)
            if self.depth_writer is not None and depth_data is not None and cv2 is not None:
                self.depth_writer.write(self._depth_to_bgr(depth_data))
            if eyetrack_data is not None and self._eyetrack_path:
                serializable = self._prepare_eyetrack_for_json(eyetrack_data)
                serializable["timestamp"] = timestamp
                with open(self._eyetrack_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")
        except Exception as exc:  # pragma: no cover - file IO issues
            self.logger.debug("Persist sample failed: %s", exc)

    # ------------------------------------------------------------------
    def _generate_mock_rgb(self):  # pragma: no cover - exercised in simulation
        height = self.rgb_resolution[1]
        width = self.rgb_resolution[0]
        self._sim_phase = (self._sim_phase + 0.12) % (2 * np.pi)
        y = np.linspace(0, 1, height, dtype=np.float32)
        x = np.linspace(0, 1, width, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        base = (np.sin(self._sim_phase + xv * np.pi * 2) + 1) * 127
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(base + yv * 30, 0, 255)
        frame[..., 1] = np.clip(255 - base * yv, 0, 255)
        frame[..., 2] = np.clip((xv * 255 + self._sim_phase * 40) % 255, 0, 255)
        return frame

    def _generate_mock_depth(self):  # pragma: no cover - exercised in simulation
        height = self.depth_resolution[1]
        width = self.depth_resolution[0]
        depth = np.linspace(0, 4000, width, dtype=np.float32)
        depth = np.tile(depth, (height, 1))
        noise = np.random.normal(0, 80, size=(height, width))
        depth = np.clip(depth + noise, 0, 4095)
        return depth.astype(np.uint16)

    def _generate_mock_eyetrack(self) -> Dict[str, Any]:  # pragma: no cover - simulation path
        return {
            "gaze_point": [float(np.random.uniform(0, 640)), float(np.random.uniform(0, 480))],
            "head_pose": [
                float(np.random.uniform(-30, 30)),
                float(np.random.uniform(-30, 30)),
                float(np.random.uniform(-30, 30)),
                float(np.random.uniform(-0.3, 0.3)),
                float(np.random.uniform(-0.3, 0.3)),
                float(np.random.uniform(0.3, 1.0)),
            ],
            "timestamp": time.time(),
            "valid": bool(np.random.choice([True, False], p=[0.8, 0.2])),
        }

    # ------------------------------------------------------------------
    def _depth_to_bgr(self, depth_image):  # pragma: no cover - depends on cv2
        if cv2 is None:
            return depth_image
        depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
        gray_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
        if gray_bgr.shape[:2] != (self.depth_resolution[1], self.depth_resolution[0]):
            gray_bgr = cv2.resize(gray_bgr, self.depth_resolution, interpolation=cv2.INTER_LINEAR)
        return gray_bgr

    def _prepare_eyetrack_for_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        serializable: Dict[str, Any] = {}
        for key, value in data.items():
            if np is not None and isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, (np.integer,)):
                serializable[key] = int(value)  # type: ignore[arg-type]
            elif isinstance(value, (np.floating,)):
                serializable[key] = float(value)  # type: ignore[arg-type]
            elif isinstance(value, (np.bool_,)):
                serializable[key] = bool(value)  # type: ignore[arg-type]
            else:
                serializable[key] = value
        return serializable

    def _extract_eyetrack_features(self, data: Dict[str, Any]) -> list[float]:
        try:
            gaze = data.get("gaze_point") or data.get("gaze") or [0.0, 0.0]
            head = data.get("head_pose") or data.get("head") or [0.0] * 6
            features = list(gaze) + list(head)
            if len(features) < 8:
                features.extend([0.0] * (8 - len(features)))
            return features[:8]
        except Exception:  # pragma: no cover - defensive
            return [0.0] * 8

    # ------------------------------------------------------------------
    def _trim_queue(self, queue: list[Any], max_len: int) -> None:
        excess = len(queue) - max_len
        if excess > 0:
            del queue[:excess]

    def _save_remaining_data(self) -> None:
        try:
            if self.rgb_writer is not None and cv2 is not None:
                self.rgb_writer.release()
            if self.depth_writer is not None and cv2 is not None:
                self.depth_writer.release()
        except Exception:  # pragma: no cover - best effort cleanup
            pass

    def _cleanup_devices(self) -> None:
        try:
            if self.rs_pipeline is not None:
                self.rs_pipeline.stop()
        except Exception:  # pragma: no cover
            pass
        finally:
            self.rs_pipeline = None
        try:
            if self.tobii_device is not None:
                self.tobii_device.stop()
        except Exception:  # pragma: no cover
            pass
        finally:
            self.tobii_device = None

    # Public getters ---------------------------------------------------
    def get_current_data(self) -> Dict[str, Any]:
        with self._data_lock:
            return {
                "rgb": list(self._rgb_queue),
                "depth": list(self._depth_queue),
                "eyetrack": list(self._eyetrack_queue),
                "timestamps": list(self._timestamp_queue),
                "queue_length": len(self._rgb_queue),
                "is_full": len(self._rgb_queue) >= self.queue_length,
            }

    def get_latest_sample(self) -> Dict[str, Any]:
        with self._data_lock:
            if not self._rgb_queue:
                return {}
            return {
                "rgb": self._rgb_queue[-1],
                "depth": self._depth_queue[-1] if self._depth_queue else None,
                "eyetrack": self._eyetrack_queue[-1] if self._eyetrack_queue else None,
                "timestamp": self._timestamp_queue[-1] if self._timestamp_queue else None,
            }

    def get_file_paths(self) -> Dict[str, str]:
        paths: Dict[str, str] = {}
        if self._rgb_path and os.path.exists(self._rgb_path):
            paths["rgb"] = os.path.abspath(self._rgb_path)
        if self._depth_path and os.path.exists(self._depth_path):
            paths["depth"] = os.path.abspath(self._depth_path)
        if self._eyetrack_path and os.path.exists(self._eyetrack_path):
            paths["eyetrack"] = os.path.abspath(self._eyetrack_path)
        if self._metadata_path and os.path.exists(self._metadata_path):
            paths["metadata"] = os.path.abspath(self._metadata_path)
        return paths

    def get_summary(self) -> CollectorSummary:
        with self._data_lock:
            total_samples = len(self._rgb_queue)
            capacity = self.queue_length
            fill_percentage = (total_samples / capacity * 100.0) if capacity else 0.0
            return CollectorSummary(
                total_samples=total_samples,
                queue_capacity=capacity,
                fill_percentage=fill_percentage,
                is_running=self.running,
                save_directory=self.save_dir,
            )


class MultimodalService:
    """High-level service exposed to UI via the command router."""

    def __init__(self, *, bus: Optional[EventBus] = None, logger: Optional[logging.Logger] = None, eeg_service=None) -> None:
        self.logger = logger or logging.getLogger("service.multimodal")
        self.bus = bus
        self._collector: Optional[MultiModalDataCollector] = None
        self._stream_publisher = MultimodalStreamPublisher(bus, logger=self.logger.getChild("stream"))
        self._lock = threading.RLock()
        self._snapshot_thread_name = "multimodal-snapshot"
        self._snapshot_stop = threading.Event()
        self._snapshot_interval = 1.2
        self._snapshot_active = False
        self._snapshot_requested = False
        self._thread_pool = get_thread_pool()
        
        # EEG服务集成（用于轮询脑负荷数据）
        self._eeg_service = eeg_service
        self._eeg_poll_thread_name = "multimodal-eeg-poller"
        self._eeg_poll_stop = threading.Event()
        self._eeg_poll_interval = 3.0  # 每3秒轮询一次（降低频率，避免过于频繁）
        self._eeg_poll_active = False

    # Internal helpers -------------------------------------------------
    def _handle_stream_frame(self, frame: Any, timestamp: str) -> None:
        """Callback from collector to stream RGB frames for UI preview."""
        self._stream_publisher.submit_frame(frame, timestamp)

    # Lifecycle --------------------------------------------------------
    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        username = payload.get("username") or "anonymous"
        save_dir = payload.get("save_dir")
        part = int(payload.get("part", 1))
        queue_duration = float(payload.get("queue_duration", 5.0))
        snapshot_interval = float(payload.get("snapshot_interval", 1.2))
        eeg_poll_interval = float(payload.get("eeg_poll_interval", 3.0))  # EEG轮询间隔（默认3秒，降低频率）
        
        with self._lock:
            if self._collector and self._collector.running:
                self.logger.info("Multimodal collector already running")
                self._snapshot_interval = max(0.5, snapshot_interval)
                self._eeg_poll_interval = max(0.5, eeg_poll_interval)
                self._snapshot_requested = True
                self._ensure_snapshot_broadcast()
                self._ensure_eeg_polling()  # 确保EEG轮询已启动
                return {"status": "already-running", "save_dir": self._collector.save_dir}
            try:
                self._collector = MultiModalDataCollector(
                    username,
                    part=part,
                    queue_duration=queue_duration,
                    save_dir=save_dir,
                    logger=self.logger.getChild("collector"),
                    frame_callback=self._handle_stream_frame,
                )
                self._stream_publisher.start()
                self._collector.start()
                self._snapshot_interval = max(0.5, snapshot_interval)
                self._eeg_poll_interval = max(0.5, eeg_poll_interval)
                self._snapshot_requested = True
                self._ensure_snapshot_broadcast()
                self._ensure_eeg_polling()  # 启动EEG轮询
            except Exception as exc:
                self.logger.error("Failed to start multimodal collector: %s", exc)
                self._stream_publisher.stop()
                self._collector = None
                raise
        return {"status": "running", "save_dir": self._collector.save_dir}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._collector:
                return {"status": "idle"}
            self._collector.stop()
            self._stream_publisher.stop()
            self._stop_snapshot_broadcast()
            self._stop_eeg_polling()  # 停止EEG轮询
            self._snapshot_requested = False
        return {"status": "stopped"}

    def cleanup(self) -> Dict[str, Any]:
        with self._lock:
            if not self._collector:
                return {"status": "idle"}
            self._collector.stop()
            self._collector = None
            self._stream_publisher.stop()
            self._stop_snapshot_broadcast()
            self._stop_eeg_polling()  # 停止EEG轮询
            self._snapshot_requested = False
        return {"status": "released"}

    def _ensure_snapshot_broadcast(self) -> None:
        if self.bus is None:
            self.logger.debug("Event bus unavailable, snapshot broadcast disabled")
            return
        if self._snapshot_active:
            return
        self._snapshot_stop.clear()
        thread = self._thread_pool.register_managed_thread(
            self._snapshot_thread_name,
            self._snapshot_loop,
            daemon=True
        )
        self._snapshot_active = True
        thread.start()
        self.logger.debug("Snapshot broadcaster started (interval=%.2fs)", self._snapshot_interval)

    def _stop_snapshot_broadcast(self) -> None:
        if not self._snapshot_active:
            return
        self._snapshot_stop.set()
        self._thread_pool.unregister_managed_thread(self._snapshot_thread_name, timeout=2.0)
        self._snapshot_active = False
        self._snapshot_stop.clear()

    def _ensure_eeg_polling(self) -> None:
        """启动EEG数据轮询线程，定期获取脑负荷数据并发送推理请求"""
        if self.bus is None or self._eeg_service is None:
            self.logger.debug("EventBus或EEG服务不可用，EEG轮询已禁用")
            return
        if self._eeg_poll_active:
            return
        self._eeg_poll_stop.clear()
        thread = self._thread_pool.register_managed_thread(
            self._eeg_poll_thread_name,
            self._eeg_polling_loop,
            daemon=True
        )
        self._eeg_poll_active = True
        thread.start()
        self.logger.info("✅ EEG数据轮询已启动 (间隔=%.2fs)", self._eeg_poll_interval)

    def _stop_eeg_polling(self) -> None:
        """停止EEG数据轮询"""
        if not self._eeg_poll_active:
            return
        self._eeg_poll_stop.set()
        self._thread_pool.unregister_managed_thread(self._eeg_poll_thread_name, timeout=2.0)
        self._eeg_poll_active = False
        self._eeg_poll_stop.clear()
        self.logger.info("⏹️ EEG数据轮询已停止")

    def _eeg_polling_loop(self) -> None:
        """EEG轮询主循环：定期获取2秒窗口数据并发送推理请求"""
        self.logger.debug("EEG轮询线程已启动")
        try:
            while not self._eeg_poll_stop.is_set():
                # 使用IO线程池提交轮询任务，避免阻塞
                self._thread_pool.submit_io_task(self._poll_eeg_and_publish)
                interval = self._eeg_poll_interval
                if interval <= 0:
                    interval = 1.0
                if self._eeg_poll_stop.wait(interval):
                    break
        finally:
            self._eeg_poll_active = False
            self.logger.debug("EEG轮询线程已停止")

    def _poll_eeg_and_publish(self) -> None:
        """在IO线程中轮询EEG数据并发布推理请求"""
        if self._eeg_service is None or self.bus is None:
            return
        
        try:
            # 获取最近2秒的窗口数据 (2秒 @ 500Hz采样率 = 1000样本)
            # 注意: EEG采集是500Hz，但模型期望250Hz，所以我们获取2秒数据后可能需要降采样
            window_data = self._eeg_service.get_recent_window(seconds=2.0, sample_rate=500.0)
            
            if not window_data:
                return  # 静默跳过空数据
            
            # 提取双通道数据
            ch1_data = window_data.get("ch1", [])
            ch2_data = window_data.get("ch2", [])
            
            # 检查数据是否足够（至少500个样本用于推理）
            sample_count = len(ch1_data)
            
            if sample_count < 500:
                return  # 静默等待更多数据,避免频繁日志
            
            if not ch1_data or not ch2_data:
                return  # 静默跳过空通道
            
            if len(ch1_data) != len(ch2_data):
                self.logger.warning(f"⚠️EEG通道长度不匹配 {len(ch1_data)}≠{len(ch2_data)}")
                return
            
            # 转换为 [n_samples, 2] 格式 (与EEG模型期望一致)
            if np is not None:
                eeg_signal = np.column_stack([ch1_data, ch2_data]).tolist()
            else:
                eeg_signal = [[ch1_data[i], ch2_data[i]] for i in range(len(ch1_data))]
            
            # 发布EEG_REQUEST事件到EventBus，触发推理
            payload = {
                "mode": "memory",  # 内存模式
                "eeg_signal": eeg_signal,
                "sample_count": sample_count,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            self.bus.publish(Event(EventTopic.EEG_REQUEST, payload))
            # 压缩日志: 只输出样本数,不输出"已发送"等冗余信息
            self.logger.debug(f"📤EEG {sample_count}样本")
            
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug(f"⚠️EEG轮询: {exc}")

    def _snapshot_loop(self) -> None:
        self.logger.debug("Snapshot broadcaster loop running")
        try:
            while not self._snapshot_stop.is_set():
                # 使用IO线程池提交快照发布任务,避免阻塞采集线程
                self._thread_pool.submit_io_task(self._publish_snapshot)
                interval = self._snapshot_interval
                if interval <= 0:
                    interval = 1.0
                if self._snapshot_stop.wait(interval):
                    break
        finally:
            self._snapshot_active = False
            self.logger.debug("Snapshot broadcaster loop stopped")

    def _publish_snapshot(self) -> None:
        """在IO线程中发布快照,避免阻塞主采集循环."""
        try:
            snapshot = self._build_snapshot()
            if self.bus is not None:
                payload = dict(snapshot)
                payload.setdefault("status", "idle")
                payload["published_at"] = datetime.utcnow().isoformat()
                self.bus.publish(Event(EventTopic.MULTIMODAL_SNAPSHOT, payload))
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.debug("Failed to publish snapshot: %s", exc)

    # Information ------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        collector = self._collector
        if not collector:
            return {"status": "idle"}
        summary = collector.get_summary()
        return {
            "status": "running" if summary.is_running else "stopped",
            "total_samples": summary.total_samples,
            "queue_capacity": summary.queue_capacity,
            "fill_percentage": summary.fill_percentage,
            "save_dir": summary.save_directory,
        }

    def hardware_capabilities(self) -> Dict[str, Any]:
        """Expose底层硬件依赖的可用性，用于启动阶段日志输出。"""
        simulate_defaults = {
            "opencv_available": cv2 is not None,
            "numpy_available": np is not None,
            "realsense_driver": rs is not None,
            "tobii_driver": HAS_TOBII,
        }
        collector = self._collector
        if not collector:
            return {
                **simulate_defaults,
                "active": False,
            }
        return {
            **simulate_defaults,
            "active": collector.running,
            "simulate_rgb_depth": collector.simulate_rgb_depth,
            "simulate_eyetrack": collector.simulate_eyetrack,
        }

    def snapshot(self) -> Dict[str, Any]:
        return self._build_snapshot()

    def file_paths(self) -> Dict[str, Any]:
        collector = self._collector
        if not collector:
            return {"paths": {}, "status": "idle"}
        return {"paths": collector.get_file_paths(), "status": "running" if collector.running else "stopped"}

    # ------------------------------------------------------------------
    # 注释: 文件模式下,疲劳度推理由 UnifiedInferenceService 处理
    # 如需启用模拟评分,取消下面代码的注释
    # def _get_fatigue_score(self, data: Dict[str, Any], elapsed: float) -> float:
    #     """获取疲劳度分数 (模拟估算)
    #     
    #     注意: 在文件模式下,疲劳度推理由 UnifiedInferenceService 处理,
    #     此方法只返回启发式估算值,实际推理结果通过 DETECTION_RESULT 事件发布
    #     
    #     Args:
    #         data: 采集的多模态数据
    #         elapsed: 采集时长(秒)
    #     
    #     Returns:
    #         疲劳度分数 (0-100) - 仅用于快照预览,非最终推理结果
    #     """
    #     # 文件模式下,返回启发式估算值(用于快照预览)
    #     # 实际的模型推理由 UnifiedInferenceService 负责
    #     return estimate_fatigue_score(
    #         data.get("rgb") or [],
    #         data.get("depth") or [],
    #         data.get("eyetrack") or [],
    #         elapsed_time=elapsed,
    #     )

    # ------------------------------------------------------------------
    def _build_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            collector = self._collector
        if not collector:
            return {"status": "idle"}
        data = collector.get_current_data()
        latest = collector.get_latest_sample()
        
        # Calculate elapsed time since collection started
        elapsed = time.time() - collector._start_time if collector._start_time > 0 else 0.0
        
        # 文件模式: 不通过WebSocket传输图像数据,只传输文件路径
        # 获取当前保存的文件路径（用于文件模式推理）
        part_suffix = f"{collector.part}"
        save_dir = collector.save_dir if collector.save_dir else None
        rgb_video_path = str(Path(save_dir) / f"rgb{part_suffix}.avi") if save_dir else None
        depth_video_path = str(Path(save_dir) / f"depth{part_suffix}.avi") if save_dir else None
        eyetrack_json_path = str(Path(save_dir) / f"eyetrack{part_suffix}.json") if save_dir else None
        
        # 获取时间戳和帧计数
        timestamps = data.get("timestamps") or []
        latest_timestamp = timestamps[-1] if timestamps else None
        # 从队列数据中获取实际帧数
        rgb_samples = len(data.get("rgb") or [])
        depth_samples = len(data.get("depth") or [])
        eyetrack_count = len(data.get("eyetrack") or [])
        
        # 内存模式优化: 直接传递内存中的numpy数组给推理服务
        # 这样可以避免"写文件→读文件"的重复I/O
        rgb_frames_memory = data.get("rgb") or []
        depth_frames_memory = data.get("depth") or []
        eyetrack_memory = data.get("eyetrack") or []
        
        return {
            "status": "running" if collector.running else "stopped",
            "queue_length": data.get("queue_length", 0),
            "rgb_samples": rgb_samples,
            "depth_samples": depth_samples,
            "eyetrack_samples": eyetrack_count,
            "latest_timestamp": latest_timestamp,
            # 注意: fatigue_score 已移除,前端应从 DETECTION_RESULT 事件获取真实推理结果
            # 如需启用模拟评分,取消下面行的注释并恢复 _get_fatigue_score 方法
            # "fatigue_score": self._get_fatigue_score(data, elapsed),
            "elapsed_time": round(elapsed, 2),
            
            # 内存模式（优先）：直接传递numpy数组，避免文件I/O
            "memory_mode": True,
            "rgb_frames_memory": rgb_frames_memory,
            "depth_frames_memory": depth_frames_memory,
            "eyetrack_memory": eyetrack_memory,
            
            # 文件路径模式（备用）：用于存档和备份推理
            "file_mode": True,
            "rgb_video_path": rgb_video_path,
            "depth_video_path": depth_video_path,
            "eyetrack_json_path": eyetrack_json_path,
            
            # 推理元数据
            "timestamp": latest_timestamp,
            "frame_count": rgb_samples,
        }

    # 文件模式: 不再需要预览编码
    # def _encode_preview(self, frame: Any) -> Optional[str]:
    #     """已弃用 - 文件模式下不通过WebSocket发送图像"""
    #     pass


__all__ = ["MultimodalService"]
