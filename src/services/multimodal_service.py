"""Backend-managed multimodal data collection service."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

@dataclass
class CollectorSummary:
    total_samples: int
    queue_capacity: int
    fill_percentage: float
    is_running: bool
    save_directory: Optional[str]



class MultiModalDataCollector:
    """Hardware-facing data collector running in the backend process."""

    def __init__(
        self,
        username: str = "anonymous",
        *,
        part: int = 1,
        save_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if np is None:  # pragma: no cover - defensive guard
            raise RuntimeError("numpy is required for multimodal collection")

        self.username = username
        self.part = part
        self.save_dir = save_dir + '/fatigue'
        self.logger = logger or logging.getLogger("service.multimodal.collector")
        self._thread_pool = get_thread_pool()
        self._thread_name = f"multimodal-collector-{username}-p{part}"

        # Sliding window queues
        self.target_fps = 15  # Target video frame rate (frames per second)
        self._data_lock = threading.Lock()

        # Async writer / eyetracker thread queues (for real-device mode)
        self._rgb_write_queue: queue.Queue = queue.Queue(maxsize=180)
        self._depth_write_queue: queue.Queue = queue.Queue(maxsize=180)
        # eyetrack samples produced by a dedicated thread (if enabled)
        self._eyetrack_thread_queue: queue.Queue = queue.Queue(maxsize=256)

        # Background threads (created at start if hardware available)
        self._eyereader_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._writer_thread_name = f"{self._thread_name}-writer"
        self._eyereader_thread_name = f"{self._thread_name}-eyereader"

        # Runtime state
        self._rgb_frame_count = 0  # Track actual frames written to RGB video
        self._depth_frame_count = 0  # Track actual frames written to Depth video
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

        self._init_devices()

    # ------------------------------------------------------------------
    def _init_devices(self) -> None:
        if rs is not None and cv2 is not None:
            try:
                self._init_realsense()
            except Exception as exc:  # pragma: no cover - hardware failures
                self.logger.warning("RealSense init failed")

    def _init_realsense(self) -> None:  # pragma: no cover - depends on hardware
        assert rs is not None and cv2 is not None
        ctx = rs.context()
        if not ctx.query_devices():
            raise RuntimeError("no RealSense device detected")
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        color_width, color_height = self.rgb_resolution
        depth_width, depth_height = self.depth_resolution
        self.rs_config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)
        self.rs_config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
        self.rs_profile = self.rs_pipeline.start(self.rs_config)
        depth_sensor = self.rs_profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.rs_align = rs.align(rs.stream.color)
        self.logger.info("RealSense device ready")

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
            self._video_reader_run,
            daemon=True
        )
        # Start background writer thread (if using real CV writer)
        if cv2 is not None and self.rgb_writer is not None:
            self._writer_thread = self._thread_pool.register_managed_thread(
                self._writer_thread_name,
                self._video_writer_run,
                daemon=True,
            )
            self._writer_thread.start()

        # Start eyetrack reader thread if running on real hardware
        if HAS_TOBII:
            self._eyereader_thread = self._thread_pool.register_managed_thread(
                self._eyereader_thread_name,
                self._eyetrack_reader_run,
                daemon=True,
            )
            self._eyereader_thread.start()

        thread.start()
        self.logger.debug("Multimodal collector started")

    def stop(self, *, join_timeout: float = 2.0) -> None:
        if not self.running:
            return
        self.running = False
        self._stop_event.set()
        # signal background threads to stop and join them
        try:
            if self._eyereader_thread and self._eyereader_thread.is_alive():
                self._eyereader_thread.join(timeout=1.0)
            self._thread_pool.unregister_managed_thread(self._eyereader_thread_name, timeout=1.0)
        except Exception:
            pass
        try:
            if self._writer_thread and self._writer_thread.is_alive():
                # writer loop drains queues then exits
                self._writer_thread.join(timeout=2.0)
            self._thread_pool.unregister_managed_thread(self._writer_thread_name, timeout=2.0)
        except Exception:
            pass
        self._thread_pool.unregister_managed_thread(self._thread_name, timeout=join_timeout)
        self._save_remaining_data()
        self._cleanup_devices()
        self.logger.debug("Multimodal collector stopped")

    # ------------------------------------------------------------------
    def _eyetrack_reader_run(self) -> None:
        device = None
        f = None
        try:
            try:
                device = TobiiDevice()
                device.start()
            except Exception as exc:
                self.logger.warning("Eyetrack background thread failed to start device: %s", exc)
                return

            # open eyetrack JSON for append
            if self._eyetrack_path:
                try:
                    os.makedirs(os.path.dirname(self._eyetrack_path), exist_ok=True)
                    f = open(self._eyetrack_path, "a", encoding="utf-8")
                except Exception as exc:
                    self.logger.debug("Failed to open eyetrack json for writing: %s", exc)

            while not self._stop_event.is_set():
                try:
                    ok, sample = device.read()
                except Exception as exc:
                    self.logger.debug("Eyetrack read error: %s", exc)
                    break
                if ok and sample is not None:
                    # enqueue latest (replace oldest on full)
                    try:
                        self._eyetrack_thread_queue.put_nowait(sample)
                    except queue.Full:
                        try:
                            _ = self._eyetrack_thread_queue.get_nowait()
                            self._eyetrack_thread_queue.put_nowait(sample)
                        except Exception:
                            pass
                    # write JSON line immediately (device native rate: 60-90Hz)
                    if f is not None:
                        try:
                            eyetrack_dict = self._extract_eyetrack_features(sample)
                            f.write(json.dumps(eyetrack_dict, ensure_ascii=False) + "\n")
                        except Exception as exc:
                            self.logger.debug("Failed to write eyetrack data: %s", exc)
        finally:
            try:
                if f:
                    f.close()
            except Exception:
                pass
            try:
                if device is not None:
                    device.stop()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _video_writer_run(self) -> None:
        try:
            while not self._stop_event.is_set() or (not self._rgb_write_queue.empty() or not self._depth_write_queue.empty()):
                did = False
                try:
                    rgb_item = self._rgb_write_queue.get(timeout=0.1)
                except queue.Empty:
                    rgb_item = None
                if rgb_item is not None:
                    did = True
                    frame, ts = rgb_item
                    try:
                        if self.rgb_writer is not None and cv2 is not None:
                            self.rgb_writer.write(frame)
                            self._rgb_frame_count += 1
                    except Exception as exc:
                        self.logger.debug("RGB frame write failed: %s", exc)

                try:
                    depth_item = self._depth_write_queue.get(timeout=0.05)
                except queue.Empty:
                    depth_item = None
                if depth_item is not None:
                    did = True
                    depth_frame, ts = depth_item
                    try:
                        if self.depth_writer is not None and cv2 is not None:
                            bgr = self._depth_to_bgr(depth_frame)
                            self.depth_writer.write(bgr)
                            self._depth_frame_count += 1
                    except Exception as exc:
                        self.logger.debug("Depth frame write failed: %s", exc)

                if not did:
                    time.sleep(0.005)
        finally:
            # Log final frame counts
            duration = time.time() - self._start_time if self._start_time > 0 else 0
            self.logger.info(
                "Video writer stopped: RGB=%d frames, Depth=%d frames, Duration=%.2fs, "
                "Expected=%.2fs at %d fps",
                self._rgb_frame_count, self._depth_frame_count, duration,
                self._rgb_frame_count / self.target_fps if self._rgb_frame_count > 0 else 0,
                self.target_fps
            )

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
            "target_fps": self.target_fps,
        }
        metadata_path = base_dir / f"metadata{part_suffix}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._metadata_path = str(metadata_path)

    # ------------------------------------------------------------------
    def _video_reader_run(self) -> None:
        """Main collection loop with strict 15 FPS timing."""
        target_interval = 1.0 / self.target_fps  # 1/15 ≈ 0.0667 seconds per frame
        next_frame_time = time.perf_counter()
        
        while not self._stop_event.is_set():
            now = time.perf_counter()
            
            # Only collect if we've reached the next scheduled frame time
            if now >= next_frame_time:
                try:
                    self._collect_sample()
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.debug("Collection iteration failed: %s", exc)
                
                # Schedule next frame at exact interval
                next_frame_time += target_interval
                
                # If we've fallen behind, skip ahead to avoid accumulating delay
                if next_frame_time < now:
                    self.logger.warning("Frame collection falling behind, resetting timing")
                    next_frame_time = now + target_interval
            
            # Sleep until next frame is due (with small polling interval)
            remaining = next_frame_time - time.perf_counter()
            if remaining > 0:
                self._stop_event.wait(timeout=min(remaining, 0.01))

    def _collect_sample(self) -> None:
        """Collect one frame of RGB/Depth/Eyetrack data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect RGB and Depth every frame (15 fps)
        rgb_data, depth_data = self._collect_aligned_images()
        
        # Persist to video files
        self._persist_sample(rgb_data, depth_data, timestamp)


    def _collect_aligned_images(self) -> Tuple[Optional[Any], Optional[Any]]:
        if rs is not None and cv2 is not None and self.rs_pipeline is not None:
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
        return None, None

    def _persist_sample(
        self,
        rgb_data: Optional[Any],
        depth_data: Optional[Any],
        timestamp: str,
    ) -> None:
        try:
            # Enqueue frames for background writer to avoid blocking capture loop
            if self.rgb_writer is not None and rgb_data is not None and cv2 is not None:
                try:
                    self._rgb_write_queue.put_nowait((rgb_data, timestamp))
                except queue.Full:
                    # drop oldest then enqueue
                    try:
                        _ = self._rgb_write_queue.get_nowait()
                        self._rgb_write_queue.put_nowait((rgb_data, timestamp))
                    except Exception:
                        pass
            if self.depth_writer is not None and depth_data is not None and cv2 is not None:
                try:
                    self._depth_write_queue.put_nowait((depth_data, timestamp))
                except queue.Full:
                    try:
                        _ = self._depth_write_queue.get_nowait()
                        self._depth_write_queue.put_nowait((depth_data, timestamp))
                    except Exception:
                        pass
        except Exception as exc:  # pragma: no cover - file IO issues
            self.logger.debug("Persist sample failed: %s", exc)

    # ------------------------------------------------------------------
    def _depth_to_bgr(self, depth_image):  # pragma: no cover - depends on cv2
        """Convert depth frame to BGR image with range clipping for face detection.
        
        Applies depth range filtering to preserve facial details:
        - Sets lower bound (min_range) to filter background
        - Sets upper bound (max_range) to filter too-close noise
        - Only keeps depth values within [min_range, max_range]
        - Values outside this range are set to 0 (black)
        """
        if cv2 is None:
            return depth_image
        
        # Convert to float32 for processing
        depth_array = depth_image.astype(np.float32)
        
        # Define depth range for optimal face detection
        min_range = 400.0  # millimetres - filter out too close objects
        max_range = 700.0  # millimetres - filter out background
        
        # Clip to valid range [min_range, max_range]
        depth_clipped = np.clip(depth_array, min_range, max_range)
        
        # Filter out values outside the range (set to 0)
        # This creates a "depth window" focused on face distance
        mask = (depth_array >= min_range) & (depth_array <= max_range)
        depth_clipped = depth_clipped * mask
        
        # Normalize to use full 8-bit range for better contrast
        # Map [min_range, max_range] to [0, 255]
        depth_normalized = (depth_clipped - min_range) / (max_range - min_range)
        depth_normalized = np.clip(depth_normalized, 0.0, 1.0)
        depth_8u = (depth_normalized * 255).astype(np.uint8)
        
        # Convert grayscale to BGR
        gray_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
        
        # Resize if needed
        if gray_bgr.shape[:2] != (self.depth_resolution[1], self.depth_resolution[0]):
            gray_bgr = cv2.resize(gray_bgr, self.depth_resolution, interpolation=cv2.INTER_LINEAR)
        
        return gray_bgr

    def _extract_eyetrack_features(self, data: Dict[str, Any]) -> Dict[str, Any]:

        try:
            # 提取注视点 (gaze_point)
            gaze = data.get("gaze_point") or data.get("gaze") or [0.0, 0.0]
            if not isinstance(gaze, list):
                gaze = [0.0, 0.0]
            
            # 提取头部姿态 (head_pose: [x, y, z, rot_x, rot_y, rot_z])
            head = data.get("head_pose") or data.get("head") or [0.0] * 6
            if not isinstance(head, list):
                head = [0.0] * 6
            
            # 组合特征: [gaze_x, gaze_y] + [head_x, head_y, head_z, rot_x, rot_y, rot_z]
            features = list(gaze[:2]) + list(head[:6])
            
            # 确保长度为8
            if len(features) < 8:
                features.extend([0.0] * (8 - len(features)))
            features = features[:8]
            
            # 构建返回字典
            result = {
                "system_time": time.time(),
                "eyetrack_data": features
            }
            
            return result
            
        except Exception as exc:
            self.logger.debug("Error extracting eyetrack features: %s", exc)
            return {
                "system_time": time.time(),
                "eyetrack_data": [0.0] * 8
            }

    # ------------------------------------------------------------------

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

    def __init__(self, *, bus: Optional[EventBus] = None, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("service.multimodal")
        self.bus = bus
        self._collector: Optional[MultiModalDataCollector] = None
        self._lock = threading.RLock()
        self._snapshot_thread_name = "multimodal-snapshot"
        self._snapshot_stop = threading.Event()
        self._snapshot_interval = 1.2
        self._snapshot_active = False
        self._snapshot_requested = False
        self._thread_pool = get_thread_pool()

    # Internal helpers -------------------------------------------------

    # Lifecycle --------------------------------------------------------
    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        username = payload.get("username") or "anonymous"
        save_dir = payload.get("save_dir")
        part = int(payload.get("part", 1))
        snapshot_interval = float(payload.get("snapshot_interval", 1.2))
        
        with self._lock:
            if self._collector and self._collector.running:
                self.logger.debug("Multimodal collector already running")
                self._snapshot_interval = max(0.5, snapshot_interval)
                self._snapshot_requested = True
                self._ensure_snapshot_broadcast()
                return {"status": "already-running", "save_dir": self._collector.save_dir}
            try:
                self._collector = MultiModalDataCollector(
                    username,
                    part=part,
                    save_dir=save_dir,
                    logger=self.logger.getChild("collector"),
                )
                self._collector.start()
                self._snapshot_interval = max(0.5, snapshot_interval)
                self._snapshot_requested = True
                self._ensure_snapshot_broadcast()
            except Exception as exc:
                self.logger.error("Failed to start multimodal collector: %s", exc)
                self._collector = None
                raise
        return {"status": "running", "save_dir": self._collector.save_dir}

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._collector:
                return {"status": "idle"}
            
            # 保存录制信息用于疲劳度评估
            username = self._collector.username
            save_dir = self._collector.save_dir
            
            self._collector.stop()
            self._stop_snapshot_broadcast()
            self._snapshot_requested = False
            
            # ✅ 录制完成后，触发疲劳度评估（只触发一次）
            # 使用 session_dir 作为去重键，防止同一会话多次触发
            if self.bus and save_dir:
                # 获取session目录（save_dir的父目录）
                session_dir = str(Path(save_dir).parent)
                
                # 去重检查：使用 session_dir 作为键
                if not hasattr(self, '_assessed_sessions'):
                    self._assessed_sessions = set()
                
                if session_dir in self._assessed_sessions:
                    self.logger.debug(f"⏭️ 疲劳度评估已触发，跳过重复请求: {session_dir}")
                else:
                    self._assessed_sessions.add(session_dir)
                    self.logger.info(f"📊 录制完成，触发疲劳度评估: {session_dir}, 被试ID: {username}")
                    
                    # 发布疲劳度评估请求事件
                    self.bus.publish(Event(
                        topic=EventTopic.FATIGUE_ASSESSMENT_REQUEST,
                        payload={
                            "request_id": uuid.uuid4().hex,
                            "session_dir": session_dir,
                            "subject_id": username,  # 使用username作为subject_id
                            "timestamp": time.time()
                        }
                    ))
        
        return {"status": "stopped"}

    def cleanup(self) -> Dict[str, Any]:
        with self._lock:
            if not self._collector:
                return {"status": "idle"}
            self._collector.stop()
            self._collector = None
            self._stop_snapshot_broadcast()
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
        }

    def snapshot(self) -> Dict[str, Any]:
        return self._build_snapshot()

    def file_paths(self) -> Dict[str, Any]:
        collector = self._collector
        if not collector:
            return {"paths": {}, "status": "idle"}
        return {"paths": collector.get_file_paths(), "status": "running" if collector.running else "stopped"}

    def _build_snapshot(self) -> Dict[str, Any]:
        """Build snapshot with file paths only (no in-memory data)."""
        with self._lock:
            collector = self._collector
        if not collector:
            return {"status": "idle"}
        
        # Calculate elapsed time since collection started
        elapsed = time.time() - collector._start_time if collector._start_time > 0 else 0.0
        
        # 文件模式: 只传输文件路径，不传输内存数据
        part_suffix = f"{collector.part}"
        save_dir = collector.save_dir if collector.save_dir else None
        rgb_video_path = str(Path(save_dir) / f"rgb{part_suffix}.avi") if save_dir else None
        depth_video_path = str(Path(save_dir) / f"depth{part_suffix}.avi") if save_dir else None
        eyetrack_json_path = str(Path(save_dir) / f"eyetrack{part_suffix}.json") if save_dir else None
        
        # 获取当前时间戳
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            "status": "running" if collector.running else "stopped",
            "rgb_frame_count": collector._rgb_frame_count,
            "depth_frame_count": collector._depth_frame_count,
            "elapsed_time": round(elapsed, 2),
            
            # 文件路径模式：用于推理
            "file_mode": True,
            "rgb_video_path": rgb_video_path,
            "depth_video_path": depth_video_path,
            "eyetrack_json_path": eyetrack_json_path,
            
            # 推理元数据
            "timestamp": current_timestamp,
            "target_fps": collector.target_fps,
        }


__all__ = ["MultimodalService"]
