"""Backend-side management for the Maibobo blood-pressure monitor."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from serial.tools import list_ports
    import serial  # type: ignore
except Exception:  # pragma: no cover - serial is optional at runtime
    list_ports = None  # type: ignore
    serial = None  # type: ignore

try:
    from ..devices.maibobo import MaiboboDevice
    HAS_MAIBOBO_DRIVER = True
except Exception:  # pragma: no cover - hardware SDK not installed
    MaiboboDevice = None  # type: ignore
    HAS_MAIBOBO_DRIVER = False

from ..core.thread_pool import get_thread_pool


class BloodPressureService:
    """Runs acquisition for the Maibobo blood-pressure monitor."""

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("service.bp")
        self._lock = threading.RLock()
        self._thread_pool = get_thread_pool()
        self._thread_name = "bp-collector"
        self._stop_event = threading.Event()
        self._device: Optional[MaiboboDevice] = None  # type: ignore[assignment]
        self._latest_reading: Optional[Dict[str, Any]] = None
        self._running = False
        self._completed = False
        self._mode = "hardware"
        self._current_port: Optional[str] = None
        self._last_error: Optional[str] = None
        self._preferred_port = "COM8"

    # ------------------------------------------------------------------
    def start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Start a blood-pressure acquisition session."""

        simulation = bool(payload.get("simulation"))
        allow_simulation = bool(payload.get("allow_simulation", True))
        requested_port = (payload.get("port") or "").strip() or None
        timeout = int(payload.get("timeout", 1))

        with self._lock:
            if self._running:
                self.logger.info("Blood-pressure collector already running")
                return {
                    "status": "already-running",
                    "mode": self._mode,
                    "port": self._current_port,
                }

            self._latest_reading = None
            self._completed = False
            self._last_error = None
            self._mode = "simulation" if simulation else "hardware"
            self._current_port = requested_port

            if not simulation:
                if not HAS_MAIBOBO_DRIVER:
                    message = "Maibobo 设备驱动未安装"
                    self._last_error = message
                    if not allow_simulation:
                        raise RuntimeError(message)
                    self.logger.warning("Hardware driver missing; falling back to simulation")
                    simulation = True
                    self._mode = "simulation"
                else:
                    if not self._current_port:
                        self._current_port = self._preferred_port
                    detected_port = self._detect_available_port()
                    if detected_port:
                        self._current_port = detected_port
                    if not self._current_port:
                        message = "未检测到可用的血压仪端口"
                        self._last_error = message
                        if not allow_simulation:
                            raise RuntimeError(message)
                        self.logger.warning("No blood-pressure device detected; switching to simulation")
                        simulation = True
                        self._mode = "simulation"

            if not simulation and self._current_port:
                try:
                    self._device = MaiboboDevice(self._current_port, timeout=timeout)  # type: ignore[call-arg]
                    self._device.start()
                except Exception as exc:
                    self._device = None
                    self._last_error = str(exc)
                    self.logger.error("Failed to start Maibobo device on %s: %s", self._current_port, exc)
                    if not allow_simulation:
                        raise
                    simulation = True
                    self._mode = "simulation"

            self._stop_event.clear()
            self._running = True
            thread = self._thread_pool.register_managed_thread(
                self._thread_name,
                self._run_collector,
                daemon=True,
                args=(simulation,)
            )
            thread.start()

            return {
                "status": "running",
                "mode": self._mode,
                "port": self._current_port,
            }

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if not self._running and not self._device:
                return {"status": "idle"}
            self._stop_event.set()
        
        # 获取已注册的线程
        thread = self._thread_pool.get_managed_thread(self._thread_name)
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        
        with self._lock:
            self._shutdown_device()
            self._running = False
        
        # 注销线程
        self._thread_pool.unregister_managed_thread(self._thread_name, timeout=0.5)
        
        return {"status": "stopped"}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            result = {
                "status": self._current_status(),
                "mode": self._mode,
                "port": self._current_port,
                "latest": self._latest_reading,
                "completed": self._completed,
                "error": self._last_error,
            }
        return result

    def status(self) -> Dict[str, Any]:
        with self._lock:
            detected_port = self._detect_available_port()
            available_ports: List[str] = []
            if self._preferred_port:
                available_ports.append(self._preferred_port)
            return {
                "running": self._running,
                "mode": self._mode,
                "port": self._current_port or detected_port,
                "available_ports": available_ports,
                "error": self._last_error,
                "driver_available": HAS_MAIBOBO_DRIVER,
            }

    # ------------------------------------------------------------------
    def _run_collector(self, simulation: bool) -> None:
        try:
            if simulation:
                self._simulate_until_complete()
            else:
                self._read_until_complete()
        except Exception as exc:  # pragma: no cover - defensive guard
            self.logger.exception("Blood-pressure collector crashed: %s", exc)
            with self._lock:
                self._last_error = str(exc)
        finally:
            with self._lock:
                self._running = False
            self._shutdown_device()

    def _simulate_until_complete(self) -> None:
        random.seed(time.time())
        end_time = time.time() + random.uniform(2.0, 4.0)
        while not self._stop_event.is_set() and time.time() < end_time:
            time.sleep(0.2)
        if self._stop_event.is_set():
            return
        reading = {
            "systolic": random.randint(110, 130),
            "diastolic": random.randint(70, 85),
            "pulse": random.randint(65, 95),
            "mode": "simulation",
            "timestamp": time.time(),
        }
        with self._lock:
            self._latest_reading = reading
            self._completed = True
            self._last_error = None
        self.logger.info(
            "血压测试模拟结果: 收缩压=%s, 舒张压=%s, 脉搏=%s",
            reading["systolic"],
            reading["diastolic"],
            reading["pulse"],
        )

    def _read_until_complete(self) -> None:
        if not self._device:
            return
        self.logger.info("开始读取血压仪数据：端口 %s", self._current_port)
        start_time = time.time()
        timeout = start_time + 120.0
        while not self._stop_event.is_set() and time.time() < timeout:
            try:
                success, frame = self._device.read()
                if not success or frame is None:
                    time.sleep(0.3)
                    continue
                reading = self._parse_frame(frame)
                if not reading:
                    time.sleep(0.3)
                    continue
                reading["mode"] = "hardware"
                reading["timestamp"] = time.time()
                with self._lock:
                    self._latest_reading = reading
                    self._completed = True
                    self._last_error = None
                self.logger.info(
                    "读取到血压仪数据: 收缩压=%s, 舒张压=%s, 脉搏=%s",
                    reading["systolic"],
                    reading["diastolic"],
                    reading["pulse"],
                )
                return
            except Exception as exc:
                self.logger.warning("读取血压仪数据失败: %s", exc)
                with self._lock:
                    self._last_error = str(exc)
                time.sleep(1.0)
        self.logger.warning("血压测试超时，未获得有效数据")

    def _parse_frame(self, frame: Any) -> Optional[Dict[str, int]]:
        try:
            if hasattr(frame, "systolic") and hasattr(frame, "diastolic") and hasattr(frame, "pulse"):
                return {
                    "systolic": int(frame.systolic),
                    "diastolic": int(frame.diastolic),
                    "pulse": int(frame.pulse),
                }
            if isinstance(frame, (list, tuple)) and len(frame) >= 11:
                return {
                    "systolic": int(frame[8]),
                    "diastolic": int(frame[10]),
                    "pulse": int(frame[2]),
                }
            if isinstance(frame, (list, tuple)) and len(frame) >= 3:
                return {
                    "systolic": int(frame[0]),
                    "diastolic": int(frame[1]),
                    "pulse": int(frame[2]),
                }
        except Exception as exc:
            self.logger.debug("无法解析血压仪帧数据: %s", exc)
        return None

    def _shutdown_device(self) -> None:
        device = self._device
        self._device = None
        if not device:
            return
        try:
            device.stop()
            self.logger.info("血压仪设备已停止")
        except Exception as exc:
            self.logger.warning("关闭血压仪设备时出错: %s", exc)

    def _current_status(self) -> str:
        if self._completed:
            return "completed"
        if self._running:
            return "running"
        if self._last_error:
            return "error"
        return "idle"

    def _detect_available_port(self) -> Optional[str]:
        if list_ports is None:
            return None
        target_port = self._current_port or self._preferred_port
        if not target_port:
            return None
        if serial is None:
            return target_port
        try:
            with serial.Serial(target_port, timeout=1):
                return target_port
        except Exception:
            return None


__all__ = ["BloodPressureService", "HAS_MAIBOBO_DRIVER"]
