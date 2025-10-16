"""Routes UI-originated command events to backend services."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..core.thread_pool import get_thread_pool
from .av_service import AVService
from .bp_service import BloodPressureService
from .db_service import DatabaseService, DatabaseUnavailable
from .eeg_service import EEGService
from .multimodal_service import MultimodalService
from .tts_service import TTSService
from .emotion_service import EmotionService


class UICommandRouter:
    """Dispatches WebSocket UI commands to backend service handlers."""

    def __init__(self, bus: EventBus, *, logger: Optional[logging.Logger] = None) -> None:
        self.bus = bus
        self.logger = logger or logging.getLogger("ui.command")
        self._thread_pool = get_thread_pool()
        self._subscription = None
        self.av_service = AVService(bus, logger=self.logger.getChild("av"))
        self.bp_service = BloodPressureService(logger=self.logger.getChild("bp"))
        self.db_service = DatabaseService(logger=self.logger.getChild("db"))
        self.eeg_service = EEGService(bus=bus, logger=self.logger.getChild("eeg"))
        # 将EEG服务传递给Multimodal服务，以支持EEG数据轮询
        self.multimodal_service = MultimodalService(
            bus=bus,
            logger=self.logger.getChild("multimodal"),
            eeg_service=self.eeg_service
        )
        self.tts_service = TTSService(logger=self.logger.getChild("tts"))
        self.emotion_service = EmotionService(
            bus=bus,
            logger=self.logger.getChild("emotion")
        )
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        self._subscription = self.bus.subscribe(EventTopic.UI_COMMAND, self._on_command)
        self._running = True
        self._log_startup_status()

    def stop(self) -> None:
        if not self._running:
            return
        self.bus.unsubscribe(EventTopic.UI_COMMAND, self._on_command)  # type: ignore[arg-type]
        self._subscription = None
        self._running = False
        self.av_service.shutdown()
        self.bp_service.stop()
        self.eeg_service.stop_recording()
        self.tts_service.shutdown()

    # ------------------------------------------------------------------
    def _on_command(self, event: Event) -> None:
        if not event.payload:
            return
        # 使用IO线程池处理UI命令(IO密集型:数据库查询、WebSocket响应)
        self._thread_pool.submit_io_task(self._dispatch_command, event)

    def _log_startup_status(self) -> None:
        """Log a concise summary of connected devices when the backend boots."""
        self.logger.info("后端服务已启动，正在检测各模块状态…")

        try:
            bp_status = self.bp_service.status()
            self.logger.info(
                "血压仪 -> driver=%s running=%s port=%s 可用端口=%s 错误=%s",
                bp_status.get("driver_available"),
                bp_status.get("running"),
                bp_status.get("port"),
                bp_status.get("available_ports"),
                bp_status.get("error"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("血压仪状态检测失败: %s", exc)

        try:
            eeg_diag = self.eeg_service.diagnostics()
            self.logger.info(
                "EEG -> 驱动可用=%s 强制模拟=%s 运行中=%s",
                eeg_diag.get("hardware_driver_available"),
                eeg_diag.get("force_simulation"),
                eeg_diag.get("running"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("EEG 状态检测失败: %s", exc)

        try:
            multimodal_diag = self.multimodal_service.hardware_capabilities()
            self.logger.info(
                "多模态 -> RealSense=%s Tobii=%s OpenCV=%s 正在运行=%s (RGB仿真=%s, 眼动仿真=%s)",
                multimodal_diag.get("realsense_driver"),
                multimodal_diag.get("tobii_driver"),
                multimodal_diag.get("opencv_available"),
                multimodal_diag.get("active"),
                multimodal_diag.get("simulate_rgb_depth"),
                multimodal_diag.get("simulate_eyetrack"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("多模态模块状态检测失败: %s", exc)

        try:
            av_diag = self.av_service.diagnostics()
            self.logger.info(
                "音视频 -> OpenCV=%s PyAudio=%s 允许模拟=%s 当前模拟=%s",
                av_diag.get("opencv_available"),
                av_diag.get("pyaudio_available"),
                av_diag.get("allow_synthetic"),
                av_diag.get("synthetic_mode"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("音视频模块状态检测失败: %s", exc)

        try:
            tts_diag = self.tts_service.diagnostics()
            self.logger.info(
                "语音播报 -> 默认后端=%s PowerShell=%s pyttsx3=%s",
                tts_diag.get("default_backend"),
                tts_diag.get("powershell_available"),
                tts_diag.get("pyttsx3_available"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("TTS 模块状态检测失败: %s", exc)

        try:
            db_diag = self.db_service.diagnostics()
            status_text = "禁用" if db_diag.get("disabled") else "启用"
            self.logger.info(
                "数据库 -> %s host=%s db=%s",
                status_text,
                db_diag.get("configured_host"),
                db_diag.get("configured_database"),
            )
        except DatabaseUnavailable as exc:
            self.logger.warning("数据库不可用: %s", exc)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning("数据库状态检测失败: %s", exc)

    def _dispatch_command(self, event: Event) -> None:
        payload = event.payload or {}
        action = payload.get("action")
        request_id = payload.get("id")
        client_token = payload.get("client_token")
        body = payload.get("payload") or {}
        self.logger.debug("Handling UI command %s (id=%s)", action, request_id)
        status = "ok"
        result: Dict[str, Any] | None = None
        try:
            result = self._handle_action(action, body)
        except Exception as exc:  # pragma: no cover - defensive
            status = "error"
            result = {"error": str(exc)}
            self.logger.exception("Command %s failed", action)
        response_payload = {
            "id": request_id,
            "status": status,
            "result": result,
            "client_token": client_token,
            "action": action,
        }
        self.bus.publish(Event(EventTopic.UI_RESPONSE, response_payload))

    def _handle_action(self, action: str, body: Dict[str, Any]):
        if action == "av.start_preview":
            self.av_service.start_preview(**body)
            return {"message": "preview-started"}
        if action == "av.stop_preview":
            result = self.av_service.stop_recording()
            self.av_service.stop_preview()
            return {
                "message": "preview-stopped",
                "audio_paths": result.audio_paths,
                "video_paths": result.video_paths,
            }
        if action == "av.start_recording":
            return self.av_service.start_recording()
        if action == "av.stop_recording":
            result = self.av_service.stop_recording()
            return {
                "audio_paths": result.audio_paths,
                "video_paths": result.video_paths,
            }
        if action == "av.list_paths":
            result = self.av_service.session_paths()
            return {
                "audio_paths": result.audio_paths,
                "video_paths": result.video_paths,
            }
        if action == "av.audio_level":
            return {"level": self.av_service.current_audio_level()}
        if action == "db.insert_test_record":
            return self.db_service.insert_test_record(body)
        if action == "db.update_test_record":
            return self.db_service.update_test_record(body)
        if action == "db.get_user_history":
            return self.db_service.get_user_history(body)
        if action == "bp.start":
            return self.bp_service.start(body)
        if action == "bp.stop":
            return self.bp_service.stop()
        if action == "bp.snapshot":
            return self.bp_service.snapshot()
        if action == "bp.status":
            return self.bp_service.status()
        if action == "multimodal.start":
            return self.multimodal_service.start(body)
        if action == "multimodal.stop":
            return self.multimodal_service.stop()
        if action == "multimodal.cleanup":
            return self.multimodal_service.cleanup()
        if action == "multimodal.status":
            return self.multimodal_service.status()
        if action == "multimodal.snapshot":
            return self.multimodal_service.snapshot()
        if action == "multimodal.paths":
            return self.multimodal_service.file_paths()
        if action == "tts.speak":
            return self.tts_service.speak(body)
        if action == "eeg.start":
            save_dir = body.get("save_dir", "")
            return self.eeg_service.start_recording(save_dir)
        if action == "eeg.stop":
            return self.eeg_service.stop_recording()
        if action == "eeg.snapshot":
            return self.eeg_service.get_snapshot()
        if action == "eeg.file_paths":
            return self.eeg_service.get_file_paths()
        if action == "emotion.analyze":
            audio_paths = body.get("audio_paths", [])
            video_paths = body.get("video_paths", [])
            text_data = body.get("text_data", [])
            timeout = body.get("timeout", 15.0)
            return self.emotion_service.analyze_emotion(
                audio_paths=audio_paths,
                video_paths=video_paths,
                text_data=text_data,
                timeout=timeout
            )
        raise ValueError(f"Unsupported action: {action}")


__all__ = ["UICommandRouter"]
