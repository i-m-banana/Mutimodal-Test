"""Main testing workflow page for the refactored UI application."""

from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Callable, Optional, Dict

import yaml
from yaml import FullLoader

from .. import config
from ..qt import (
    QEasingCurve,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QMetaObject,
    QPushButton,
    QSize,
    QSizePolicy,
    QShortcut,
    QSpacerItem,
    QStackedWidget,
    QTimer,
    QVBoxLayout,
    QWidget,
    Qt,
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPropertyAnimation,
    pyqtSignal,
    qta,
)

from ..utils.widgets import AudioLevelMeter, ScoreChartWidget
from ..utils.responsive import scale, scale_size, scale_font
from ..widgets.camera_preview import CameraPreviewWidget
from ...widgets.schulte_grid import SchulteGridWidget
from...widgets.score_page import ScorePage
from ...services.backend_client import get_backend_client
from ...utils_common.thread_process_manager import get_thread_manager

# ---------------------------------------------------------------------------
# Configuration exports (preserve legacy names for migrated code)
# ---------------------------------------------------------------------------
logger = config.logger
DEBUG_MODE = config.DEBUG_MODE
SKIP_DATABASE = config.SKIP_DATABASE
HAS_MULTIMODAL = config.HAS_MULTIMODAL
HAS_SPEECH_RECOGNITION = config.HAS_SPEECH_RECOGNITION
HAS_BP_BACKEND = config.HAS_BP_BACKEND
BP_SIMULATION = config.BP_SIMULATION
HAS_MAIBOBO_BACKEND = config.HAS_MAIBOBO_BACKEND  # 保留旧常量名供兼容
SCORES_CSV_FILE = config.SCORES_CSV_FILE
add_audio_for_recognition = config.add_audio_for_recognition
clear_recognition_results = config.clear_recognition_results
get_recognition_results = config.get_recognition_results
stop_recognition = config.stop_recognition

av_start_collection = config.av_start_collection
av_stop_collection = config.av_stop_collection
av_start_recording = config.av_start_recording
av_stop_recording = config.av_stop_recording
av_get_current_frame = config.av_get_current_frame
av_get_audio_paths = config.av_get_audio_paths
av_get_video_paths = config.av_get_video_paths
av_get_current_audio_level = config.av_get_current_audio_level

multidata_start_collection = config.multidata_start_collection
multidata_stop_collection = config.multidata_stop_collection
multidata_get_snapshot = config.multidata_get_snapshot

eeg_start_collection = config.eeg_start_collection
eeg_stop_collection = config.eeg_stop_collection
eeg_get_snapshot = config.eeg_get_snapshot
eeg_get_file_paths = config.eeg_get_file_paths

bp_start_measurement = config.bp_start_measurement
bp_stop_measurement = config.bp_stop_measurement
bp_get_snapshot = config.bp_get_snapshot
bp_get_status = config.bp_get_status

_build_session_dir = config.build_session_dir


class TestPage(QWidget):
    """
    The main testing page, featuring a multi-step process with voice questions
    and score visualization. Refactored into a dedicated module.
    """
    
    # 用于从异步线程安全地调度UI回调的信号
    _invoke_later_signal = pyqtSignal(object, int)  # (callback, delay_ms)

    def __init__(self) -> None:
        super().__init__()

        questionnaire_file = config.QUESTIONNAIRE_YAML_FILE
        with open(questionnaire_file, encoding="utf-8") as handle:
            all_questions = yaml.load(handle, Loader=FullLoader)

        questions_num = 5
        self.questions = []
        for _ in range(questions_num):
            self.questions.append(
                all_questions.pop(random.randint(0, len(all_questions) - 1))
            )
        # TTS队列
        # 记录已经朗读的题目
        self.spoken_questions = set()

        # 初始化 TTS 配置（实际引擎在后台线程内创建，避免跨线程的 COM 问题）
        self.tts_queue = Queue()
        self._tts_rate = 150
        self._tts_volume = 1.0
        self._tts_voice = os.getenv("UI_TTS_VOICE", "").strip()
        default_backend = "powershell" if sys.platform.startswith("win") else "pyttsx3"
        backend_pref = os.getenv("UI_TTS_BACKEND", default_backend).strip().lower()
        self._tts_backend = backend_pref or default_backend

        # 后台线程处理朗读
        self.thread_manager = get_thread_manager()
        self.tts_task_id = self.thread_manager.submit_data_task(
            self._tts_loop,
            task_name="TTS朗读处理"
        )

        self.current_question = 0
        self.current_step = 0

        self.setAutoFillBackground(True)

        self._setup_properties()
        self._init_ui()
        self._connect_signals()

        self._setup_mic_button_animation()
        self.update_step_ui()

        self._dot_animations = []  # 用于保留动画对象，防止 GC

        # 连接异步调度信号,确保从任何线程调用_invoke_later都安全
        self._invoke_later_signal.connect(self._handle_invoke_later_signal)

        self.load_history_scores()
        logger.info("TestPage 初始化完成。")
        self._is_shutting_down = False

    def _tts_loop(self):
        backend_pref = getattr(self, "_tts_backend", "pyttsx3") or "pyttsx3"
        backend_pref = backend_pref.lower()
        client = None

        while True:
            text = self.tts_queue.get()
            if text is None:
                break

            preview = text if len(text) <= 20 else text[:20] + "..."
            logger.info(f"开始朗读：{preview}")

            if client is None:
                try:
                    client = get_backend_client()
                except Exception as exc:
                    logger.error("获取后端 TTS 客户端失败: %s", exc)
                    self.tts_queue.task_done()
                    client = None
                    continue

            timeout_seconds = max(12.0, min(200.0, len(text) / 3.5 + 15.0))
            payload = {
                "text": text,
                "voice": self._tts_voice or None,
                "rate": self._tts_rate,
                "volume": self._tts_volume,
                "backend": backend_pref,
                "timeout": timeout_seconds,
            }

            future = client.send_command_future("tts.speak", payload)
            try:
                result = future.result(timeout=timeout_seconds + 5.0)
                backend_used = result.get("backend", backend_pref)
                elapsed = result.get("elapsed")
                if isinstance(elapsed, (int, float)):
                    logger.info("朗读完成（%s），用时 %.2f 秒", backend_used, elapsed)
                    expected = max(len(text) / 5.0, 1.0)
                    if elapsed < expected * 0.35:
                        logger.warning(
                            "朗读用时异常偏短（%.2fs），文本长度 %d，请检查系统音量或语音包是否可用。",
                            elapsed,
                            len(text)
                        )
                else:
                    logger.info("朗读完成（%s）", backend_used)
            except Exception as exc:
                logger.error("TTS 后端朗读失败: %s", exc)
            finally:
                self.tts_queue.task_done()

    def _setup_properties(self):
        """初始化测试页面的所有状态变量。"""
        self.steps = ['语音答题', '血压测试', '舒特格测试', '分数展示']
        self.current_step = 0
        self.current_question = 0
        self.is_recording = False
        self.score = None  # 将在舒尔特测试完成后计算
        self.history_scores = []
        # 音频录制已转移到AVCollector，这里只保留定时器用于更新UI
        self.audio_timer = QTimer(self)
        self.camera_preview: Optional[CameraPreviewWidget] = None
        self.schulte_camera_preview: Optional[CameraPreviewWidget] = None
        # 会话与录制文件管理
        self.session_timestamp = None
        self.session_dir = None
        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None
        # 当前登录用户名（默认匿名）
        self.current_user = 'anonymous'

        # 多模态数据采集相关（不再使用独立预览窗口）
        self.multimodal_collector = None
        self._multimodal_poll_timer = QTimer(self)
        self._multimodal_poll_timer.setInterval(1200)
        self._multimodal_poll_timer.timeout.connect(self._poll_multimodal_snapshot)
        self._multimodal_poll_active = False
        self._multimodal_last_status: Optional[str] = None
        self._last_multimodal_snapshot_monotonic: Optional[float] = None
        self._last_fatigue_score: Optional[float] = None
        self._last_brain_load_score: Optional[float] = None  # 保存最后的脑负荷分数
        self._last_fatigue_log_time: Optional[float] = None
        self._multimodal_gap_warned: bool = False
        
        # 实时分数累积列表（用于计算平均值）
        self._fatigue_scores_list: list[float] = []  # 疲劳度实时分数列表
        self._brain_load_scores_list: list[float] = []  # 脑负荷实时分数列表
        
        # 情绪分数（测试结束时分析一次）
        self._emotion_score: Optional[float] = None
        self._emotion_analysis_triggered: bool = False  # 防止重复触发情绪分析

        # 数据库交互状态
        self._db_warning_logged = False
        self._db_disabled = SKIP_DATABASE
        self._row_id_future = None
        self._pending_db_updates = []
        self.row_id = None  # 数据库记录尚未创建前保持空值

        # 血压后端采集状态
        self.bp_simulation_enabled = BP_SIMULATION
        self.bp_forced_port = config.BP_PORT
        self.bp_available_port = None
        self.bp_measurement_active = False
        self.bp_poll_timer = QTimer(self)
        self.bp_poll_timer.setInterval(600)
        self.bp_poll_timer.timeout.connect(self._poll_bp_snapshot)
        self._bp_error_reported = False
        self._bp_snapshot_warned = False

        # 舒特测试结果实例属性（用于信号穿透保存）
        self.schulte_elapsed = None  # 用时（秒）
        self.schulte_accuracy = None  # 准确率（百分比）

        # 环节时间戳记录
        self.part_timestamps = []

        # 测试流程状态标志
        self.test_started = False

    def _invoke_later(self, callback: Callable[[], None], delay_ms: int = 0) -> None:
        """Run `callback` on the UI thread after the given delay.
        
        This method can be safely called from any thread (main or worker threads).
        It uses Qt signals to ensure callbacks are always executed on the main Qt thread.
        """
        # 使用信号发送到主线程,无论从哪个线程调用都安全
        self._invoke_later_signal.emit(callback, delay_ms)
    
    def _handle_invoke_later_signal(self, callback: Callable[[], None], delay_ms: int) -> None:
        """Handle _invoke_later_signal in the main thread.
        
        This slot is guaranteed to run on the main thread due to Qt's signal/slot mechanism.
        """
        timeout = max(0, int(delay_ms))
        if timeout == 0:
            # 立即执行
            try:
                callback()
            except Exception as e:
                logger.error(f"执行立即回调时出错: {e}", exc_info=True)
        else:
            # 延迟执行 - 现在我们在主线程中,可以安全使用QTimer
            QTimer.singleShot(timeout, lambda: self._safe_callback(callback))
    
    def _safe_callback(self, callback: Callable[[], None]) -> None:
        """Execute a callback with error handling."""
        try:
            callback()
        except Exception as e:
            logger.error(f"执行延迟回调时出错: {e}", exc_info=True)

    def _start_multimodal_monitoring(self, *, force: bool = False) -> None:
        """启动或重新启动多模态数据监控（仅内嵌显示，非阻塞）。"""
        try:
            if not HAS_MULTIMODAL:
                return

            timer_active = False
            try:
                timer_active = self._multimodal_poll_timer.isActive()
            except Exception:
                timer_active = False

            if not force and self._multimodal_poll_active and timer_active:
                # 正常情况下已经在轮询，无需重复启动
                return

            if self._multimodal_poll_active and not timer_active:
                logger.warning("检测到多模态监控标记为活动但定时器未运行，自动重新启动")

            if force and timer_active:
                # 防御性地重置定时器，避免潜在的 stuck 状态
                self._multimodal_poll_timer.stop()
                timer_active = False

            logger.info("启动多模态数据监控（内嵌显示模式）")
            self._multimodal_poll_active = True
            self._multimodal_last_status = None
            # 重置一次性日志标志，避免复用旧状态导致不更新
            for attr in ("_multimodal_first_data", "_fatigue_score_cast_failed",
                        "_no_scores_warned"):
                if hasattr(self, attr):
                    delattr(self, attr)
            self._last_multimodal_snapshot_monotonic = None
            self._last_fatigue_score = None
            self._last_fatigue_log_time = None
            self._multimodal_gap_warned = False

            if not timer_active:
                self._multimodal_poll_timer.start()

            self._poll_multimodal_snapshot()
        except Exception as e:
            logger.error(f"启动多模态监控失败: {e}")
            self._multimodal_poll_active = False

    def _stop_multimodal_monitoring(self) -> None:
        """停止多模态数据监控（安全，不抛出异常）"""
        try:
            if self._multimodal_poll_timer.isActive():
                self._multimodal_poll_timer.stop()
            self._multimodal_poll_active = False
            self._multimodal_last_status = None
            logger.info("多模态数据监控已停止")
        except Exception as e:
            logger.debug(f"停止多模态监控时出错: {e}")

    def _save_speech_recognition_results(self) -> None:
        """保存语音识别结果到数据库和文件（在情绪分析前调用）"""
        if not HAS_SPEECH_RECOGNITION:
            return
        
        try:
            # 获取语音识别结果
            record_payload = get_recognition_results()
            if not record_payload:
                logger.debug("没有语音识别结果需要保存")
                return
            
            logger.info(f"💾 保存 {len(record_payload)} 条语音识别结果")
            
            # 写入到文件
            try:
                record_txt = os.path.join(self.session_dir, 'emotion', "record.txt")
                os.makedirs(os.path.dirname(record_txt), exist_ok=True)
                with open(record_txt, 'w', encoding='utf-8') as f:
                    f.write(str(record_payload))
                logger.info(f"✅ 语音识别结果已写入文件: {record_txt}")
            except Exception as exc:
                logger.warning(f"写入语音识别记录文本失败: {exc}")
            
            # 更新到数据库
            try:
                self._queue_db_update(
                    {'record_text': record_payload},
                    "保存语音识别结果到数据库失败"
                )
                logger.info("✅ 语音识别结果已更新到数据库")
            except Exception as exc:
                logger.warning(f"更新语音识别结果到数据库失败: {exc}")
            
        except Exception as e:
            logger.error(f"保存语音识别结果失败: {e}")
    
    def _trigger_emotion_analysis(self) -> None:
        """
        📍 触发情绪分析 - 在答题结束、切换到血压测试时调用
        
        收集音视频和文本数据，发送到后端进行情绪推理
        """
        # 防止重复触发（一个测试会话只分析一次）
        if self._emotion_analysis_triggered:
            logger.debug("情绪分析已经触发过，跳过重复调用")
            return
        
        try:
            # 收集音视频路径
            audio_paths = getattr(self, '_audio_paths', [])
            video_paths = getattr(self, '_video_paths', [])
            
            # 收集文本识别结果
            text_data = []
            if HAS_SPEECH_RECOGNITION:
                try:
                    text_data = get_recognition_results()
                except Exception as e:
                    logger.warning(f"获取语音识别结果失败: {e}")
            
            logger.info(
                f"准备进行情绪分析: {len(audio_paths)} 个音频, "
                f"{len(video_paths)} 个视频, {len(text_data)} 个文本"
            )
            
            # 检查是否有数据
            if not audio_paths and not video_paths and not text_data:
                logger.warning("没有可用的音视频或文本数据，跳过情绪分析")
                return
            
            # 标记为已触发，防止重复调用
            self._emotion_analysis_triggered = True
            
            # 异步发送情绪分析请求，不阻塞UI
            def analyze_async():
                try:
                    from ...services.backend_proxy import emotion_analyze
                    
                    logger.info("正在进行情绪分析...")
                    result = emotion_analyze(
                        audio_paths=audio_paths,
                        video_paths=video_paths,
                        text_data=text_data,
                        timeout=30.0  # 情绪推理需要较长时间(5个样本约5秒)
                    )
                    
                    emotion_score = result.get("emotion_score", 0.0)
                    emotion_label = result.get("emotion_label", "unknown")
                    confidence = result.get("confidence", 0.0)
                    
                    logger.info(
                        f"情绪分析完成: {emotion_label} "
                        f"(score={emotion_score:.3f}, confidence={confidence:.3f})"
                    )
                    
                    # 保存情绪分数
                    self._emotion_score = emotion_score
                    logger.info(f"✅ 情绪分数已保存: {emotion_score:.2f}")
                    
                except Exception as exc:
                    logger.error(f"情绪分析失败: {exc}", exc_info=True)
            
            # 使用线程池执行，不阻塞主线程
            import threading
            thread = threading.Thread(target=analyze_async, daemon=True)
            thread.start()
            
        except Exception as exc:
            logger.error(f"触发情绪分析失败: {exc}", exc_info=True)

    def _poll_multimodal_snapshot(self) -> None:
        """轮询多模态数据快照，仅更新内嵌显示（安全，失败不影响UI）"""
        if not HAS_MULTIMODAL:
            self._multimodal_poll_timer.stop()
            self._multimodal_poll_active = False
            return

        try:
            snapshot = multidata_get_snapshot()
        except Exception as exc:
            logger.debug(f"获取多模态采集状态失败: {exc}")
            return

        if not snapshot:
            logger.debug("多模态快照为空")
            return

        try:
            status = (snapshot.get("status") or "idle").lower()

            # 首次收到数据时记录日志
            if not hasattr(self, '_multimodal_first_data'):
                logger.info(f"多模态数据轮询已启动，当前状态: {status}")
                self._multimodal_first_data = True

            # ⚠️ 注意：多模态快照中的分数数据已废弃
            # 现在疲劳度和脑负荷通过 DETECTION_RESULT 事件独立推送
            # 保留此代码仅用于兼容性检查
            
            # 废弃：不再从快照中读取分数，因为：
            # 1. 疲劳度通过 model_fatigue 的 DETECTION_RESULT 事件推送
            # 2. 脑负荷通过 model_eeg 的 DETECTION_RESULT 事件推送
            # 3. 两者完全独立，互不依赖
            
            # fatigue = snapshot.get("fatigue_score")  # 已废弃
            # brain = snapshot.get("brain_load_score")  # 已废弃
            
            # 检查快照数据（仅用于调试）
            if not hasattr(self, '_snapshot_check_warned'):
                if "fatigue_score" in snapshot or "brain_load_score" in snapshot:
                    logger.debug("检测到快照中仍包含分数数据（已不使用）")
                self._snapshot_check_warned = True
            # 检查采集状态
            if status != "running" and self._multimodal_poll_active:
                self._multimodal_poll_timer.stop()
                self._multimodal_poll_active = False
                logger.info("多模态采集已停止，停止轮询")

        except Exception as exc:
            logger.error(f"处理多模态快照数据时出错: {exc}")

    def _on_detection_result(self, payload: Dict) -> None:
        """处理模型推理结果 (DETECTION_RESULT事件)
        
        Args:
            payload: 推理结果数据,格式:
                {
                    "detector": "model_fatigue",
                    "status": "detected", 
                    "predictions": {
                        "fatigue_score": 51.38,
                        "prediction_class": 1
                    },
                    "timestamp": ...,
                    "frame_count": 30
                }
        """
        try:
            detector = payload.get("detector", "")
            status = payload.get("status", "")
            predictions = payload.get("predictions", {})
            
            # 处理疲劳度推理结果（独立更新，不依赖脑负荷）
            if detector == "model_fatigue" and status == "detected":
                fatigue_score = predictions.get("fatigue_score")
                prediction_class = predictions.get("prediction_class")
                
                if fatigue_score is not None:
                    logger.info(f"📊 收到疲劳度推理结果: score={fatigue_score:.2f}, class={prediction_class}")
                    
                    # 保存疲劳度分数（最后一次）
                    self._last_fatigue_score = fatigue_score
                    
                    # 累积到列表中用于计算平均值
                    self._fatigue_scores_list.append(fatigue_score)
                    
                    # 只更新疲劳度显示，不影响脑负荷
                    self._update_fatigue_only(fatigue_score)
                else:
                    logger.warning("⚠️ 疲劳度推理结果中没有 fatigue_score 字段")
            
            # 处理EEG脑负荷推理结果（独立更新，不依赖疲劳度）
            elif detector == "model_eeg" and status == "detected":
                brain_load_score = predictions.get("brain_load_score")
                state = predictions.get("state")
                
                if brain_load_score is not None:
                    logger.info(f"🧠 收到EEG脑负荷推理结果: score={brain_load_score:.2f}, state={state}")
                    
                    # 保存脑负荷分数（最后一次）
                    self._last_brain_load_score = brain_load_score
                    
                    # 累积到列表中用于计算平均值
                    self._brain_load_scores_list.append(brain_load_score)
                    
                    # 只更新脑负荷显示，不影响疲劳度
                    self._update_brain_load_only(brain_load_score)
                else:
                    logger.warning("⚠️ EEG推理结果中没有 brain_load_score 字段")
            
        except Exception as exc:
            logger.error(f"处理推理结果时出错: {exc}", exc_info=True)

    def _update_fatigue_only(self, score_f) -> None:
        """只更新疲劳度显示（安全，失败不影响UI）"""
        try:
            score_value_f = float(score_f)
            logger.debug(f"更新疲劳度显示: {score_value_f}")

            # 根据疲劳度设置不同颜色
            if score_value_f < 30:
                color_f = "#27ae60"  # 绿色 - 正常
                bg_color_f = "#d5f4e6"
            elif score_value_f < 60:
                color_f = "#f39c12"  # 橙色 - 警告
                bg_color_f = "#fef5e7"
            else:
                color_f = "#e74c3c"  # 红色 - 疲劳
                bg_color_f = "#fadbd8"

            # 更新语音答题页面的疲劳度显示
            if hasattr(self, 'fatigue_info_label') and self.fatigue_info_label:
                try:
                    self.fatigue_info_label.setText(f"疲劳度: {score_value_f:.1f}")
                    self.fatigue_info_label.setStyleSheet(f"""
                        QLabel {{
                            color: {color_f};
                            padding: 8px;
                            background-color: {bg_color_f};
                            border-radius: 8px;
                            font-weight: bold;
                        }}
                    """)
                except Exception as e:
                    logger.error(f"更新语音答题页疲劳度标签失败: {e}")

            # 更新舒尔特页面的疲劳度显示
            if hasattr(self, 'schulte_fatigue_label') and self.schulte_fatigue_label:
                try:
                    self.schulte_fatigue_label.setText(f"疲劳度: {score_value_f:.1f}")
                    self.schulte_fatigue_label.setStyleSheet(f"""
                        QLabel {{
                            color: {color_f};
                            padding: 8px;
                            background-color: {bg_color_f};
                            border-radius: 8px;
                            font-weight: bold;
                        }}
                    """)
                except Exception as e:
                    logger.error(f"更新舒尔特页疲劳度标签失败: {e}")

        except Exception as exc:
            logger.error(f"更新疲劳度显示失败: {exc}")

    def _update_brain_load_only(self, score_b) -> None:
        """只更新脑负荷显示（安全，失败不影响UI）"""
        try:
            score_value_b = float(score_b)
            logger.debug(f"更新脑负荷显示: {score_value_b}")

            # 根据脑负荷设置不同颜色
            if score_value_b < 30:
                color_b = "#27ae60"  # 绿色 - 正常
                bg_color_b = "#d5f4e6"
            elif score_value_b < 60:
                color_b = "#f39c12"  # 橙色 - 警告
                bg_color_b = "#fef5e7"
            else:
                color_b = "#e74c3c"  # 红色 - 高负荷
                bg_color_b = "#fadbd8"

            # 更新语音答题页面的脑负荷显示
            if hasattr(self, 'brain_load_info_label') and self.brain_load_info_label:
                try:
                    self.brain_load_info_label.setText(f"脑负荷: {score_value_b:.1f}")
                    self.brain_load_info_label.setStyleSheet(f"""
                        QLabel {{
                            color: {color_b};
                            padding: 8px;
                            background-color: {bg_color_b};
                            border-radius: 8px;
                            font-weight: bold;
                        }}
                    """)
                except Exception as e:
                    logger.error(f"更新语音答题页脑负荷标签失败: {e}")

            # 更新舒尔特页面的脑负荷显示
            if hasattr(self, 'schulte_brain_load_label') and self.schulte_brain_load_label:
                try:
                    self.schulte_brain_load_label.setText(f"脑负荷: {score_value_b:.1f}")
                    self.schulte_brain_load_label.setStyleSheet(f"""
                        QLabel {{
                            color: {color_b};
                            padding: 8px;
                            background-color: {bg_color_b};
                            border-radius: 8px;
                            font-weight: bold;
                        }}
                    """)
                except Exception as e:
                    logger.error(f"更新舒尔特页脑负荷标签失败: {e}")

        except Exception as exc:
            logger.error(f"更新脑负荷显示失败: {exc}")

    def _update_fatigue_display(self, score_f, score_b) -> None:
        """更新疲劳度和脑负荷显示（已废弃，保留用于兼容性）
        
        注意：此方法已废弃，建议使用 _update_fatigue_only 和 _update_brain_load_only
        """
        try:
            # 转换为浮动数值
            score_value_f = float(score_f)
            score_value_b = float(score_b)

            logger.debug(f"收到疲劳度数据: {score_value_f}")
            logger.debug(f"收到脑负荷数据: {score_value_b}")

            # 调试：检查当前步骤
            logger.debug(f"当前步骤: {self.current_step}")
            logger.debug(f"是否有 fatigue_info_label: {hasattr(self, 'fatigue_info_label')}")
            logger.debug(f"是否有 schulte_fatigue_label: {hasattr(self, 'schulte_fatigue_label')}")
            logger.debug(f"是否有 brain_load_info_label: {hasattr(self, 'brain_load_info_label')}")
            logger.debug(f"是否有 schulte_brain_load_label: {hasattr(self, 'schulte_brain_load_label')}")

            # 根据疲劳度设置不同颜色
            if score_value_f < 30:
                color_f = "#27ae60"  # 绿色 - 正常
                bg_color_f = "#d5f4e6"
            elif score_value_f < 60:
                color_f = "#f39c12"  # 橙色 - 警告
                bg_color_f = "#fef5e7"
            else:
                color_f = "#e74c3c"  # 红色 - 疲劳
                bg_color_f = "#fadbd8"

            # 根据脑负荷设置不同颜色
            if score_value_b < 30:
                color_b = "#27ae60"  # 绿色 - 正常
                bg_color_b = "#d5f4e6"
            elif score_value_b < 60:
                color_b = "#f39c12"  # 橙色 - 警告
                bg_color_b = "#fef5e7"
            else:
                color_b = "#e74c3c"  # 红色 - 疲劳
                bg_color_b = "#fadbd8"

            # 设置样式
            style_f = f"""
                     QLabel {{
                         color: {color_f};
                         background-color: {bg_color_f};
                         padding: 8px;
                         border-radius: 8px;
                         font-weight: bold;
                     }}
                 """

            style_b = f"""
                     QLabel {{
                         color: {color_b};
                         background-color: {bg_color_b};
                         padding: 8px;
                         border-radius: 8px;
                         font-weight: bold;
                     }}
                 """

            # 更新内嵌的疲劳度显示（第一页答题界面）
            if hasattr(self, 'fatigue_info_label'):
                self.fatigue_info_label.setText(f"疲劳度: {int(score_value_f)}%")
                self.fatigue_info_label.setStyleSheet(style_f)

                if not hasattr(self, '_fatigue_updated'):
                    logger.info(f"✅ 第一页疲劳度显示已更新: {int(score_value_f)}%")
                    self._fatigue_updated = True
                else:
                    logger.debug(f"第一页疲劳度更新: {int(score_value_f)}%")
            else:
                logger.warning("⚠️ 第一页 fatigue_info_label 不存在！")

            # 更新舒尔特页面的疲劳度显示
            if hasattr(self, 'schulte_fatigue_label'):
                self.schulte_fatigue_label.setText(f"疲劳度: {int(score_value_f)}%")
                self.schulte_fatigue_label.setStyleSheet(style_f)

                if not hasattr(self, '_schulte_fatigue_updated'):
                    logger.info(f"✅ 舒尔特页疲劳度显示已更新: {int(score_value_f)}%")
                    self._schulte_fatigue_updated = True
                else:
                    logger.debug(f"舒尔特页疲劳度更新: {int(score_value_f)}%")
            else:
                logger.debug("舒尔特页 schulte_fatigue_label 尚未创建")

            # 更新脑负荷显示
            if hasattr(self, 'brain_load_info_label'):
                self.brain_load_info_label.setText(f"脑负荷: {int(score_value_b)}%")
                self.brain_load_info_label.setStyleSheet(style_b)

                if not hasattr(self, '_brain_load_updated'):
                    logger.info(f"✅ 第一页脑负荷显示已更新: {int(score_value_b)}%")
                    self._brain_load_updated = True
                else:
                    logger.debug(f"第一页脑负荷更新: {int(score_value_b)}%")
            else:
                logger.warning("⚠️ 第一页 brain_load_info_label 不存在！")

            # 更新舒尔特页面的脑负荷显示
            if hasattr(self, 'schulte_brain_load_label'):
                self.schulte_brain_load_label.setText(f"脑负荷: {int(score_value_b)}%")
                self.schulte_brain_load_label.setStyleSheet(style_b)

                if not hasattr(self, '_schulte_brain_load_updated'):
                    logger.info(f"✅ 舒尔特页脑负荷显示已更新: {int(score_value_b)}%")
                    self._schulte_brain_load_updated = True
                else:
                    logger.debug(f"舒尔特页脑负荷更新: {int(score_value_b)}%")
            else:
                logger.debug("舒尔特页 schulte_brain_load_label 尚未创建")

        except Exception as exc:
            logger.error(f"更新疲劳度和脑负荷显示失败: {exc}")

    def _init_ui(self):
        """初始化用户界面。"""
        self.main_layout = QVBoxLayout(self)
        
        # 使用固定边距和间距
        self.main_layout.setContentsMargins(scale(15), scale(15), scale(15), scale(15))
        self.main_layout.setSpacing(scale(15))

        # 顶部步骤导航
        self.step_container = self._create_step_navigator()
        self.main_layout.addWidget(self.step_container)

        # 问题进度条
        self.question_container = self._create_question_progress_bar()
        self.main_layout.addWidget(self.question_container)

        # 主内容区
        content_container = self._create_main_content_area()
        self.main_layout.addWidget(content_container, 1)
        # 底部按钮
        self.bottom_button_container = self._create_bottom_buttons()
        self.main_layout.addWidget(self.bottom_button_container, 0, Qt.AlignCenter)
    def _connect_signals(self):
        """连接所有控件的信号到槽函数。"""
        self.audio_timer.timeout.connect(self._process_audio)
        self.btn_next.clicked.connect(self._next_step_or_question)
        self.btn_finish.clicked.connect(self._finish_test)
        self.btn_mic.clicked.connect(self._toggle_recording)
        
        # 连接后端推理结果信号 (用于获取真实的疲劳度分数)
        backend_client = get_backend_client()
        backend_client.detection_result.connect(self._on_detection_result)

    def _setup_mic_button_animation(self):
        """为麦克风按钮创建光晕（阴影模糊）动画，以避免布局抖动。"""
        self.mic_shadow = QGraphicsDropShadowEffect()
        self.mic_shadow.setBlurRadius(20)
        self.mic_shadow.setColor(QColor(66, 165, 245, 180))
        self.mic_shadow.setOffset(0, 0)
        self.btn_mic.setGraphicsEffect(self.mic_shadow)

        self.mic_anim = QPropertyAnimation(self.mic_shadow, b"blurRadius")
        self.mic_anim.setDuration(1200)
        self.mic_anim.setStartValue(15)
        self.mic_anim.setEndValue(35)
        self.mic_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.mic_anim.setLoopCount(-1)

        self.mic_anim_reverse = QPropertyAnimation(self.mic_shadow, b"blurRadius")
        self.mic_anim_reverse.setDuration(1200)
        self.mic_anim_reverse.setStartValue(35)
        self.mic_anim_reverse.setEndValue(15)
        self.mic_anim_reverse.setEasingCurve(QEasingCurve.InOutQuad)

        self.mic_anim.finished.connect(self.mic_anim_reverse.start)
        self.mic_anim_reverse.finished.connect(self.mic_anim.start)

    def _setup_debug_shortcut(self):
        try:
            self._skip_shortcut = QShortcut(QKeySequence("Q"), self)
            self._skip_shortcut.setContext(Qt.ApplicationShortcut)
            self._skip_shortcut.activated.connect(self._handle_debug_shortcut)
        except Exception as e:
            logger.warning(f"注册调试快捷键失败: {e}")

    def _disable_db_writes(self, reason: str):
        if not self._db_warning_logged:
            logger.warning(reason)
            logger.warning("后续数据库写入已禁用；请检查 MySQL 服务或设置 UI_SKIP_DATABASE=1 后重启应用。")
        self._db_warning_logged = True
        self._db_disabled = True
        self._row_id_future = None
        self._pending_db_updates.clear()
        if hasattr(self, 'score_page') and self.score_page:
            try:
                self.score_page.set_force_mock(True)
            except Exception as exc:
                logger.debug("切换分数页数据模式失败: %s", exc)

    def _handle_db_failure(self, error: Exception, context: str):
        logger.error(f"{context}: {error}")
        message = str(error)
        lower = message.lower()
        if any(keyword in lower for keyword in ["10061", "2003", "connection refused", "econnrefused", "timeout"]):
            self._disable_db_writes("检测到数据库连接被拒绝，已暂停后续数据库写入以避免界面卡顿。")
        elif "skip_database" in lower or "disabled" in lower:
            self._disable_db_writes(message or "数据库写入已禁用")

    def _send_db_command(self, action: str, payload: dict, *, context: str,
                          on_success=None):
        if self._db_disabled:
            return None
        try:
            client = get_backend_client()
        except Exception as exc:
            self._handle_db_failure(exc, context)
            return None

        future = client.send_command_future(action, payload)

        def _dispatch_result(fut):
            try:
                result = fut.result()
            except Exception as exc:
                # 修复闭包变量捕获问题：使用默认参数捕获 exc
                self._invoke_later(lambda error=exc, ctx=context: self._handle_db_failure(error, ctx))
                return
            if on_success:
                # 同样修复 result 的捕获
                self._invoke_later(lambda res=result: on_success(res or {}))

        future.add_done_callback(_dispatch_result)
        return future

    def _flush_pending_db_updates(self, row_id: int) -> None:
        if not self._pending_db_updates:
            return
        callbacks = list(self._pending_db_updates)
        self._pending_db_updates.clear()
        for callback in callbacks:
            try:
                callback(row_id)
            except Exception as exc:
                logger.error(f"延迟数据库更新执行失败: {exc}")

    def _ensure_db_row(self):
        """确保数据库记录已创建（仅创建一次，后续使用更新）"""
        if self._db_disabled or self.row_id:
            return
        if self._row_id_future:
            logger.debug("数据库记录创建请求已在处理中，跳过重复创建")
            return

        # 只包含必填字段，其他数据通过后续更新添加
        payload = {
            "name": self.current_user or 'anonymous',
        }

        def _on_created(result: dict):
            row_id = result.get("row_id")
            if not row_id:
                logger.warning("数据库返回的记录ID无效，后续更新将被忽略。")
                return
            self.row_id = row_id
            self._row_id_future = None
            logger.info(f"✅ 数据库记录已创建，ID: {row_id}")
            # 执行所有待处理的更新
            self._flush_pending_db_updates(row_id)

        logger.info("📝 创建新的数据库记录...")
        self._row_id_future = self._send_db_command(
            "db.insert_test_record",
            payload,
            context="创建数据库记录失败",
            on_success=_on_created,
        )

    def _queue_db_update(self, update_payload: dict, context: str) -> None:
        if self._db_disabled:
            return

        def _dispatch(row_id: int) -> None:
            payload = dict(update_payload)
            payload["row_id"] = row_id
            self._send_db_command("db.update_test_record", payload, context=context)

        if self.row_id:
            _dispatch(self.row_id)
        else:
            self._pending_db_updates.append(_dispatch)
            self._ensure_db_row()

    # --- UI 创建辅助方法 ---
    def _create_step_navigator(self):
        container = QWidget()
        container.setObjectName("card")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(scale(12), scale(10), scale(12), scale(10))
        self.step_labels = []
        self.step_opacity_effects = []
        for i, step_name in enumerate(self.steps):
            widget, number_label, text_label = QWidget(), QLabel(str(i + 1)), QLabel(step_name)
            h_layout = QHBoxLayout(widget)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(scale(4))
            number_label.setAlignment(Qt.AlignCenter)
            number_label.setFixedSize(35, 35)
            h_layout.addWidget(number_label)
            h_layout.addWidget(text_label)
            self.step_labels.append((number_label, text_label))

            # 添加透明度效果
            opacity_effect = QGraphicsOpacityEffect(number_label)
            number_label.setGraphicsEffect(opacity_effect)
            self.step_opacity_effects.append(opacity_effect)

            layout.addWidget(widget, 1)
            if i < len(self.steps) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setObjectName("separatorLine")
                layout.addWidget(line)
        return container

    def _create_question_progress_bar(self):
        container = QWidget()
        container.setObjectName("card")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(scale(12), scale(8), scale(12), scale(8))

        self.question_dots = []
        for i in range(len(self.questions)):
            dot = QLabel()
            dot.setFixedSize(24, 24)
            # 初始黑点
            dot.setPixmap(qta.icon('fa5s.circle', color='#212121').pixmap(20, 20))
            self.question_dots.append(dot)
            layout.addWidget(dot, 0, Qt.AlignCenter)
        return container

    def mark_question_done(self, index: int):
        """将指定题目标记为绿色对号"""
        if 0 <= index < len(self.question_dots):
            dot = self.question_dots[index]
            # 设置绿色对号 pixmap
            pixmap = qta.icon('fa5s.check', color='#4CAF50').pixmap(20, 20)
            dot.setPixmap(pixmap)

            # 淡入动画
            effect = QGraphicsOpacityEffect(dot)
            dot.setGraphicsEffect(effect)
            anim = QPropertyAnimation(effect, b"opacity", self)
            anim.setDuration(400)
            anim.setStartValue(0)
            anim.setEndValue(1)
            anim.start()
            self._dot_animations.append(anim)  # 保留引用

    def _create_main_content_area(self):
        container = QWidget()
        self.content_layout = QHBoxLayout(container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(scale(15))
        
        # 创建左侧摄像头视图（内嵌模式）
        self.camera_widget = self._create_camera_view()
        self.content_layout.addWidget(self.camera_widget, 0)
        
        self.answer_stack = QStackedWidget()
        self._create_answer_area_widgets()
        self.content_layout.addWidget(self.answer_stack, 1)

        self.score_page = ScorePage(username=self.current_user)
        self.answer_stack.addWidget(self.score_page)

        return container

    def _create_camera_view(self):
        """创建摄像头视图，包含画面和疲劳度、脑负荷信息"""
        inner_widget = QWidget()
        vlayout = QVBoxLayout(inner_widget)
        vlayout.setSpacing(scale(8))
        vlayout.setContentsMargins(0, 0, 0, 0)

        vlayout.addStretch(1)

        # 摄像头画面 - 缩小尺寸以匹配右侧高度
        cam_width, cam_height = scale_size(560, 420)
        self.camera_preview = CameraPreviewWidget(cam_width, cam_height, placeholder_text="摄像头画面加载中...")
        self.camera_preview.label.setObjectName("cameraView")
        self.camera_preview.label.setStyleSheet(
            """
            QLabel#cameraView {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 10px;
                font-size: 14px;
            }
            """
        )
        vlayout.addWidget(self.camera_preview, 0, Qt.AlignCenter)

        # 疲劳度信息容器 - 调整尺寸和间距
        fatigue_container = QFrame()
        fatigue_container.setObjectName("fatigueContainer")
        fatigue_container.setFixedWidth(cam_width)
        fatigue_container.setStyleSheet("""
                  QFrame#fatigueContainer {
                      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                          stop:0 #ffffff, stop:1 #f8f9fa);
                      border: 2px solid #e0e0e0;
                      border-radius: 10px;
                      padding: 10px;
                  }
              """)

        fatigue_layout = QVBoxLayout(fatigue_container)
        fatigue_layout.setSpacing(scale(6))
        margin = scale(6)
        fatigue_layout.setContentsMargins(margin, margin, margin, margin)

        # 疲劳度标题
        title_label = QLabel("🧠 疲劳度监测")
        title_font = QFont()
        title_font.setPointSize(scale_font(11))
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 4px;")
        fatigue_layout.addWidget(title_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #bdc3c7;")
        fatigue_layout.addWidget(separator)

        # 疲劳度显示（大号）
        self.fatigue_info_label = QLabel("疲劳度: --")
        info_font = QFont()
        info_font.setPointSize(scale_font(13))
        info_font.setBold(True)
        self.fatigue_info_label.setFont(info_font)
        self.fatigue_info_label.setAlignment(Qt.AlignCenter)
        self.fatigue_info_label.setStyleSheet("""
                  QLabel {
                      color: #7f8c8d;
                      padding: 8px;
                      background-color: #ecf0f1;
                      border-radius: 8px;
                  }
              """)
        fatigue_layout.addWidget(self.fatigue_info_label)

        vlayout.addWidget(fatigue_container, 0, Qt.AlignCenter)

        # 脑负荷信息容器 - 与疲劳度信息容器相同
        brain_load_container = QFrame()
        brain_load_container.setObjectName("brainLoadContainer")
        brain_load_container.setFixedWidth(cam_width)
        brain_load_container.setStyleSheet("""
                  QFrame#brainLoadContainer {
                      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                          stop:0 #ffffff, stop:1 #f8f9fa);
                      border: 2px solid #e0e0e0;
                      border-radius: 10px;
                      padding: 10px;
                  }
              """)

        brain_load_layout = QVBoxLayout(brain_load_container)
        brain_load_layout.setSpacing(scale(6))
        brain_load_layout.setContentsMargins(margin, margin, margin, margin)

        # 脑负荷标题
        brain_load_title_label = QLabel("🧠 脑负荷监测")
        brain_load_title_font = QFont()
        brain_load_title_font.setPointSize(scale_font(11))
        brain_load_title_font.setBold(True)
        brain_load_title_label.setFont(brain_load_title_font)
        brain_load_title_label.setAlignment(Qt.AlignCenter)
        brain_load_title_label.setStyleSheet("color: #2c3e50; padding: 4px;")
        brain_load_layout.addWidget(brain_load_title_label)

        # 分隔线
        brain_load_separator = QFrame()
        brain_load_separator.setFrameShape(QFrame.HLine)
        brain_load_separator.setFrameShadow(QFrame.Sunken)
        brain_load_separator.setStyleSheet("background-color: #bdc3c7;")
        brain_load_layout.addWidget(brain_load_separator)

        # 脑负荷显示（大号）
        self.brain_load_info_label = QLabel("脑负荷: --")
        brain_load_info_font = QFont()
        brain_load_info_font.setPointSize(scale_font(13))
        brain_load_info_font.setBold(True)
        self.brain_load_info_label.setFont(brain_load_info_font)
        self.brain_load_info_label.setAlignment(Qt.AlignCenter)
        self.brain_load_info_label.setStyleSheet("""
                  QLabel {
                      color: #7f8c8d;
                      padding: 8px;
                      background-color: #ecf0f1;
                      border-radius: 8px;
                  }
              """)
        brain_load_layout.addWidget(self.brain_load_info_label)

        vlayout.addWidget(brain_load_container, 0, Qt.AlignCenter)

        # 实时监测中... 提示放在疲劳度和脑负荷显示下方
        tip_label = QLabel("实时监测中...")
        tip_font = QFont()
        tip_font.setPointSize(scale_font(8))
        tip_label.setFont(tip_font)
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("color: #95a5a6; padding: 4px;")
        vlayout.addWidget(tip_label)

        vlayout.addStretch(1)

        # ✅ 新增一层水平布局，用于让整个块在水平方向居中
        outer_widget = QWidget()
        hlayout = QHBoxLayout(outer_widget)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addStretch(1)  # 左侧空白
        hlayout.addWidget(inner_widget)  # 中间摄像头列
        hlayout.addStretch(1)  # 右侧空白

        return outer_widget

    def _create_camera_view_for_schulte(self):
        """为舒尔特页面创建摄像头视图（与第一页保持一致）"""
        inner_widget = QWidget()
        vlayout = QVBoxLayout(inner_widget)
        vlayout.setSpacing(scale(8))
        vlayout.setContentsMargins(0, 0, 0, 0)

        # 顶部拉伸
        vlayout.addStretch(1)

        # 摄像头画面（与第一页相同尺寸）- 缩小尺寸以匹配右侧高度
        cam_width, cam_height = scale_size(560, 420)
        self.schulte_camera_preview = CameraPreviewWidget(cam_width, cam_height, placeholder_text="摄像头画面")
        self.schulte_camera_preview.label.setObjectName("schulteCameraView")
        self.schulte_camera_preview.label.setStyleSheet(
            """
            QLabel#schulteCameraView {
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 2px solid #34495e;
                border-radius: 10px;
                font-size: 14px;
            }
            """
        )
        vlayout.addWidget(self.schulte_camera_preview, 0, Qt.AlignCenter)

        # 疲劳度信息容器（与第一页相同样式）- 调整尺寸和间距
        fatigue_container = QFrame()
        fatigue_container.setObjectName("schulteFatigueContainer")
        fatigue_container.setFixedWidth(cam_width)
        fatigue_container.setStyleSheet("""
                  QFrame#schulteFatigueContainer {
                      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                          stop:0 #ffffff, stop:1 #f8f9fa);
                      border: 2px solid #e0e0e0;
                      border-radius: 10px;
                      padding: 10px;
                  }
              """)

        fatigue_layout = QVBoxLayout(fatigue_container)
        fatigue_layout.setSpacing(scale(6))
        margin = scale(6)
        fatigue_layout.setContentsMargins(margin, margin, margin, margin)

        # 疲劳度标题
        title_label = QLabel("🧠 疲劳度监测")
        title_font = QFont()
        title_font.setPointSize(scale_font(11))
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 4px;")
        fatigue_layout.addWidget(title_label)

        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #bdc3c7;")
        fatigue_layout.addWidget(separator)

        # 疲劳度显示（大号）
        self.schulte_fatigue_label = QLabel("疲劳度: --")
        info_font = QFont()
        info_font.setPointSize(scale_font(13))
        info_font.setBold(True)
        self.schulte_fatigue_label.setFont(info_font)
        self.schulte_fatigue_label.setAlignment(Qt.AlignCenter)
        self.schulte_fatigue_label.setStyleSheet("""
                  QLabel {
                      color: #7f8c8d;
                      padding: 8px;
                      background-color: #ecf0f1;
                      border-radius: 8px;
                  }
              """)
        fatigue_layout.addWidget(self.schulte_fatigue_label)

        # 脑负荷信息容器 - 与疲劳度信息容器相同
        brain_load_container = QFrame()
        brain_load_container.setObjectName("schulteBrainLoadContainer")
        brain_load_container.setFixedWidth(cam_width)
        brain_load_container.setStyleSheet("""
                  QFrame#schulteBrainLoadContainer {
                      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                          stop:0 #ffffff, stop:1 #f8f9fa);
                      border: 2px solid #e0e0e0;
                      border-radius: 10px;
                      padding: 10px;
                  }
              """)

        brain_load_layout = QVBoxLayout(brain_load_container)
        brain_load_layout.setSpacing(scale(6))
        brain_load_layout.setContentsMargins(margin, margin, margin, margin)

        # 脑负荷标题
        brain_load_title_label = QLabel("🧠 脑负荷监测")
        brain_load_title_font = QFont()
        brain_load_title_font.setPointSize(scale_font(11))
        brain_load_title_font.setBold(True)
        brain_load_title_label.setFont(brain_load_title_font)
        brain_load_title_label.setAlignment(Qt.AlignCenter)
        brain_load_title_label.setStyleSheet("color: #2c3e50; padding: 4px;")
        brain_load_layout.addWidget(brain_load_title_label)

        # 分隔线
        brain_load_separator = QFrame()
        brain_load_separator.setFrameShape(QFrame.HLine)
        brain_load_separator.setFrameShadow(QFrame.Sunken)
        brain_load_separator.setStyleSheet("background-color: #bdc3c7;")
        brain_load_layout.addWidget(brain_load_separator)

        # 脑负荷显示（大号）
        self.schulte_brain_load_label = QLabel("脑负荷: --")
        brain_load_info_font = QFont()
        brain_load_info_font.setPointSize(scale_font(13))
        brain_load_info_font.setBold(True)
        self.schulte_brain_load_label.setFont(brain_load_info_font)
        self.schulte_brain_load_label.setAlignment(Qt.AlignCenter)
        self.schulte_brain_load_label.setStyleSheet("""
                  QLabel {
                      color: #7f8c8d;
                      padding: 8px;
                      background-color: #ecf0f1;
                      border-radius: 8px;
                  }
              """)
        brain_load_layout.addWidget(self.schulte_brain_load_label)

        # 提示信息
        tip_label = QLabel("实时监测中...")
        tip_font = QFont()
        tip_font.setPointSize(scale_font(8))
        tip_label.setFont(tip_font)
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("color: #95a5a6; padding: 4px;")
        brain_load_layout.addWidget(tip_label)

        # 将疲劳度和脑负荷容器添加到布局
        vlayout.addWidget(fatigue_container, 0, Qt.AlignCenter)
        vlayout.addWidget(brain_load_container, 0, Qt.AlignCenter)

        vlayout.addStretch(1)

        # ✅ 新增一层水平布局，用于让整个块在水平方向居中
        outer_widget = QWidget()
        hlayout = QHBoxLayout(outer_widget)
        hlayout.setContentsMargins(0, 0, 0, 0)
        hlayout.addStretch(1)  # 左侧空白
        hlayout.addWidget(inner_widget)  # 中间摄像头列
        hlayout.addStretch(1)  # 右侧空白

        return outer_widget

    def _create_answer_area_widgets(self):
        # 语音答题页面
        page_qna = QWidget()
        layout_qna = QVBoxLayout(page_qna)
        layout_qna.setAlignment(Qt.AlignCenter)
        layout_qna.setSpacing(scale(15))

        # 题目标签
        self.lbl_question = QLabel("Question Text")
        self.lbl_question.setObjectName("questionLabel")
        self.lbl_question.setWordWrap(True)
        self.lbl_question.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(20)
        font.setBold(True)
        self.lbl_question.setFont(font)

        # 麦克风按钮
        self.btn_mic = QPushButton()
        self.btn_mic.setObjectName("micButtonCallToAction")
        self.btn_mic.setFixedSize(130, 130)
        self.btn_mic.setIconSize(QSize(60, 60))
        self.btn_mic.setCursor(Qt.PointingHandCursor)
        self.btn_mic.setIcon(qta.icon('fa5s.microphone-alt', color='white'))

        # 音量显示
        self.audio_level = AudioLevelMeter()
        self.audio_level.setFixedWidth(350)

        # 录音状态标签
        self.lbl_recording_status = QLabel("请点击上方按钮开始录音")
        self.lbl_recording_status.setObjectName("statusLabel")
        self.lbl_recording_status.setAlignment(Qt.AlignCenter)

        # 布局顺序
        layout_qna.addStretch(2)
        layout_qna.addWidget(self.lbl_question)
        layout_qna.addStretch(1)
        layout_qna.addWidget(self.btn_mic, 0, Qt.AlignCenter)
        layout_qna.addWidget(self.audio_level, 0, Qt.AlignCenter)
        layout_qna.addWidget(self.lbl_recording_status, 0, Qt.AlignCenter)
        layout_qna.addStretch(2)

        self.answer_stack.addWidget(page_qna)

        # 血压测试页面
        page_blood_pressure = self._create_blood_pressure_page()
        self.answer_stack.addWidget(page_blood_pressure)

        # 舒特格测试页面
        page_schulte = self._create_schulte_page()
        self.answer_stack.addWidget(page_schulte)

        # 信息确认页面
        page_confirm = self._create_info_page()
        self.answer_stack.addWidget(page_confirm)

        # 分数展示页面
        page_score = self._create_score_page()
        self.answer_stack.addWidget(page_score)

    def _create_info_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.check-circle', color='#4CAF50').pixmap(80, 80))
        title_label = QLabel("信息已记录")
        title_label.setObjectName("h1")
        subtitle_label = QLabel("系统已保存您的回答，请进入下一步。")
        subtitle_label.setObjectName("subtitle")
        layout.addStretch()
        layout.addWidget(icon_label, 0, Qt.AlignCenter)
        layout.addWidget(title_label, 0, Qt.AlignCenter)
        layout.addWidget(subtitle_label, 0, Qt.AlignCenter)
        layout.addStretch()
        return page

    def _create_blood_pressure_page(self):
        """创建血压脉搏测试页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # 标题
        title_label = QLabel("血压脉搏测试")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)

        # 说明文字
        description_label = QLabel(
            "请按照左侧血压仪说明，将您的手臂放置在仪器测量位置\n\n点击开始测试按钮开始测量"
        )
        description_label.setObjectName("subtitle")
        description_label.setAlignment(Qt.AlignCenter)
        description_label.setWordWrap(True)

        # 设备状态区域
        status_container = QWidget()
        status_layout = QVBoxLayout(status_container)
        status_layout.setSpacing(10)

        # 设备连接状态
        self.bp_status_label = QLabel("正在检测血压仪器连接...")
        self.bp_status_label.setObjectName("statusLabel")
        self.bp_status_label.setAlignment(Qt.AlignCenter)

        # 测试进度显示
        self.bp_progress_label = QLabel("等待开始测试")
        self.bp_progress_label.setObjectName("subtitle")
        self.bp_progress_label.setAlignment(Qt.AlignCenter)

        status_layout.addWidget(self.bp_status_label)
        status_layout.addWidget(self.bp_progress_label)

        # 测试控制区域
        control_container = QWidget()
        control_layout = QVBoxLayout(control_container)
        control_layout.setSpacing(15)

        # 开始/停止测试按钮
        self.bp_start_button = QPushButton("开始测试")
        self.bp_start_button.setObjectName("successButton")
        self.bp_start_button.setFixedSize(150, 50)
        self.bp_start_button.clicked.connect(self._toggle_bp_test)
        self.bp_start_button.setEnabled(False)  # 初始禁用

        # 圆形进度指示器
        self.bp_progress_circle = QLabel()
        self.bp_progress_circle.setFixedSize(80, 80)
        self.bp_progress_circle.setAlignment(Qt.AlignCenter)
        self.bp_progress_circle.setStyleSheet("""
            QLabel {
                border: 4px solid #E0E0E0;
                border-radius: 40px;
                background-color: #F5F5F5;
                color: #666;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        self.bp_progress_circle.setText("准备")

        control_layout.addWidget(self.bp_progress_circle, 0, Qt.AlignCenter)
        control_layout.addWidget(self.bp_start_button, 0, Qt.AlignCenter)

        # 结果显示区域
        self.result_container = QWidget()
        self.result_container.setVisible(False)
        result_layout = QVBoxLayout(self.result_container)
        result_layout.setSpacing(15)

        # 结果标题
        result_title = QLabel("测试结果")
        result_title.setObjectName("h2")
        result_title.setAlignment(Qt.AlignCenter)

        # 结果卡片
        self.result_card = QWidget()
        self.result_card.setObjectName("card")
        self.result_card.setFixedSize(400, 200)
        result_card_layout = QVBoxLayout(self.result_card)
        result_card_layout.setSpacing(15)

        # 收缩压
        self.systolic_label = QLabel("收缩压: -- mmHg")
        self.systolic_label.setObjectName("statusLabel")
        self.systolic_label.setAlignment(Qt.AlignCenter)
        self.systolic_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2;")

        # 舒张压
        self.diastolic_label = QLabel("舒张压: -- mmHg")
        self.diastolic_label.setObjectName("statusLabel")
        self.diastolic_label.setAlignment(Qt.AlignCenter)
        self.diastolic_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #1976D2;")

        # 脉搏
        self.pulse_label = QLabel("脉搏: -- 次/分")
        self.pulse_label.setObjectName("statusLabel")
        self.pulse_label.setAlignment(Qt.AlignCenter)
        self.pulse_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #4CAF50;")

        result_card_layout.addWidget(self.systolic_label)
        result_card_layout.addWidget(self.diastolic_label)
        result_card_layout.addWidget(self.pulse_label)

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_card, 0, Qt.AlignCenter)

        # 布局组装
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(status_container, 0, Qt.AlignCenter)
        layout.addWidget(control_container, 0, Qt.AlignCenter)
        layout.addWidget(self.result_container, 0, Qt.AlignCenter)
        layout.addStretch(2)

        # 初始化血压测试相关变量
        self.bp_test_running = False
        self.bp_test_timer = QTimer()
        self.bp_test_timer.timeout.connect(self._update_bp_test_progress)
        self.bp_test_progress = 0
        self.bp_test_duration = 60  # 测试持续时间（秒）

        # 血压测试结果
        self.bp_results = {
            'systolic': None,  # 收缩压
            'diastolic': None,  # 舒张压
            'pulse': None  # 脉搏
        }

        # 启动定时器检测血压仪状态
        self.bp_timer = QTimer()
        self.bp_timer.timeout.connect(self._check_bp_device)
        self.bp_timer.start(1000)  # 每秒检测一次

        return page

    def _check_bp_device(self):
        """
        检测血压仪器连接状态
        尝试检测 maibobo 脉搏仪设备
        """
        if getattr(self, "bp_simulation_enabled", False):
            self.bp_status_label.setText("血压仪器已连接 ✅ (模拟模式)")
            self.bp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.bp_start_button.setEnabled(True)
            self.bp_available_port = "SIMULATION"
            return

        try:
            status = bp_get_status() if HAS_BP_BACKEND else {}
        except Exception as exc:
            logger.debug("查询血压后端状态失败: %s", exc)
            self.bp_status_label.setText("血压后端未响应 ❌")
            self.bp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            if not self.bp_test_running:
                self.bp_start_button.setEnabled(False)
            return

        forced_port = (self.bp_forced_port or "").strip()
        available_ports = status.get("available_ports") or []
        port = forced_port or (status.get("port") or "").strip()
        mode = status.get("mode") or ("simulation" if self.bp_simulation_enabled else "hardware")
        error = status.get("error")
        running = bool(status.get("running"))

        if port:
            self.bp_available_port = port
        elif available_ports:
            self.bp_available_port = available_ports[0]
        else:
            self.bp_available_port = None

        if running:
            label_mode = "模拟模式" if mode == "simulation" else f"端口: {self.bp_available_port or '未知'}"
            self.bp_status_label.setText(f"血压仪器测试中 ⏳ ({label_mode})")
            self.bp_status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
            self.bp_start_button.setEnabled(self.bp_test_running)
            return

        if error and not self.bp_simulation_enabled:
            self.bp_status_label.setText(f"血压仪器不可用 ❌ ({error})")
            self.bp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            if not self.bp_test_running:
                self.bp_start_button.setEnabled(False)
            return

        if self.bp_available_port:
            if mode == "simulation":
                suffix = "模拟模式"
            else:
                suffix = f"端口: {self.bp_available_port}"
            self.bp_status_label.setText(f"血压仪器已连接 ✅ ({suffix})")
            self.bp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if not self.bp_test_running:
                self.bp_start_button.setEnabled(True)
        else:
            self.bp_status_label.setText("血压仪器未连接，请确认设备连接状态 📥")
            self.bp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            if not self.bp_test_running:
                self.bp_start_button.setEnabled(False)

    def _toggle_bp_test(self):
        """切换血压测试状态（开始/停止）"""
        if not self.bp_test_running:
            self._start_bp_test()
        else:
            self._stop_bp_test()

    def _start_bp_test(self):
        """开始血压测试"""
        try:
            if not self.bp_simulation_enabled and not HAS_BP_BACKEND:
                QMessageBox.warning(self, "设备错误", "血压后端服务不可用，无法开始测试")
                return

            if self.bp_simulation_enabled:
                self.bp_status_label.setText("血压仪器已连接 ✅ (模拟模式)")
                self.bp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                if not self.bp_available_port:
                    self.bp_available_port = "SIMULATION"

            self.bp_results = {
                'systolic': None,
                'diastolic': None,
                'pulse': None,
            }

            self.bp_test_running = True
            self.bp_start_button.setText("停止测试")
            self.bp_start_button.setObjectName("finishButton")
            self.bp_start_button.style().unpolish(self.bp_start_button)
            self.bp_start_button.style().polish(self.bp_start_button)

            self.bp_test_progress = 0
            self.bp_progress_label.setText("测试进行中...")
            self.bp_progress_circle.setText("0%")

            self.result_container.setVisible(False)

            self.bp_test_timer.start(100)

            self._bp_error_reported = False
            self._bp_snapshot_warned = False

            port_candidate = (self.bp_forced_port or self.bp_available_port or "").strip() or None
            try:
                response = bp_start_measurement(
                    port=port_candidate,
                    simulation=bool(self.bp_simulation_enabled),
                    allow_simulation=True,
                    timeout=1,
                )
                mode = response.get("mode", "hardware")
                resolved_port = response.get("port") or port_candidate or "SIMULATION"
                self.bp_available_port = resolved_port
                self.bp_measurement_active = True
                if not self.bp_poll_timer.isActive():
                    self.bp_poll_timer.start()
                logger.info("血压测试已开始（模式：%s，端口：%s）", mode, resolved_port)
                if mode == "simulation":
                    self.bp_status_label.setText("血压仪器已连接 ✅ (模拟模式)")
                    self.bp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            except Exception as exc:
                logger.error("启动血压测试失败: %s", exc)
                QMessageBox.critical(self, "设备错误", f"启动血压仪失败: {exc}")
                self._stop_bp_test()
                return

            logger.info("血压测试已开始")

        except Exception as e:
            logger.error(f"开始血压测试失败: {e}")
            self._stop_bp_test()

    def _stop_bp_test(self):
        """停止血压测试"""
        try:
            self.bp_test_running = False
            self.bp_start_button.setText("开始测试")
            self.bp_start_button.setObjectName("successButton")
            self.bp_start_button.style().unpolish(self.bp_start_button)
            self.bp_start_button.style().polish(self.bp_start_button)

            self.bp_test_timer.stop()

            if self.bp_measurement_active:
                try:
                    bp_stop_measurement()
                except Exception as exc:
                    logger.debug("停止血压后端失败: %s", exc)
            self._stop_bp_polling()

            self.bp_progress_label.setText("测试已停止")
            self.bp_progress_circle.setText("停止")

            logger.info("血压测试已停止")

        except Exception as e:
            logger.error(f"停止血压测试失败: {e}")

    def _stop_bp_polling(self) -> None:
        try:
            if self.bp_poll_timer.isActive():
                self.bp_poll_timer.stop()
        except Exception as exc:
            logger.debug(f"停止血压轮询时出错: {exc}")
        self.bp_measurement_active = False

    def _poll_bp_snapshot(self) -> None:
        if not HAS_BP_BACKEND or not self.bp_test_running:
            self._stop_bp_polling()
            return

        try:
            snapshot = bp_get_snapshot()
        except Exception as exc:
            if not self._bp_snapshot_warned:
                logger.debug(f"获取血压快照失败: {exc}")
                self._bp_snapshot_warned = True
            return

        status = (snapshot.get("status") or "").lower()
        latest = snapshot.get("latest") or {}
        error = snapshot.get("error")
        mode = snapshot.get("mode")

        if status != "running":
            self.bp_measurement_active = False

        if mode == "simulation" and not self.bp_simulation_enabled:
            self.bp_status_label.setText("血压仪器已连接 ✅ (模拟模式)")
            self.bp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")

        if error:
            error_text = str(error)
            if "maibobo" in error_text.lower():
                self._auto_skip_bp_test("未检测到血压仪驱动，自动跳过此环节")
                return

            if not self._bp_error_reported:
                logger.error(f"血压监测发生错误: {error}")
                QMessageBox.warning(self, "血压测试失败", error_text)
                self._bp_error_reported = True
            self._stop_bp_test()
            return

        if latest and self.bp_results.get('systolic') is None:
            try:
                systolic = int(latest.get('systolic'))
                diastolic = int(latest.get('diastolic'))
                pulse = int(latest.get('pulse'))
            except Exception as exc:
                logger.debug(f"解析血压快照失败: {exc}")
            else:
                self.bp_results = {
                    'systolic': systolic,
                    'diastolic': diastolic,
                    'pulse': pulse,
                }
                logger.info(
                    "血压测试完成: 收缩压=%s, 舒张压=%s, 脉搏=%s",
                    systolic,
                    diastolic,
                    pulse,
                )
                self._invoke_later(self._complete_bp_test)
                return

        if status in {"idle", "completed", "error"} and not latest:
            self._stop_bp_polling()

    def _complete_bp_test(self):
        """完成血压测试，显示结果"""
        try:
            self._stop_bp_test()

            if (hasattr(self, 'bp_results') and
                    self.bp_results and
                    self.bp_results.get('systolic') is not None):

                self.systolic_label.setText(f"收缩压: {self.bp_results['systolic']} mmHg")
                self.diastolic_label.setText(f"舒张压: {self.bp_results['diastolic']} mmHg")
                self.pulse_label.setText(f"脉搏: {self.bp_results['pulse']} 次/分")

                systolic = self.bp_results['systolic']
                diastolic = self.bp_results['diastolic']

                if systolic < 120 and diastolic < 80:
                    color = "#4CAF50"
                elif systolic < 130 and diastolic < 85:
                    color = "#FF9800"
                else:
                    color = "#F44336"

                self.systolic_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
                self.diastolic_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

                self.result_container.setVisible(True)

                self.bp_progress_label.setText("测试完成 ✅")
                self.bp_progress_circle.setText("完成")
                self.bp_progress_circle.setStyleSheet("""
                    QLabel {
                        border: 4px solid #4CAF50;
                        border-radius: 40px;
                        background-color: #E8F5E8;
                        color: #4CAF50;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
                self.btn_next.setText("进入舒特格测试")
                self.btn_next.setEnabled(True)

                logger.info(f"血压测试完成: 收缩压={systolic}, 舒张压={diastolic}, 脉搏={self.bp_results['pulse']}")

                self._save_bp_results_to_db()

            else:
                self.bp_progress_label.setText("测试失败 ❌")
                self.bp_progress_circle.setText("失败")
                self.bp_progress_circle.setStyleSheet("""
                    QLabel {
                        border: 4px solid #F44336;
                        border-radius: 40px;
                        background-color: #FFEBEE;
                        color: #F44336;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)

                QMessageBox.warning(self, "测试失败", "未能获取有效的血压数据，请检查设备连接或重新测试")

        except Exception as e:
            logger.error(f"完成血压测试失败: {e}")
            self.bp_progress_label.setText("测试出错 ❌")
            self.bp_progress_circle.setText("错误")

    def _update_bp_test_progress(self):
        """更新血压测试进度"""
        if not self.bp_test_running:
            return

        try:
            self.bp_test_progress += 0.1
            progress_percent = min(100, int((self.bp_test_progress / self.bp_test_duration) * 100))

            self.bp_progress_circle.setText(f"{progress_percent}%")

            if progress_percent < 30:
                color = "#FF9800"
            elif progress_percent < 70:
                color = "#2196F3"
            else:
                color = "#4CAF50"

            self.bp_progress_circle.setStyleSheet(f"""
                QLabel {{
                    border: 4px solid {color};
                    border-radius: 40px;
                    background-color: #F5F5F5;
                    color: {color};
                    font-size: 12px;
                    font-weight: bold;
                }}
            """)

            if self.bp_test_progress >= self.bp_test_duration:
                logger.warning("血压测试超时")
                self._complete_bp_test()

        except Exception as e:
            logger.error(f"更新血压测试进度失败: {e}")

    def _save_bp_results_to_db(self):
        """将血压测试结果保存到数据库"""
        try:
            if not self.bp_results['systolic']:
                logger.warning("没有血压测试结果可保存")
                return

            blood_data = f"{self.bp_results['systolic']}/{self.bp_results['diastolic']}/{self.bp_results['pulse']}"

            self._queue_db_update({"blood": blood_data}, "保存血压测试结果到数据库失败")
            logger.info(f"血压测试结果将通过后端写入数据库: {blood_data}")

        except Exception as e:
            logger.error(f"保存血压测试结果失败: {e}")

    def _auto_skip_bp_test(self, reason: str) -> None:
        logger.warning("血压测试无法正常运行：%s，已自动跳过。", reason)
        self.bp_results = {
            'systolic': 120,
            'diastolic': 80,
            'pulse': 75,
        }
        self.bp_status_label.setText(reason)
        self.bp_status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self._bp_error_reported = True
        self._complete_bp_test()

    def _handle_debug_shortcut(self) -> bool:
        try:
            if self.current_step == 0:
                logger.info("🔧 测试后门触发：按下 Q，语音问答视为完成")
                
                # 停止音视频录制并获取路径
                try:
                    logger.info("📹 正在停止音视频录制...")
                    av_stop_recording()
                    self._audio_paths = av_get_audio_paths()
                    self._video_paths = av_get_video_paths()
                    logger.info(f"✅ 音视频录制已停止: {len(self._audio_paths)} 个音频, {len(self._video_paths)} 个视频")
                except Exception as e:
                    logger.error(f"停止音视频录制失败: {e}")
                    # 初始化为空列表,避免后续错误
                    if not hasattr(self, '_audio_paths'):
                        self._audio_paths = []
                    if not hasattr(self, '_video_paths'):
                        self._video_paths = []
                
                # 保存音视频路径到数据库
                self._persist_av_paths_to_db()
                
                # 切换到下一步
                self.current_step = 1
                self.update_step_ui()
                return True

            if self.current_step == 1:
                logger.info("测试后门触发：按下 Q，血压测试视为完成")
                self.bp_results = {
                    'systolic': 120,
                    'diastolic': 80,
                    'pulse': 75,
                }
                self._complete_bp_test()
                # 立即推进到下一步，避免重复触发情绪分析
                self.current_step = 2
                self.update_step_ui()
                return True

            if self.current_step == 2:
                logger.info("测试后门触发：按下 Q，舒尔特测试视为完成")
                self._on_schulte_result(30.0, 85.0)
                self._on_schulte_completed()
                return True
        except Exception as e:
            logger.warning(f"执行调试快捷操作失败: {e}")
        return False

    def keyPressEvent(self, event):
        """全局监听键盘事件，用于测试调试后门"""
        if event.key() == Qt.Key_Q and not event.isAutoRepeat():
            if self._handle_debug_shortcut():
                return
        super().keyPressEvent(event)

    def _create_schulte_page(self):
        """创建舒特格测试页面（带摄像头和疲劳度显示）"""
        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(20)
        
        # 左侧：摄像头和疲劳度（复用第一页的样式）
        self.schulte_camera_widget = self._create_camera_view_for_schulte()
        main_layout.addWidget(self.schulte_camera_widget, 0)
        
        # 右侧：舒尔特测试区域
        schulte_container = QWidget()
        schulte_layout = QVBoxLayout(schulte_container)
        schulte_layout.setAlignment(Qt.AlignCenter)
        schulte_layout.setSpacing(20)
        
        self.schulte_widget = SchulteGridWidget(self.current_user)
        self.schulte_widget.test_completed.connect(self._on_schulte_completed)
        self.schulte_widget.test_result_ready.connect(self._on_schulte_result)
        
        schulte_layout.addWidget(self.schulte_widget)
        main_layout.addWidget(schulte_container, 1)
        
        return page

    def _create_score_page(self):
        page_score = QWidget()
        layout_score = QVBoxLayout(page_score)
        layout_score.setAlignment(Qt.AlignCenter)
        layout_score.setSpacing(5)
        trophy_icon = QLabel()
        trophy_icon.setPixmap(qta.icon('fa5s.trophy', color='#FFC107').pixmap(50, 50))
        score_title = QLabel("本次评估分数")
        score_title.setObjectName("h2")
        self.score_value_label = QLabel("0")
        self.score_value_label.setObjectName("scoreValue")
        self.score_chart = ScoreChartWidget()
        self.score_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout_score.addStretch(1)
        layout_score.addWidget(trophy_icon, 0, Qt.AlignCenter)
        layout_score.addWidget(score_title, 0, Qt.AlignCenter)
        layout_score.addWidget(self.score_value_label, 0, Qt.AlignCenter)
        layout_score.addWidget(self.score_chart, 5)
        layout_score.addStretch(1)
        return page_score

    def _create_bottom_buttons(self):
        container = QWidget()
        layout = QHBoxLayout(container)
        self.btn_next = QPushButton("下一题")
        self.btn_next.setObjectName("successButton")
        self.btn_next.setIcon(qta.icon('fa5s.arrow-right'))
        self.btn_next.setFixedWidth(200)
        self.btn_finish = QPushButton("完成评估")
        self.btn_finish.setObjectName("finishButton")
        self.btn_finish.setIcon(qta.icon('fa5s.flag-checkered'))
        self.btn_finish.setFixedWidth(200)
        self.btn_finish.setVisible(False)
        layout.addWidget(self.btn_next)
        layout.addWidget(self.btn_finish)
        return container

    def update_step_ui(self):
        for i, (num_label, text_label) in enumerate(self.step_labels):
            target_opacity = 1.0 if i == self.current_step else 0.5
            anim = QPropertyAnimation(self.step_opacity_effects[i], b"opacity")
            anim.setDuration(400)
            anim.setStartValue(self.step_opacity_effects[i].opacity())
            anim.setEndValue(target_opacity)
            anim.setEasingCurve(QEasingCurve.InOutQuad)
            anim.start(QPropertyAnimation.DeleteWhenStopped)

            if i == self.current_step:
                num_label.setStyleSheet("""
                        QLabel {
                            background-color: #1976D2;
                            color: white;
                            border-radius: 17px;
                            font-weight: bold;
                            font-size: 16px;
                        }
                    """)
                text_label.setStyleSheet("color: #1976D2; font-weight: bold;")
            else:
                num_label.setStyleSheet("""
                        QLabel {
                            background-color: #E0E0E0;
                            color: #212121;
                            border-radius: 17px;
                            font-weight: normal;
                            font-size: 16px;
                        }
                    """)
                text_label.setStyleSheet("color: #757575; font-weight: normal;")

        self.question_container.setVisible(self.current_step == 0)
        if self.current_step == 0:
            for i, dot in enumerate(self.question_dots):
                if i < self.current_question:
                    self.mark_question_done(i)
                elif i == self.current_question:
                    icon = qta.icon('fa5s.circle', color='#212121')
                    dot.setPixmap(icon.pixmap(24, 24))
                    dot.setAlignment(Qt.AlignCenter)
                else:
                    icon = qta.icon('fa5s.circle', color='#212121')
                    dot.setPixmap(icon.pixmap(24, 24))
                    dot.setAlignment(Qt.AlignCenter)

        main_window = self.window()
        brain_load_bar = getattr(main_window, "brain_load_bar", None)
        if brain_load_bar:
            brain_load_bar.setVisible(self.current_step != 3)
        
        # 控制摄像头组件的可见性（血压阶段隐藏疲劳度/摄像头）
        self.camera_widget.setVisible(self.current_step == 0)
        if hasattr(self, 'schulte_camera_widget'):
            self.schulte_camera_widget.setVisible(self.current_step == 2)

        if self.current_step == 0:
            if not self.audio_timer.isActive():
                self.audio_timer.start(50)
        else:
            if self.audio_timer.isActive():
                self.audio_timer.stop()
                self.audio_level.set_level(0)

        if self.current_step == 0:
            self.answer_stack.setCurrentIndex(0)
            self.lbl_question.setText(self.questions[self.current_question])
            self.btn_next.setText(
                "下一题" if self.current_question < len(self.questions) - 1 else "完成答题"
            )
            self.btn_next.setVisible(True)
            self.btn_next.setEnabled(False)
            self.btn_finish.setVisible(False)
            if self.test_started:
                self._speak_current_question()

        elif self.current_step == 1:
            self.answer_stack.setCurrentIndex(1)
            if hasattr(self, 'bp_results') and self.bp_results['systolic'] is not None:
                self.btn_next.setText("进入舒特格测试")
                self.btn_next.setEnabled(True)
            else:
                self.btn_next.setText("请先完成血压测试")
                self.btn_next.setEnabled(False)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
            
            # 📍 在切换到血压测试时，先保存语音识别结果，再触发情绪分析
            self._save_speech_recognition_results()
            self._trigger_emotion_analysis()
        elif self.current_step == 2:
            self.answer_stack.setCurrentIndex(2)
            self.btn_next.setVisible(False)
            self.btn_finish.setVisible(False)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
        elif self.current_step == 3:
            self.answer_stack.setCurrentWidget(self.score_page)
            # 异步更新分数页数据，避免阻塞UI
            def update_scores_async():
                try:
                    # 正确的调用顺序：先设置用户，再发送测试结果，最后更新显示
                    self.score_page._set_user(self.current_user)
                    self._send_scores_to_score_page()
                    self.score_page._update_scores()
                except Exception as e:
                    logger.error(f"更新分数页失败: {e}")
            
            # 先更新基本UI，然后异步加载数据
            self.btn_next.setVisible(False)
            self.btn_finish.setVisible(True)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
            self.score_value_label.setText(str(self.score) if self.score is not None else "计算中...")
            self.score_chart.update_chart(self.history_scores)
            
            # 使用 QTimer 异步更新分数页，不阻塞UI切换
            self._invoke_later(update_scores_async, 50)

        # 多模态监控生命周期管理：独立于测试流程
        # 只在真正结束时停止，其他时候让定时器自然运行
        if HAS_MULTIMODAL:
            if self.current_step == 3:
                # 测试完全结束，停止监控
                self._stop_multimodal_monitoring()
                logger.debug("update_step_ui → 第3步完成，多模态监控已停止")
            # 移除所有其他干预：让监控独立运行，不受步骤切换影响
            # 这样可以避免在血压测试、答题等操作时意外停止数据更新

        self._update_camera_previews_for_step()

    def start_test(self):
        # 摄像头预览在 AV 采集准备好后启动
        self.audio_timer.start(50)
        self.current_step = 0
        self.current_question = 0
        self.btn_finish.setVisible(False)

        self.spoken_questions = set()
        
        # 重置分数累积列表
        self._fatigue_scores_list = []
        self._brain_load_scores_list = []
        self._emotion_score = None
        self._emotion_analysis_triggered = False  # 重置情绪分析触发标志
        logger.info("已重置分数累积列表和情绪分析标志")

        if HAS_SPEECH_RECOGNITION:
            try:
                stop_recognition()
                clear_recognition_results()
            except Exception as exc:
                logger.warning("重置语音识别队列失败: %s", exc)

        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)

        try:
            self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = 'recordings'
            user_dir = self.current_user or 'anonymous'
            self.session_dir = _build_session_dir(base_dir, user_dir, self.session_timestamp)
            logger.info(f"语音答题会话目录: {self.session_dir}")
        except Exception as e:
            logger.error(f"创建会话目录失败: {e}")
            self.session_dir = 'recordings'
            os.makedirs(self.session_dir, exist_ok=True)

        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None

        self.test_started = True
        self.update_step_ui()

        # 确保多模态监控在语音问答阶段就已启动
        if HAS_MULTIMODAL:
            self._start_multimodal_monitoring()

        # 使用线程异步启动AV采集，完成后启动摄像头更新（非阻塞）
        def start_av_async():
            try:
                # 确保后端客户端已连接（异步，不阻塞UI）
                from ...services.backend_client import get_backend_client
                backend_client = get_backend_client()
                
                # 尝试启动后端（如果支持的话）
                try:
                    backend_client.ensure_started()
                except Exception as e:
                    logger.warning(f"后端自动启动失败: {e}")
                
                # 等待后端连接建立（减少超时时间，避免长时间阻塞）
                logger.info("等待后端服务器连接...")
                connection_ok = False
                try:
                    connection_ok = backend_client.wait_for_connection(timeout=3.0)
                except Exception as e:
                    logger.warning(f"等待后端连接时出错: {e}")
                
                if not connection_ok:
                    logger.warning("⚠️ 后端连接超时（3秒），UI将继续运行但摄像头功能可能不可用")
                    if DEBUG_MODE:
                        logger.info("调试模式下可以使用模拟数据")
                    else:
                        logger.warning("非调试模式，请手动启动后端: python -m src.main --root .")
                    # 不抛出异常，让UI继续运行
                else:
                    logger.info("✅ 后端服务器连接成功")
                
                # 尝试启动AV采集（即使后端未连接也尝试，可能使用本地摄像头）
                try:
                    av_start_collection(
                        save_dir=self.session_dir,
                        camera_index=config.ACTIVE_CAMERA_INDEX,
                        video_fps=30.0,
                        input_device_index=config.ACTIVE_AUDIO_DEVICE_INDEX,
                    )
                    logger.info("AV采集器已启动")
                except Exception as e:
                    logger.error(f"启动 AV 采集器失败: {e}")
                    logger.info("UI将继续运行，但摄像头功能不可用")
                
                # 延迟启动预览，确保数据流稳定（使用QTimer在主线程执行）
                self._invoke_later(self._start_camera_preview, 500)
                
            except Exception as e:
                logger.error(f"AV采集异步启动过程出错: {e}")
                # 即使出错也启动摄像头预览（显示占位符）
                self._invoke_later(self._start_camera_preview, 500)
        
        self.thread_manager.submit_data_task(
            start_av_async,
            task_name="启动AV采集"
        )

        if HAS_MULTIMODAL:
            # 使用线程异步启动多模态采集，避免阻塞UI
            def start_multimodal_async():
                try:
                    result = multidata_start_collection(
                        self.current_user,
                        part=1,
                        save_dir=self.session_dir,
                    )
                    self.multimodal_collector = result
                    if result and result.get("status") in {"running", "already-running"}:
                        logger.info("多模态数据采集已启动，用户: %s", self.current_user)
                        logger.info("多模态数据保存目录: %s\\fatigue", self.session_dir)
                        # 【重要修改】立即在主线程中启动监控，从测试开始就获取脑负荷和疲劳度数据
                        # 延迟800ms确保采集器完全启动并开始产生数据
                        self._invoke_later(self._start_multimodal_monitoring, 800)
                        logger.info("✅ 多模态监控将在800ms后启动，从语音答题开始就可以看到脑负荷和疲劳度数据")
                    else:
                        logger.warning("多模态数据采集启动失败: %s", result)
                except Exception as e:
                    logger.error(f"启动多模态数据采集时出错: {e}")
                    logger.info("UI将继续运行，但疲劳度监测功能不可用")
            
            # 提交到后台线程执行（非阻塞）
            self.thread_manager.submit_data_task(
                start_multimodal_async,
                task_name="启动多模态采集"
            )
        
        # EEG采集也使用异步方式（非阻塞），由后端统一管理硬件连接
        def start_eeg_async():
            try:
                from ...services.backend_proxy import eeg_start
                eeg_start(username=self.current_user, save_dir=self.session_dir, part=1)
                logger.info(f"EEG采集已启动，保存目录: {self.session_dir}\\eeg")
            except Exception as e:
                logger.error(f"启动EEG采集失败: {e}")
                logger.info("UI将继续运行，但EEG功能不可用")
        
        self.thread_manager.submit_data_task(
            start_eeg_async,
            task_name="启动EEG采集"
        )

    def _start_camera_preview(self) -> None:
        """启动当前步骤所需的摄像头预览（异步，非阻塞）。"""
        try:
            self._update_camera_previews_for_step()
        except Exception as e:
            logger.error(f"启动摄像头预览失败: {e}")
            logger.info("摄像头预览将显示占位符")

    def _stop_camera_preview(self) -> None:
        """停止所有摄像头预览（安全，不抛出异常）。"""
        try:
            if self.camera_preview:
                self.camera_preview.stop_preview()
        except Exception as e:
            logger.debug(f"停止camera_preview时出错: {e}")
        
        try:
            if self.schulte_camera_preview:
                self.schulte_camera_preview.stop_preview()
        except Exception as e:
            logger.debug(f"停止schulte_camera_preview时出错: {e}")

    def _update_camera_previews_for_step(self) -> None:
        """根据当前步骤切换摄像头预览（安全启动，失败不影响UI）。"""
        if not self.test_started:
            self._stop_camera_preview()
            return

        try:
            if self.current_step == 0:
                if self.schulte_camera_preview:
                    self.schulte_camera_preview.stop_preview()
                if self.camera_preview:
                    self.camera_preview.start_preview()
            elif self.current_step == 2:
                if self.camera_preview:
                    self.camera_preview.stop_preview()
                if self.schulte_camera_preview:
                    self.schulte_camera_preview.start_preview()
            else:
                self._stop_camera_preview()
        except Exception as e:
            logger.error(f"切换摄像头预览时出错: {e}")
            logger.info("摄像头将显示占位符，但不影响其他功能")

    def _start_video_recording(self, target_path: str = None):
        try:
            av_start_recording()
        except Exception as e:
            logger.error(f"开始音视频录制失败: {e}")

    def _stop_video_recording(self):
        try:
            av_stop_recording()
            self._audio_paths = av_get_audio_paths()
            self._video_paths = av_get_video_paths()
            if HAS_SPEECH_RECOGNITION:
                if self._audio_paths:
                    latest_audio = self._audio_paths[-1]
                    deadline = time.time() + 2.0
                    while not os.path.exists(latest_audio) and time.time() < deadline:
                        time.sleep(0.1)

                    if os.path.exists(latest_audio):
                        try:
                            question_idx = self.current_question + 1
                            question_text = (
                                self.questions[self.current_question]
                                if 0 <= self.current_question < len(self.questions)
                                else ""
                            )
                            add_audio_for_recognition(
                                latest_audio,
                                question_idx,
                                question_text,
                            )
                        except Exception as e:
                            logger.error("加入语音识别队列失败: %s", e)
                    else:
                        logger.error("录音文件未生成，无法加入识别队列: %s", latest_audio)
                else:
                    logger.warning("语音识别队列未入队：未检测到最新音频片段。")
        except Exception as e:
            logger.error(f"停止音视频录制失败: {e}")

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if self.mic_anim.state() == QPropertyAnimation.Running:
            self.mic_anim.stop()
        self.mic_shadow.setEnabled(False)
        self.is_recording = True

        self.btn_mic.setObjectName("micButtonRecording")
        self.btn_mic.setIcon(qta.icon('fa5s.stop', color='white'))

        self.btn_mic.style().unpolish(self.btn_mic)
        self.btn_mic.style().polish(self.btn_mic)

        self.lbl_recording_status.setText("正在录音...")
        logger.info("开始音视频录制...")

        self.audio_timer.start(50)

        self._start_video_recording()

    def _stop_recording(self):
        if not self.is_recording:
            return
        self.is_recording = False
        self.audio_timer.stop()
        self._stop_video_recording()

        self.btn_mic.setObjectName("micButtonStopped")
        self.btn_mic.setIcon(qta.icon('fa5s.check', color='white'))
        self.btn_mic.style().unpolish(self.btn_mic)
        self.btn_mic.style().polish(self.btn_mic)
        self.mic_shadow.setEnabled(False)

        self.btn_next.setEnabled(True)
        self.lbl_recording_status.setText("录制已完成，请进入下一题")
        logger.info("音视频录制完毕。")
        self.audio_level.set_level(0)

        def restore_button():
            self.mic_shadow.setEnabled(True)
            self.btn_mic.setObjectName("micButtonCallToAction")
            self.btn_mic.setIcon(qta.icon('fa5s.microphone-alt', color='white'))
            self.btn_mic.style().unpolish(self.btn_mic)
            self.btn_mic.style().polish(self.btn_mic)

            if self.mic_anim.state() != QPropertyAnimation.Running:
                self.mic_anim.start()

        self._invoke_later(restore_button, 1000)

    def _process_audio(self):
        try:
            level = av_get_current_audio_level()
            self.audio_level.set_level(level)
        except Exception as e:
            logger.warning(f"获取音频电平时发生错误: {e}")
            self.audio_level.set_level(0)

    def _speak_current_question(self):
        if not self.test_started:
            return
        idx = self.current_question
        if idx in self.spoken_questions:
            return
        try:
            text = self.questions[idx]
        except Exception as exc:
            logger.warning(f"获取题目文本失败: {exc}")
            return
        self.spoken_questions.add(idx)
        try:
            self.tts_queue.put(text)
            preview = text if len(text) <= 20 else text[:20] + "..."
            logger.info(f"已提交朗读任务：第 {idx + 1} 题 -> {preview}")
        except Exception as exc:
            logger.warning(f"提交朗读任务失败: {exc}")

    def _next_step_or_question(self):
        if self.current_step == 0:
            if self.current_question < len(self.questions) - 1:
                self.current_question += 1
                self.update_step_ui()
                if self.test_started:
                    self._speak_current_question()
            else:
                self.current_step += 1
                call_timestamp = time.time()
                self.part_timestamps.append(call_timestamp)
                try:
                    self._close_camera()
                except Exception as e:
                    logger.warning(f"关闭摄像头失败: {e}")
                
                # 停止音视频录制并获取路径
                try:
                    logger.info("📹 正在停止音视频录制...")
                    av_stop_recording()
                    self._audio_paths = av_get_audio_paths()
                    self._video_paths = av_get_video_paths()
                    logger.info(f"✅ 音视频录制已停止: {len(self._audio_paths)} 个音频, {len(self._video_paths)} 个视频")
                except Exception as e:
                    logger.error(f"停止音视频录制失败: {e}")
                    # 初始化为空列表,避免后续错误
                    if not hasattr(self, '_audio_paths'):
                        self._audio_paths = []
                    if not hasattr(self, '_video_paths'):
                        self._video_paths = []
                
                try:
                    # 在切换至舒特格阶段前，短暂停止上一阶段采集以重新编号
                    multidata_stop_collection()
                except Exception as stop_exc:
                    logger.warning(f"收尾 part=1 多模态采集失败: {stop_exc}")
                # 修复: 保持脑负荷/疲劳度轮询持续到舒特格测试结束
                self.update_step_ui()
                
                # 保存音视频路径到数据库
                self._persist_av_paths_to_db()
        elif self.current_step == 1:
            call_timestamp = time.time()
            self.part_timestamps.append(call_timestamp)
            
            # ⚠️ 注释掉重新初始化逻辑，避免在切换到舒尔特阶段时重启多模态采集
            # 原因: 重启会导致疲劳度分数重新从初始值开始，影响连续性
            # 改进: 保持多模态采集持续运行，从答题阶段到舒尔特阶段无缝过渡
            if HAS_MULTIMODAL:
                try:
                    # try:
                    #     # 在切换至舒特格阶段前，短暂停止上一阶段采集以重新编号
                    #     multidata_stop_collection()
                    # except Exception as stop_exc:
                    #     logger.warning(f"收尾 part=1 多模态采集失败: {stop_exc}")
                    result = multidata_start_collection(
                        self.current_user,
                        part=2,
                        save_dir=self.session_dir,
                    )
                    self.multimodal_collector = result
                    status = (result or {}).get("status", "").lower()
                    if status in {"running", "already-running"}:
                        logger.info("多模态数据采集 part=2 已启动，用户: %s", self.current_user)
                        logger.info("多模态数据保存目录: %s", self.session_dir)
            
                        timer_active = False
                        try:
                            timer_active = self._multimodal_poll_timer.isActive()
                        except Exception:
                            timer_active = False
            
                        if not self._multimodal_poll_active or not timer_active:
                            if self._multimodal_poll_active and not timer_active:
                                logger.warning("多模态监控定时器未运行，将强制重新启动监控")
                            else:
                                logger.info("启动多模态监控（从舒尔特方格开始）")
                            self._start_multimodal_monitoring(force=True)
                        else:
                            logger.info("✅ 多模态监控已在运行，无需重复启动")
                    else:
                        logger.warning("多模态数据采集启动失败: %s", result)
                except Exception as e:
                    logger.error(f"启动多模态数据采集时出错: {e}")
                
                logger.info("✅ 保持多模态采集持续运行（从答题阶段到舒尔特阶段无缝过渡）")
                self.current_step += 1
                self.update_step_ui()

    def _on_schulte_completed(self):
        logger.info("舒特格测试完成，自动进入分数展示页面")
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)
        try:
            multidata_stop_collection()
        except Exception as e:
            logger.warning(f"停止多模态采集器失败: {e}")
        finally:
            self._stop_multimodal_monitoring()
        
        # 停止EEG采集并保存路径到数据库
        try:
            eeg_stop_collection()
            logger.info("EEG采集已停止")
            # 获取EEG文件路径并保存到数据库
            eeg_paths = eeg_get_file_paths()
            if eeg_paths:
                logger.info(f"✅ 获取到EEG文件路径: {eeg_paths}")
                self._persist_eeg_paths_to_db(eeg_paths)
            else:
                logger.warning("未获取到EEG文件路径")
        except Exception as e:
            logger.error(f"停止EEG采集或保存路径时出错: {e}")
        
        self.current_step += 1
        if self.current_step == 3:
            self.save_score()
        self.update_step_ui()

    def _finish_test(self):
        self.test_started = False
        self._stop_camera_preview()
        if HAS_MULTIMODAL:
            try:
                multidata_stop_collection()
                self.multimodal_collector = None
                logger.info("多模态数据采集已停止")
                self._persist_multimodal_paths_to_db()
                from ...services.backend_proxy import cleanup_collector
                cleanup_collector()
            except Exception as e:
                logger.error(f"停止多模态数据采集时出错: {e}")
            finally:
                self._stop_multimodal_monitoring()
        
        # 停止EEG采集并保存文件路径
        try:
            eeg_stop_collection()
            logger.info("EEG采集已完全停止")
            # 获取EEG文件路径并保存到数据库
            eeg_paths = eeg_get_file_paths()
            if eeg_paths:
                logger.info(f"获取到EEG文件路径: {eeg_paths}")
                self._persist_eeg_paths_to_db(eeg_paths)
        except Exception as e:
            logger.error(f"停止EEG采集或保存路径时出错: {e}")
        
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)
        if self.part_timestamps:
            import json
            call_timestamp_json_path = os.path.join(self.session_dir, 'eeg')
            os.makedirs(call_timestamp_json_path, exist_ok=True)
            call_timestamp_json_path = os.path.join(call_timestamp_json_path, 'part_timestamps.json')
            call_timestamps_formatted = [
                {
                    'timestamp': ts,
                    'datetime': datetime.fromtimestamp(ts).isoformat(),
                    'call_index': i
                }
                for i, ts in enumerate(self.part_timestamps)
            ]
            with open(call_timestamp_json_path, 'w', encoding='utf-8') as f:
                json.dump(call_timestamps_formatted, f, ensure_ascii=False, indent=2)
            logger.info(f"调用时间戳JSON数据已保存: {call_timestamp_json_path}")
        QMessageBox.information(self, "评估完成", "感谢您的参与！")
        self.btn_finish.setEnabled(False)
        self._invoke_later(self._auto_close_page, 2000)

    def _auto_close_page(self):
        self._is_shutting_down = True
        try:
            main_window = self.window()
            if main_window:
                main_window.close()
            else:
                self.close()
            logger.info("评估完成后自动关闭页面")
        except Exception as e:
            logger.error(f"自动关闭页面失败: {e}")

    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#F4F7F9"))
        gradient.setColorAt(1, QColor("#E5E9EC"))
        painter.fillRect(self.rect(), gradient)
        super().paintEvent(event)

    def _shutdown_active_services(self) -> None:
        """停止所有正在运行的采集/监测任务。"""
        self._stop_camera_preview()
        self.audio_timer.stop()
        if self.is_recording:
            self._stop_recording()

        if hasattr(self, 'schulte_widget'):
            self.schulte_widget.reset_for_next_stage()

        if HAS_MULTIMODAL:
            try:
                multidata_stop_collection()
                self.multimodal_collector = None
                self._persist_multimodal_paths_to_db()
                from ...services.backend_proxy import cleanup_collector
                cleanup_collector()
            except Exception as e:
                logger.error(f"页面隐藏时停止多模态数据采集失败: {e}")
            finally:
                self._stop_multimodal_monitoring()

        try:
            from ...services.backend_proxy import eeg_stop, eeg_paths
            eeg_stop()
            paths_result = eeg_paths()
            paths = paths_result.get("paths", []) if isinstance(paths_result, dict) else []
            if paths:
                self._persist_eeg_paths_to_db(paths)
        except Exception as e:
            logger.error(f"页面隐藏时停止EEG采集失败: {e}")

        try:
            if self.schulte_accuracy and self.schulte_elapsed:
                self._on_schulte_result(self.schulte_elapsed, self.schulte_accuracy)
        except Exception as e:
            logger.error(f"舒尔特结果写入数据库失败: {e}")

    def hideEvent(self, event):
        super().hideEvent(event)
        try:
            window = self.window()
            if window and window.isMinimized():
                logger.debug("TestPage 已最小化，保持采集任务运行")
            elif getattr(self, "_is_shutting_down", False):
                logger.debug("TestPage 正在关闭，资源回收将在 closeEvent 中处理")
            else:
                logger.debug("TestPage 暂时隐藏但未退出评估，保持采集任务运行")
        except Exception:
            logger.debug("隐藏事件处理中无法获取窗口状态，默认保持采集运行")

    def closeEvent(self, event):
        self._is_shutting_down = True
        try:
            self._shutdown_active_services()
        finally:
            super().closeEvent(event)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, "_is_shutting_down", False):
            logger.debug("TestPage 重新显示，重置关停标记")
        self._is_shutting_down = False

    def _close_camera(self):
        try:
            self._stop_camera_preview()
            try:
                av_stop_recording()
            except Exception:
                pass
            logger.info("语音答题环节结束，停止录制")
        except Exception as e:
            logger.warning(f"关闭摄像头时出现问题: {e}")

    def _persist_av_paths_to_db(self):
        """保存音视频路径到数据库（使用更新而不是插入，避免重复创建记录）"""
        if self._db_disabled:
            return

        try:
            update_payload = {
                "video": list(self._video_paths),
                "audio": list(self._audio_paths),
            }
            
            logger.info(f"准备保存音视频路径: {len(self._video_paths)} 视频, {len(self._audio_paths)} 音频")
            
            # 使用排队更新机制，如果记录不存在会自动创建
            self._queue_db_update(update_payload, "保存音视频路径失败")
            
            logger.info("✅ 音视频路径已加入数据库更新队列")
            
        except Exception as e:
            logger.exception(f"❌ 保存音视频路径时发生异常: {e}")

    def _persist_multimodal_paths_to_db(self):
        """保存多模态数据文件路径到数据库（RGB/Depth/Eyetrack）
        
        注意：语音识别结果已在情绪分析前保存，这里不再重复保存
        """
        try:
            if not HAS_MULTIMODAL:
                logger.warning("多模态数据采集模块不可用，跳过数据库写入。")
                return

            from ...services.backend_proxy import get_multimodal_file_paths
            file_paths_result = get_multimodal_file_paths()
            file_paths = file_paths_result.get("paths", {}) if isinstance(file_paths_result, dict) else {}

            if not file_paths:
                logger.warning("未获取到多模态数据文件路径")
                return

            # 清理语音识别结果（避免内存泄漏），结果已在 _save_speech_recognition_results 中保存
            try:
                clear_recognition_results()
                logger.debug("已清理语音识别结果缓存")
            except Exception as e:
                logger.debug(f"清理语音识别结果失败: {e}")

            update_payload = {}
            if file_paths.get('rgb'):
                update_payload['rgb'] = file_paths.get('rgb')
            if file_paths.get('depth'):
                update_payload['depth'] = file_paths.get('depth')
            if file_paths.get('eyetrack'):
                update_payload['tobii'] = file_paths.get('eyetrack')

            if not update_payload:
                logger.debug("多模态文件路径为空，跳过数据库更新。")
                return

            self._queue_db_update(update_payload, "更新多模态数据路径到数据库失败")

        except Exception as e:
            logger.error(f"写入多模态数据路径到数据库失败: {e}")

    def _persist_eeg_paths_to_db(self, eeg_paths: dict):
        """保存 EEG 数据文件路径到数据库（增强版，带同步等待）"""
        if self._db_disabled:
            logger.debug("数据库已禁用，跳过 EEG 路径保存")
            return
        
        try:
            # 提取路径（兼容多种格式）
            update_payload = {}
            
            # 格式 1: {'ch1_txt': 'path1', 'ch2_txt': 'path2'}
            if 'ch1_txt' in eeg_paths or 'ch2_txt' in eeg_paths:
                if eeg_paths.get('ch1_txt'):
                    update_payload['eeg1'] = eeg_paths['ch1_txt']
                if eeg_paths.get('ch2_txt'):
                    update_payload['eeg2'] = eeg_paths['ch2_txt']
            
            # 格式 2: {'eeg_json_path': 'path1', 'eeg_csv_path': 'path2'}
            elif 'eeg_json_path' in eeg_paths or 'eeg_csv_path' in eeg_paths:
                if eeg_paths.get('eeg_json_path'):
                    update_payload['eeg1'] = eeg_paths['eeg_json_path']
                if eeg_paths.get('eeg_csv_path'):
                    update_payload['eeg2'] = eeg_paths['eeg_csv_path']
            
            # 格式 3: 列表形式 ['path1', 'path2']
            elif isinstance(eeg_paths, list):
                if len(eeg_paths) > 0 and eeg_paths[0]:
                    update_payload['eeg1'] = eeg_paths[0]
                if len(eeg_paths) > 1 and eeg_paths[1]:
                    update_payload['eeg2'] = eeg_paths[1]
            
            if not update_payload:
                logger.warning(f"⚠️ EEG 路径为空或格式不支持: {eeg_paths}")
                return
            
            logger.info(f"准备保存 EEG 路径: {update_payload}")
            
            # 如果数据库行还未创建，同步等待最多 3 秒
            if not self.row_id:
                logger.info("⏳ 等待数据库行创建...")
                import time
                max_wait = 30  # 最多等待 3 秒 (30 * 0.1s)
                wait_count = 0
                while not self.row_id and wait_count < max_wait:
                    time.sleep(0.1)
                    wait_count += 1
                
                if not self.row_id:
                    logger.error("❌ 等待数据库行创建超时，EEG 路径将被加入待处理队列")
                    # 仍然尝试排队
                    self._queue_db_update(update_payload, "保存 EEG 路径失败（等待超时）")
                    return
                else:
                    logger.info(f"✅ 数据库行已创建 (row_id={self.row_id})")
            
            # 使用排队机制
            self._queue_db_update(update_payload, "写入EEG路径到数据库失败")
            logger.info(f"✅ EEG 路径已加入数据库更新队列 (row_id={self.row_id}): {update_payload}")
            
        except Exception as e:
            logger.exception(f"❌ 保存 EEG 路径时发生异常: {e}")
            self._handle_db_failure(e, "写入EEG路径到数据库失败")

    def save_score(self):
        try:
            if self.score is not None:
                with open(SCORES_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), self.score])
                self.history_scores.append(self.score)
                logger.info(f"分数已保存到CSV文件: {self.score}")
            else:
                logger.warning("分数尚未计算，跳过CSV保存")
        except Exception as e:
            logger.error(f"保存分数时出错: {e}")

    def load_history_scores(self):
        self.history_scores = []
        if not os.path.exists(SCORES_CSV_FILE):
            return
        try:
            with open(SCORES_CSV_FILE, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        self.history_scores.append(int(row[1]))
        except Exception as e:
            logger.error(f"读取历史分数时出错: {e}")

    def set_current_user(self, username: str):
        self.current_user = username or 'anonymous'
        if hasattr(self, 'schulte_widget') and self.schulte_widget:
            try:
                self.schulte_widget.set_username(self.current_user)
            except Exception as e:
                logger.warning(f"同步用户名到舒特格控件失败: {e}")

    def _on_schulte_result(self, elapsed_seconds: float, accuracy_percent: float):
        try:
            self.schulte_elapsed = float(elapsed_seconds)
            self.schulte_accuracy = float(accuracy_percent)

            time_score = max(0, min(100, 100 - (self.schulte_elapsed - 30) * 2))
            accuracy_score = self.schulte_accuracy
            self.score = int(accuracy_score * 0.7 + time_score * 0.3)

            logger.info(f"舒特结果: 用时={self.schulte_elapsed:.2f}s, 准确率={self.schulte_accuracy:.1f}%, 计算得分={self.score}")

            ptime = os.path.abspath(self.session_dir)
            ptime = os.path.join(ptime, 'eeg', 'part_timestamps.txt')

            update_payload = {
                "accuracy": self.schulte_accuracy,
                "elapsed": self.schulte_elapsed,
                "score": self.score,
                "ptime": ptime,
            }
            self._queue_db_update(update_payload, "保存舒特结果到数据库失败")
        except Exception as e:
            logger.warning(f"处理舒特结果信号失败: {e}")
    
    def _calculate_average_scores(self) -> Dict[str, Optional[float]]:
        """
        计算疲劳度和脑负荷的平均分数
        
        Returns:
            包含平均分数的字典:
            {
                "fatigue_avg": 平均疲劳度分数 (0-100),
                "brain_load_avg": 平均脑负荷分数 (0-100),
                "fatigue_count": 疲劳度样本数量,
                "brain_load_count": 脑负荷样本数量
            }
        """
        result = {
            "fatigue_avg": None,
            "brain_load_avg": None,
            "fatigue_count": 0,
            "brain_load_count": 0
        }
        
        # 计算疲劳度平均值
        if self._fatigue_scores_list:
            result["fatigue_avg"] = sum(self._fatigue_scores_list) / len(self._fatigue_scores_list)
            result["fatigue_count"] = len(self._fatigue_scores_list)
            logger.info(
                f"疲劳度平均分数: {result['fatigue_avg']:.2f} "
                f"(基于 {result['fatigue_count']} 个样本)"
            )
        else:
            logger.warning("没有收集到疲劳度分数数据")
        
        # 计算脑负荷平均值
        if self._brain_load_scores_list:
            result["brain_load_avg"] = sum(self._brain_load_scores_list) / len(self._brain_load_scores_list)
            result["brain_load_count"] = len(self._brain_load_scores_list)
            logger.info(
                f"脑负荷平均分数: {result['brain_load_avg']:.2f} "
                f"(基于 {result['brain_load_count']} 个样本)"
            )
        else:
            logger.warning("没有收集到脑负荷分数数据")
        
        return result
    
    def _prepare_score_data(self) -> Dict[str, any]:
        """
        准备传递给分数展示页面的所有数据
        
        Returns:
            包含所有测试结果的字典
        """
        # 计算平均分数
        avg_scores = self._calculate_average_scores()
        
        # 准备数据
        score_data = {
            # 疲劳检测 (平均值)
            "疲劳检测": avg_scores["fatigue_avg"] if avg_scores["fatigue_avg"] is not None else 0,
            
            # 情绪分数
            "情绪": self._emotion_score if self._emotion_score is not None else 0,
            
            # 脑负荷 (平均值)
            "脑负荷": avg_scores["brain_load_avg"] if avg_scores["brain_load_avg"] is not None else 0,
            
            # 舒尔特准确率
            "舒尔特准确率": self.schulte_accuracy if self.schulte_accuracy is not None else 0,
            
            # 血压数据
            "收缩压": self.bp_results.get("systolic", 0) if hasattr(self, 'bp_results') else 0,
            "舒张压": self.bp_results.get("diastolic", 0) if hasattr(self, 'bp_results') else 0,
            "脉搏": self.bp_results.get("pulse", 0) if hasattr(self, 'bp_results') else 0,
            
            # 舒尔特综合得分
            "舒尔特综合得分": self.score if self.score is not None else 0,
            
            # 元数据
            "_metadata": {
                "fatigue_sample_count": avg_scores["fatigue_count"],
                "brain_load_sample_count": avg_scores["brain_load_count"],
                "has_emotion_score": self._emotion_score is not None,
                "has_schulte_result": self.schulte_accuracy is not None,
                "has_bp_result": hasattr(self, 'bp_results') and self.bp_results.get('systolic') is not None,
            }
        }
        
        logger.info(f"准备分数数据完成: {score_data}")
        return score_data
    
    def _send_scores_to_score_page(self):
        """
        将所有测试分数发送到分数展示页面,并保存推理结果到数据库
        """
        try:
            # 准备数据
            score_data = self._prepare_score_data()
            
            # 保存推理结果到数据库
            self._save_inference_scores_to_db(score_data)
            
            # 发送到分数页面
            if not hasattr(self, 'score_page') or not self.score_page:
                logger.warning("分数页面未初始化，无法发送分数数据")
                return
            
            if hasattr(self.score_page, 'set_test_results'):
                self.score_page.set_test_results(score_data)
                logger.info("✅ 测试结果已发送到分数展示页面")
            else:
                logger.warning("分数页面没有 set_test_results 方法")
                
        except Exception as e:
            logger.error(f"发送分数到分数页面失败: {e}", exc_info=True)
    
    def _save_inference_scores_to_db(self, score_data: dict):
        """
        将疲劳检测、脑负荷、情绪推理结果保存到数据库
        
        Args:
            score_data: 包含所有分数的字典
        """
        try:
            if self._db_disabled:
                logger.debug("数据库已禁用,跳过保存推理结果")
                return
            
            # 提取推理结果
            update_payload = {
                "fatigue_score": score_data.get("疲劳检测", 0),
                "brain_load_score": score_data.get("脑负荷", 0),
                "emotion_score": score_data.get("情绪", 0),
            }
            
            # 过滤掉0值(表示没有数据)
            update_payload = {k: v for k, v in update_payload.items() if v > 0}
            
            if not update_payload:
                logger.debug("没有有效的推理结果需要保存到数据库")
                return
            
            # 更新数据库记录
            self._queue_db_update(
                update_payload,
                "保存推理结果到数据库失败"
            )
            
            logger.info(f"📊 推理结果已保存到数据库: {update_payload}")
            
        except Exception as e:
            logger.error(f"保存推理结果到数据库失败: {e}", exc_info=True)


__all__ = ["TestPage"]
