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
    QTextEdit,
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
    QRegion,
    QBitmap,
    QPixmap,
)

from ..utils.widgets import AudioLevelMeter, ScoreChartWidget
from ..utils.responsive import scale, scale_size, scale_font
from ...widgets.camera_preview import CameraPreviewWidget
from ...widgets.schulte_grid import SchulteGridWidget
from...widgets.score_page import ScorePage
from ...services.backend_client import get_backend_client
from ...utils_common.thread_process_manager import get_thread_manager

# ---------------------------------------------------------------------------
# 自定义圆形标签类
# ---------------------------------------------------------------------------
class CircleLabel(QLabel):
    """完美圆形的数字标签"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
    
    def resizeEvent(self, event):
        """在尺寸变化时重新应用圆形遮罩"""
        super().resizeEvent(event)
        size = min(self.width(), self.height())
        # 确保是正方形
        self.setFixedSize(size, size)
        # 应用圆形遮罩
        region = QRegion(0, 0, size, size, QRegion.Ellipse)
        self.setMask(region)

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

        # 🔄 改为加载朗读文本配置（而不是问答题目）
        reading_texts_file = config.BASE_DIR/ "data" / "text" / "reading_texts.yaml"
        try:
            with open(reading_texts_file, encoding="utf-8") as handle:
                reading_config = yaml.load(handle, Loader=FullLoader)
            
            # 获取默认文本
            default_index = reading_config.get('default_text_index', 0)
            texts_list = reading_config.get('texts', [])
            
            if texts_list and 0 <= default_index < len(texts_list):
                selected_text = texts_list[default_index]
                self.reading_text_title = selected_text.get('title', '朗读文本')
                self.reading_text_content = selected_text.get('content', '').strip()
            else:
                raise ValueError("配置文件中没有可用文本")
        except Exception as e:
            logger.warning(f"加载朗读文本配置失败: {e}，使用默认文本")
            # 默认文本（如果配置文件读取失败）
            self.reading_text_title = "航空安全规范"
            self.reading_text_content = """乘员应在进入航空器前，接受安检并配合现场工作人员的引导。严禁携带易燃易爆、有毒、腐蚀性等危险品登机。

登机后，请对号入座，并系好安全带。飞行过程中，须遵守"系好安全带"信号指示，未经允许不得离开座位。

发现可疑人员或物品，应及时报告机组成员。不得传播虚假信息、制造恐慌。

请勿携带或在机上使用禁止类电子设备，遵守所有安全广播和公告要求。"""
        # 兼容旧语音识别接口：保留 questions 列表结构
        self.questions = [self.reading_text_content]
        
        # 朗读状态标志
        self.reading_completed = False
        
        # ❌ 不再需要 TTS 功能（用户自己朗读，不需要机器播放）
        self.thread_manager = get_thread_manager()
        
        self.current_question = 0  # 保留变量名兼容性，实际已无多个问题
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
        logger.debug("TestPage 初始化完成。")
        self._is_shutting_down = False

    def _setup_properties(self):
        """初始化测试页面的所有状态变量。"""
        # 🔄 完整的测试流程步骤(包括基线和SART)
        self.all_stages = ['多模态疲劳检测', '情绪检测', '血压脉搏检测', '舒尔特专注度检测', '分数展示']
        self.steps = ['情绪检测', '血压脉搏检测', '舒尔特专注度检测', '分数展示']  # 当前test.py内部的步骤

        # � 记录每个阶段的完成状态
        self.stage_completed = {
            '多模态疲劳检测': False,  # 包含基线校准+SART实验
            '情绪检测': False,        # 原朗读录音改名
            '血压脉搏检测': False,    # 原血压测试改名
            '舒尔特专注度检测': False  # 原舒尔特测试改名
        }

        self.current_step = 0
        self.current_question = 0  # 保留兼容性，但不再有多个问题
        self.is_recording = False
        self.score = None  # 将在舒尔特测试完成后计算
        self.history_scores = []
        # 音频录制已转移到AVCollector，这里只保留定时器用于更新UI
        self.audio_timer = QTimer(self)
        self.camera_preview: Optional[CameraPreviewWidget] = None
        # ✅ 两个独立的摄像头widget，都从同一个AV服务获取帧数据
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
        
        # SART模式配置（从命令行参数读取）
        self.sart_mode = "short"  # 默认短时模式
        self.sart_duration = 300  # 默认5分钟

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
        self._timestamps_file_path = None  # 时间戳文件路径
        self._text_qa_start_timestamp_recorded = False  # 朗读录音开始时间戳是否已记录

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
    
    def _save_timestamp_immediately(self, call_timestamp: float) -> None:
        """实时保存时间戳到JSON文件（每次添加时间戳立即写入）
        
        Args:
            call_timestamp: 时间戳（Unix时间戳）
        """
        try:
            # 确保文件路径已初始化
            if self._timestamps_file_path is None:
                if not hasattr(self, 'session_dir') or not self.session_dir:
                    logger.warning("session_dir 未初始化，无法保存时间戳")
                    return
                
                eeg_dir = os.path.join(self.session_dir, 'eeg')
                os.makedirs(eeg_dir, exist_ok=True)
                self._timestamps_file_path = os.path.join(eeg_dir, 'part_timestamps.json')
            
            # 添加到列表
            self.part_timestamps.append(call_timestamp)
            
            # 格式化所有时间戳
            call_timestamps_formatted = [
                {
                    'timestamp': ts,
                    'datetime': datetime.fromtimestamp(ts).isoformat(),
                    'call_index': i
                }
                for i, ts in enumerate(self.part_timestamps)
            ]
            
            # 立即写入文件
            import json
            with open(self._timestamps_file_path, 'w', encoding='utf-8') as f:
                json.dump(call_timestamps_formatted, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ 时间戳实时保存: call_index={len(self.part_timestamps)-1}, file={self._timestamps_file_path}")
            
        except Exception as e:
            logger.error(f"实时保存时间戳失败: {e}", exc_info=True)

    def _start_multimodal_monitoring(self, *, force: bool = False) -> None:
        """启动或重新启动多模态数据监控（仅内嵌显示，非阻塞）。"""
        try:
            if not HAS_MULTIMODAL:
                return
            
            # ✅ 确保数据库记录已创建（防御性检查）
            # 如果用户直接跳转到基线/SART阶段而没有先登录，这里会触发创建
            try:
                if not self._db_disabled and not self.row_id and not self._row_id_future:
                    logger.info("📝 开始多模态监控前确保数据库记录已创建...")
                    self._ensure_db_row()
            except Exception as e:
                logger.warning(f"⚠️ 创建数据库记录失败（将在后续尝试）: {e}")

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

            logger.debug("启动多模态数据监控（内嵌显示模式）")
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
            logger.debug("多模态数据监控已停止")
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
                    
                    logger.debug("正在进行情绪分析...")
                    result = emotion_analyze(
                        audio_paths=audio_paths,
                        video_paths=video_paths,
                        text_data=text_data,
                        timeout=30.0  # 情绪推理需要较长时间(5个样本约5秒)
                    )
                    
                    emotion_score = result.get("emotion_score", 0.0)
                    emotion_label = result.get("emotion_label", "unknown")
                    confidence = result.get("confidence", 0.0)
                    
                    logger.debug(
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
                logger.debug(f"多模态数据轮询已启动，当前状态: {status}")
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
                logger.debug("多模态采集已停止，停止轮询")

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
                    # logger.info(f"📊 收到疲劳度推理结果: score={fatigue_score:.2f}, class={prediction_class}")
                    
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
                    # logger.info(f"🧠 收到EEG脑负荷推理结果: score={brain_load_score:.2f}, state={state}")
                    
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
        # 背景渐变由全局样式表(style.qss)设置
        
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

        logger.debug("📝 创建新的数据库记录...")
        self._row_id_future = self._send_db_command(
            "db.insert_test_record",
            payload,
            context="创建数据库记录失败",
            on_success=_on_created,
        )

    def _queue_db_update(self, update_payload: dict, context: str) -> None:
        """排队数据库更新（无回调）"""
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
    
    def _queue_db_update_with_callback(self, update_payload: dict, context: str, on_success=None) -> None:
        """排队数据库更新（带成功回调）"""
        if self._db_disabled:
            return

        def _dispatch(row_id: int) -> None:
            payload = dict(update_payload)
            payload["row_id"] = row_id
            self._send_db_command("db.update_test_record", payload, context=context, on_success=on_success)

        if self.row_id:
            _dispatch(self.row_id)
        else:
            self._pending_db_updates.append(_dispatch)
            self._ensure_db_row()

    # --- UI 创建辅助方法 ---
    def _create_step_navigator(self):
        """创建完整的阶段导航栏(包括基线、SART、文本、血压、舒尔特)"""
        container = QWidget()
        container.setObjectName("stepNavigator")  # 改用独特的名称避免与全局card样式冲突
        # 设置圆角属性
        container.setAttribute(Qt.WA_StyledBackground, True)
        layout = QHBoxLayout(container)
        # 增加上下内边距，让导航栏更高，圆角效果更明显
        layout.setContentsMargins(scale(12), scale(8), scale(12), scale(8))
        layout.setSpacing(scale(6))
        
        self.stage_buttons = {}  # 保存每个阶段的按钮引用
        self.step_labels = []  # 兼容旧代码
        self.step_opacity_effects = []  # 兼容旧代码
        
        # 设置渐变背景和圆角（独立样式，不受全局card影响）
        # 导航栏高度 = scale(8)*2 + scale(45) ≈ 61px
        # 圆角设置为30px，形成左右半圆效果
        container.setStyleSheet("""
            QWidget#stepNavigator {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7FDBFF,
                    stop:0.5 #A0E7FF,
                    stop:1 #C0F0FF
                );
                border-radius: 30px;
                border: none;
            }
        """)
        
        # 添加右下黑色阴影效果（悬浮效果）- 调整参数避免遮挡圆角
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)  # 减小模糊半径
        shadow.setXOffset(3)      # 减小偏移
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 80))  # 降低透明度
        container.setGraphicsEffect(shadow)
        
        # 创建所有阶段的导航按钮(包括"分数展示")
        display_stages = self.all_stages  # 显示所有5个阶段
        
        for i, stage_name in enumerate(display_stages):
            # 创建横向布局容器：数字在左，文字在右
            stage_widget = QWidget()
            stage_widget.setStyleSheet("background: transparent;")
            stage_layout = QHBoxLayout(stage_widget)
            stage_layout.setContentsMargins(scale(10), scale(4), scale(10), scale(4))
            stage_layout.setSpacing(scale(8))
            stage_layout.setAlignment(Qt.AlignCenter)
            
            # 数字标签（使用自定义圆形标签）- 老年模式字体更大
            size = scale(45)
            number_label = CircleLabel(str(i + 1))
            number_label.setObjectName("stageNumber")
            number_label.setFixedSize(size, size)
            number_label.setStyleSheet("""
                QLabel#stageNumber {
                    background-color: rgb(227, 247, 253);
                    border: none;
                    font-size: 30px;
                    font-weight: bold;
                    color: rgb(77, 171, 201);
                }
            """)
            
            # 阶段名称标签（老年模式 - 超大字号）
            name_label = QLabel(stage_name)
            name_label.setObjectName("stageName")
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            name_label.setStyleSheet("""
                QLabel#stageName {
                    background: transparent;
                    font-size: 32px;
                    font-weight: bold;
                    color: rgb(77, 171, 201);
                }
            """)
            
            stage_layout.addWidget(number_label, 0, Qt.AlignCenter)
            stage_layout.addWidget(name_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
            
            # 创建可点击按钮（透明覆盖层）
            stage_btn = QPushButton(stage_widget)
            stage_btn.setObjectName("stageNavButton")
            stage_btn.setCursor(Qt.PointingHandCursor)
            stage_btn.setStyleSheet("""
                QPushButton#stageNavButton {
                    background: transparent;
                    border: none;
                }
            """)
            stage_btn.setGeometry(0, 0, stage_widget.width(), stage_widget.height())
            
            # 绑定点击事件
            stage_btn.clicked.connect(lambda checked, s=stage_name: self._on_stage_nav_clicked(s))
            
            # 保存引用（用于后续更新状态）
            self.stage_buttons[stage_name] = {
                'widget': stage_widget,
                'number': number_label,
                'name': name_label,
                'button': stage_btn
            }
            
            layout.addWidget(stage_widget, 1)
            
            # 添加分隔线(最后一个不加)
            if i < len(display_stages) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFixedWidth(2)
                line.setFixedHeight(scale(45))
                line.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border: none;")
                layout.addWidget(line, 0, Qt.AlignCenter)
        
        return container

    def _create_question_progress_bar(self):
        """创建问题进度条（朗读模式下不显示，返回空容器）"""
        container = QWidget()
        # 不设置 card 样式，避免显示白条
        container.setStyleSheet("background: transparent;")
        container.setVisible(False)  # 🔄 朗读模式下隐藏进度条
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)  # 移除边距

        self.question_dots = []  # 保留空列表以避免其他代码报错
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

    def _on_stage_nav_clicked(self, stage_name: str):
        """处理阶段导航点击事件 - 支持自由跳转"""
        logger.info(f"🔘 用户点击导航: {stage_name}")
        
        # ✅ 血压测试进行中时，禁止页面跳转
        if hasattr(self, 'bp_test_running') and self.bp_test_running:
            QMessageBox.warning(
                self, 
                "测试进行中", 
                "血压测试正在进行中，请等待测试完成后再进行其他操作。"
            )
            logger.warning("⚠️ 血压测试进行中，禁止页面跳转")
            return

        # ✅ 完全自由跳转，无需按顺序完成
        if stage_name == '多模态疲劳检测':
            self._jump_to_multimodal_fatigue()
        elif stage_name == '情绪检测':
            self._jump_to_emotion()
        elif stage_name == '血压脉搏检测':
            self._jump_to_blood_pressure()
        elif stage_name == '舒尔特专注度检测':
            self._jump_to_schulte()
        elif stage_name == '分数展示':
            self._jump_to_score_display()

    def _jump_to_multimodal_fatigue(self):
        """跳转到多模态疲劳检测阶段(显示基线校准提示页面)"""
        logger.info("🔄 跳转到多模态疲劳检测阶段（显示基线提示）")
        
        # 重置基线和SART页面状态（防止上次的完成提示残留）
        try:
            main_window = self.window()
            if hasattr(main_window, 'baseline_page'):
                main_window.baseline_page.reset()
                logger.debug("✅ 已重置基线校准页面状态")
            if hasattr(main_window, 'sart_page'):
                main_window.sart_page.reset()
                logger.debug("✅ 已重置SART实验页面状态")
        except Exception as e:
            logger.error(f"❌ 重置页面状态失败: {e}")
        
        # 显示基线校准提示页面（answer_stack中的第0个widget）
        self.answer_stack.setCurrentIndex(0)
        # 隐藏摄像头和按钮（提示页面不需要）
        self._hide_camera_and_buttons()
        self._update_stage_nav_status()
    
    def _on_start_baseline_clicked(self):
        """基线校准开始按钮点击处理"""
        logger.info("🚀 用户点击开始基线校准")
        # 跳转到全屏基线校准页面
        try:
            main_window = self.window()
            if hasattr(main_window, 'show_baseline_page'):
                # ✅ 使用 show_baseline_page() 方法，会正确设置 session_info
                main_window.show_baseline_page()
                logger.debug("✅ 已通过show_baseline_page()跳转到基线校准页面")
            elif hasattr(main_window, 'stack'):
                # 后备方案：直接跳转（但可能没有正确设置session_dir）
                logger.warning("⚠️ show_baseline_page()不存在，使用后备方案")
                main_window.stack.setCurrentIndex(2)
                logger.info("✅ 已跳转到基线校准页面（全屏）")
            else:
                logger.warning("⚠️ 无法找到主窗口堆栈")
                QMessageBox.information(self, "提示", "无法启动基线校准")
        except Exception as e:
            logger.error(f"❌ 跳转基线校准页面失败: {e}")
            QMessageBox.warning(self, "错误", f"启动失败：{e}")
    
    def _on_start_sart_clicked(self):
        """SART实验开始按钮点击处理"""
        logger.info("🚀 用户点击开始SART实验")
        # 跳转到全屏SART实验页面
        try:
            main_window = self.window()
            if hasattr(main_window, 'show_sart_page'):
                # ✅ 使用 show_sart_page() 方法，会正确设置 session_info
                main_window.show_sart_page()
                logger.debug("✅ 已通过show_sart_page()跳转到SART实验页面")
            elif hasattr(main_window, 'stack'):
                # 后备方案：直接跳转（但可能没有正确设置session_dir）
                logger.warning("⚠️ show_sart_page()不存在，使用后备方案")
                main_window.stack.setCurrentIndex(3)
                logger.debug("✅ 已跳转到SART实验页面（全屏）")
            else:
                logger.warning("⚠️ 无法找到主窗口堆栈")
                QMessageBox.information(self, "提示", "无法启动SART实验")
        except Exception as e:
            logger.error(f"❌ 跳转SART页面失败: {e}")
            QMessageBox.warning(self, "错误", f"启动失败：{e}")
    
    def _jump_to_emotion(self):
        """跳转到情绪检测阶段(原朗读录音阶段)"""
        logger.info("🔄 跳转到情绪检测阶段")
        
        # 🔧 修复布局问题：从舒尔特页面跳转时，先切换到血压页面（隐藏摄像头）
        # 然后再跳转到情绪检测页面，避免布局被拉宽
        if self.current_step == 2:  # 如果当前在舒尔特页面
            logger.debug("从舒尔特页面跳转，先重置布局")
            # 先切换到血压页面（step=1），隐藏舒尔特摄像头
            self.current_step = 1
            self.update_step_ui()
            # 使用极短的延迟（仅10ms）让布局重置
            QTimer.singleShot(10, lambda: self._complete_jump_to_emotion())
        else:
            # 从其他页面跳转，直接切换
            self._complete_jump_to_emotion()
    
    def _complete_jump_to_emotion(self):
        """完成跳转到情绪检测（内部方法）"""
        self.current_step = 0  # 情绪检测是第一个步骤
        self.update_step_ui()  # 内部会调用_update_stage_nav_status()
    
    def _jump_to_blood_pressure(self):
        """跳转到血压脉搏检测阶段"""
        logger.info("🔄 跳转到血压脉搏检测阶段")
        
        # ✅ 重置血压测试状态（确保每次进入都是干净的状态）
        try:
            # 停止正在进行的测试（如果有）
            if hasattr(self, 'bp_test_running') and self.bp_test_running:
                self._stop_bp_test()
            
            # 重置结果数据
            self.bp_results = {
                'systolic': None,
                'diastolic': None,
                'pulse': None,
            }
            
            # 重置UI控件状态
            if hasattr(self, 'bp_start_button'):
                self.bp_start_button.setText("开始测试")
                self.bp_start_button.setObjectName("successButton")
                self.bp_start_button.style().unpolish(self.bp_start_button)
                self.bp_start_button.style().polish(self.bp_start_button)
            
            if hasattr(self, 'bp_progress_label'):
                self.bp_progress_label.setText("等待开始测试...")
            
            if hasattr(self, 'bp_progress_circle'):
                self.bp_progress_circle.setText("0%")
            
            # 隐藏结果区域，显示测试控制区域
            if hasattr(self, 'result_container'):
                self.result_container.setVisible(False)
            if hasattr(self, 'bp_status_container'):
                self.bp_status_container.setVisible(True)
            if hasattr(self, 'bp_control_container'):
                self.bp_control_container.setVisible(True)
            
            # 重置卡片内按钮状态
            if hasattr(self, 'bp_next_button'):
                self.bp_next_button.setText("请先完成血压测试")
                self.bp_next_button.setEnabled(False)
            
            logger.info("✅ 血压测试状态已重置")
        except Exception as e:
            logger.error(f"❌ 重置血压测试状态失败: {e}")
        
        self.current_step = 1  # 血压脉搏检测是第二个步骤
        self.update_step_ui()  # 内部会调用_update_stage_nav_status()
    
    def _jump_to_schulte(self):
        """跳转到舒尔特专注度检测阶段"""
        logger.info("🔄 跳转到舒尔特专注度检测阶段")
        
        # 每次进入都重新初始化舒尔特widget（避免状态卡住）
        self._reinit_schulte_widget()
        
        self.current_step = 2  # 舒尔特专注度检测是第三个步骤
        self.update_step_ui()  # 内部会调用_update_stage_nav_status()
    
    def _jump_to_score_display(self):
        """跳转到分数展示阶段"""
        logger.info("🔄 跳转到分数展示阶段")
        self.current_step = 3  # 分数展示是第四个步骤
        self.update_step_ui()  # 内部会调用_update_stage_nav_status()
    
    def _hide_camera_and_buttons(self):
        """隐藏摄像头和底部按钮（用于提示页面）"""
        if hasattr(self, 'camera_widget'):
            self.camera_widget.setVisible(False)
        if hasattr(self, 'schulte_camera_widget'):
            self.schulte_camera_widget.setVisible(False)
        if hasattr(self, 'btn_next'):
            self.btn_next.setVisible(False)
        if hasattr(self, 'btn_finish'):
            self.btn_finish.setVisible(False)

    def _update_stage_nav_status(self):
        """更新导航栏的视觉状态"""
        # 根据answer_stack的当前索引判断当前阶段
        current_index = self.answer_stack.currentIndex()

        # answer_stack索引映射到全局阶段
        index_to_stage = {
            0: '多模态疲劳检测',    # 基线提示页面
            1: '多模态疲劳检测',    # SART提示页面
            2: '情绪检测',          # 朗读录音
            3: '血压脉搏检测',      # 血压测试
            4: '舒尔特专注度检测',  # 舒尔特测试
            5: None,                # 信息确认页
            6: '分数展示'           # 分数页面
        }

        current_stage = index_to_stage.get(current_index, None)

        for stage_name, components in self.stage_buttons.items():
            number_label = components['number']
            name_label = components['name']

            is_current = (stage_name == current_stage)
            is_completed = self.stage_completed.get(stage_name, False)

            # 根据状态设置样式（圆形无边框设计）
            if is_current:
                # 当前阶段：水绿色填充圆圈，白色数字
                number_label.setStyleSheet("""
                    QLabel#stageNumber {
                        background-color: rgb(77, 171, 201);
                        border: none;
                        font-size: 24px;
                        font-weight: bold;
                        color: white;
                    }
                """)
                name_label.setStyleSheet("""
                    QLabel#stageName {
                        background: transparent;
                        font-size: 24px;
                        font-weight: bold;
                        color: rgb(77, 171, 201);
                    }
                """)
            else:
                # 其他阶段（无论是否完成）：浅青绿色圆圈，水绿色数字
                # 完成后只有文字变绿，数字底色保持浅青绿色
                number_label.setStyleSheet("""
                    QLabel#stageNumber {
                        background-color: rgb(227, 247, 253);
                        border: none;
                        font-size: 24px;
                        font-weight: bold;
                        color: rgb(77, 171, 201);
                    }
                """)

                # 文字颜色：已完成的变绿色，未开始的保持水绿色
                if is_completed:
                    name_label.setStyleSheet("""
                        QLabel#stageName {
                            background: transparent;
                            font-size: 24px;
                            font-weight: bold;
                            color: #4CAF50;
                        }
                    """)
                else:
                    name_label.setStyleSheet("""
                        QLabel#stageName {
                            background: transparent;
                            font-size: 24px;
                            font-weight: bold;
                            color: rgb(77, 171, 201);
                        }
                    """)

    def mark_stage_completed(self, stage_name: str):
        """标记阶段为已完成

        Args:
            stage_name: 阶段名称，支持新旧命名自动映射
        """
        # 🔄 兼容旧命名，自动映射到新命名
        name_mapping = {
            '基线校准': '多模态疲劳检测',
            'SART实验': '多模态疲劳检测',
            '朗读录音': '情绪检测',
            '血压测试': '血压脉搏检测',
            '舒尔特测试': '舒尔特专注度检测'
        }

        # 如果是旧命名，映射到新命名
        mapped_stage = name_mapping.get(stage_name, stage_name)

        self.stage_completed[mapped_stage] = True
        logger.debug(f"✅ 阶段已完成: {stage_name} → {mapped_stage}")
        self._update_stage_nav_status()

    def _create_main_content_area(self):
        container = QWidget()
        self.content_layout = QHBoxLayout(container)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(scale(15))
        
        # 创建左侧摄像头视图（内嵌模式）
        self.camera_widget = self._create_camera_view()
        # 使用最大宽度限制而不是固定宽度，这样隐藏时不占空间
        cam_width = scale_size(560, 420)[0]  # 获取摄像头宽度
        self.camera_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.content_layout.addWidget(self.camera_widget, 1)  # stretch factor = 0

        self.answer_stack = QStackedWidget()
        self._create_answer_area_widgets()
        self.content_layout.addWidget(self.answer_stack, 1)  # stretch factor = 1

        return container

    def _create_camera_view(self):
        """创建摄像头视图，包含画面和疲劳度、脑负荷信息"""
        inner_widget = QWidget()
        vlayout = QVBoxLayout(inner_widget)
        vlayout.setSpacing(scale(8))
        vlayout.setContentsMargins(0, 0, 0, 0)

        vlayout.addStretch(1)

        # 摄像头画面 - 统一尺寸与舒尔特页面保持一致
        cam_width, cam_height = scale_size(620, 480)
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

        # ❌ 疲劳度信息容器已移除（仅后台记录数据）
        # 创建隐藏的疲劳度标签以保持代码兼容性
        self.fatigue_info_label = QLabel("疲劳度: --")
        self.fatigue_info_label.setVisible(False)  # 隐藏不显示
        margin = scale(6)

        # ✅ 脑负荷信息容器 - 模仿舒尔特右边框样式(白色背景+底部阴影)
        brain_load_container = QFrame()
        brain_load_container.setObjectName("brainLoadContainer")
        brain_load_container.setFixedWidth(cam_width)
        brain_load_container.setStyleSheet("""
                  QFrame#brainLoadContainer {
                      background-color: #ffffff;
                      border: 2px solid #e0e0e0;
                      border-radius: 10px;
                      padding: 10px;
                  }
              """)
        
        # 添加底部阴影效果(模仿舒尔特框)
        brain_load_shadow = QGraphicsDropShadowEffect()
        brain_load_shadow.setBlurRadius(15)
        brain_load_shadow.setXOffset(0)
        brain_load_shadow.setYOffset(5)  # 底部阴影
        brain_load_shadow.setColor(QColor(0, 0, 0, 60))
        brain_load_container.setGraphicsEffect(brain_load_shadow)

        brain_load_layout = QVBoxLayout(brain_load_container)
        brain_load_layout.setSpacing(scale(6))
        brain_load_layout.setContentsMargins(margin, margin, margin, margin)

        # 脑负荷标题（统一使用超大字号，与舒尔特页面一致）
        brain_load_title_label = QLabel("🧠 脑负荷监测")
        brain_load_title_label.setAlignment(Qt.AlignCenter)
        brain_load_title_label.setStyleSheet("""
            color: #2c3e50;
            padding: 8px;
            font-size: 24px;
            font-weight: bold;
        """)
        brain_load_layout.addWidget(brain_load_title_label)

        # 分隔线
        brain_load_separator = QFrame()
        brain_load_separator.setFrameShape(QFrame.HLine)
        brain_load_separator.setFrameShadow(QFrame.Sunken)
        brain_load_separator.setStyleSheet("background-color: #bdc3c7;")
        brain_load_layout.addWidget(brain_load_separator)

        # 脑负荷显示（超大号）
        self.brain_load_info_label = QLabel("脑负荷: --")
        self.brain_load_info_label.setAlignment(Qt.AlignCenter)
        self.brain_load_info_label.setStyleSheet("""
                  QLabel {
                      color: #7f8c8d;
                      padding: 12px;
                      background-color: #ecf0f1;
                      border-radius: 8px;
                      font-size: 22px;
                      font-weight: bold;
                  }
              """)
        brain_load_layout.addWidget(self.brain_load_info_label)

        # 提示信息（放在脑负荷容器内部，与舒尔特页面保持一致）
        tip_label = QLabel("实时监测中...")
        tip_font = QFont()
        tip_font.setPointSize(scale_font(8))
        tip_label.setFont(tip_font)
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("color: #95a5a6; padding: 8px;")
        brain_load_layout.addWidget(tip_label)

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

    def _create_camera_view_for_schulte(self):
        """为舒尔特页面创建摄像头视图（与情绪检测页面完全一致，独立widget但共享AV数据源）"""
        inner_widget = QWidget()
        vlayout = QVBoxLayout(inner_widget)
        vlayout.setSpacing(scale(8))
        vlayout.setContentsMargins(0, 0, 0, 0)

        vlayout.addStretch(1)

        # 摄像头画面 - 与情绪检测页面使用完全相同的尺寸和样式
        cam_width, cam_height = scale_size(560, 420)
        self.schulte_camera_preview = CameraPreviewWidget(cam_width, cam_height, placeholder_text="摄像头画面加载中...")
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

        # ======================== 疲劳度信息(隐藏,仅后台记录) ========================
        # 创建隐藏的标签以维持代码兼容性
        self.schulte_fatigue_label = QLabel("--")
        self.schulte_fatigue_label.hide()  # 隐藏显示

        # 脑负荷信息容器 - 与情绪检测页面完全一致
        brain_load_container = QFrame()
        brain_load_container.setObjectName("schulteBrainLoadContainer")
        brain_load_container.setFixedWidth(cam_width)
        brain_load_container.setStyleSheet("""
                  QFrame#schulteBrainLoadContainer {
                      background-color: #ffffff;
                      border: 2px solid #e0e0e0;
                      border-radius: 10px;
                      padding: 10px;
                  }
              """)
        
        # 添加底部阴影效果
        schulte_brain_shadow = QGraphicsDropShadowEffect()
        schulte_brain_shadow.setBlurRadius(15)
        schulte_brain_shadow.setXOffset(0)
        schulte_brain_shadow.setYOffset(5)  # 底部阴影
        schulte_brain_shadow.setColor(QColor(0, 0, 0, 60))
        brain_load_container.setGraphicsEffect(schulte_brain_shadow)

        brain_load_layout = QVBoxLayout(brain_load_container)
        brain_load_layout.setSpacing(scale(6))
        margin = scale(6)
        brain_load_layout.setContentsMargins(margin, margin, margin, margin)

        # 脑负荷标题（与情绪检测页面完全一致）
        brain_load_title_label = QLabel("🧠 脑负荷监测")
        brain_load_title_label.setAlignment(Qt.AlignCenter)
        brain_load_title_label.setStyleSheet("""
            color: #2c3e50;
            padding: 8px;
            font-size: 24px;
            font-weight: bold;
        """)
        brain_load_layout.addWidget(brain_load_title_label)

        # 分隔线
        brain_load_separator = QFrame()
        brain_load_separator.setFrameShape(QFrame.HLine)
        brain_load_separator.setFrameShadow(QFrame.Sunken)
        brain_load_separator.setStyleSheet("background-color: #bdc3c7;")
        brain_load_layout.addWidget(brain_load_separator)

        # 脑负荷显示（与情绪检测页面完全一致）
        self.schulte_brain_load_label = QLabel("脑负荷: --")
        self.schulte_brain_load_label.setAlignment(Qt.AlignCenter)
        self.schulte_brain_load_label.setStyleSheet("""
                  QLabel {
                      color: #7f8c8d;
                      padding: 12px;
                      background-color: #ecf0f1;
                      border-radius: 8px;
                      font-size: 22px;
                      font-weight: bold;
                  }
              """)
        brain_load_layout.addWidget(self.schulte_brain_load_label)

        # 提示信息（与情绪检测页面完全一致）
        tip_label = QLabel("实时监测中...")
        tip_font = QFont()
        tip_font.setPointSize(scale_font(8))
        tip_label.setFont(tip_font)
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("color: #95a5a6; padding: 8px;")
        brain_load_layout.addWidget(tip_label)

        # 将脑负荷容器添加到布局
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

    def _create_baseline_prompt_page(self):
        """创建基线校准提示页面（带白色圆角外框）"""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(scale(20), scale(20), scale(20), scale(20))
        page_layout.setSpacing(scale(12))

        # 创建白色圆角矩形容器（外框 - 增加圆角和阴影）
        content_frame = QFrame()
        content_frame.setObjectName("baselinePromptFrame")
        content_frame.setStyleSheet("""
            QFrame#sartPromptFrame {
                background-image: url("ui/assets/sart");
                background-repeat: no-repeat;
                background-position: center;
                background-origin: content;
                background-size: contain;
                border-radius: 60px;
            }
        """)

        content_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 添加底部阴影效果
        baseline_shadow = QGraphicsDropShadowEffect()
        baseline_shadow.setBlurRadius(20)
        baseline_shadow.setXOffset(0)
        baseline_shadow.setYOffset(8)
        baseline_shadow.setColor(QColor(0, 0, 0, 80))
        content_frame.setGraphicsEffect(baseline_shadow)

        # 容器内部布局
        layout = QVBoxLayout(content_frame)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(scale(30))
        layout.setContentsMargins(scale(40), scale(32), scale(40), scale(32))

        # 添加顶部弹性空间
        layout.addStretch(1)

        # 标题（更大字号，确保单行显示）✅
        title_label = QLabel("请注视屏幕中央的十字，进行30s静息基线采集。")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(False)  # ✅ 禁止换行
        title_label.setStyleSheet("""
            color: #2c3e50;
            font-size: 50px;
            font-weight: bold;
            padding: 20px;
        """)  # ✅ 从32px增加到40px
        layout.addWidget(title_label)

        # 添加中间弹性空间
        layout.addStretch(2)

        # 十字符号（超大号）
        cross_label = QLabel("+")
        cross_label.setAlignment(Qt.AlignCenter)
        cross_label.setStyleSheet("""
            color: #000000;
            font-size: 300px;
            font-weight: bold;
        """)
        layout.addWidget(cross_label)

        # 添加底部弹性空间
        layout.addStretch(2)

        # 开始按钮（尺寸加大，与校准页面统一）✅
        self.btn_start_baseline = QPushButton("点击开始")
        self.btn_start_baseline.setObjectName("primaryButton")
        self.btn_start_baseline.setFixedSize(scale(280), scale(80))  # ✅ 从240×60增加到280×80
        self.btn_start_baseline.setCursor(Qt.PointingHandCursor)
        self.btn_start_baseline.setStyleSheet("""
            QPushButton#primaryButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 32px;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover {
                background-color: #3498DB;
            }
            QPushButton#primaryButton:pressed {
                background-color: #2E86C1;
            }
        """)  # ✅ 字体从24px增加到32px，圆角从15px增加到20px
        self.btn_start_baseline.clicked.connect(self._on_start_baseline_clicked)
        layout.addWidget(self.btn_start_baseline, 0, Qt.AlignCenter)

        # 添加底部一点空间
        layout.addStretch(1)

        page_layout.addWidget(content_frame, 1)
        return page

    def _create_sart_prompt_page(self):
        """创建SART实验提示页面（用整张图片替代中间内容）"""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(scale(16), scale(16), scale(16), scale(16))
        page_layout.setSpacing(scale(12))

        # 创建白色圆角矩形容器
        content_frame = QFrame()
        content_frame.setObjectName("sartPromptFrame")
        content_frame.setStyleSheet("""
            QFrame#sartPromptFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 60px;
            }
        """)

        # 添加阴影
        sart_shadow = QGraphicsDropShadowEffect()
        sart_shadow.setBlurRadius(20)
        sart_shadow.setXOffset(0)
        sart_shadow.setYOffset(8)
        sart_shadow.setColor(QColor(0, 0, 0, 80))
        content_frame.setGraphicsEffect(sart_shadow)

        # 主布局
        layout = QVBoxLayout(content_frame)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(scale(30))
        layout.setContentsMargins(scale(60), scale(40), scale(60), scale(40))

        # 顶部标题（字体更大、单行显示）✅
        title_label = QLabel("基线校准结束，点击按钮开始多模态疲劳检测。")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(False)  # 禁止换行
        title_label.setStyleSheet("""
            color: #2c3e50;
            font-size: 42px;
            font-weight: bold;
            padding: 20px;
        """)
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addStretch(1)

        # ✅ 中间图片部分
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: none; background-color: transparent;")

        image_path = str(config.BASE_DIR / "assets" / "sart.png")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 按比例缩放，尽量填满中间区域（保留边距）
                scaled_pixmap = pixmap.scaled(
                    scale(1000), scale(600),  # 根据你的窗口大小调整这两个值
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("⚠️ 图片加载失败")
                image_label.setStyleSheet("font-size: 24px; color: #e74c3c;")
        else:
            image_label.setText("⚠️ 未找到图片: ui/assets/sart")
            image_label.setStyleSheet("font-size: 24px; color: #e74c3c;")

        # 添加大图并让它扩展空间
        layout.addStretch(1)
        layout.addWidget(image_label, 1, Qt.AlignCenter)
        layout.addStretch(1)

        # ✅ 大按钮部分（与其他页面统一）
        self.btn_start_sart = QPushButton("我已了解规则，开始测试")
        self.btn_start_sart.setObjectName("sartPrimaryButton")
        self.btn_start_sart.setFixedSize(scale(500), scale(330))  # 大按钮
        self.btn_start_sart.setCursor(Qt.PointingHandCursor)
        self.btn_start_sart.setStyleSheet("""
            QPushButton#sartPrimaryButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 25px;
                font-size: 32px;
                font-weight: bold;
            }
            QPushButton#sartPrimaryButton:hover {
                background-color: #3498DB;
            }
            QPushButton#sartPrimaryButton:pressed {
                background-color: #2E86C1;
            }
        """)
        self.btn_start_sart.clicked.connect(self._on_start_sart_clicked)
        layout.addWidget(self.btn_start_sart, 0, Qt.AlignCenter)
        layout.addStretch(1)

        page_layout.addWidget(content_frame)
        return page

    def _create_answer_area_widgets(self):
        # 🆕 基线校准提示页面
        page_baseline_prompt = self._create_baseline_prompt_page()
        self.answer_stack.addWidget(page_baseline_prompt)
        
        # 🆕 SART实验提示页面
        page_sart_prompt = self._create_sart_prompt_page()
        self.answer_stack.addWidget(page_sart_prompt)
        
        # 🔄 朗读录音页面 - 使用卡片容器(模仿舒尔特右边框)
        page_qna = QWidget()
        # ✅ 设置透明背景，与舒尔特页面保持一致
        page_qna.setStyleSheet("QWidget { background-color: transparent; }")
        layout_qna = QVBoxLayout(page_qna)
        layout_qna.setAlignment(Qt.AlignCenter)
        layout_qna.setSpacing(scale(20))
        layout_qna.setContentsMargins(scale(20), scale(20), scale(20), scale(20))

        # 创建白色卡片容器(模仿舒尔特右边框样式)
        card_container = QFrame()
        card_container.setObjectName("emotionCardContainer")
        # 🔧 修复：设置最大宽度限制，防止被内容撑得过宽
        card_container.setMaximumWidth(scale(1200))  # 限制最大宽度
        card_container.setStyleSheet("""
            QFrame#emotionCardContainer {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                padding: 20px;
            }
        """)
        
        # 添加底部阴影效果(模仿舒尔特框)
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(15)
        card_shadow.setXOffset(0)
        card_shadow.setYOffset(5)
        card_shadow.setColor(QColor(0, 0, 0, 60))
        card_container.setGraphicsEffect(card_shadow)
        
        card_layout = QVBoxLayout(card_container)
        card_layout.setSpacing(scale(20))
        card_layout.setContentsMargins(scale(15), scale(15), scale(15), scale(15))

        # 标题：朗读文本(老年模式 - 超大字体)
        title_label = QLabel("📖 请朗读以下文本")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(80)  # ✅ 从32增大到80，更适合老年人
        title_font.setBold(True)
        title_label.setFont(title_font)
        card_layout.addWidget(title_label)

        # 文本显示区域（可滚动的文本框）
        self.lbl_reading_text = QTextEdit()
        self.lbl_reading_text.setReadOnly(True)
        self.lbl_reading_text.setObjectName("readingTextDisplay")
        self.lbl_reading_text.setMinimumWidth(scale(900))
        self.lbl_reading_text.setMinimumHeight(scale(500))  # ✅ 从300增加到350，给更大字体更多空间
        self.lbl_reading_text.setMaximumHeight(scale(500))  # ✅ 从400增加到500
        
        # 设置文本样式(老年模式 - 更大字体)
        text_font = QFont()
        text_font.setPointSize(28)  # ✅ 从20增大到28，显著提升可读性
        self.lbl_reading_text.setFont(text_font)
        self.lbl_reading_text.setStyleSheet("""
            QTextEdit#readingTextDisplay {
                background-color: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                padding: 38px;  
                line-height: 1.4;  
                color: #212529;
                font-size: 27px;
            }
        """)
        
        # 设置文本内容
        self.lbl_reading_text.setPlainText(self.reading_text_content)
        self.lbl_reading_text.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        
        card_layout.addWidget(self.lbl_reading_text, 0, Qt.AlignCenter)

        # 控制区域 - 横向布局(左:麦克风+音量条, 右:完成录音按钮)
        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(scale(40))
        control_layout.setAlignment(Qt.AlignCenter)
        
        # 左侧:麦克风按钮、状态和音量条(垂直排列)
        left_control = QWidget()
        left_layout = QVBoxLayout(left_control)
        left_layout.setSpacing(scale(15))
        left_layout.setAlignment(Qt.AlignCenter)

        # 麦克风按钮
        self.btn_mic = QPushButton()
        self.btn_mic.setObjectName("micButtonCallToAction")
        self.btn_mic.setFixedSize(130, 130)
        self.btn_mic.setIconSize(QSize(60, 60))
        self.btn_mic.setCursor(Qt.PointingHandCursor)
        self.btn_mic.setIcon(qta.icon('fa5s.microphone-alt', color='white'))

        # 录音状态标签(老年模式 - 更大字体)
        self.lbl_recording_status = QLabel("点击录音按钮开始朗读并录音")
        self.lbl_recording_status.setObjectName("statusLabel")
        self.lbl_recording_status.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(16)
        self.lbl_recording_status.setFont(status_font)
        
        # 音量显示(放在麦克风下面)
        self.audio_level = AudioLevelMeter()
        self.audio_level.setFixedWidth(350)

        left_layout.addWidget(self.btn_mic, 0, Qt.AlignCenter)
        left_layout.addWidget(self.lbl_recording_status, 0, Qt.AlignCenter)
        left_layout.addWidget(self.audio_level, 0, Qt.AlignCenter)
        
        # 右侧:完成录音按钮(变大,与左侧对齐)
        right_control = QWidget()
        right_layout = QVBoxLayout(right_control)
        right_layout.setSpacing(scale(10))
        right_layout.setAlignment(Qt.AlignCenter)
        
        # 完成录音按钮(增大尺寸)
        self.btn_next = QPushButton("完成录音")
        self.btn_next.setObjectName("successButton")
        self.btn_next.setIcon(qta.icon('fa5s.arrow-right'))
        self.btn_next.setFixedSize(scale(280), scale(90))  # 从200x70增加到280x90,更大更醒目
        self.btn_next.setStyleSheet("""
            QPushButton#successButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 28px;
                font-weight: bold;
            }
            QPushButton#successButton:hover {
                background-color: #45a049;
            }
            QPushButton#successButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton#successButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        # ❌ 不要在这里连接信号！已在 _connect_signals() 中连接
        # self.btn_next.clicked.connect(self._next_step_or_question)
        self.btn_next.setEnabled(False)
        right_layout.addWidget(self.btn_next, 0, Qt.AlignCenter)

        control_layout.addWidget(left_control)
        control_layout.addWidget(right_control)
        
        card_layout.addWidget(control_container)
        
        # 将卡片添加到主页面
        layout_qna.addStretch(1)
        layout_qna.addWidget(card_container, 0, Qt.AlignCenter)
        layout_qna.addStretch(1)

        self.answer_stack.addWidget(page_qna)

        # 血压测试页面
        page_blood_pressure = self._create_blood_pressure_page()
        self.answer_stack.addWidget(page_blood_pressure)

        # 舒特格测试页面
        self.page_schulte = self._create_schulte_page()
        self.answer_stack.addWidget(self.page_schulte)

        # 信息确认页面
        page_confirm = self._create_info_page()
        self.answer_stack.addWidget(page_confirm)

        # 分数展示页面（使用ScorePage组件）
        self.score_page = ScorePage(username=self.current_user)
        self.answer_stack.addWidget(self.score_page)

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
        """创建血压脉搏测试页面(卡片布局 - 左图右文在一个卡片中)"""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setAlignment(Qt.AlignCenter)
        page_layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))
        
        # 创建白色卡片容器(模仿舒尔特右边框样式)
        card_container = QFrame()
        card_container.setObjectName("bpCardContainer")
        # 🔧 修复：设置固定大小，防止内容变化导致卡片尺寸变化
        card_container.setFixedSize(scale(1200), scale(600))  # 固定宽度1200，高度600
        card_container.setStyleSheet("""
            QFrame#bpCardContainer {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                padding: 30px;
            }
        """)
        
        # 添加底部阴影效果
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(15)
        card_shadow.setXOffset(0)
        card_shadow.setYOffset(5)
        card_shadow.setColor(QColor(0, 0, 0, 60))
        card_container.setGraphicsEffect(card_shadow)
        
        # 卡片内的横向布局(左图右文)
        main_layout = QHBoxLayout(card_container)
        main_layout.setSpacing(scale(25))  # 紧凑间距
        main_layout.setContentsMargins(scale(20), scale(20), scale(20), scale(20))
        
        # 左侧:血压仪图片提示（固定高度）
        left_widget = QWidget()
        left_widget.setFixedWidth(scale(450))  # 固定宽度
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(scale(15))
        
        # 图片标签
        image_label = QLabel()
        image_label.setObjectName("bpImageLabel")
        image_label.setAlignment(Qt.AlignCenter)
        
        # 加载图片 (相对于ui目录,显示更大,无边框)
        image_path = "assets/maibobo/maibobo.jpg"
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 🔧 修复：进一步减小图片尺寸以适配1080p显示器
                scaled_pixmap = pixmap.scaled(
                    scale(400), scale(450),  # 从450x510进一步减小到400x450
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
                # 移除边框和背景,只显示图片
                image_label.setStyleSheet("""
                    QLabel#bpImageLabel {
                        border: none;
                        background-color: transparent;
                        padding: 0px;
                    }
                """)
            else:
                image_label.setText("图片加载失败")
                image_label.setStyleSheet("font-size: 18px; color: #999;")
        else:
            image_label.setText("⚠️\n图片文件不存在")
            image_label.setStyleSheet("font-size: 20px; color: #f44336;")
        
        # 图片说明(更大字体)
        tip_label = QLabel("请将手臂放置在仪器测量位置")
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #333;")
        
        left_layout.addStretch(1)
        left_layout.addWidget(image_label, 0, Qt.AlignCenter)
        left_layout.addWidget(tip_label, 0, Qt.AlignCenter)
        left_layout.addStretch(1)
        
        # 右侧:测试控制和结果(垂直布局，固定宽度)
        right_widget = QWidget()
        right_widget.setFixedWidth(scale(670))  # 固定宽度（1200 - 450 - 20*2边距 - 25间距 - 60padding）
        layout = QVBoxLayout(right_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(scale(25))

        # 标题
        title_label = QLabel("血压脉搏检测")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 40px; font-weight: bold;")

        # 说明文字
        # description_label = QLabel(
        #     "将手臂放置在测量位置\n\n点击开始测试按钮"
        # )
        # description_label.setObjectName("subtitle")
        # description_label.setAlignment(Qt.AlignCenter)
        # description_label.setWordWrap(True)
        # description_label.setStyleSheet("font-size: 24px; color: #666;")

        # 设备状态区域
        self.bp_status_container = QWidget()  # 保存引用以便隐藏
        status_layout = QVBoxLayout(self.bp_status_container)
        status_layout.setSpacing(scale(12))

        # 设备连接状态(改为蓝色,字体更大)
        self.bp_status_label = QLabel("正在检测血压仪器连接...")
        self.bp_status_label.setObjectName("statusLabel")
        self.bp_status_label.setAlignment(Qt.AlignCenter)
        self.bp_status_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #1976D2;")

        # 测试进度显示(字体更大)
        self.bp_progress_label = QLabel("等待开始测试")
        self.bp_progress_label.setObjectName("subtitle")
        self.bp_progress_label.setAlignment(Qt.AlignCenter)
        self.bp_progress_label.setStyleSheet("font-size: 28px; color: #666;")

        status_layout.addWidget(self.bp_status_label)
        status_layout.addWidget(self.bp_progress_label)

        # 测试控制区域
        self.bp_control_container = QWidget()  # 保存引用以便隐藏
        control_layout = QVBoxLayout(self.bp_control_container)
        control_layout.setSpacing(scale(20))

        # 圆形进度指示器(还原回原来的样式)
        self.bp_progress_circle = QLabel()
        self.bp_progress_circle.setFixedSize(100, 100)  # 还原原始尺寸
        self.bp_progress_circle.setAlignment(Qt.AlignCenter)
        self.bp_progress_circle.setStyleSheet("""
            QLabel {
                border: 4px solid #E0E0E0;
                border-radius: 50px;
                background-color: #F5F5F5;
                color: #666;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.bp_progress_circle.setText("准备")
        
        # 开始/停止测试按钮(更大)
        self.bp_start_button = QPushButton("开始测试")
        self.bp_start_button.setObjectName("successButton")
        self.bp_start_button.setFixedSize(scale(220), scale(70))  # 增大按钮
        self.bp_start_button.setStyleSheet("font-size: 26px; font-weight: bold; background-color: #5DADE2;")  # 老年模式
        self.bp_start_button.clicked.connect(self._toggle_bp_test)
        self.bp_start_button.setEnabled(False)  # 初始禁用

        control_layout.addWidget(self.bp_progress_circle, 0, Qt.AlignCenter)
        control_layout.addWidget(self.bp_start_button, 0, Qt.AlignCenter)

        # 结果显示区域（固定高度，避免显示/隐藏时布局变化）
        self.result_container = QWidget()
        self.result_container.setVisible(False)
        self.result_container.setFixedHeight(scale(380))  # 固定高度 = "测试完成"标签 + 卡片高度 + 间距
        result_layout = QVBoxLayout(self.result_container)
        result_layout.setSpacing(scale(15))  # "测试完成"和卡片之间的间距
        result_layout.setContentsMargins(0, 0, 0, 0)  # 移除外边距

        # "测试完成"标签（测试成功时才显示）
        self.bp_complete_label = QLabel("测试完成 ✅")
        self.bp_complete_label.setObjectName("subtitle")
        self.bp_complete_label.setAlignment(Qt.AlignCenter)
        self.bp_complete_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")

        # 结果卡片（固定大小）
        self.result_card = QWidget()
        self.result_card.setObjectName("card")
        self.result_card.setFixedSize(scale(550), scale(300))  # 适度增加高度，更舒适
        result_card_layout = QVBoxLayout(self.result_card)
        result_card_layout.setSpacing(scale(12))  # 增加三行数据之间的间距
        result_card_layout.setContentsMargins(scale(20), scale(25), scale(20), scale(25))  # 增加上下内边距

        # 收缩压
        self.systolic_label = QLabel("收缩压: -- mmHg")
        self.systolic_label.setObjectName("statusLabel")
        self.systolic_label.setAlignment(Qt.AlignCenter)
        self.systolic_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #1976D2;")  # 统一大字体

        # 舒张压
        self.diastolic_label = QLabel("舒张压: -- mmHg")
        self.diastolic_label.setObjectName("statusLabel")
        self.diastolic_label.setAlignment(Qt.AlignCenter)
        self.diastolic_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #1976D2;")  # 统一大字体

        # 脉搏
        self.pulse_label = QLabel("脉搏: -- 次/分")
        self.pulse_label.setObjectName("statusLabel")
        self.pulse_label.setAlignment(Qt.AlignCenter)
        self.pulse_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #4CAF50;")  # 统一大字体

        # 在卡片内添加"进入下一步"按钮
        self.bp_next_button = QPushButton("请先完成血压测试")
        self.bp_next_button.setObjectName("bpNextButton")
        self.bp_next_button.setFixedSize(scale(240), scale(60))
        self.bp_next_button.setCursor(Qt.PointingHandCursor)
        self.bp_next_button.setStyleSheet("""
            QPushButton#bpNextButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton#bpNextButton:hover {
                background-color: #3498DB;
            }
            QPushButton#bpNextButton:pressed {
                background-color: #2E86C1;
            }
            QPushButton#bpNextButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)
        self.bp_next_button.setEnabled(False)  # 初始禁用
        self.bp_next_button.clicked.connect(self._next_step_or_question)

        result_card_layout.addWidget(self.systolic_label)
        result_card_layout.addWidget(self.diastolic_label)
        result_card_layout.addWidget(self.pulse_label)
        result_card_layout.addSpacing(scale(15))  # 增加数据和按钮之间的间距
        result_card_layout.addWidget(self.bp_next_button, 0, Qt.AlignCenter)

        result_layout.addWidget(self.bp_complete_label)
        result_layout.addWidget(self.result_card, 0, Qt.AlignCenter)

        # 右侧布局组装
        layout.addStretch(1)
        layout.addWidget(title_label)
        # layout.addWidget(description_label)
        layout.addWidget(self.bp_status_container, 0, Qt.AlignCenter)
        layout.addWidget(self.bp_control_container, 0, Qt.AlignCenter)
        layout.addWidget(self.result_container, 0, Qt.AlignCenter)
        layout.addStretch(1)
        
        # 主布局组装:左图右文都在卡片内（不使用stretch因子，因为已固定宽度）
        main_layout.addWidget(left_widget, 0)  # 左侧图片区域（固定宽度450）
        main_layout.addWidget(right_widget, 0)  # 右侧控制区域（固定宽度670）
        
        # 将卡片添加到页面(居中显示)
        page_layout.addStretch(1)
        page_layout.addWidget(card_container, 0, Qt.AlignCenter)
        page_layout.addStretch(1)

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
            # ✅ 血压测试进行中不允许手动停止
            QMessageBox.warning(
                self, 
                "测试进行中", 
                "血压测试正在进行中，无法手动停止。\n请等待测试自动完成。"
            )
            logger.warning("⚠️ 血压测试进行中，禁止手动停止")
            return

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
            # ✅ 测试进行中：禁用按钮，防止点击
            self.bp_start_button.setText("测试进行中...")
            self.bp_start_button.setEnabled(False)  # 禁用按钮
            self.bp_start_button.setObjectName("disabledButton")
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

            logger.debug("血压测试已开始")

        except Exception as e:
            logger.error(f"开始血压测试失败: {e}")
            self._stop_bp_test()

    def _stop_bp_test(self):
        """停止血压测试（仅在测试完成或出错时内部调用）"""
        try:
            self.bp_test_running = False
            # ✅ 测试结束：恢复按钮可用状态
            self.bp_start_button.setText("开始测试")
            self.bp_start_button.setEnabled(True)  # 恢复可用
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

            logger.debug("血压测试已停止")

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
                logger.debug(
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

                # 保持统一的大字体（30px），只改变颜色
                self.systolic_label.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")
                self.diastolic_label.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")
                self.pulse_label.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")

                # 显示结果，隐藏状态和控制区域
                self.result_container.setVisible(True)
                self.bp_status_container.setVisible(False)
                self.bp_control_container.setVisible(False)
                
                # 隐藏加载圆圈和开始测试按钮（已在control_container中，但为保险起见也单独设置）
                self.bp_progress_circle.setVisible(False)
                self.bp_start_button.setVisible(False)
                
                # 启用卡片内的"进入下一步"按钮
                if hasattr(self, 'bp_next_button'):
                    self.bp_next_button.setText("进入舒特格测试")
                    self.bp_next_button.setEnabled(True)

                # 不再需要更新bp_progress_label，因为已经隐藏了status_container
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
                
                # 失败时保持status和control容器可见，允许重新测试
                self.bp_status_container.setVisible(True)
                self.bp_control_container.setVisible(True)
                self.bp_progress_circle.setVisible(True)
                self.bp_start_button.setVisible(True)
                self.bp_start_button.setText("重新测试")
                self.bp_start_button.setEnabled(True)

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
                
                # 📍 确保文本问答开始时间戳已记录（如果还没记录则补记录）
                if not getattr(self, '_text_qa_start_timestamp_recorded', False):
                    call_timestamp = time.time()
                    self._save_timestamp_immediately(call_timestamp)
                    logger.info(f"📍 补记录文本问答开始时间戳(Q键跳过，页面未真正开始): {call_timestamp}")
                    self._text_qa_start_timestamp_recorded = True
                
                # 停止音视频录制并获取路径
                try:
                    logger.debug("📹 正在停止音视频录制...")
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
                
                # 📍 记录文本问答结束时间戳
                call_timestamp = time.time()
                self._save_timestamp_immediately(call_timestamp)
                logger.info(f"📍 已记录文本问答结束时间戳(Q键跳过): {call_timestamp}")
                
                # 保存音视频路径到数据库
                self._persist_av_paths_to_db()
                
                # 切换到下一步
                self.current_step = 1
                self.update_step_ui()
                return True

            if self.current_step == 1:
                logger.info("测试后门触发：按下 Q，血压测试视为完成")
                
                # ✅ 如果血压测试正在运行，先停止它
                if hasattr(self, 'bp_test_running') and self.bp_test_running:
                    self._stop_bp_test()
                
                # 📍 记录血压测试开始时间戳（如果还没进入血压页面就跳过）
                # 正常流程：文本QA结束 → 血压开始 → 血压结束 → 舒尔特开始
                # 跳过场景：可能在血压页面加载时就按Q,需要补开始时间戳
                expected_timestamps_before_bp = 7  # 系统+设备校准+基线+SART+文本QA = 7个
                if len(self.part_timestamps) < expected_timestamps_before_bp + 1:
                    # 缺少血压开始时间戳，补记录
                    call_timestamp = time.time()
                    self._save_timestamp_immediately(call_timestamp)
                    logger.info(f"📍 补记录血压测试开始时间戳(Q键跳过): {call_timestamp}")
                
                # 📍 记录血压测试结束时间戳
                call_timestamp = time.time()
                self._save_timestamp_immediately(call_timestamp)
                logger.info(f"📍 已记录血压测试结束时间戳(Q键跳过): {call_timestamp}")
                
                # ✅ 设置模拟血压值
                self.bp_results = {
                    'systolic': 120,
                    'diastolic': 80,
                    'pulse': 75,
                }
                logger.info(f"✅ 已设置模拟血压值: {self.bp_results}")
                
                # ✅ 完成血压测试并显示结果
                self._complete_bp_test()
                
                # ⚠️ 不要立即切换到下一步，让用户看到结果
                # self.current_step = 2
                # self.update_step_ui()
                logger.info("✅ 血压测试已跳过，显示模拟结果")
                return True

            if self.current_step == 2:
                logger.info("测试后门触发：按下 Q，舒尔特测试视为完成")
                
                # 📍 检查是否已记录舒尔特开始时间戳（通过时间戳数量判断）
                # 如果舒尔特还没开始（刚进入页面就按Q），需要先记录开始时间戳
                expected_timestamps_before_schulte = 9  # 系统+设备校准+基线+SART+文本QA+血压 = 9个
                if len(self.part_timestamps) < expected_timestamps_before_schulte + 1:
                    # 缺少舒尔特开始时间戳，补记录
                    call_timestamp = time.time()
                    self._save_timestamp_immediately(call_timestamp)
                    logger.info(f"📍 补记录舒尔特测试开始时间戳: {call_timestamp}")

                try:
                    self._save_speech_recognition_results()
                    self._trigger_emotion_analysis()
                    logger.info("✅ 快捷键跳过：已保存语音识别结果并触发情绪分析")
                except Exception as e:
                    logger.warning(f"快捷键跳过时保存结果失败: {e}")
                
                
                self._on_schulte_result(30.0, 85.0)
                self._on_schulte_completed()
                return True
        except Exception as e:
            logger.warning(f"执行调试快捷操作失败: {e}")
        return False

    def keyPressEvent(self, event):
        """全局监听键盘事件，用于测试调试后门和基线/SART启动"""
        # 空格键：触发当前显示页面的开始按钮
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            current_index = self.answer_stack.currentIndex()
            if current_index == 0:  # 基线校准提示页面
                self.btn_start_baseline.click()
                return
            elif current_index == 1:  # SART提示页面
                self.btn_start_sart.click()
                return
        
        # 调试后门
        if event.key() == Qt.Key_Q and not event.isAutoRepeat():
            if self._handle_debug_shortcut():
                return
        super().keyPressEvent(event)

    def _create_schulte_page(self):
        page = QWidget()
        main_layout = QHBoxLayout(page)
        main_layout.setContentsMargins(scale(8), 0, scale(8), 0)
        main_layout.setSpacing(scale(20))

        # ✅ 左侧弹簧
        main_layout.addStretch(1)

        # 左列：摄像头
        self.schulte_camera_widget = self._create_camera_view_for_schulte()
        cam_width = scale_size(560, 420)[0]
        self.schulte_camera_widget.setMaximumWidth(cam_width)
        self.schulte_camera_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        main_layout.addWidget(self.schulte_camera_widget, 0, Qt.AlignRight)

        # 右列：舒尔特方格
        self.schulte_container = QWidget()
        schulte_layout = QVBoxLayout(self.schulte_container)
        schulte_layout.setContentsMargins(0, 0, 0, 0)
        schulte_layout.setSpacing(scale(10))

        self.schulte_widget = SchulteGridWidget(self.current_user)
        self.schulte_widget.test_completed.connect(self._on_schulte_completed)
        self.schulte_widget.test_result_ready.connect(self._on_schulte_result)
        schulte_layout.addWidget(self.schulte_widget, 0, Qt.AlignCenter)

        main_layout.addWidget(self.schulte_container, 0, Qt.AlignLeft)

        # ✅ 右侧弹簧
        main_layout.addStretch(1)

        return page

    def _reinit_schulte_widget(self):
        """重新初始化舒尔特widget（每次进入页面时调用，避免状态卡住）"""
        if not hasattr(self, 'schulte_container') or not hasattr(self, 'schulte_widget'):
            logger.warning("舒尔特容器或widget不存在，跳过重新初始化")
            return
        
        try:
            # 获取布局
            layout = self.schulte_container.layout()
            if layout is None:
                logger.warning("舒尔特容器没有布局")
                return
            
            # 移除旧的widget
            layout.removeWidget(self.schulte_widget)
            self.schulte_widget.deleteLater()
            
            # 创建新的widget
            self.schulte_widget = SchulteGridWidget(self.current_user)
            self.schulte_widget.test_completed.connect(self._on_schulte_completed)
            self.schulte_widget.test_result_ready.connect(self._on_schulte_result)
            
            # 添加到布局
            layout.addWidget(self.schulte_widget)
            
            logger.info("✅ 舒尔特widget已重新初始化")
        except Exception as e:
            logger.error(f"❌ 重新初始化舒尔特widget失败: {e}", exc_info=True)

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
        layout.setContentsMargins(0, scale(15), 0, scale(20))
        
        # 通用的下一步按钮(用于血压、舒尔特等其他页面)
        self.btn_next_bottom = QPushButton("下一步")
        self.btn_next_bottom.setObjectName("successButton")
        self.btn_next_bottom.setIcon(qta.icon('fa5s.arrow-right'))
        self.btn_next_bottom.setFixedSize(scale(320), scale(80))
        self.btn_next_bottom.setStyleSheet("""
            QPushButton#successButton {
                font-size: 26px;
                font-weight: bold;
            }
        """)
        self.btn_next_bottom.clicked.connect(self._next_step_or_question)
        
        # 完成评估按钮（用于最后的页面）
        self.btn_finish = QPushButton("完成评估")
        self.btn_finish.setObjectName("finishButton")
        self.btn_finish.setIcon(qta.icon('fa5s.flag-checkered'))
        self.btn_finish.setFixedSize(scale(320), scale(80))
        self.btn_finish.setStyleSheet("""
            QPushButton#finishButton {
                font-size: 26px;
                font-weight: bold;
            }
        """)
        self.btn_finish.setVisible(False)
        layout.addWidget(self.btn_next_bottom)
        layout.addWidget(self.btn_finish)
        return container

    def update_step_ui(self):
        # 兼容旧代码：保留原有的步骤标签更新(如果存在)
        if hasattr(self, 'step_labels') and self.step_labels:
            for i, (num_label, text_label) in enumerate(self.step_labels):
                target_opacity = 1.0 if i == self.current_step else 0.5
                if hasattr(self, 'step_opacity_effects') and i < len(self.step_opacity_effects):
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
        
        # 控制摄像头组件的可见性
        # ✅ 情绪检测阶段显示左侧摄像头（已设置固定宽度，不影响布局）
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
            # 🔄 朗读录音阶段（情绪检测）
            self.answer_stack.setCurrentIndex(2)  # 朗读页面是索引2（0基线提示，1SART提示）
            
            # 显示朗读文本（已在 _create_answer_area_widgets 中设置）
            if hasattr(self, 'lbl_reading_text'):
                self.lbl_reading_text.setPlainText(self.reading_text_content)
            
            # 情绪检测页面使用卡片内的btn_next,隐藏底部按钮
            if hasattr(self, 'btn_next'):
                self.btn_next.setText("完成录音")
                self.btn_next.setVisible(True)
                self.btn_next.setEnabled(False)  # 初始禁用，录音完成后启用
            if hasattr(self, 'btn_next_bottom'):
                self.btn_next_bottom.setVisible(False)  # 隐藏底部按钮
            self.btn_finish.setVisible(False)
            
            # ❌ 不再需要 TTS 朗读
            # if self.test_started:
            #     self._speak_current_question()

        elif self.current_step == 1:
            self.answer_stack.setCurrentIndex(3)  # 血压页面现在是索引3（0基线提示，1SART提示，2朗读）
            
            # 重置血压页面控件的可见性（当重新进入或从其他页面返回时）
            if hasattr(self, 'bp_progress_circle'):
                self.bp_progress_circle.setVisible(True)
            if hasattr(self, 'bp_start_button'):
                self.bp_start_button.setVisible(True)
            if hasattr(self, 'bp_status_container'):
                self.bp_status_container.setVisible(True)
            if hasattr(self, 'bp_control_container'):
                self.bp_control_container.setVisible(True)
            if hasattr(self, 'result_container'):
                # 如果已有测试结果则显示，否则隐藏
                if hasattr(self, 'bp_results') and self.bp_results.get('systolic') is not None:
                    self.result_container.setVisible(True)
                    # 已有结果时，隐藏状态和控制区域
                    self.bp_status_container.setVisible(False)
                    self.bp_control_container.setVisible(False)
                    # 更新卡片内按钮文字
                    if hasattr(self, 'bp_next_button'):
                        self.bp_next_button.setText("进入舒特格测试")
                        self.bp_next_button.setEnabled(True)
                else:
                    self.result_container.setVisible(False)
                    # 重置卡片内按钮状态
                    if hasattr(self, 'bp_next_button'):
                        self.bp_next_button.setText("请先完成血压测试")
                        self.bp_next_button.setEnabled(False)
            
            # 血压页面隐藏底部按钮（使用卡片内按钮代替）
            if hasattr(self, 'btn_next_bottom'):
                self.btn_next_bottom.setVisible(False)
            self.btn_finish.setVisible(False)  # 血压阶段不显示完成测试按钮
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()

        elif self.current_step == 2:
            logger.debug(f"📍 update_step_ui: current_step=2 (舒尔特)，设置 answer_stack index=4")
            self.answer_stack.setCurrentIndex(4)  # 舒尔特页面现在是索引4
            if hasattr(self, 'btn_next_bottom'):
                self.btn_next_bottom.setVisible(False)
            self.btn_finish.setVisible(False)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
        elif self.current_step == 3:
            self.answer_stack.setCurrentWidget(self.score_page)  # 分数页面现在是索引6
            # 异步更新分数页数据，避免阻塞UI
            def update_scores_async():
                try:
                    # 正确的调用顺序：先设置用户，再发送测试结果，最后更新显示
                    self.score_page._set_user(self.current_user)
                    self._send_scores_to_score_page()
                    self.score_page._update_scores()
                except Exception as e:
                    logger.error(f"更新分数页失败: {e}")
            
            # 先更新基本UI
            if hasattr(self, 'btn_next_bottom'):
                self.btn_next_bottom.setVisible(False)
            self.btn_finish.setVisible(True)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
            
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
        
        # 🔄 最后更新阶段导航状态（确保answer_stack已切换完成）
        self._update_stage_nav_status()

    def start_test(self):
        # 摄像头预览在 AV 采集准备好后启动
        self.audio_timer.start(50)
        self.current_step = 0
        self.current_question = 0
        self.btn_finish.setVisible(False)

        # 🔄 重置朗读录音状态
        self.reading_completed = False

        self.mark_stage_completed('多模态疲劳检测')
        
        # 重置分数累积列表
        self._fatigue_scores_list = []
        self._emotion_score = None
        self._emotion_analysis_triggered = False  # 重置情绪分析触发标志
        logger.debug("已重置分数累积列表和情绪分析标志")

        # ❌ 不再需要语音识别功能（用户自己朗读，不需要识别）
        # if HAS_SPEECH_RECOGNITION:
        #     stop_recognition()

        # 📍 记录朗读录音开始时间戳
        self._text_qa_start_timestamp_recorded = False

        try:
            # 如果 session_dir 已经存在（由 SART 页面创建），则直接使用
            if not self.session_dir or not hasattr(self, 'session_timestamp'):
                self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_dir = 'recordings'
                user_dir = self.current_user or 'anonymous'
                self.session_dir = _build_session_dir(base_dir, user_dir, self.session_timestamp)
                logger.info(f"创建新的会话目录: {self.session_dir}")
            else:
                logger.debug(f"使用已有会话目录: {self.session_dir}")
        except Exception as e:
            logger.error(f"处理会话目录失败: {e}")
            self.session_dir = 'recordings'
            os.makedirs(self.session_dir, exist_ok=True)

        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None

        self.test_started = True
        self.update_step_ui()

        # ✅ 疲劳度监控已在基线校准阶段启动，这里只需确保轮询继续运行
        if HAS_MULTIMODAL:
            # 检查监控是否已在运行，如果没有则启动
            timer_active = False
            try:
                timer_active = self._multimodal_poll_timer.isActive()
            except Exception:
                timer_active = False
            
            if not self._multimodal_poll_active or not timer_active:
                config.logger.debug("疲劳度监控未运行，启动监控轮询")
                self._start_multimodal_monitoring()
            else:
                config.logger.debug("✅ 疲劳度监控已在运行（从基线阶段继续）")

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
                logger.debug("等待后端服务器连接...")
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
                    logger.debug("✅ 后端服务器连接成功")
                
                # 尝试启动AV采集（即使后端未连接也尝试，可能使用本地摄像头）
                try:
                    logger.info(f"🎥 准备启动 AV 采集，session_dir={self.session_dir}")
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
                        logger.debug("✅ 多模态监控将在800ms后启动，从语音答题开始就可以看到脑负荷和疲劳度数据")
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
        

    def start_eeg_collection(self) -> None:
        self._brain_load_scores_list = []
        # EEG采集也使用异步方式（非阻塞），由后端统一管理硬件连接
        # ⚠️ 注意：如果EEG已经在基线/SART阶段启动，这里会返回 "already-running"，这是正常的
        def start_eeg_async():
            try:
                from ...services.backend_proxy import eeg_start
                result = eeg_start(username=self.current_user, save_dir=self.session_dir, part=1)
                if result.get('status') == 'already-running':
                    logger.debug(f"✅ EEG采集已在运行中，继续使用现有连接: {result.get('save_dir')}")
                else:
                    logger.info(f"✅ EEG采集已启动，保存目录: {self.session_dir}\\eeg")
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
                logger.debug("✅ 已停止情绪检测摄像头预览")
        except Exception as e:
            logger.debug(f"停止camera_preview时出错: {e}")
        
        try:
            if self.schulte_camera_preview:
                self.schulte_camera_preview.stop_preview()
                logger.debug("✅ 已停止舒尔特摄像头预览")
        except Exception as e:
            logger.debug(f"停止schulte_camera_preview时出错: {e}")

    def _update_camera_previews_for_step(self) -> None:
        """根据当前步骤切换摄像头预览（两个独立widget，根据步骤启停）。"""
        try:
            if self.current_step == 0:
                # 情绪检测阶段：启动情绪检测摄像头，停止舒尔特摄像头
                if self.schulte_camera_preview:
                    self.schulte_camera_preview.stop_preview()
                    logger.debug("已停止舒尔特摄像头")
                if self.camera_preview:
                    self.camera_preview.start_preview()
                    logger.debug("✅ 已启动情绪检测摄像头预览")
                    
            elif self.current_step == 2:
                # 舒尔特测试阶段：停止情绪检测摄像头，启动舒尔特摄像头
                if self.camera_preview:
                    self.camera_preview.stop_preview()
                    logger.debug("已停止情绪检测摄像头")
                if self.schulte_camera_preview:
                    self.schulte_camera_preview.start_preview()
                    logger.debug("✅ 已启动舒尔特摄像头预览")
            else:
                # 其他阶段：停止所有摄像头预览
                if self.camera_preview:
                    self.camera_preview.stop_preview()
                if self.schulte_camera_preview:
                    self.schulte_camera_preview.stop_preview()
                logger.debug("✅ 其他阶段：已停止所有摄像头预览")
                
        except Exception as e:
            logger.error(f"切换摄像头预览时出错: {e}", exc_info=True)
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
            # 原有停止逻辑
            self._stop_recording()

            # 视觉：停止“录制中”效果 → 恢复麦克风图标
            try:
                # 如果用了光晕动画，停止它（可选）
                if hasattr(self, "mic_anim") and self.mic_anim.state() == QPropertyAnimation.Running:
                    self.mic_anim.stop()
            except Exception:
                pass

            # 恢复麦克风图标与提示
            try:
                self.btn_mic.setIcon(qta.icon('fa5s.microphone-alt', color='white'))
            except Exception:
                pass
            self.btn_mic.setToolTip("点击开始录音")

            self.is_recording = False

        else:
            # 原有开始逻辑
            self._start_recording()

            # 视觉：显示“录制中”的图标（红点/圆）
            try:
                # 任选一个你喜欢的录制图标；三选一：
                # 1) 红色圆点
                self.btn_mic.setIcon(qta.icon('fa5s.circle', color='#e74c3c'))
                # 2) 靶心圆点（更像“录制”）
                # self.btn_mic.setIcon(qta.icon('fa5s.dot-circle', color='#e74c3c'))
                # 3) 黑胶样式（如果你喜欢）
                # self.btn_mic.setIcon(qta.icon('fa5s.record-vinyl', color='#e74c3c'))
            except Exception:
                pass
            self.btn_mic.setToolTip("录音中，点击结束")

            # 可选：开启现有的光晕动画，让“正在录制”更醒目（不喜欢就注释）
            try:
                if hasattr(self, "mic_anim"):
                    self.mic_anim.start()
            except Exception:
                pass

            self.is_recording = True

    def _start_recording(self):
        if self.mic_anim.state() == QPropertyAnimation.Running:
            self.mic_anim.stop()
        self.mic_shadow.setEnabled(False)
        self.is_recording = True

        self.btn_mic.setObjectName("micButtonRecording")
        self.btn_mic.setIcon(qta.icon('fa5s.stop', color='white'))

        self.btn_mic.style().unpolish(self.btn_mic)
        self.btn_mic.style().polish(self.btn_mic)

        self.lbl_recording_status.setText("正在录音，请朗读上方文本...")
        logger.info("开始朗读录音...")

        # 📍 记录朗读录音开始时间戳（第一次录音时）
        if not self._text_qa_start_timestamp_recorded:
            call_timestamp = time.time()
            self._save_timestamp_immediately(call_timestamp)
            logger.info(f"📍 已记录朗读录音开始时间戳: {call_timestamp}")
            self._text_qa_start_timestamp_recorded = True

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
        self.lbl_recording_status.setText("录制已完成，点击「完成录音」进入下一步")
        logger.info("朗读录音完毕。")
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

    # ❌ 不再需要 _speak_current_question（用户自己朗读）
    # def _speak_current_question(self):
    #     已删除 TTS 朗读相关代码

    def _next_step_or_question(self):
        logger.debug(f"🔍 _next_step_or_question 被调用: current_step={self.current_step}")

        if self.current_step == 0:
            # 🔄 朗读录音阶段，不再有多个问题，直接进入下一步
            # 📍 记录朗读录音结束时间戳
            call_timestamp = time.time()
            self._save_timestamp_immediately(call_timestamp)
            logger.info(f"📍 已记录朗读录音结束时间戳: {call_timestamp}")

            self.reading_completed = True
            self.mark_stage_completed('情绪检测')  # ✅ 标记阶段完成(新命名)
            
            # ✅ 立即更新分数页面，传递情绪检测结果
            try:
                if hasattr(self, 'score_page') and self.score_page:
                    logger.info(f"📊 情绪检测完成，立即更新分数页面 (emotion={self._emotion_score})")
                    self._send_scores_to_score_page()
                    logger.debug("✅ 情绪检测结果已发送到分数页面")
            except Exception as e:
                logger.error(f"发送情绪结果到分数页面失败: {e}", exc_info=True)
            
            self.current_step += 1
            logger.debug(f"✅ 情绪检测完成，current_step 增加到: {self.current_step}")

            try:
                self._close_camera()
            except Exception as e:
                logger.warning(f"关闭摄像头失败: {e}")

            # 停止音视频录制并获取路径
            try:
                logger.debug("📹 正在停止音视频录制...")
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
                self._trigger_emotion_analysis()
            except Exception as e:
                logger.warning(f"触发情绪分析失败: {e}")

            # ✅ 停止疲劳度推理与多模态数据采集（EEG 保持运行）
            if HAS_MULTIMODAL:
                try:
                    self._stop_multimodal_monitoring()
                    logger.debug("✅ 朗读阶段结束，已停止疲劳度监控定时器")
                except Exception as e:
                    logger.warning(f"停止疲劳度监控失败: {e}")

                try:
                    stop_result = multidata_stop_collection()
                    status = (stop_result or {}).get("status", "unknown")
                    logger.info(f"✅ 朗读阶段结束，已停止多模态采集 (状态={status})")
                    self.multimodal_collector = None

                    try:
                        self._persist_multimodal_paths_to_db(clear_recognition_cache=False)
                        logger.info("💾 朗读阶段结束，多模态数据路径已写入数据库")
                    except Exception as persist_exc:
                        logger.warning(f"多模态数据路径写入数据库失败: {persist_exc}")
                except Exception as stop_exc:
                    logger.warning(f"停止多模态数据采集失败: {stop_exc}")

            # 📍 记录血压测试开始时间戳
            call_timestamp = time.time()
            self._save_timestamp_immediately(call_timestamp)
            logger.info(f"📍 已记录血压测试开始时间戳: {call_timestamp}")

            logger.debug(f"🔄 准备调用 update_step_ui()，当前 current_step={self.current_step}")
            self.update_step_ui()
            logger.debug(f"✅ update_step_ui() 调用完成，answer_stack.currentIndex={self.answer_stack.currentIndex()}")

            # 保存音视频路径到数据库
            self._persist_av_paths_to_db()
        elif self.current_step == 1:
            # 📍 记录血压测试结束时间戳
            call_timestamp = time.time()
            self._save_timestamp_immediately(call_timestamp)
            logger.info(f"📍 已记录血压测试结束时间戳: {call_timestamp}")

            self.mark_stage_completed('血压脉搏检测')  # ✅ 标记阶段完成(新命名)
            
            # ✅ 立即更新分数页面，传递血压测试结果
            try:
                if hasattr(self, 'score_page') and self.score_page:
                    logger.info(f"📊 血压测试完成，立即更新分数页面 (bp={self.bp_results})")
                    self._send_scores_to_score_page()
                    logger.debug("✅ 血压测试结果已发送到分数页面")
            except Exception as e:
                logger.error(f"发送血压结果到分数页面失败: {e}", exc_info=True)

            # ❌ 舒尔特阶段不再需要疲劳度监控（已在朗读阶段结束时停止）
            # 💡 EEG采集仍在后台运行，只是不进行疲劳度分数推理

            # 📍 记录舒尔特测试开始时间戳
            call_timestamp = time.time()
            self._save_timestamp_immediately(call_timestamp)
            logger.info(f"📍 已记录舒尔特测试开始时间戳: {call_timestamp}")

            # 📍 在切换到舒尔特测试时，保存语音识别结果
            self._save_speech_recognition_results()

            self.current_step += 1
            self.update_step_ui()

    def _on_schulte_completed(self):
        logger.debug("舒特格测试完成，自动进入分数展示页面")
        
        # 📍 记录舒尔特测试结束时间戳
        call_timestamp = time.time()
        self._save_timestamp_immediately(call_timestamp)
        logger.info(f"📍 已记录舒尔特测试结束时间戳: {call_timestamp}")
        
        self.mark_stage_completed('舒尔特专注度检测')  # ✅ 标记阶段完成(新命名)
        
        # ✅ 立即更新分数页面，传递舒尔特测试结果
        try:
            if hasattr(self, 'score_page') and self.score_page:
                logger.info(f"📊 舒尔特测试完成，立即更新分数页面 (score={self.score}, accuracy={self.schulte_accuracy}%)")
                self._send_scores_to_score_page()
                logger.info("✅ 舒尔特测试结果已发送到分数页面")
        except Exception as e:
            logger.error(f"发送舒尔特结果到分数页面失败: {e}", exc_info=True)
        
        # ✅ 舒尔特阶段结束，确保疲劳度监控已停止（防御性代码，实际在朗读阶段已停止）
        try:
            self._stop_multimodal_monitoring()
            multidata_stop_collection()
            logger.debug("舒尔特测试完成，已确认疲劳度监控与多模态采集已停止")
        except Exception as e:
            logger.warning(f"舒尔特阶段停止疲劳度监控或采集失败: {e}")
        
        # 停止EEG采集并保存路径到数据库
        try:
            eeg_stop_collection()
            logger.info("EEG采集已停止")
            # 获取EEG文件路径并保存到数据库
            eeg_paths = eeg_get_file_paths()
            if eeg_paths:
                # logger.info(f"✅ 获取到EEG文件路径: {eeg_paths}")
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
        
        # # 停止EEG采集并保存文件路径
        # try:
        #     eeg_stop_collection()
        #     logger.info("EEG采集已完全停止")
        #     # 获取EEG文件路径并保存到数据库
        #     eeg_paths = eeg_get_file_paths()
        #     if eeg_paths:
        #         logger.info(f"获取到EEG文件路径: {eeg_paths}")
        #         self._persist_eeg_paths_to_db(eeg_paths)
        # except Exception as e:
        #     logger.error(f"停止EEG采集或保存路径时出错: {e}")
        
        # 📍 最后的完成时间戳（已通过实时保存自动写入）
        call_timestamp = time.time()
        self._save_timestamp_immediately(call_timestamp)
        logger.info(f"📍 已记录评估完成时间戳: {call_timestamp}")
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
            
            logger.debug(f"准备保存音视频路径: {len(self._video_paths)} 视频, {len(self._audio_paths)} 音频")
            
            # 使用排队更新机制，如果记录不存在会自动创建
            self._queue_db_update(update_payload, "保存音视频路径失败")
            
            logger.info("✅ 音视频路径已加入数据库更新队列")
            
        except Exception as e:
            logger.exception(f"❌ 保存音视频路径时发生异常: {e}")

    def _persist_multimodal_paths_to_db(self, *, clear_recognition_cache: bool = True):
        """保存多模态数据文件路径到数据库（RGB/Depth/Eyetrack）
        
        Args:
            clear_recognition_cache: 是否在写入后清理语音识别缓存。
                朗读阶段结束时需要保留语音识别结果用于后续情绪分析，因此允许调用方禁用缓存清理。
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

            # 根据需要清理语音识别结果缓存
            if clear_recognition_cache:
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
            
            # logger.info(f"准备保存 EEG 路径: {update_payload}")
            
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
                logger.debug(f"分数已保存到CSV文件: {self.score}")
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
        
        # ✅ 用户登录后立即创建数据库记录，确保后续测试数据能正常保存
        # 这样无论用户从哪个阶段开始测试，都能正常保存数据
        try:
            if not self._db_disabled and not self.row_id:
                logger.info(f"📝 用户 '{self.current_user}' 登录，创建新的数据库记录...")
                self._ensure_db_row()
        except Exception as e:
            logger.error(f"❌ 创建数据库记录失败: {e}", exc_info=True)
        
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
        
        ✅ 注意：疲劳度数据收集范围
        - 开始：基线校准开始时
        - 结束：文本朗读完成时
        - 包含阶段：基线校准 + SART实验 + 文本朗读
        - 不包含：舒尔特方格阶段
        
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
            logger.debug(
                f"疲劳度平均分数: {result['fatigue_avg']:.2f} "
                f"(基于 {result['fatigue_count']} 个样本)"
            )
        else:
            logger.warning("没有收集到疲劳度分数数据")
        
        # 计算脑负荷平均值
        if self._brain_load_scores_list:
            result["brain_load_avg"] = sum(self._brain_load_scores_list) / len(self._brain_load_scores_list)
            result["brain_load_count"] = len(self._brain_load_scores_list)
            logger.debug(
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
            # 疲劳检测 (平均值) - 仅朗读录音阶段测试
            "疲劳检测": avg_scores["fatigue_avg"] if avg_scores["fatigue_avg"] is not None else 0,
            
            # 情绪分数 - 仅朗读录音阶段测试
            "情绪": self._emotion_score if self._emotion_score is not None else 0,
            
            # 脑负荷 (平均值) - 贯穿整个流程
            "脑负荷": avg_scores["brain_load_avg"] if avg_scores["brain_load_avg"] is not None else 0,
            
            # 舒尔特准确率 - 舒尔特测试阶段
            "舒尔特准确率": self.schulte_accuracy if self.schulte_accuracy is not None else 0,
            
            # 血压数据 - 血压测试阶段
            "收缩压": self.bp_results.get("systolic", 0) if hasattr(self, 'bp_results') else 0,
            "舒张压": self.bp_results.get("diastolic", 0) if hasattr(self, 'bp_results') else 0,
            "脉搏": self.bp_results.get("pulse", 0) if hasattr(self, 'bp_results') else 0,
            
            # 舒尔特综合得分
            "舒尔特综合得分": self.score if self.score is not None else 0,
            
            # 🔄 阶段完成状态(用于控制分数页面显示) - 新命名
            "_stage_completed": {
                "多模态疲劳检测": self.stage_completed.get('多模态疲劳检测', False),
                "情绪检测": self.stage_completed.get('情绪检测', False),
                "血压脉搏检测": self.stage_completed.get('血压脉搏检测', False),
                "舒尔特专注度检测": self.stage_completed.get('舒尔特专注度检测', False),
            },
            
            # 元数据
            "_metadata": {
                "fatigue_sample_count": avg_scores["fatigue_count"],
                "brain_load_sample_count": avg_scores["brain_load_count"],
                "has_emotion_score": self._emotion_score is not None,
                "has_schulte_result": self.schulte_accuracy is not None,
                "has_bp_result": hasattr(self, 'bp_results') and self.bp_results.get('systolic') is not None,
            }
        }
        
        logger.debug(f"准备分数数据完成: {score_data}")
        logger.debug(f"阶段完成状态: {score_data['_stage_completed']}")
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
            
            # 定义成功回调，在数据库更新完成后刷新ScorePage
            def _on_saved(result: dict):
                logger.info(f"📊 推理结果已保存到数据库: {update_payload}")
                # 数据库更新完成后，通知ScorePage刷新历史数据
                if hasattr(self, 'score_page') and hasattr(self.score_page, '_refresh_data'):
                    try:
                        self.score_page._refresh_data()
                        logger.debug("✅ 已通知ScorePage刷新历史数据")
                    except Exception as e:
                        logger.warning(f"刷新ScorePage历史数据失败: {e}")
            
            # 更新数据库记录，并在成功后执行回调
            self._queue_db_update_with_callback(
                update_payload,
                "保存推理结果到数据库失败",
                on_success=_on_saved
            )
            
        except Exception as e:
            logger.error(f"保存推理结果到数据库失败: {e}", exc_info=True)


__all__ = ["TestPage"]
