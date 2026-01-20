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
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QShortcut,
    QSpacerItem,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import (
    QEasingCurve,
    QMetaObject,
    QSize,
    QTimer,
    Qt,
    QPropertyAnimation,
    pyqtSignal,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QBitmap,
    QPixmap,
)
import qtawesome as qta

from ..utils.widgets import AudioLevelMeter
from ..utils.responsive import scale, scale_size, scale_font
from ...widgets.camera_preview import CameraPreviewWidget
from ...widgets.schulte_grid import SchulteGridWidget
from...widgets.score_page import ScorePage
from ...widgets.navigation_bar import StageNavigationBar
from .baseline_prompt import BaselinePromptPage
from .sart_prompt import SARTPromptPage
from .emotion import EmotionDetectionPage
from .schulte import SchultePage
from .blood_pressure import BloodPressurePage
from ...services.backend_client import get_backend_client
from ...services.database_service import DatabaseService
from ...managers.session_manager import SessionManager
from ...managers.test_db_manager import TestDBManager
from ...utils_common.ui_thread_pool import get_ui_thread_pool

# ---------------------------------------------------------------------------
# Configuration exports
# ---------------------------------------------------------------------------
logger = config.logger
DEBUG_MODE = config.DEBUG_MODE
SKIP_DATABASE = config.SKIP_DATABASE
HAS_MULTIMODAL = config.HAS_MULTIMODAL
HAS_SPEECH_RECOGNITION = config.HAS_SPEECH_RECOGNITION
HAS_BP_BACKEND = config.HAS_BP_BACKEND
BP_SIMULATION = config.BP_SIMULATION
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
        
        # 朗读状态标志
        self.reading_completed = False
        
        self.thread_pool = get_ui_thread_pool()
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
        self.is_recording = False
        self.score = None  # 将在舒尔特测试完成后计算
        self.history_scores = []
        self.audio_timer = QTimer(self)
        self.camera_preview: Optional[CameraPreviewWidget] = None
        self.schulte_camera_preview: Optional[CameraPreviewWidget] = None
        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None
        self.current_user = 'anonymous'
        
        self.session_manager = SessionManager.get_instance()
        
        # SART模式配置（从命令行参数读取）
        self.sart_mode = "short"  # 默认短时模式
        self.sart_duration = 60  # 默认1分钟

        # 情绪分数（测试结束时分析一次）
        self._emotion_score: Optional[float] = None
        self._emotion_analysis_triggered: bool = False

        # 疲劳度评估结果（离线评估完成后接收一次）
        self._fatigue_assessment_result: Optional[float] = None
        
        # 脑负荷分数列表（实时推理，累积多次结果）
        self._brain_load_scores_list: list[float] = []

        # ✅ 数据库服务（替代原有的数据库交互状态变量）
        self._db_disabled = SKIP_DATABASE
        self.db_service = DatabaseService(parent=self, username=self.current_user)
        if SKIP_DATABASE:
            self.db_service.disable_writes("用户设置了 SKIP_DATABASE，数据库写入已禁用")
        
        # ✅ 数据库管理器（封装所有数据库操作逻辑）
        self.db_manager = TestDBManager(self.db_service, db_disabled=self._db_disabled)

        # 血压后端采集状态
        self.bp_simulation_enabled = BP_SIMULATION
        self.bp_forced_port = config.BP_PORT
        self.bp_available_port = None
        self.bp_measurement_active = False
        # 血压轮询定时器由 bp_page 管理，这里不再需要
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
    
    @property
    def row_id(self) -> Optional[int]:
        """数据库记录 ID（通过 db_service 获取，保持向后兼容）"""
        return self.db_service.get_row_id()

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
            if self._timestamps_file_path is None:
                eeg_dir = self.session_manager.get_eeg_dir()
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
            
            try:
                emotion_dir = self.session_manager.get_emotion_dir()
                os.makedirs(emotion_dir, exist_ok=True)
                record_txt = os.path.join(emotion_dir, "record.txt")
                with open(record_txt, 'w', encoding='utf-8') as f:
                    f.write(str(record_payload))
                logger.info(f"✅ 语音识别结果已写入文件: {record_txt}")
            except Exception as exc:
                logger.warning(f"写入语音识别记录文本失败: {exc}")
            
            # 更新到数据库
            try:
                self.db_service.update_test_record(
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

        # 主内容区
        content_container = self._create_main_content_area()
        self.main_layout.addWidget(content_container, 1)
        # 底部按钮
        self.bottom_button_container = self._create_bottom_buttons()
        self.main_layout.addWidget(self.bottom_button_container, 0, Qt.AlignCenter)
    def _connect_signals(self):
        """连接所有控件的信号到槽函数。"""
        self.audio_timer.timeout.connect(self._process_audio)
        self.btn_finish.clicked.connect(self._finish_test)
        
        # 连接疲劳度评估结果信号（只接收离线评估的最终结果）
        from ...services.backend_client import get_backend_client
        backend_client = get_backend_client()
        backend_client.detection_result.connect(self._on_fatigue_assessment_result)
    
    def _on_fatigue_assessment_result(self, payload: Dict) -> None:
        """
        接收检测结果（包括离线疲劳度评估和实时脑负荷推理）
        
        支持两种类型的结果:
        1. 疲劳度评估 (detector="model_fatigue"): 离线评估,只接收一次
        2. 脑负荷推理 (detector="model_eeg"): 实时推理,持续接收并累积
        
        Args:
            payload: 包含检测结果的字典
                疲劳度: {
                    "detector": "model_fatigue",
                    "predictions": {
                        "fatigue_score": 53.77,
                        "prediction_class": "正常 😊"
                    }
                }
                脑负荷: {
                    "detector": "model_eeg",
                    "predictions": {
                        "brain_load_score": 45.2,
                        "state": "calibrated",
                        "num_windows": 3
                    }
                }
        """
        try:
            detector = payload.get("detector")
            predictions = payload.get("predictions", {})
            
            # 处理疲劳度评估结果（离线,一次性）
            if detector == "model_fatigue":
                fatigue_score = predictions.get("fatigue_score")
                if fatigue_score is not None:
                    self._fatigue_assessment_result = float(fatigue_score)
                    logger.info(f"✅ 接收到疲劳度评估结果: {self._fatigue_assessment_result:.2f}/90")
            
            # 处理脑负荷推理结果（实时,累积）
            elif detector == "model_eeg":
                brain_load_score = predictions.get("brain_load_score")
                if brain_load_score is not None and brain_load_score > 0:
                    score_value = float(brain_load_score)
                    self._brain_load_scores_list.append(score_value)
                    logger.debug(f"接收到脑负荷分数: {score_value:.2f}, 累积数量: {len(self._brain_load_scores_list)}")
                    
                    # 更新UI显示
                    self._update_brain_load_only(score_value)
            
        except Exception as e:
            logger.error(f"处理检测结果失败: {e}", exc_info=True)

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

    # --- 数据库操作包装方法 ---

    def _queue_db_update(self, update_payload: dict, context: str) -> None:
        """数据库更新包装器（向后兼容）"""
        self.db_manager.queue_update(update_payload, context)
    
    def _queue_db_update_with_callback(self, update_payload: dict, context: str, on_success=None) -> None:
        """带回调的数据库更新包装器（向后兼容）"""
        self.db_manager.queue_update_with_callback(update_payload, context, on_success)
    
    def _handle_db_failure(self, error: Exception, context: str) -> None:
        """数据库错误处理（向后兼容）"""
        self.db_manager.handle_failure(error, context)

    # --- UI 创建辅助方法 ---
    def _create_step_navigator(self):
        """创建完整的阶段导航栏(使用独立组件)"""
        # 创建导航栏组件
        navigator = StageNavigationBar(self.all_stages, self)
        navigator.stage_clicked.connect(self._on_stage_nav_clicked)
        
        # 保存引用以便后续更新状态
        self.stage_navigator = navigator
        self.stage_buttons = navigator.stage_buttons  # 保持兼容性
        
        return navigator

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
        
        # ✅ 重置血压测试状态（使用组件的reset_test方法）
        try:
            if hasattr(self, 'bp_page'):
                self.bp_page.reset_test()
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
        """更新导航栏的视觉状态（使用组件方法）"""
        # 根据answer_stack的当前索引判断当前阶段
        current_index = self.answer_stack.currentIndex()

        # answer_stack索引映射到全局阶段
        index_to_stage = {
            0: '多模态疲劳检测',    # 基线提示页面
            1: '多模态疲劳检测',    # SART提示页面
            2: '情绪检测',          # 朗读录音
            3: '血压脉搏检测',      # 血压测试
            4: '舒尔特专注度检测',  # 舒尔特测试
            5: '分数展示'           # 分数页面
        }

        current_stage = index_to_stage.get(current_index, None)
        
        # 使用导航栏组件的更新方法
        if hasattr(self, 'stage_navigator'):
            self.stage_navigator.update_stage_status(current_stage)

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
        
        # 同时更新导航栏组件的完成状态
        if hasattr(self, 'stage_navigator'):
            self.stage_navigator.mark_stage_completed(mapped_stage)
        
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

    def _create_answer_area_widgets(self):
        """创建并添加所有答题区域页面到 answer_stack"""
        # 🆕 基线校准提示页面（使用独立模块）
        page_baseline_prompt = BaselinePromptPage(self)
        page_baseline_prompt.start_clicked.connect(self._on_start_baseline_clicked)
        self.answer_stack.addWidget(page_baseline_prompt)
        # 保存按钮引用（用于后门快捷键）
        self.btn_start_baseline = page_baseline_prompt.btn_start
        
        # 🆕 SART实验提示页面（使用独立模块）
        page_sart_prompt = SARTPromptPage(self)
        page_sart_prompt.start_clicked.connect(self._on_start_sart_clicked)
        self.answer_stack.addWidget(page_sart_prompt)
        self.btn_start_sart = page_sart_prompt.btn_start
        
        page_emotion = EmotionDetectionPage(self.reading_text_content, self)
        page_emotion.recording_requested.connect(self._toggle_recording)
        page_emotion.next_clicked.connect(self._next_step_or_question)
        self.answer_stack.addWidget(page_emotion)
        self.lbl_reading_text = page_emotion.lbl_reading_text
        self.btn_mic = page_emotion.btn_mic
        self.lbl_recording_status = page_emotion.lbl_recording_status
        self.audio_level = page_emotion.audio_level
        self.btn_next = page_emotion.btn_next

        page_blood_pressure = self._create_blood_pressure_page()
        self.answer_stack.addWidget(page_blood_pressure)

        # 舒特格测试页面
        self.page_schulte = self._create_schulte_page()
        self.answer_stack.addWidget(self.page_schulte)

        # 分数展示页面（使用ScorePage组件）
        self.score_page = ScorePage(username=self.current_user)
        self.answer_stack.addWidget(self.score_page)

    def _create_blood_pressure_page(self):
        """创建血压脉搏测试页面（使用BloodPressurePage组件）"""
        page = BloodPressurePage(
            simulation_enabled=self.bp_simulation_enabled,
            forced_port=self.bp_forced_port,
            parent=self
        )
        
        # 连接信号
        page.test_completed.connect(lambda: logger.debug("血压测试完成信号收到"))
        page.test_result_ready.connect(self._on_bp_test_result)
        page.next_clicked.connect(self._next_step_or_question)
        
        # 保存引用以便外部访问
        self.bp_page = page
        self.bp_results = page.results  # 保持兼容性
        self.bp_test_running = False  # 初始化标志
        
        # 暴露常用控件引用（保持向后兼容）
        self.bp_status_label = page.status_label
        self.bp_start_button = page.start_button
        self.bp_next_button = page.next_button
        self.bp_progress_circle = page.progress_circle
        self.bp_progress_label = page.progress_label
        self.bp_status_container = page.status_container
        self.bp_control_container = page.control_container
        self.result_container = page.result_container
        self.systolic_label = page.systolic_label
        self.diastolic_label = page.diastolic_label
        self.pulse_label = page.pulse_label
        
        # 保存定时器引用
        self.bp_timer = page.device_check_timer
        self.bp_test_timer = page.test_timer
        self.bp_poll_timer = page.poll_timer
        
        return page
    
    def _on_bp_test_result(self, results: dict):
        """血压测试结果回调"""
        self.bp_results = results
        logger.debug(f"收到血压测试结果: {results}")
        
        # 保存到数据库
        try:
            systolic = results.get('systolic')
            diastolic = results.get('diastolic')
            pulse = results.get('pulse')
            
            if systolic is not None and diastolic is not None and pulse is not None:
                blood_data = f"{systolic}/{diastolic}/{pulse}"
                self._queue_db_update({"blood": blood_data}, "保存血压测试结果到数据库失败")
                logger.info(f"血压测试结果将写入数据库: {blood_data}")
            else:
                logger.warning("血压测试结果不完整，无法保存到数据库")
        except Exception as e:
            logger.error(f"保存血压测试结果失败: {e}")
        
        # 更新 test_running 标志
        self.bp_test_running = self.bp_page.test_running

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
                
                # ✅ 如果血压测试正在运行，先停止它（使用组件方法）
                if hasattr(self, 'bp_page') and self.bp_page.test_running:
                    self.bp_page.stop_test()
                
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
                
                # ✅ 让血压组件自动跳过测试
                if hasattr(self, 'bp_page'):
                    self.bp_page.auto_skip_test("测试后门：按下 Q 键跳过")
                
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
        self.schulte_camera_widget = self._create_camera_view_for_schulte()
        page = SchultePage(self.current_user, self.schulte_camera_widget, self)
        page.test_completed.connect(self._on_schulte_completed)
        page.test_result_ready.connect(self._on_schulte_result)
        self.schulte_widget = page.schulte_widget
        return page

    def _reinit_schulte_widget(self):
        """重新初始化舒尔特widget（每次进入页面时调用，避免状态卡住）"""
        if hasattr(self, 'page_schulte') and hasattr(self.page_schulte, 'reinit_widget'):
            self.page_schulte.reinit_widget()
            self.schulte_widget = self.page_schulte.schulte_widget
            logger.info("✅ 舒尔特widget已通过SchultePage重新初始化")
        else:
            logger.warning("SchultePage不存在或没有reinit_widget方法")

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

        self._update_camera_previews_for_step()
        
        # 🔄 最后更新阶段导航状态（确保answer_stack已切换完成）
        self._update_stage_nav_status()

    def start_test(self):
        # 摄像头预览在 AV 采集准备好后启动
        self.audio_timer.start(50)
        self.current_step = 0
        self.btn_finish.setVisible(False)

        # 🔄 重置朗读录音状态
        self.reading_completed = False

        self.mark_stage_completed('多模态疲劳检测')
        
        # 重置疲劳度评估、脑负荷和情绪分析状态
        self._fatigue_assessment_result = None
        self._brain_load_scores_list = []
        self._emotion_score = None
        self._emotion_analysis_triggered = False
        logger.debug("已重置疲劳度评估、脑负荷和情绪分析标志")

        # 📍 记录朗读录音开始时间戳
        self._text_qa_start_timestamp_recorded = False

        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None

        self.test_started = True
        self.update_step_ui()


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
                    session_dir = self.session_manager.get_session_dir()
                    logger.info(f"🎥 准备启动 AV 采集，session_dir={session_dir}")
                    av_start_collection(
                        save_dir=session_dir,
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
        
        self.thread_pool.submit_task(start_av_async)
        

    def start_eeg_collection(self) -> None:
        self._brain_load_scores_list = []
        # EEG采集也使用异步方式（非阻塞），由后端统一管理硬件连接
        # ⚠️ 注意：如果EEG已经在基线/SART阶段启动，这里会返回 "already-running"，这是正常的
        def start_eeg_async():
            try:
                from ...services.backend_proxy import eeg_start
                session_dir = self.session_manager.get_session_dir()
                result = eeg_start(username=self.current_user, save_dir=session_dir, part=1)
                if result.get('status') == 'already-running':
                    logger.debug(f"✅ EEG采集已在运行中，继续使用现有连接: {result.get('save_dir')}")
                else:
                    eeg_dir = self.session_manager.get_eeg_dir()
                    logger.info(f"✅ EEG采集已启动，保存目录: {eeg_dir}")
            except Exception as e:
                logger.error(f"启动EEG采集失败: {e}")
                logger.info("UI将继续运行，但EEG功能不可用")
        
        self.thread_pool.submit_task(start_eeg_async)

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
                            add_audio_for_recognition(
                                latest_audio,
                                1,  # 只有一段朗读文本
                                self.reading_text_content,
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
        
        # ℹ️ EEG采集已在舒尔特测试完成时停止，无需重复停止
        
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
        """保存音视频路径到数据库"""
        self.db_manager.persist_av_paths(self._video_paths, self._audio_paths)

    def _persist_multimodal_paths_to_db(self, *, clear_recognition_cache: bool = True):
        """保存多模态数据文件路径到数据库（RGB/Depth/Eyetrack）"""
        self.db_manager.persist_multimodal_paths(clear_recognition_cache)

    def _persist_eeg_paths_to_db(self, eeg_paths: dict):
        """保存EEG数据文件路径到数据库"""
        self.db_manager.persist_eeg_paths(eeg_paths, wait_for_row=True)

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
        
        # ✅ 更新数据库服务的用户名并创建记录
        try:
            if not self.db_service.is_disabled() and not self.db_service.get_row_id():
                logger.info(f"📝 用户 '{self.current_user}' 登录，创建新的数据库记录...")
                self.db_service.create_test_record(self.current_user)
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

            ptime = os.path.join(self.session_manager.get_eeg_dir(), 'part_timestamps.txt')

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
        
        ✅ 疲劳度数据来源（离线评估）：
        - 使用后端离线评估的融合结果（RGB + EEG）
        - 评估时机：朗读阶段结束后触发一次性评估
        - 评估范围：基线校准 + SART实验 + 文本朗读
        - 不包含：舒尔特方格阶段
        
        ✅ 脑负荷数据来源（实时推理）：
        - 使用EEG实时推理结果
        - 推理时机：EEG采集期间持续推理
        - 累积方式：接收多次推理结果并计算平均值
        - 覆盖范围：整个测试流程（基线校准 + SART + 朗读 + 舒尔特）
        
        Returns:
            包含平均分数的字典:
            {
                "fatigue_avg": 疲劳度分数 (0-90, 离线评估一次),
                "brain_load_avg": 脑负荷分数 (0-100, 实时推理平均值),
                "fatigue_count": 1 (离线评估只返回一次),
                "brain_load_count": N (实时推理累积次数)
            }
        """
        result = {
            "fatigue_avg": None,
            "brain_load_avg": None,
            "fatigue_count": 0,
            "brain_load_count": 0
        }
        
        # 使用离线疲劳度评估结果
        if self._fatigue_assessment_result is not None:
            result["fatigue_avg"] = self._fatigue_assessment_result
            result["fatigue_count"] = 1
            logger.debug(
                f"疲劳度评估结果: {result['fatigue_avg']:.2f}/90 (离线评估)"
            )
        else:
            logger.warning("未接收到疲劳度评估结果")
        
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
        """将疲劳检测、脑负荷、情绪推理结果保存到数据库"""
        # 定义成功回调，在数据库更新完成后刷新ScorePage
        def _on_saved(result: dict):
            logger.info(f"📊 推理结果已保存到数据库")
            # 数据库更新完成后，通知ScorePage刷新历史数据
            if hasattr(self, 'score_page') and hasattr(self.score_page, '_refresh_data'):
                try:
                    self.score_page._refresh_data()
                    logger.debug("✅ 已通知ScorePage刷新历史数据")
                except Exception as e:
                    logger.warning(f"刷新ScorePage历史数据失败: {e}")
        
        # 使用数据库管理器保存
        self.db_manager.persist_inference_scores(score_data, on_success=_on_saved)


__all__ = ["TestPage"]
