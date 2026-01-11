"""Baseline calibration page - 30s eyes-open resting state baseline."""

from __future__ import annotations

import time

from .. import config
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFrame
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFont, QKeyEvent
from ..utils.responsive import scale, scale_font
from ...utils_common.thread_process_manager import get_thread_manager
from ...services.session_manager import SessionManager

HAS_MULTIMODAL = config.HAS_MULTIMODAL
multidata_start_collection = config.multidata_start_collection
multidata_stop_collection = config.multidata_stop_collection


class BaselineCalibrationPage(QWidget):
    """基线校准页面 - 30秒静息基线采集（眼开注视十字）"""
    
    baseline_finished = pyqtSignal()  # 基线完成信号
    
    BASELINE_DURATION = 30  # 基线时长（秒）
    
    def __init__(self) -> None:
        super().__init__()
        # 设置对象名称，用于QSS选择器
        self.setObjectName("baselinePage")
        self._init_ui()
        self._setup_timer()
        
        self.remaining_time = self.BASELINE_DURATION
        self.is_running = False
        self.waiting_for_continue = False
        self.part_timestamps = []
        self.current_user = None
        
        self.session_manager = SessionManager.get_instance()
        self.thread_manager = get_thread_manager()
    
    def _init_ui(self) -> None:
        """初始化UI - 浅色渐变背景，黑色文字，简洁样式"""
        # 设置页面属性标识
        self.setProperty("page_type", "baseline")
        
        # 设置明显的渐变背景（从左上白色到右下深青色）+ 文字颜色
        self.setStyleSheet("""
            BaselineCalibrationPage {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 1),
                    stop:0.5 rgba(200, 235, 245, 0.6),
                    stop:1 rgba(127, 219, 255, 0.5)
                );
            }
            #baselinePage QLabel {
                color: #000000 !important;
                background-color: transparent !important;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(scale(50), scale(50), scale(50), scale(50))
        layout.setSpacing(scale(40))
        layout.setAlignment(Qt.AlignCenter)
        
        # 添加顶部弹性空间，让十字居中
        layout.addStretch(1)
        
        # 说明文字（移除，不再显示）
        self.instruction_label = QLabel("")
        self.instruction_label.setVisible(False)  # 隐藏提示文字
        layout.addWidget(self.instruction_label)
        
        # 十字注视点(直接显示)- 占据主要空间
        self.fixation_cross = QLabel("+")
        self.fixation_cross.setAlignment(Qt.AlignCenter)
        cross_font = QFont("阿里健康体2.0 中文 45 R", 250, 75)  # 固定250px超大字号
        self.fixation_cross.setFont(cross_font)
        self.fixation_cross.setStyleSheet(
            "color: #000000 !important; background-color: transparent !important; font-size: 250px !important; font-weight: bold !important;"
        )
        self.fixation_cross.setVisible(True)  # 直接显示
        layout.addWidget(self.fixation_cross, stretch=1)  # 占据主要空间
        
        # 完成提示标签(初始隐藏)- 居中显示
        self.completion_label = QLabel("")
        self.completion_label.setAlignment(Qt.AlignCenter)
        completion_font = QFont("阿里健康体2.0 中文 45 R", 36, 75)  # 固定36px
        self.completion_label.setFont(completion_font)
        self.completion_label.setStyleSheet(
            "color: #000000 !important; background-color: transparent !important; font-size: 36px !important; font-weight: bold !important;"
        )
        self.completion_label.setVisible(False)
        layout.addWidget(self.completion_label, stretch=1)  # 占据主要空间，与十字准心位置一致
        
        # 添加底部弹性空间，将倒计时推到底部
        layout.addStretch(1)
        
        # 倒计时显示（初始隐藏）
        self.countdown_label = QLabel(f"剩余 {self.BASELINE_DURATION} 秒")
        self.countdown_label.setAlignment(Qt.AlignCenter)
        countdown_font = QFont("阿里健康体2.0 中文 45 R", 20)  # 固定20px字号
        self.countdown_label.setFont(countdown_font)
        self.countdown_label.setStyleSheet(
            "color: #666666 !important; background-color: transparent !important; padding-bottom: 10px; font-size: 20px !important;"
        )
        self.countdown_label.setVisible(False)  # 初始隐藏
        layout.addWidget(self.countdown_label)  # 添加到布局
    
    def _setup_timer(self) -> None:
        """设置倒计时定时器"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_countdown)
        self.timer.setInterval(1000)  # 每秒更新
    
    def _start_baseline(self) -> None:
        """开始基线采集"""
        config.logger.info("📍 开始30秒基线校准")
        
        # 确保十字显示，隐藏倒计时
        self.instruction_label.setVisible(False)
        self.fixation_cross.setVisible(True)
        self.countdown_label.setVisible(False)  # 隐藏倒计时
        
        # 重置倒计时
        self.remaining_time = self.BASELINE_DURATION
        self.is_running = True
        
        # 记录基线开始时间戳（触发10）
        call_timestamp = time.time()
        if hasattr(self, '_save_timestamp_callback') and self._save_timestamp_callback:
            self._save_timestamp_callback(call_timestamp)
        else:
            self.part_timestamps.append(call_timestamp)
        config.logger.info(f"📍 已记录基线开始时间戳: {call_timestamp}")
        
        if HAS_MULTIMODAL:
            try:
                config.logger.debug("🚀 启动疲劳度检测（基线校准开始）")
                result = multidata_start_collection(
                    self.current_user,
                    part=0,
                    save_dir=self.session_manager.get_session_dir(),
                )
                status = (result or {}).get("status", "").lower()
                if status in {"running", "already-running"}:
                    config.logger.info(f"✅ 疲劳度检测已启动，用户: {self.current_user}")
                else:
                    config.logger.warning(f"⚠️ 疲劳度检测启动失败: {result}")
            except Exception as e:
                config.logger.error(f"❌ 启动疲劳度检测失败: {e}", exc_info=True)
        
        # ⚠️ 注意：EEG采集应该在进入基线页面前就已经启动
        # 这里不再启动EEG，只记录时间戳用于后续分段分析
        
        # 启动定时器
        self.timer.start()
    
    def _update_countdown(self) -> None:
        """更新倒计时"""
        self.remaining_time -= 1
        self.countdown_label.setText(f"剩余 {self.remaining_time} 秒")
        
        if self.remaining_time <= 0:
            self._finish_baseline()
    
    def _finish_baseline(self) -> None:
        """完成基线采集"""
        self.timer.stop()
        self.is_running = False
        
        # 记录基线结束时间戳（触发11）
        call_timestamp = time.time()
        if hasattr(self, '_save_timestamp_callback') and self._save_timestamp_callback:
            self._save_timestamp_callback(call_timestamp)
        else:
            self.part_timestamps.append(call_timestamp)
        config.logger.info(f"📍 已记录基线结束时间戳: {call_timestamp}")
        
        # ⚠️ 注意：EEG采集继续运行，不在这里停止
        # 会在整个测试流程结束时统一停止
        
        config.logger.info("✅ 基线校准完成")
        
        # 显示完成信息,等待用户按键
        self.fixation_cross.setVisible(False)
        self.countdown_label.setVisible(False)
        self.completion_label.setText("基线校准完成\n\n按任意键继续")
        self.completion_label.setVisible(True)
        
        # 设置标志,表示等待按键确认
        self.waiting_for_continue = True
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """键盘事件处理"""
        # 如果正在等待按键继续,任意键都触发继续
        if self.waiting_for_continue and not event.isAutoRepeat():
            config.logger.info("用户按键确认,继续下一阶段")
            self.waiting_for_continue = False
            self.baseline_finished.emit()
            return
        
        # Q键跳过或中断
        if event.key() == Qt.Key_Q and not event.isAutoRepeat():
            if not self.is_running:
                config.logger.info("⏭️ 用户跳过基线校准")
                # 记录开始和结束时间戳（快速标记）
                call_timestamp = time.time()
                if hasattr(self, '_save_timestamp_callback') and self._save_timestamp_callback:
                    self._save_timestamp_callback(call_timestamp)  # 开始
                    self._save_timestamp_callback(call_timestamp)  # 结束
                else:
                    self.part_timestamps.append(call_timestamp)  # 开始
                    self.part_timestamps.append(call_timestamp)  # 结束
                
                # 显示跳过提示,等待按键继续
                self.instruction_label.setVisible(False)
                self.fixation_cross.setVisible(False)
                self.countdown_label.setVisible(False)
                self.completion_label.setText("基线校准已跳过\n\n按任意键继续")
                self.completion_label.setVisible(True)
                self.waiting_for_continue = True
                
            elif self.is_running:
                # 正在运行时按Q也可以跳过
                config.logger.info("⏭️ 用户中断基线校准")
                self._finish_baseline()
        
        super().keyPressEvent(event)
    
    def set_part_timestamps(self, timestamps: list, save_callback=None) -> None:
        """设置时间戳列表（与TestPage共享）
        
        Args:
            timestamps: 时间戳列表引用
            save_callback: 保存时间戳的回调函数（可选）
        """
        self.part_timestamps = timestamps
        self._save_timestamp_callback = save_callback
    
    def showEvent(self, event) -> None:
        """页面显示时自动开始基线校准"""
        super().showEvent(event)
        # 只在第一次显示或重置后开始
        if not self.is_running and not self.waiting_for_continue:
            # 延迟500ms后自动开始，让用户有时间准备
            QTimer.singleShot(500, self._auto_start_baseline)
    
    def _auto_start_baseline(self) -> None:
        """自动开始基线校准（页面显示后触发）"""
        if not self.is_running and not self.waiting_for_continue:
            config.logger.debug("🚀 页面显示，自动开始基线校准")
            self._start_baseline()
    
    def set_session_info(self, session_dir: str, current_user: str) -> None:
        """设置会话信息
        
        Args:
            session_dir: 会话目录路径(兼容性参数,实际不使用)
            current_user: 当前用户名
        """
        self.current_user = current_user
        config.logger.info(f"✅ 基线页面已设置用户: {current_user}")
        config.logger.info(f"📂 基线数据将保存到: {self.session_manager.get_baseline_dir()}")
    
    def reset(self) -> None:
        """重置页面状态（供下次使用）"""
        self.timer.stop()
        self.is_running = False
        self.waiting_for_continue = False  # 重置等待标志
        self.remaining_time = self.BASELINE_DURATION
        
        self.instruction_label.setVisible(False)  # 不显示提示
        self.fixation_cross.setVisible(True)  # 直接显示十字
        self.completion_label.setVisible(False)
        self.completion_label.setText("")
        self.countdown_label.setVisible(False)  # 倒计时初始隐藏，开始后才显示
        self.countdown_label.setText("")
        
        # 重置倒计时字体
        countdown_font = QFont("阿里健康体2.0 中文 45 R", 20)  # 固定20px
        self.countdown_label.setFont(countdown_font)
        self.countdown_label.setStyleSheet(
            "color: #666666 !important; background-color: transparent !important; padding-bottom: 10px; font-size: 20px !important;"
        )


__all__ = ["BaselineCalibrationPage"]
