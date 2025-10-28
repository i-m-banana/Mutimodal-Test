"""SART (Sustained Attention to Response Task) experiment page - Qt version."""

from __future__ import annotations

import random
import time
from typing import List

from .. import config
from ..qt import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QTimer,
    Qt, QFont, QFrame, pyqtSignal, QKeyEvent
)
from ..utils.responsive import scale, scale_font
from ...utils_common.thread_process_manager import get_thread_manager


class SARTPage(QWidget):
    """SART实验页面 - 持续注意力反应测试"""
    
    sart_finished = pyqtSignal()  # SART完成信号
    
    # 实验参数
    DIGITS = list("123456789")
    NOGO_DIGIT = "3"  # NoGo数字
    GO_PROB = 0.88  # Go概率
    STIM_MS = 250  # 数字呈现时长(ms)
    ISI_MS = 900  # 空屏间隔(ms)
    TOTAL_DURATION = 300  # 总时长(秒) - 默认5分钟
    
    def __init__(self, mode: str = "short", duration: int = None) -> None:
        """
        Args:
            mode: 实验模式 ("short" 或 "long")
            duration: 自定义时长(秒)，None则使用默认值
        """
        super().__init__()
        
        # 设置对象名称，用于QSS选择器
        self.setObjectName("sartPage")
        
        self.mode = mode
        self.total_duration = duration if duration else (300 if mode == "short" else 1500)
        self.waiting_for_continue = False  # 新增:等待按键继续的标志
        self.part_timestamps = []  # 与TestPage共享的时间戳列表
        self.session_dir = None  # 会话目录
        self.current_user = None  # 当前用户
        
        # 获取线程管理器（用于异步调用后端）
        self.thread_manager = get_thread_manager()
        
        self._init_ui()
        self._setup_timers()
        self._reset_state()
    
    def _init_ui(self) -> None:
        """初始化UI - 白色背景，黑色文字，简洁样式"""
        # 取消黑色背景设置，使用默认白色背景
        # 只设置页面属性标识
        self.setProperty("page_type", "sart")
        
        # 不再设置背景色，让它保持默认白色
        # 只确保文字是黑色
        self.setStyleSheet("""
            #sartPage QLabel {
                color: #000000 !important;
                background-color: transparent !important;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(scale(50), scale(50), scale(50), scale(50))
        layout.setSpacing(scale(20))
        
        # 添加顶部弹性空间，让说明文字居中
        layout.addStretch(1)
        
        # 说明文字（初始显示）- 居中
        mode_text = "低负荷采集" if self.mode == "short" else "疲劳诱发"
        duration_text = f"{self.total_duration // 60}分钟" if self.total_duration >= 60 else f"{self.total_duration}秒"
        
        self.instruction_label = QLabel(
            f"SART 任务 ({mode_text}, 时长: {duration_text})\n\n"
            "规则：看到除'3'外的数字按【空格】，看到'3'不要按。\n\n"
            "请保持安静、注视中央，不说话。\n\n"
            "按【空格】开始测试"
        )
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        instruction_font = QFont("Arial", 24)  # 固定24px字号
        self.instruction_label.setFont(instruction_font)
        self.instruction_label.setStyleSheet(
            "color: #000000 !important; line-height: 2.0; background-color: transparent !important; font-size: 24px !important;"
        )
        layout.addWidget(self.instruction_label)
        
        # 数字刺激显示 - 中央
        self.digit_label = QLabel("")
        self.digit_label.setAlignment(Qt.AlignCenter)
        digit_font = QFont("Arial", 250, QFont.Bold)  # 固定250px超大字号
        self.digit_label.setFont(digit_font)
        self.digit_label.setStyleSheet(
            "color: #000000 !important; background-color: transparent !important; font-size: 250px !important; font-weight: bold !important;"
        )
        self.digit_label.setVisible(False)
        layout.addWidget(self.digit_label, stretch=1)  # 占据主要空间
        
        # 完成提示标签（初始隐藏）- 居中显示
        self.completion_label = QLabel("")
        self.completion_label.setAlignment(Qt.AlignCenter)
        completion_font = QFont("Arial", 36, QFont.Bold)  # 固定36px
        self.completion_label.setFont(completion_font)
        self.completion_label.setStyleSheet(
            "color: #000000 !important; background-color: transparent !important; font-size: 36px !important; font-weight: bold !important;"
        )
        self.completion_label.setVisible(False)
        layout.addWidget(self.completion_label, stretch=1)  # 占据主要空间，与数字位置一致
        
        # 添加底部弹性空间
        layout.addStretch(1)
        
        # 进度信息 - 底部
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        progress_font = QFont("Arial", 20)  # 固定20px字号
        self.progress_label.setFont(progress_font)
        self.progress_label.setStyleSheet(
            "color: #666666 !important; background-color: transparent !important; padding-bottom: 10px; font-size: 20px !important;"
        )
        self.progress_label.setVisible(False)
        layout.addWidget(self.progress_label)
        
        # 统计信息（调试用，初始隐藏）- 最底部
        self.stats_label = QLabel("")
        self.stats_label.setAlignment(Qt.AlignCenter)
        stats_font = QFont("Arial", 16)  # 固定16px字号
        self.stats_label.setFont(stats_font)
        self.stats_label.setStyleSheet(
            "color: #444444 !important; background-color: transparent !important; padding-bottom: 5px; font-size: 16px !important;"
        )
        self.stats_label.setVisible(config.DEBUG_MODE)
        layout.addWidget(self.stats_label)
    
    def _setup_timers(self) -> None:
        """设置定时器"""
        # 总时间定时器
        self.total_timer = QTimer(self)
        self.total_timer.timeout.connect(self._finish_sart)
        
        # 试次定时器（控制数字呈现和空屏）
        self.trial_timer = QTimer(self)
        self.trial_timer.timeout.connect(self._run_trial)
        self.trial_timer.setInterval(self.STIM_MS + self.ISI_MS)
    
    def _reset_state(self) -> None:
        """重置实验状态"""
        self.is_running = False
        self.start_time = 0
        self.trial_count = 0
        self.correct_count = 0
        self.error_count = 0
        
        # 记录每次试次的结果
        self.trial_records: List[dict] = []
        
        # 当前试次状态
        self.current_digit = ""
        self.current_is_nogo = False
        self.trial_start_time = 0
        self.key_pressed = False
        self.response_time = None
    
    def start_test(self) -> None:
        """开始SART测试"""
        config.logger.info(f"📍 开始SART测试 (模式={self.mode}, 时长={self.total_duration}秒)")
        
        # 隐藏说明
        self.instruction_label.setVisible(False)
        self.progress_label.setVisible(True)
        if config.DEBUG_MODE:
            self.stats_label.setVisible(True)
        
        # 重置状态
        self._reset_state()
        self.is_running = True
        self.start_time = time.time()
        
        # 记录SART开始时间戳（触发1 for short, 20 for long）
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)
        config.logger.info(f"📍 已记录SART开始时间戳: {call_timestamp} (模式={self.mode})")
        
        # ⚠️ 注意：EEG采集应该已经在运行中
        # 这里不再启动EEG，只记录时间戳用于后续分段分析
        
        # 启动定时器
        self.total_timer.start(self.total_duration * 1000)
        self.trial_timer.start()
    
    def _run_trial(self) -> None:
        """运行一次试次"""
        if not self.is_running:
            return
        
        # 随机生成数字（Go或NoGo）
        if random.random() < self.GO_PROB:
            self.current_digit = random.choice([d for d in self.DIGITS if d != self.NOGO_DIGIT])
            self.current_is_nogo = False
        else:
            self.current_digit = self.NOGO_DIGIT
            self.current_is_nogo = True
        
        # 重置试次状态
        self.trial_count += 1
        self.key_pressed = False
        self.response_time = None
        self.trial_start_time = time.time()
        
        # 显示数字
        self.digit_label.setText(self.current_digit)
        self.digit_label.setVisible(True)
        
        # 数字呈现250ms后空屏
        QTimer.singleShot(self.STIM_MS, self._show_blank)
    
    def _show_blank(self) -> None:
        """显示空屏（ISI）"""
        self.digit_label.setVisible(False)
        
        # 在空屏期结束时记录结果
        QTimer.singleShot(self.ISI_MS, self._record_trial)
        
        # 更新进度和统计
        self._update_progress()
    
    def _record_trial(self) -> None:
        """记录当前试次结果"""
        # 判断正确性
        if self.current_is_nogo:
            # NoGo试次：不按为正确
            correct = not self.key_pressed
        else:
            # Go试次：按空格为正确
            correct = self.key_pressed
        
        if correct:
            self.correct_count += 1
        else:
            self.error_count += 1
        
        # 保存记录
        record = {
            "trial": self.trial_count,
            "digit": self.current_digit,
            "is_nogo": self.current_is_nogo,
            "key_pressed": self.key_pressed,
            "response_time": self.response_time,
            "correct": correct,
            "timestamp": time.time(),
        }
        self.trial_records.append(record)
    
    def _update_progress(self) -> None:
        """更新进度显示"""
        elapsed = time.time() - self.start_time
        remaining = max(0, self.total_duration - elapsed)
        
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        
        self.progress_label.setText(f"剩余时间: {minutes:02d}:{seconds:02d}")
        
        if config.DEBUG_MODE and self.trial_count > 0:
            accuracy = (self.correct_count / self.trial_count) * 100
            self.stats_label.setText(
                f"试次: {self.trial_count} | "
                f"正确: {self.correct_count} | "
                f"错误: {self.error_count} | "
                f"准确率: {accuracy:.1f}%"
            )
    
    def _finish_sart(self) -> None:
        """完成SART测试"""
        self.trial_timer.stop()
        self.total_timer.stop()
        self.is_running = False
        
        # 记录SART结束时间戳（触发4 for short, 21 for long）
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)
        config.logger.info(f"📍 已记录SART结束时间戳: {call_timestamp} (模式={self.mode})")
        
        # ⚠️ 注意：EEG采集继续运行，不在这里停止
        # 会在整个测试流程结束时统一停止
        
        # 计算最终统计
        accuracy = (self.correct_count / self.trial_count * 100) if self.trial_count > 0 else 0
        config.logger.info(
            f"✅ SART测试完成: 试次={self.trial_count}, "
            f"正确={self.correct_count}, 错误={self.error_count}, "
            f"准确率={accuracy:.1f}%"
        )
        
        # 保存结果到文件
        self._save_results()
        
        # 显示完成信息,等待用户按键
        self.digit_label.setVisible(False)
        self.progress_label.setVisible(False)
        self.completion_label.setText(f"任务结束\n\n准确率: {accuracy:.1f}%\n\n按任意键继续")
        self.completion_label.setVisible(True)
        
        # 设置标志,表示等待按键确认
        self.waiting_for_continue = True
    
    def _ensure_sart_directory(self) -> str:
        """确保 sart 目录存在，返回目录路径"""
        import os
        
        # 获取会话目录
        session_dir = getattr(self, 'session_dir', None)
        config.logger.info(f"📂 SART：当前session_dir = {session_dir}")
        
        if not session_dir:
            # 如果没有设置会话目录，使用默认路径
            config.logger.warning("⚠️ SART 会话目录未设置，使用默认路径")
            session_dir = 'recordings/default'
        
        # 在会话目录下创建 sart 子目录
        sart_dir = os.path.join(session_dir, 'sart')
        sart_dir_abs = os.path.abspath(sart_dir)
        
        config.logger.info(f"📂 SART：创建目录 {sart_dir}")
        config.logger.info(f"📂 SART：绝对路径 {sart_dir_abs}")
        
        try:
            os.makedirs(sart_dir_abs, exist_ok=True)
            config.logger.info(f"✅ SART目录创建成功: {sart_dir_abs}")
            
            # 验证目录确实存在
            if os.path.exists(sart_dir_abs) and os.path.isdir(sart_dir_abs):
                config.logger.info(f"✅ SART目录存在验证通过")
            else:
                config.logger.error(f"❌ SART目录创建后不存在！路径: {sart_dir_abs}")
                
        except Exception as e:
            config.logger.error(f"❌ 创建SART目录失败: {e}", exc_info=True)
        
        return sart_dir_abs  # 🐛 修复：返回绝对路径而不是相对路径
    
    def _save_results(self) -> None:
        """保存SART结果到CSV文件"""
        try:
            import os
            import csv
            from datetime import datetime
            
            # 确保 sart 目录存在
            sart_dir = self._ensure_sart_directory()
            
            # 🐛 添加详细调试日志
            config.logger.info(f"🔍 SART保存调试：")
            config.logger.info(f"  - sart_dir = {sart_dir}")
            config.logger.info(f"  - 绝对路径 = {os.path.abspath(sart_dir)}")
            config.logger.info(f"  - 目录存在？ {os.path.exists(sart_dir)}")
            config.logger.info(f"  - 当前工作目录 = {os.getcwd()}")
            config.logger.info(f"  - trial_records数量 = {len(self.trial_records)}")
            
            # 保存试次记录
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(sart_dir, f'sart_events_{timestamp}.csv')
            csv_path_abs = os.path.abspath(csv_path)
            
            # 确保有记录才保存
            if not self.trial_records:
                config.logger.warning("⚠️ SART 没有试次记录，跳过保存")
                return
            
            config.logger.info(f"📂 SART保存：准备写入 {len(self.trial_records)} 条记录到 {csv_path}")
            config.logger.info(f"📂 SART绝对路径：{csv_path_abs}")
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'trial', 'timestamp', 'digit', 'is_nogo', 
                    'key_pressed', 'response_time', 'correct'
                ])
                writer.writeheader()
                writer.writerows(self.trial_records)
            
            # 验证文件是否真的存在
            if os.path.exists(csv_path_abs):
                file_size = os.path.getsize(csv_path_abs)
                config.logger.info(f"✅ SART结果已保存: {csv_path_abs} (大小: {file_size} 字节)")
            else:
                config.logger.error(f"❌ 文件写入后不存在！路径: {csv_path_abs}")
            
        except Exception as e:
            config.logger.error(f"❌ 保存SART结果失败: {e}", exc_info=True)
    
    def set_part_timestamps(self, timestamps: list) -> None:
        """设置时间戳列表（与TestPage共享）
        
        Args:
            timestamps: 时间戳列表引用
        """
        self.part_timestamps = timestamps
    
    def set_mode(self, mode: str, duration: int = None) -> None:
        """
        动态设置 SART 模式
        
        Args:
            mode: "short" (5分钟) 或 "long" (25分钟)
            duration: 自定义时长(秒)，None则使用默认值
        """
        self.mode = mode
        self.total_duration = duration if duration else (300 if mode == "short" else 1500)
        
        mode_text = "低负荷采集" if mode == "short" else "疲劳诱发"
        duration_text = f"{self.total_duration // 60}分钟" if self.total_duration >= 60 else f"{self.total_duration}秒"
        
        config.logger.info(f"✅ SART 模式已设置为: {mode} ({mode_text}, 时长: {duration_text})")
        
        # 更新说明文字
        self.instruction_label.setText(
            f"SART 任务 ({mode_text}, 时长: {duration_text})\n\n"
            "规则：看到除'3'外的数字按【空格】，看到'3'不要按。\n\n"
            "请保持安静、注视中央，不说话。\n\n"
            "按【空格】开始测试"
        )
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        """键盘事件处理"""
        # 如果正在等待按键继续,任意键都触发继续
        if self.waiting_for_continue and not event.isAutoRepeat():
            config.logger.info("用户按键确认,继续下一阶段")
            self.waiting_for_continue = False
            self.sart_finished.emit()
            return
        
        # 空格键：开始测试或响应刺激
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # 未开始时按空格启动测试
            if not self.is_running and self.instruction_label.isVisible():
                self.start_test()
            # 运行中按空格响应刺激
            elif self.is_running and not self.key_pressed:
                self.key_pressed = True
                self.response_time = (time.time() - self.trial_start_time) * 1000  # ms
                config.logger.debug(f"按键响应: RT={self.response_time:.1f}ms")
        
        # Q键跳过
        elif event.key() == Qt.Key_Q and not event.isAutoRepeat():
            if not self.is_running:
                config.logger.info("⏭️ 用户跳过SART测试")
                # 记录开始和结束时间戳（快速标记）
                call_timestamp = time.time()
                self.part_timestamps.append(call_timestamp)  # 开始
                self.part_timestamps.append(call_timestamp)  # 结束
                
                # 即使跳过也要创建 sart 目录并尝试保存数据（如果有试次记录）
                self._ensure_sart_directory()
                
                # 🐛 修复：即使跳过也要尝试保存已有的试次记录
                if self.trial_records:
                    config.logger.info(f"⚠️ 跳过前已有 {len(self.trial_records)} 条试次记录，正在保存...")
                    self._save_results()
                else:
                    config.logger.info("⚠️ 跳过时无试次记录，不保存CSV文件")
                
                # 显示跳过提示,等待按键继续
                self.instruction_label.setVisible(False)
                self.digit_label.setVisible(False)
                self.completion_label.setText("SART任务已跳过\n\n按任意键继续")
                self.completion_label.setVisible(True)
                self.waiting_for_continue = True
                
            elif self.is_running:
                # 正在运行时按Q提前结束
                config.logger.info("⏭️ 用户中断SART测试")
                self._finish_sart()
        
        super().keyPressEvent(event)
    
    def set_session_dir(self, session_dir: str) -> None:
        """设置会话目录（用于保存结果）"""
        self.session_dir = session_dir
    
    def set_session_info(self, session_dir: str, current_user: str) -> None:
        """设置会话信息（用于EEG采集和结果保存）
        
        Args:
            session_dir: 会话目录路径（例如：recordings/admin/20251024_185949）
            current_user: 当前用户名
        """
        self.session_dir = session_dir
        self.current_user = current_user
        config.logger.info(f"✅ SART页面已设置会话信息: user={current_user}, dir={session_dir}")
        config.logger.info(f"📂 SART结果将保存到: {session_dir}/sart/")
    
    def reset(self) -> None:
        """重置页面（供下次使用）"""
        self.trial_timer.stop()
        self.total_timer.stop()
        self.waiting_for_continue = False  # 重置等待标志
        self._reset_state()
        
        self.instruction_label.setVisible(True)
        self.digit_label.setVisible(False)
        self.completion_label.setVisible(False)
        self.completion_label.setText("")
        self.progress_label.setVisible(False)
        self.stats_label.setVisible(False)


__all__ = ["SARTPage"]
