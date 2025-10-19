import sys
import os
import csv
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, 
    QLabel, QMessageBox, QSpacerItem, QSizePolicy, QApplication, QWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QPalette, QPainter

# 获取全局logger
logger = logging.getLogger()

# 使用config中的路径
try:
    from ..app import config
    SCHULTE_CSV_FILE = config.SCHULTE_SCORES_CSV_FILE
except ImportError:
    # 兜底方案：直接计算路径
    BASE_DIR = Path(__file__).resolve().parent.parent
    SCHULTE_CSV_FILE = BASE_DIR / "data" / "users" / "schulte_scores.csv"

class SchulteButton(QPushButton):
    """舒特格测试中的单个方格按钮"""
    def __init__(self, position, background_number, test_number):
        super().__init__()
        self.position = position  # 在5x5网格中的位置 (row, col)
        self.background_number = background_number  # 背景图片编号 (1-25)
        self.test_number = test_number  # 本次测试中的数字
        self.is_clicked = False
        self.button_size = 92  # 基于当前控件尺寸动态更新
        
        self.setObjectName("schulteButton")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(40, 40)
        
        # 使用内存中的pixmap而非临时文件
        self.original_pixmap = None
        self.scaled_pixmap = None
        self.transparent_pixmap = None
        self._last_scaled_size = None  # (w, h)
        
        # 载入原始背景图（一次IO）
        self._load_original_pixmap()
        
        # 初始显示：透明（无数字）
        self.setText("" if not self.test_number else str(self.test_number))
        self.setStyleSheet(self._get_normal_style())

    def paintEvent(self, event):
        """自定义绘制：背景图在最底层，文本在顶层。"""
        # 先让样式绘制边框/背景，但避免绘制文本与图标
        saved_text = self.text()
        self.setText("")
        saved_icon = self.icon()
        self.setIcon(QIcon())
        super().paintEvent(event)
        # 恢复文本/icon 属性（不让Qt重绘）
        self.setText(saved_text)
        self.setIcon(saved_icon)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 绘制背景图，充满整个按钮区域
        pix = None
        if self.is_clicked and self.transparent_pixmap is not None:
            pix = self.transparent_pixmap
        elif self.scaled_pixmap is not None:
            pix = self.scaled_pixmap
        if pix is not None:
            target_rect = self.rect()  # 紧密填充，不留内边距
            painter.drawPixmap(target_rect, pix, pix.rect())
        
        # 绘制数字文本在顶层
        painter.setFont(self.font())
        # 阴影（半透明深色）
        shadow_color = QColor(0, 0, 0, 180)
        painter.setPen(shadow_color)
        painter.drawText(self.rect().translated(1, 1), Qt.AlignCenter, saved_text)
        # 文字（白色）
        painter.setPen(QColor("white"))
        painter.drawText(self.rect(), Qt.AlignCenter, saved_text)
        painter.end()

    def resizeEvent(self, event):
        """在按钮尺寸变化时，重建缩放图与样式，保持自适应。"""
        super().resizeEvent(event)
        try:
            w, h = max(1, self.width()), max(1, self.height())
            # 记录一个代表性尺寸用于字体/边框比例
            self.button_size = min(w, h)
            # 仅在尺寸变化时重建
            if self._last_scaled_size != (w, h):
                self._prepare_scaled_pixmaps(w, h)
                # 根据尺寸刷新样式（字体/圆角/边框）
                if self.is_clicked:
                    self.setStyleSheet(self._get_clicked_style())
                else:
                    self.setStyleSheet(self._get_normal_style())
                self.update()
        except Exception as e:
            logger.warning(f"按钮resize时更新失败: {e}")
        
    def _prepare_scaled_pixmaps(self, target_w: int, target_h: int):
        """按当前按钮大小生成普通与半透明两个版本"""
        try:
            if self.original_pixmap is None:
                self._load_original_pixmap()
            if self.original_pixmap is None or self.original_pixmap.isNull():
                self.scaled_pixmap = None
                self.transparent_pixmap = None
                return
            
            scaled_pixmap = self.original_pixmap.scaled(
                target_w, target_h,
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            self.scaled_pixmap = scaled_pixmap
            self.transparent_pixmap = self._create_transparent_version(scaled_pixmap)
            self._last_scaled_size = (target_w, target_h)
        except Exception as e:
            logger.error(f"处理背景图片时出错: {e}")
            self.scaled_pixmap = None
            self.transparent_pixmap = None
    
    def _load_original_pixmap(self):
        """仅加载一次原始背景图到内存，避免重复IO。"""
        try:
            original_path = f"assets/schult/{self.background_number}.png"
            if not os.path.exists(original_path):
                logger.warning(f"背景图片不存在: {original_path}")
                self.original_pixmap = None
                return
            pix = QPixmap(original_path)
            if pix.isNull():
                logger.warning(f"无法加载图片: {original_path}")
                self.original_pixmap = None
                return
            self.original_pixmap = pix
        except Exception as e:
            logger.error(f"加载原始图片时出错: {e}")
            self.original_pixmap = None
    
    def _create_transparent_version(self, original_pixmap):
        """创建图片的半透明版本"""
        try:
            from PyQt5.QtGui import QPainter
            
            transparent_pixmap = QPixmap(original_pixmap.size())
            transparent_pixmap.fill(Qt.transparent)
            painter = QPainter(transparent_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setOpacity(0.5)
            painter.drawPixmap(0, 0, original_pixmap)
            painter.end()
            return transparent_pixmap
        except Exception as e:
            logger.error(f"创建半透明图片时出错: {e}")
            return None
    
    def _get_normal_style(self):
        """获取正常状态的样式（按尺寸比例动态生成）。"""
        font_size = max(12, int(self.button_size * 0.20))
        border_radius = max(0, int(self.button_size * 0.06))
        border_width = max(1, int(self.button_size * 0.02))
        return f"""
        QPushButton {{
            border: {border_width}px solid #ccc;
            border-radius: {border_radius}px;
            color: white;
            font-size: {font_size}px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            border: {border_width}px solid #4CAF50;
            background-color: rgba(255,255,255,0.1);
        }}
        """
    
    def _get_clicked_style(self):
        """获取已点击状态的样式（与半透明icon配合）。"""
        font_size = max(12, int(self.button_size * 0.20))
        border_radius = max(0, int(self.button_size * 0.06))
        border_width = max(1, int(self.button_size * 0.02))
        return f"""
        QPushButton {{
            background-color: rgba(255,255,255,0.3);
            border: {border_width}px solid #2196F3;
            border-radius: {border_radius}px;
            color: #2196F3;
            font-size: {font_size}px;
            font-weight: bold;
        }}
        """
    
    def _get_fallback_style(self):
        """当图片加载失败时的备用样式（随尺寸比例）。"""
        font_size = max(12, int(self.button_size * 0.20))
        border_radius = max(0, int(self.button_size * 0.06))
        border_width = max(1, int(self.button_size * 0.02))
        return f"""
        QPushButton {{
            background-color: #E0E0E0;
            border: {border_width}px solid #ccc;
            border-radius: {border_radius}px;
            color: #333;
            font-size: {font_size}px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            border: {border_width}px solid #4CAF50;
            background-color: #F0F0F0;
        }}
        """
    
    def _get_fallback_clicked_style(self):
        """当图片加载失败时的备用点击样式（随尺寸比例）。"""
        font_size = max(12, int(self.button_size * 0.20))
        border_radius = max(0, int(self.button_size * 0.06))
        border_width = max(1, int(self.button_size * 0.02))
        return f"""
        QPushButton {{
            background-color: rgba(33,150,243,0.3);
            border: {border_width}px solid #2196F3;
            border-radius: {border_radius}px;
            color: #2196F3;
            font-size: {font_size}px;
            font-weight: bold;
        }}
        """
    
    def mark_as_clicked(self):
        """标记按钮为已点击状态"""
        self.is_clicked = True
        self.setStyleSheet(self._get_clicked_style())
        self.setEnabled(False)


class SchulteGridDialog(QDialog):
    """舒特格测试弹窗界面"""
    test_completed = pyqtSignal()  # 测试完成信号
    test_result_ready = pyqtSignal(float, float)  # (elapsed_time, accuracy)
    
    @staticmethod
    def cleanup_temp_files():
        """清理临时缩放图片文件（兼容保留，当前实现不再生成临时文件）"""
        temp_dir = "temp_scaled_images"
        if os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir)
                logger.info("已清理临时图片文件")
            except Exception as e:
                logger.warning(f"清理临时文件时出错: {e}")
    
    def __init__(self, username: str = "anonymous", parent=None):
        super().__init__(parent)
        self.username = username
        self.buttons = []  # 25个按钮的列表
        self.correct_sequence = []  # 正确点击序列（按钮索引）
        self.current_target_index = 0  # 当前应该点击的按钮在correct_sequence中的索引
        self.start_time = None
        self.total_clicks = 0
        self.test_started = False
        self.test_completed_flag = False
        
        self.timer = QTimer()
        self.elapsed_time = 0
        
        # 设计基准尺寸与缩放因子
        self._base_dialog_w = 1200
        self._base_dialog_h = 800
        self.ui_scale = 1.0
        
        # 左右分栏引用（用于后续按比例固定尺寸）
        self.left_panel = None
        self.center_container = None
        self.center_layout = None
        
        # 顶部留白（开发者可配置）
        try:
            self.left_top_padding_base_px = int(os.getenv('SCHULTE_LEFT_TOP_PADDING_PX', '200'))
        except Exception:
            self.left_top_padding_base_px = 200
        try:
            self.left_top_padding_scale = float(os.getenv('SCHULTE_LEFT_TOP_PADDING_SCALE', '1.0'))
        except Exception:
            self.left_top_padding_scale = 1.0
        self._left_top_spacer = None
        
        self._setup_dialog()
        self._init_ui()
        self._connect_signals()
        # 初始化后应用一次缩放布局（确保网格为正方形）
        QTimer.singleShot(0, self._apply_scaled_layout)
    
    def _setup_dialog(self):
        """设置弹窗的基本属性，并根据屏幕分辨率确定固定尺寸。"""
        self.setWindowTitle("舒特格测试")
        self.setModal(True)  # 设置为模态弹窗
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        
        # 读取屏幕分辨率并计算缩放因子
        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()
                screen_w, screen_h = geo.width(), geo.height()
            else:
                # 兜底值
                screen_w, screen_h = 1920, 1080
            # 以 1920x1080 为参考，按较小边比例缩放
            scale = min(screen_w / 1920.0, screen_h / 1080.0)
            # 限制缩放范围，避免过大或过小
            self.ui_scale = scale
        except Exception as e:
            logger.warning(f"读取屏幕分辨率失败，使用默认缩放: {e}")
            self.ui_scale = 1.0
        
        final_w = int(self._base_dialog_w * self.ui_scale)
        final_h = int(self._base_dialog_h * self.ui_scale)
        # 固定窗口尺寸，禁止用户拉伸
        self.setFixedSize(final_w, final_h)
        
        # 设置弹窗样式
        self.setObjectName("schulteDialog")
        self.setStyleSheet("""
            QDialog#schulteDialog {
                background-color: #F5F5F5;
                border-radius: 15px;
            }
        """)
    
    def set_username(self, username: str):
        """更新当前测试的用户名"""
        self.username = username or "anonymous"
        
    def _init_ui(self):
        """初始化UI（左侧文字引导，右侧紧凑25宫格）"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)
        
        # 顶部标题
        title_label = QLabel("舒尔特方格注意力测试")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 将错误提示移动到标题下方
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 中部左右分栏
        center_container = QWidget()
        center_layout = QHBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(20)
        
        # 左侧：文字引导+状态+开始按钮
        left_panel = QWidget()
        left_panel.setObjectName("card")
        left_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)
        # 顶部不再留白，直接展示标题与规则
        self._left_top_spacer = None
        
        rule_title = QLabel("测试目的与规则")
        rule_title.setObjectName("h2")
        rule_title.setAlignment(Qt.AlignLeft)
        # 使用样式表基于缩放因子设置字号，优先级高于全局QSS
        try:
            title_px = max(24, int(18 * getattr(self, 'ui_scale', 1.0)))
            rule_title.setStyleSheet(f"font-size: {title_px}px; font-weight: 600;")
        except Exception:
            rule_title.setStyleSheet("font-size: 24px; font-weight: 600;")
        rule_text = QLabel(
            "测试目的：评估注意力与视觉搜索效率。\n\n"
            "规则：\n"
            "1. 从最小数字开始，按递增顺序点击方格。\n"
            "2. 点错会有红色提示，请继续寻找正确数字。\n"
            "3. 开始后数字会显现，完成后显示用时与准确率。"
        )
        rule_text.setObjectName("subtitle")
        rule_text.setWordWrap(True)
        try:
            text_px = max(22, int(14 * getattr(self, 'ui_scale', 1.0)))
            rule_text.setStyleSheet(f"font-size: {text_px}px;")
        except Exception:
            rule_text.setStyleSheet("font-size: 22px;")
        
        # 开始按钮
        self.start_button = QPushButton("开始测试")
        self.start_button.setObjectName("successButton")
        self.start_button.setFixedWidth(200)
        self.start_button.clicked.connect(self._start_test)
        
        # 将结果显示移动到开始按钮上方（左侧面板内）
        self.result_label = QLabel("")
        self.result_label.setObjectName("subtitle")
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setVisible(False)
        
        left_layout.addWidget(rule_title)
        left_layout.addWidget(rule_text)
        left_layout.addStretch(1)
        left_layout.addWidget(self.result_label)
        left_layout.addSpacing(8)
        
        # 下一阶段按钮（测试完成后显示）
        self.next_button = QPushButton("进入下一阶段")
        self.next_button.setObjectName("finishButton")
        self.next_button.setFixedWidth(200)
        self.next_button.setVisible(False)
        self.next_button.clicked.connect(self._on_next_stage)
        
        # 两个按钮放在左框底部居中
        buttons_container = QWidget(left_panel)
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(8)
        buttons_layout.addWidget(self.start_button, 0, Qt.AlignHCenter)
        buttons_layout.addWidget(self.next_button, 0, Qt.AlignHCenter)
        left_layout.addWidget(buttons_container, 0, Qt.AlignBottom)
        
        # 右侧：紧凑5x5网格（正方形）
        self.grid_container = QWidget()
        self.grid_container.setObjectName("card")
        self.grid_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        grid_layout = QGridLayout(self.grid_container)
        grid_layout.setSpacing(0)  # 关键：0间距
        grid_layout.setContentsMargins(0, 0, 0, 0)  # 关键：0边距
        self.grid_layout = grid_layout
        
        # 创建25个按钮（初始透明）
        for i in range(5):
            for j in range(5):
                button = SchulteButton((i, j), i * 5 + j + 1, 0)  # 初始不显示数字
                button.clicked.connect(lambda checked, btn=button: self._on_button_clicked(btn))
                button.setEnabled(False)
                self.buttons.append(button)
                grid_layout.addWidget(button, i, j)
        
        # 将左右面板加入分栏，右侧给予更大伸展比（右侧容器本身为固定正方形）
        center_layout.addWidget(left_panel, 1)
        center_layout.addWidget(self.grid_container, 0, Qt.AlignCenter)
        
        layout.addWidget(center_container, 1)
        
        # 结果显示已移动到左侧面板中
        
        # 底部按钮区域（已移除“进入下一阶段”按钮）
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.setSpacing(20)
        bottom_button_layout.addStretch()
        layout.addLayout(bottom_button_layout)
        
        # 保存引用用于缩放布局
        self.left_panel = left_panel
        self.center_container = center_container
        self.center_layout = center_layout
        
    def _apply_scaled_layout(self):
        """根据 ui_scale 固定左侧宽度，并将右侧网格设为正方形，禁止用户拉伸时仍保持比例。"""
        try:
            # 固定左侧面板宽度（随缩放变化）
            left_w = int(380 * self.ui_scale)
            self.left_panel.setFixedWidth(left_w)
            # 调整左右间距随缩放
            self.center_layout.setSpacing(int(20 * self.ui_scale))
            
            # 计算中心区域内可用于网格的宽高
            # 使用已布局后的尺寸，确保测得的是最终可用空间
            available_w = max(200, self.center_container.width() - self.left_panel.width() - self.center_layout.spacing())
            available_h = max(200, self.center_container.height())
            side = min(available_w, available_h)
            # 设为正方形固定尺寸
            self.grid_container.setFixedSize(side, side)
        except Exception as e:
            logger.warning(f"应用缩放布局失败: {e}")
    
    def _connect_signals(self):
        """连接信号"""
        self.timer.timeout.connect(self._update_timer)
        
    def _start_test(self):
        """开始测试"""
        logger.info("舒特格测试开始")
        
        # 如果之前有未完成的测试，先保存它
        if self.test_started and not self.test_completed_flag:
            self.reset_for_next_stage()
        
        # 重置状态
        self.test_started = True
        self.test_completed_flag = False
        self.current_target_index = 0
        self.total_clicks = 0
        self.elapsed_time = 0
        
        # 生成测试数字序列
        self._generate_test_numbers()
        
        # 启用所有按钮并显示数字
        for button in self.buttons:
            button.setEnabled(True)
            button.is_clicked = False
            button.setText(str(button.test_number))
            button.setStyleSheet(button._get_normal_style())
        
        # 开始计时
        self.start_time = time.time()
        self.timer.start(100)  # 每100ms更新一次
        
        # 更新UI状态（不再提示下一个应点击数字）
        self.start_button.setText("重新开始")
        self.status_label.setText("")
        self.status_label.setStyleSheet("")
        self.result_label.setVisible(False)
        self.next_button.setVisible(False)
        
        # 固定窗口下仍确保右侧为正方形（某些平台首次布局后需要再调整一次）
        QTimer.singleShot(0, self._apply_scaled_layout)
    
    def _generate_test_numbers(self):
        """生成测试数字序列（线性构建正确序列）"""
        # 从1-75中随机选择起始数字
        start_num = random.randint(1, 51)  # 确保不超过75
        test_numbers = list(range(start_num, start_num + 25))
        
        # 随机打乱并分配给按钮
        random.shuffle(test_numbers)
        
        for i, button in enumerate(self.buttons):
            button.test_number = test_numbers[i]
            # 不在此处设置文本，由 _start_test 中统一显示
        
        # 建立 数字 -> 按钮索引 的映射（O(n)）
        num_to_index = {button.test_number: i for i, button in enumerate(self.buttons)}
        
        # 线性构建正确点击序列：按从小到大的自然顺序（起始数已知且连续）
        self.correct_sequence = [(num_to_index[num], num) for num in range(start_num, start_num + 25)]
        
        logger.info(f"测试数字范围: {start_num}-{start_num+24}")
        logger.info(f"正确点击序列: {[x[1] for x in self.correct_sequence]}")
        
    def _on_button_clicked(self, clicked_button):
        """处理按钮点击"""
        if not self.test_started or self.test_completed_flag:
            return
            
        self.total_clicks += 1
        
        # 找到被点击按钮的索引
        clicked_index = self.buttons.index(clicked_button)
        expected_index, expected_number = self.correct_sequence[self.current_target_index]
        
        if clicked_index == expected_index:
            # 正确点击
            clicked_button.mark_as_clicked()
            self.current_target_index += 1
            
            if self.current_target_index >= len(self.correct_sequence):
                # 测试完成
                self._complete_test()
            else:
                # 正确时不提示下一个应点击数字
                self.status_label.setText("")
                self.status_label.setStyleSheet("")
        else:
            # 错误点击：标红并提示正确数字
            current_target_number = self.correct_sequence[self.current_target_index][1]
            QApplication.beep()
            
            self.status_label.setText(f"点击错误，应点击 {current_target_number}")
            self.status_label.setStyleSheet("color: #D32F2F; font-weight: bold;")
            
            # 错误按钮边框短暂高亮
            original_style = clicked_button.styleSheet()
            clicked_button.setStyleSheet("""
            QPushButton {
                border: 2px solid #E53935;
                border-radius: 8px;
            }
            """)
            
            def _restore_feedback():
                # 恢复提示与按钮样式（清除提示，不再显示下一目标）
                self.status_label.setText("")
                self.status_label.setStyleSheet("")
                if clicked_button.isEnabled():
                    clicked_button.setStyleSheet(clicked_button._get_normal_style())
                else:
                    clicked_button.setStyleSheet(clicked_button._get_clicked_style())
            
            QTimer.singleShot(800, _restore_feedback)
            logger.info(f"用户错误点击了 {clicked_button.test_number}，应该点击 {current_target_number}")
    
    def _update_timer(self):
        """更新计时器显示"""
        if self.test_started and not self.test_completed_flag:
            self.elapsed_time = time.time() - self.start_time
    
    def _complete_test(self):
        """完成测试"""
        self.test_completed_flag = True
        self.timer.stop()
        
        # 计算结果
        accuracy = (25 / self.total_clicks) * 100 if self.total_clicks > 0 else 0
        
        # 显示结果（移动至左侧按钮上方）
        result_text = f"用时: {self.elapsed_time:.2f}秒    准确率: {accuracy:.1f}%"
        self.result_label.setText(result_text)
        self.result_label.setVisible(True)
        self.status_label.setText("")
        self.status_label.setStyleSheet("")
        
        # 发射结果信号，供外部接收
        try:
            self.test_result_ready.emit(float(self.elapsed_time), float(accuracy))
        except Exception as e:
            logger.warning(f"发射舒特结果信号失败: {e}")
        
        # 保存结果
        self._save_result(self.elapsed_time, accuracy, self.total_clicks, True)
        
        # 显示进入下一阶段按钮
        self.next_button.setVisible(True)
        
        # 禁用所有测试按钮
        for button in self.buttons:
            button.setEnabled(False)
            
        logger.info(f"舒特格测试完成 - 用时: {self.elapsed_time:.2f}s, 准确率: {accuracy:.1f}%, 总点击: {self.total_clicks}")
    
    def _on_next_stage(self):
        """进入下一阶段，发送信号并关闭弹窗"""
        self.test_completed.emit()
        self.accept()  # 关闭弹窗
    
    def _save_result(self, completion_time, accuracy, total_clicks, completed):
        """保存测试结果到CSV文件，增加用户名字段"""
        try:
            file_exists = os.path.exists(SCHULTE_CSV_FILE)
            
            # 如果文件不存在则写入带用户名的标题
            if not file_exists:
                with open(SCHULTE_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['用户名', '时间戳', '完成时间(秒)', '准确率(%)', '总点击次数', '是否完成'])
            
            # 追加写入数据行
            with open(SCHULTE_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.username,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    f"{completion_time:.2f}",
                    f"{accuracy:.1f}",
                    total_clicks,
                    "是" if completed else "否"
                ])
            
            logger.info(f"舒特格测试结果已保存到 {SCHULTE_CSV_FILE}")
        except Exception as e:
            logger.error(f"保存舒特格测试结果时出错: {e}")
            QMessageBox.warning(self, "保存失败", f"无法保存测试结果: {e}")

    def reset_for_next_stage(self):
        """为进入下一阶段重置状态"""
        # 如果测试未完成就进入下一阶段，记录未完成状态
        if self.test_started and not self.test_completed_flag:
            # 停止计时器并计算最终时间
            if self.timer.isActive():
                self.timer.stop()
            
            # 计算当前已用时间
            if self.start_time:
                current_elapsed_time = time.time() - self.start_time
            else:
                current_elapsed_time = self.elapsed_time
                
            # 计算当前准确率（已完成的数量/总点击数）
            completed_count = self.current_target_index
            if self.total_clicks > 0:
                current_accuracy = (completed_count / self.total_clicks) * 100
            else:
                current_accuracy = 0
                
            logger.info(f"舒特格测试未完成就退出 - 已完成: {completed_count}/25, 用时: {current_elapsed_time:.2f}s, 点击数: {self.total_clicks}")
            self._save_result(current_elapsed_time, current_accuracy, self.total_clicks, False)
            
            # 重置状态
            self.test_started = False

    def closeEvent(self, event):
        """弹窗关闭事件处理"""
        # 如果测试正在进行中，保存未完成状态
        if self.test_started and not self.test_completed_flag:
            self.reset_for_next_stage()
        
        # 停止计时器
        if self.timer.isActive():
            self.timer.stop()
        
        super().closeEvent(event)

    def show_dialog(self):
        """显示弹窗的便捷方法"""
        self.exec_()

    def set_left_top_padding(self, base_px: int = None, scale: float = None):
        """
        运行时调整左侧顶部留白参数。
        - base_px: 基准像素（默认读取环境变量 SCHULTE_LEFT_TOP_PADDING_PX）
        - scale: 额外缩放系数（默认读取环境变量 SCHULTE_LEFT_TOP_PADDING_SCALE）
        调整后会立即刷新当前布局。
        """
        try:
            if base_px is not None:
                self.left_top_padding_base_px = int(base_px)
            if scale is not None:
                self.left_top_padding_scale = float(scale)
        except Exception as e:
            logger.warning(f"设置顶部留白参数失败: {e}")
        
        try:
            # 重新计算留白高度
            left_top_padding = int(self.left_top_padding_base_px * getattr(self, 'ui_scale', 1.0) * self.left_top_padding_scale)
            # 用新的 spacer 替换旧的
            if self._left_top_spacer is not None and self.left_panel is not None:
                # 从布局中移除旧 spacer（简单做法：清除并重新插入）
                left_layout = self.left_panel.layout()
                # 重建顶部布局：删除第一个条目并插入新 spacer
                if left_layout is not None and left_layout.count() > 0:
                    item = left_layout.itemAt(0)
                    left_layout.removeItem(item)
                new_spacer = QSpacerItem(20, left_top_padding, QSizePolicy.Minimum, QSizePolicy.Fixed)
                left_layout.insertItem(0, new_spacer)
                self._left_top_spacer = new_spacer
            # 刷新网格区域大小
            QTimer.singleShot(0, self._apply_scaled_layout)
        except Exception as e:
            logger.warning(f"刷新顶部留白失败: {e}")

    def keyPressEvent(self, event):
        """拦截ESC，避免对话框直接关闭。"""
        try:
            if event.key() == Qt.Key_Escape:
                event.ignore()
                return
        except Exception:
            pass
        super().keyPressEvent(event)


# 为了保持向后兼容，保留原来的类名
class SchulteGridWidget(SchulteGridDialog):
    """向后兼容的舒特格测试控件类名"""
    pass 