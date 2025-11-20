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
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize, QEvent
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QPalette, QPainter, QMovie

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
        self._image_loaded = False  # 标记图片是否已加载
        
        # 延迟加载背景图（不在__init__中加载，避免卡UI）
        # 将在第一次paintEvent或resizeEvent时加载
        
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
        """
        仅加载一次原始背景图到内存，避免重复IO。
        延迟加载策略：只在第一次需要时加载。
        """
        if self._image_loaded:  # 已经加载过，直接返回
            return
            
        try:
            original_path = f"assets/schult/{self.background_number}.png"
            if not os.path.exists(original_path):
                logger.warning(f"背景图片不存在: {original_path}")
                self.original_pixmap = None
                self._image_loaded = True  # 标记为已尝试加载
                return
            pix = QPixmap(original_path)
            if pix.isNull():
                logger.warning(f"无法加载图片: {original_path}")
                self.original_pixmap = None
                self._image_loaded = True  # 标记为已尝试加载
                return
            self.original_pixmap = pix
            self._image_loaded = True  # 加载成功
        except Exception as e:
            logger.error(f"加载原始图片时出错: {e}")
            self.original_pixmap = None
            self._image_loaded = True  # 标记为已尝试加载
    
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
        
        # 设计基准尺寸与缩放因子（增加高度以容纳更多内容）
        self._base_dialog_w = 1200
        self._base_dialog_h = 900  # 从800增加到900
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
                border-radius: 35px;
            }
        """)
        
        # 添加右下角阴影效果
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        from PyQt5.QtGui import QColor
        shadow_effect = QGraphicsDropShadowEffect()
        shadow_effect.setBlurRadius(25)  # 阴影模糊半径稍大
        shadow_effect.setOffset(6, 6)    # 阴影偏移：右下方向
        shadow_effect.setColor(QColor(0, 0, 0, 50))  # 浅黑色，透明度50
        self.setGraphicsEffect(shadow_effect)
    
    def set_username(self, username: str):
        """更新当前测试的用户名"""
        self.username = username or "anonymous"
        
    def _init_ui(self):
        """初始化UI（顶部大标题，中间网格+按钮，底部结果显示）"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 15)  # 减小底部边距（30→15）
        layout.setSpacing(15)  # 减小整体间距
        
        # 顶部大标题 - 老年版简洁文字（替换原来的"舒尔特方格注意力测试"）
        title_label = QLabel("按从小到大的顺序,快速、准确地依次点击所有数字")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        # 使用超大字号
        try:
            title_px = max(36, int(36 * getattr(self, 'ui_scale', 1.0)))
            title_label.setStyleSheet(f"font-size: {title_px}px; font-weight: 600; line-height: 1.4; color: #2c3e50;")
        except Exception:
            title_label.setStyleSheet("font-size: 36px; font-weight: 600; line-height: 1.4; color: #2c3e50;")
        layout.addWidget(title_label)
        
        # 错误提示移动到标题下方
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 中间区域：网格 + 开始按钮（居中，往上提）
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(25)  # 减小网格和按钮之间的间距
        center_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)  # 改为顶部对齐
        
        # 创建GIF播放器容器（用于叠加播放按钮）
        self.gif_container = QWidget()
        self.gif_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        # GIF标签
        self.gif_label = QLabel(self.gif_container)
        self.gif_label.setObjectName("gifLabel")
        self.gif_label.setAlignment(Qt.AlignCenter)
        self.gif_label.setStyleSheet("""
            QLabel#gifLabel {
                background-color: #2c3e50;
                border-radius: 10px;
                border: none;
            }
        """)
        
        # 创建纯透明的播放/暂停按钮（覆盖在GIF中央，初始显示）
        self.play_pause_button = QPushButton(self.gif_container)
        self.play_pause_button.setObjectName("playPauseButton")
        self.play_pause_button.setText("▶")  # 初始显示播放图标
        self.play_pause_button.setCursor(Qt.PointingHandCursor)
        self.play_pause_button.setVisible(True)  # 初始显示，提示用户可以点击
        self.play_pause_button.setStyleSheet("""
            QPushButton#playPauseButton {
                background-color: transparent;
                background: transparent;
                color: white;
                border: none;
                outline: none;
                font-size: 80px;
                font-weight: bold;
            }
            QPushButton#playPauseButton:hover {
                background-color: transparent;
                background: transparent;
                color: white;
                border: none;
                outline: none;
                font-size: 90px;
            }
            QPushButton#playPauseButton:pressed {
                background-color: transparent;
                background: transparent;
                color: white;
                border: none;
                outline: none;
                font-size: 75px;
            }
            QPushButton#playPauseButton:focus {
                background-color: transparent;
                background: transparent;
                color: white;
                border: none;
                outline: none;
            }
        """)
        self.play_pause_button.setFixedSize(150, 150)
        self.play_pause_button.clicked.connect(self._toggle_gif_playback)
        
        # 安装事件过滤器来监听鼠标悬停
        self.gif_container.installEventFilter(self)
        
        # 创建QMovie对象用于播放GIF
        # 使用绝对路径确保能找到文件
        gif_path = Path(__file__).resolve().parent.parent / "assets" / "gif" / "shuerte.gif"
        logger.debug(f"🔍 尝试加载GIF: {gif_path}")
        
        if gif_path.exists():
            self.gif_movie = QMovie(str(gif_path))
            if self.gif_movie.isValid():
                self.gif_label.setMovie(self.gif_movie)
                # 显示第一帧但不自动播放
                self.gif_movie.jumpToFrame(0)  # 跳到第一帧
                self.gif_playing = False
                logger.debug(f"✅ 舒尔特GIF已加载成功: {gif_path}")
            else:
                logger.error(f"❌ GIF文件无效: {gif_path}")
                self.gif_movie = None
                self.gif_playing = False
        else:
            logger.error(f"❌ GIF文件不存在: {gif_path}")
            self.gif_movie = None
            self.gif_playing = False
        
        center_layout.addWidget(self.gif_container, 0, Qt.AlignCenter)
        
        # 5x5网格（正方形）- 初始隐藏
        self.grid_container = QWidget()
        self.grid_container.setObjectName("card")
        self.grid_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.grid_container.setVisible(False)  # 初始隐藏网格
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
        
        center_layout.addWidget(self.grid_container, 0, Qt.AlignCenter)
        
        # 添加间距，把开始按钮往下推
        center_layout.addSpacing(40)
        
        # 开始按钮放在网格下方 - 使用水绿色（与基线校准按钮一致）
        # 高度设置为60px与底部按钮对齐
        self.start_button = QPushButton("我已了解规则,开始测试")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setFixedSize(400, 60)  # 高度改为60px与底部按钮一致
        self.start_button.clicked.connect(self._start_test)
        self.start_button.setCursor(Qt.PointingHandCursor)
        # 使用与基线校准按钮相同的水绿色
        self.start_button.setStyleSheet("""
            QPushButton#primaryButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover {
                background-color: #3498DB;
            }
            QPushButton#primaryButton:pressed {
                background-color: #2980B9;
            }
        """)
        
        # 开始按钮先添加到中间区域
        center_layout.addWidget(self.start_button, 0, Qt.AlignCenter)
        
        layout.addWidget(center_container, 1)
        
        # 添加弹性空间，把底部按钮往下推
        layout.addStretch(1)
        
        # 底部区域：结果显示、重新开始按钮和下一阶段按钮（同一行，靠下显示）
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(0, 20, 0, 0)  # 顶部留出空间
        bottom_layout.setSpacing(30)
        bottom_layout.setAlignment(Qt.AlignCenter)
        
        # 结果显示标签
        self.result_label = QLabel("")
        self.result_label.setObjectName("subtitle")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setVisible(False)
        try:
            result_px = max(24, int(24 * getattr(self, 'ui_scale', 1.0)))
            self.result_label.setStyleSheet(f"font-size: {result_px}px; font-weight: 600; color: #27ae60;")
        except Exception:
            self.result_label.setStyleSheet("font-size: 24px; font-weight: 600; color: #27ae60;")
        bottom_layout.addWidget(self.result_label)
        
        # 创建"重新开始测试"按钮（完成后显示在底部）
        self.restart_button = QPushButton("重新开始测试")
        self.restart_button.setObjectName("primaryButton")
        self.restart_button.setFixedSize(200, 60)
        self.restart_button.clicked.connect(self._start_test)
        self.restart_button.setCursor(Qt.PointingHandCursor)
        self.restart_button.setVisible(False)
        self.restart_button.setStyleSheet("""
            QPushButton#primaryButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover {
                background-color: #3498DB;
            }
            QPushButton#primaryButton:pressed {
                background-color: #2980B9;
            }
        """)
        bottom_layout.addWidget(self.restart_button)
        
        # 下一阶段按钮（测试完成后显示，与结果在同一行）
        self.next_button = QPushButton("进入下一阶段")
        self.next_button.setObjectName("finishButton")
        self.next_button.setFixedSize(200, 60)
        self.next_button.setVisible(False)
        self.next_button.clicked.connect(self._on_next_stage)
        bottom_layout.addWidget(self.next_button)
        
        layout.addWidget(bottom_container, 0)
        
        # 保存引用用于缩放布局
        self.center_container = center_container
        
    def _apply_scaled_layout(self):
        """根据 ui_scale 设置网格和GIF为正方形，保持比例。"""
        try:
            # 计算可用空间并设置为正方形（增大尺寸适合老年人）
            available_w = max(600, self.center_container.width() - 50)
            available_h = max(600, self.center_container.height() - 100)
            # 进一步增大最大尺寸限制，使网格更大
            side = min(available_w, available_h, int(850 * self.ui_scale))
            # 设为正方形固定尺寸
            self.grid_container.setFixedSize(side, side)
            # GIF容器与网格等大
            self.gif_container.setFixedSize(side, side)
            self.gif_label.setFixedSize(side, side)
            # 同时缩放GIF内容
            if self.gif_movie:
                self.gif_movie.setScaledSize(QSize(side, side))
            # 将播放按钮居中放置
            button_x = (side - self.play_pause_button.width()) // 2
            button_y = (side - self.play_pause_button.height()) // 2
            self.play_pause_button.move(button_x, button_y)
        except Exception as e:
            logger.warning(f"应用缩放布局失败: {e}")
    
    def _connect_signals(self):
        """连接信号"""
        self.timer.timeout.connect(self._update_timer)
        
    def _toggle_gif_playback(self):
        """点击按钮播放/暂停GIF"""
        if not self.gif_movie:
            logger.warning("GIF动画未加载")
            return
        
        try:
            if self.gif_playing:
                # 暂停播放
                self.gif_movie.stop()
                self.gif_playing = False
                self.play_pause_button.setText("▶")  # 显示播放图标
                self.play_pause_button.setVisible(True)  # 暂停时显示按钮
                logger.info("⏸️ 舒尔特GIF已暂停")
            else:
                # 开始播放
                self.gif_movie.start()
                self.gif_playing = True
                self.play_pause_button.setText("⏸")  # 显示暂停图标
                # 播放时不自动隐藏按钮，由鼠标悬停控制
                logger.info("▶️ 舒尔特GIF开始播放")
        except Exception as e:
            logger.error(f"❌ GIF播放控制失败: {e}", exc_info=True)
    
    def _start_test(self):
        """开始测试"""
        logger.info("舒特格测试开始")
        
        # 停止并隐藏GIF容器
        if self.gif_movie and self.gif_playing:
            self.gif_movie.stop()
            self.gif_playing = False
            self.play_pause_button.setText("▶")  # 重置为播放图标
        self.gif_container.setVisible(False)
        
        # 显示网格
        self.grid_container.setVisible(True)
        
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
        self.status_label.setText("")
        self.status_label.setStyleSheet("")
        self.result_label.setVisible(False)
        self.restart_button.setVisible(False)
        self.next_button.setVisible(False)
        # 隐藏开始按钮（测试进行中）
        self.start_button.setVisible(False)
        
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
        
        # 显示结果（底部同一行显示）
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
        
        # 显示重新开始按钮和进入下一阶段按钮（底部同一行）
        self.restart_button.setVisible(True)
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
        
        # 重置UI：隐藏网格，显示GIF
        self.grid_container.setVisible(False)
        self.gif_container.setVisible(True)
        # 重置GIF到第一帧并停止播放
        if self.gif_movie:
            self.gif_movie.stop()
            self.gif_movie.jumpToFrame(0)
            self.gif_playing = False
            self.play_pause_button.setText("▶")
    
    def eventFilter(self, obj, event):
        """事件过滤器：处理GIF容器的鼠标进入/离开事件"""
        if obj == self.gif_container:
            if event.type() == QEvent.Enter:
                # 鼠标进入GIF区域，显示播放/暂停按钮
                if hasattr(self, 'play_pause_button'):
                    self.play_pause_button.setVisible(True)
            elif event.type() == QEvent.Leave:
                # 鼠标离开GIF区域，隐藏播放/暂停按钮
                if hasattr(self, 'play_pause_button'):
                    self.play_pause_button.setVisible(False)
        
        return super().eventFilter(obj, event)
    
    def showEvent(self, event):
        """对话框显示时的事件处理"""
        super().showEvent(event)
        # GIF已经在初始化时加载，无需额外处理

    def closeEvent(self, event):
        """弹窗关闭事件处理"""
        # 如果测试正在进行中，保存未完成状态
        if self.test_started and not self.test_completed_flag:
            self.reset_for_next_stage()
        
        # 停止计时器
        if self.timer.isActive():
            self.timer.stop()
        
        # 停止GIF播放
        if hasattr(self, 'gif_movie') and self.gif_movie and self.gif_playing:
            self.gif_movie.stop()
            self.gif_playing = False
        
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