import sys
import os
import csv
import time
import logging
from datetime import datetime
import matplotlib
import pyttsx3
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from schulte_grid import SchulteGridWidget
from PyQt5.QtWidgets import QLabel, QGraphicsOpacityEffect
from queue import Queue
from threading import Thread


# 脑负荷显示
from brain_load_bar import BrainLoadBar
# 成绩显示
from score_page import ScorePage
# 多模态数据采集器
try:
    from get_multidata import (
        start_collection as multidata_start_collection,
        stop_collection as multidata_stop_collection
    )

    HAS_MULTIMODAL = True
except ImportError as e:
    HAS_MULTIMODAL = False
# 音视频采集器
from get_avdata import (
    start_collection as av_start_collection,
    stop_collection as av_stop_collection,
    start_recording as av_start_recording,
    stop_recording as av_stop_recording,
    get_current_frame as av_get_current_frame,
    get_audio_paths as av_get_audio_paths,
    get_video_paths as av_get_video_paths,
    get_current_audio_level as av_get_current_audio_level,
)
# 血压仪设备模块
try:
    from backend.devices.maibobo import MaiboboDevice

    HAS_MAIBOBO_BACKEND = True
except ImportError as e:
    HAS_MAIBOBO_BACKEND = False
# 数据库支持
try:
    from database import TestTableStore, build_store_dir
except Exception:
    TestTableStore = None  # type: ignore[assignment]
    build_store_dir = None  # type: ignore[assignment]
matplotlib.use("Qt5Agg")

# 语音识别支持（异步队列，避免阻塞）
try:
    from tools import add_audio_for_recognition, get_recognition_results, clear_recognition_results

    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False
    # 注意：这里不能使用 logger，因为 logger 还没有定义
    print("警告：语音识别功能不可用，请确保 tools.py 和 faster-whisper 已正确安装")

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
NO_CAMERA_MODE = False

# 尝试导入必要的库，如果失败则给出提示
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QStackedWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QPushButton, QMessageBox,
        QSpacerItem, QSizePolicy, QProgressBar, QFrame, QStackedLayout,
        QGraphicsDropShadowEffect
    )
    from PyQt5.QtCore import (
        Qt, QTimer, pyqtSignal, QSize, QRect, QThread, pyqtSlot, QPropertyAnimation,
        QEasingCurve, QRectF
    )
    from PyQt5.QtGui import (
        QImage, QPixmap, QIcon, QFont, QColor, QPainter, QLinearGradient,
        QBrush, QPainterPath
    )
    import cv2
    import numpy as np
    import pyaudio
    import qtawesome as qta

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所有必要的库: pip install PyQt5 opencv-python pyaudio numpy qtawesome pyttsx3")
    sys.exit(1)

# --- 全局配置 ---

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 日志设置
if not os.path.exists('logs'):
    os.makedirs('logs')
log_filename = os.path.join('logs', f'app_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()


# --- 数据管理 ---

USER_CSV_FILE = "users.csv"
SCORES_CSV_FILE = 'scores.csv'


def load_users_from_csv():
    """从CSV文件加载用户账户信息。"""
    users = {}
    if os.path.exists(USER_CSV_FILE):
        try:
            with open(USER_CSV_FILE, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        users[row[0]] = row[1]
            logger.info(f"成功从CSV加载 {len(users)} 个用户。")
        except Exception as e:
            logger.error(f"读取用户数据时出错: {e}")
    if not users:
        users = {"admin": "123456"}
        save_users_to_csv(users)
        logger.info("未找到用户文件，已创建默认用户 (admin/123456)。")
    return users


def save_users_to_csv(users):
    """将用户账户信息保存到CSV文件。"""
    try:
        with open(USER_CSV_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for username, password in users.items():
                writer.writerow([username, password])
        logger.info(f"已将 {len(users)} 个用户保存到CSV。")
    except Exception as e:
        logger.error(f"保存用户数据时出错: {e}")


# 全局用户数据
USERS = load_users_from_csv()


# --- 核心资源管理器 (单例模式) ---

class CameraInitThread(QThread):
    init_finished = pyqtSignal(bool)

    def run(self):
        logger.info("摄像头初始化线程已启动。")
        if NO_CAMERA_MODE:
            time.sleep(0.5)  # 模拟初始化等待
            self.init_finished.emit(True)
            logger.info("模拟摄像头初始化成功。")
            return
        try:
            # 通过 AV 采集器尝试启动（预览采集，不录制）
            preview_dir = os.path.join('recordings')
            os.makedirs(preview_dir, exist_ok=True)
            av_start_collection(save_dir=preview_dir, camera_index=1, video_fps=30.0, input_device_index=2)
            self.init_finished.emit(True)
            logger.info("通过 AV 采集器摄像头初始化成功。")
            return
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            self.init_finished.emit(False)


# --- 自定义UI控件 ---

def create_shadow_effect():
    """创建一个标准的阴影效果，用于卡片式设计。"""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(25)
    shadow.setColor(QColor(0, 0, 0, 50))
    shadow.setOffset(0, 4)
    return shadow


class FadingStackedWidget(QStackedWidget):
    """一个带有淡入淡出切换效果的QStackedWidget。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.m_duration = 500  # 动画时长 (ms)
        self.m_current_index = 0
        self.m_next_index = 0
        self.m_p_animation = None

    def set_animation_duration(self, duration):
        self.m_duration = duration

    @pyqtSlot()
    def on_animation_finished(self):
        self.widget(self.m_current_index).hide()
        self.setCurrentIndex(self.m_next_index)
        self.widget(self.m_next_index).setGraphicsEffect(None)
        self.m_p_animation = None

    def fade_to_index(self, index):
        if self.currentIndex() == index:
            return

        self.m_current_index = self.currentIndex()
        self.m_next_index = index

        opacity_effect_next = QGraphicsDropShadowEffect(self)
        self.widget(self.m_next_index).setGraphicsEffect(opacity_effect_next)

        self.m_p_animation = QPropertyAnimation(opacity_effect_next, b"color")
        self.m_p_animation.setDuration(self.m_duration)
        self.m_p_animation.setStartValue(QColor(0, 0, 0, 255))
        self.m_p_animation.setEndValue(QColor(0, 0, 0, 0))
        self.m_p_animation.setEasingCurve(QEasingCurve.OutQuad)
        self.m_p_animation.finished.connect(self.on_animation_finished)

        self.widget(self.m_next_index).show()
        self.widget(self.m_next_index).raise_()
        self.m_p_animation.start()


# --- 页面定义 ---

class LoginPage(QWidget):
    """登录页面UI和逻辑。"""

    def __init__(self, on_login_success_callback):
        super().__init__()
        self.on_login_success = on_login_success_callback
        self._init_ui()

    def _init_ui(self):
        """初始化UI布局和控件。"""
        self.setAutoFillBackground(True)
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignCenter)

        form_container = self._create_login_form()
        main_layout.addWidget(form_container)

    def _create_login_form(self):
        """创建登录表单的容器和内容。"""
        container = QWidget()
        container.setFixedSize(450, 550)
        container.setObjectName("loginForm")
        container.setGraphicsEffect(create_shadow_effect())

        layout = QVBoxLayout(container)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.user-shield', color='#1565C0').pixmap(60, 60))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        title = QLabel('非接触人员状态评估系统')
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("loginTitle")
        layout.addWidget(title)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.username_input = QLineEdit('admin')
        self.username_input.setPlaceholderText('用户名')
        self.username_input.setObjectName("loginInput")
        self.username_input.setTextMargins(15, 0, 15, 0)

        self.password_input = QLineEdit('123456')
        self.password_input.setPlaceholderText('密码')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("loginInput")

        # 密码可见性切换按钮
        password_layout = QHBoxLayout(self.password_input)
        password_layout.setContentsMargins(0, 0, 10, 0)
        password_layout.addStretch()
        self.toggle_password_button = QPushButton()
        self.toggle_password_button.setIcon(qta.icon('fa5s.eye-slash', color='grey'))
        self.toggle_password_button.setCursor(Qt.PointingHandCursor)
        self.toggle_password_button.setFlat(True)
        self.toggle_password_button.setCheckable(True)
        self.toggle_password_button.clicked.connect(self._toggle_password_visibility)
        password_layout.addWidget(self.toggle_password_button)

        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addSpacing(20)

        login_button = QPushButton('登 录')
        login_button.setObjectName("loginButton")
        login_button.setCursor(Qt.PointingHandCursor)
        login_button.clicked.connect(self._perform_login)
        layout.addWidget(login_button)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        copyright_label = QLabel("© 2025 智能评估系统. All Rights Reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setObjectName("copyrightLabel")
        layout.addWidget(copyright_label)

        return container

    def paintEvent(self, event):
        """Paints a consistent, professional gradient background for the test page."""
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0, QColor("#F4F6F7"))
        gradient.setColorAt(1, QColor("#EAECEE"))
        painter.fillRect(self.rect(), gradient)
        super().paintEvent(event)

    def _toggle_password_visibility(self, checked):
        """切换密码的可见性。"""
        if checked:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_button.setIcon(qta.icon('fa5s.eye', color='grey'))
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_button.setIcon(qta.icon('fa5s.eye-slash', color='grey'))

    def _perform_login(self):
        """执行登录逻辑。"""
        user = self.username_input.text()
        pwd = self.password_input.text()
        if user in USERS and USERS[user] == pwd:
            logger.info(f"用户 '{user}' 登录成功。")
            # 将用户名传递给回调
            self.on_login_success(user)
        else:
            logger.warning(f"用户 '{user}' 尝试登录失败。")
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("登录失败")
            msg.setInformativeText("用户名或密码错误，请重试。")
            msg.setWindowTitle("错误")
            msg.exec_()


# ... [其他类的代码将继续在这里] ...

class CalibrationPage(QWidget):
    """校准页面，用于在测试前检查和准备摄像头。"""
    calibration_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.cap = None
        self.camera_update_timer = QTimer(self)
        self.camera_update_timer.timeout.connect(self._update_preview_frame)

        self.stacked_layout = QStackedLayout()
        self.setLayout(self.stacked_layout)

        self._init_loading_widget()
        self._init_calibration_widget()
        self.stacked_layout.setCurrentIndex(0)

        # 线程必须保持引用
        self.camera_init_thread = CameraInitThread()
        self.camera_init_thread.init_finished.connect(self._on_camera_init_finished)

    # ----------------- Loading Widget -----------------
    def _init_loading_widget(self):
        loading_widget = QWidget()
        vbox = QVBoxLayout(loading_widget)
        vbox.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container.setFixedSize(500, 300)
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(30)

        self.loading_label = QLabel('正在初始化摄像头...')
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)

        vbox.addWidget(container)
        self.stacked_layout.addWidget(loading_widget)

        self.loading_progress = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self._update_loading_progress)

    def _update_loading_progress(self):
        if self.loading_progress < 95:
            self.loading_progress += 2
            self.progress_bar.setValue(self.loading_progress)

    # ----------------- Calibration Widget -----------------
    def _init_calibration_widget(self):
        """初始化校准主界面。"""
        calibration_widget = QWidget()
        layout = QVBoxLayout(calibration_widget)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)

        title = QLabel('设备校准')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18))
        layout.addWidget(title)

        self.camera_label = QLabel('等待摄像头开启...')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(640, 480)
        layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        self.finish_button = QPushButton("  校准完成")
        self.finish_button.setMinimumHeight(50)
        self.finish_button.setFixedWidth(220)
        self.finish_button.clicked.connect(self._on_finish_calibration)
        layout.addWidget(self.finish_button, 0, Qt.AlignCenter)

        self.stacked_layout.addWidget(calibration_widget)

    # ----------------- Event Handlers -----------------
    def showEvent(self, event):
        super().showEvent(event)
        logger.info("进入校准页面。")
        self.loading_progress = 0
        self.progress_bar.setValue(0)
        self.loading_timer.start(30)
        self.camera_init_thread.start()

    def hideEvent(self, event):
        logger.info("离开校准页面。")
        super().hideEvent(event)

    # ----------------- Camera Logic -----------------
    def _on_camera_init_finished(self, success):
        self.loading_timer.stop()
        self.progress_bar.setValue(100)
        if success:
            self.loading_label.setText("✅ 摄像头准备就绪")
            QTimer.singleShot(500, self._switch_to_calibration_view)
        else:
            self.loading_label.setText("❌ 摄像头初始化失败")
            QMessageBox.critical(self, "错误", "无法打开摄像头。请检查设备。")

    def _switch_to_calibration_view(self):
        self.stacked_layout.setCurrentIndex(1)
        self._start_camera()

    def _start_camera(self):
        if NO_CAMERA_MODE:
            # 使用灰色占位图
            placeholder = QPixmap(640, 480)
            placeholder.fill(QColor("#CCCCCC"))
            painter = QPainter(placeholder)
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 20))
            painter.drawText(placeholder.rect(), Qt.AlignCenter, "模拟摄像头画面")
            painter.end()
            self.camera_label.setPixmap(placeholder)
            self.camera_update_timer.start(100)  # 模拟刷新
            return

        # 使用 AV 采集器提供预览
        try:
            if not self.camera_update_timer.isActive():
                self.camera_update_timer.start(30)
            logger.info("Calibration 预览使用已启动的 AV 采集器")
        except Exception as e:
            logger.error(f"AV 采集器启动失败: {e}")
            self.camera_label.setText("摄像头开启失败")

    def _update_preview_frame(self):
        # 优先用 AV 采集器帧
        try:
            frame = av_get_current_frame()
            if frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.camera_label.setPixmap(pixmap.scaled(
                    self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                ))
                return
        except Exception:
            pass
        # 回退灰色占位
        try:
            w, h = 640, 480
            # 创建灰色背景 frame
            frame = np.full((h, w, 3), 204, dtype=np.uint8)

            # 绘制引导框区域
            box_w, box_h = int(w * 0.6), int(h * 0.7)
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h

            # 创建半透明遮罩
            overlay = frame.copy()
            overlay[:] = 0  # 黑色遮罩
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 中间清晰区域
            face_area = np.full((box_h, box_w, 3), 204, dtype=np.uint8)
            frame[y1:y2, x1:x2] = face_area

            # 绘制绿色引导框角
            corner_len = 30
            thickness = 4
            color = QColor("#4CAF50").getRgb()[:3]

            # 左上角
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
            # 右上角
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
            # 左下角
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
            # 右下角
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

            # 转成 QImage 显示
            qt_image = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.camera_label.setPixmap(pixmap.scaled(
                self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))

        except Exception as e:
            logger.error(f"预览更新失败: {e}")

    # 可选：真实摄像头帧处理
    def _update_camera_frame(self):
        try:
            frame = av_get_current_frame()
            if frame is None:
                return
            if self.camera_widget.isVisible():
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(self.camera_label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
                self.camera_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"更新摄像头显示失败: {e}")

    # ----------------- Finish -----------------
    def _on_finish_calibration(self):
        logger.info("用户完成设备校准。")
        self.calibration_finished.emit()


# ... [后续类的代码将继续在这里] ...
# (In your main.py, replace the old AudioLevelMeter class)

# (替换您现有的 AudioLevelMeter 类)
class AudioLevelMeter(QWidget):
    """A custom widget to display audio input level, styled via QSS."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 30)
        self.level = 0
        self.setObjectName("audioLevelMeter")

    def set_level(self, level):
        """Sets the current audio level (0-100)."""
        new_level = min(100, max(0, level))
        if self.level != new_level:
            self.level = new_level
            self.update()  # Trigger a repaint only if level changes

    def paintEvent(self, event):
        """Paints the level meter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Let the stylesheet handle the background and border.
        # It's important for the widget to be visible even at level 0.

        if self.level <= 0:
            # When there is no audio, show placeholder text
            painter.setPen(QColor("#90A4AE"))  # Grey text color
            painter.drawText(self.rect(), Qt.AlignCenter, "等待音频输入...")
            return

        # Calculate the width of the active bar
        bar_width = int(self.width() * (self.level / 100.0))
        bar_rect = QRect(0, 0, bar_width, self.height())

        # Define gradient colors based on level
        gradient = QLinearGradient(0, 0, self.width(), 0)
        if self.level < 40:
            gradient.setColorAt(0, QColor("#66BB6A"))  # Green
            gradient.setColorAt(1, QColor("#43A047"))
        elif self.level < 75:
            gradient.setColorAt(0, QColor("#FFA726"))  # Orange
            gradient.setColorAt(1, QColor("#FB8C00"))
        else:
            gradient.setColorAt(0, QColor("#EF5350"))  # Red
            gradient.setColorAt(1, QColor("#E53935"))

        # Draw the bar using a rounded path
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        path = QPainterPath()
        # Use QRectF for floating point precision in calculations
        path.addRoundedRect(QRectF(bar_rect), self.height() / 2, self.height() / 2)
        painter.drawPath(path)


class ScoreChartWidget(QWidget):
    """A widget to display historical scores using a Matplotlib line chart."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)

        # Use a modern plot style
        plt.style.use('seaborn-v0_8-whitegrid')

        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor('none')  # Transparent background

        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color:transparent;")

        self.ax = self.figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def update_chart(self, scores):
        """Clears and redraws the chart with new score data."""
        self.ax.clear()

        # Style the axes
        self.ax.set_facecolor('#F5F5F5')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#B0BEC5')
        self.ax.spines['bottom'].set_color('#B0BEC5')
        self.ax.tick_params(axis='x', colors='#546E7A')
        self.ax.tick_params(axis='y', colors='#546E7A')
        self.ax.set_ylabel('评估分数', fontsize=12, color='#37474F')
        self.ax.set_xlabel('测试次数', fontsize=12, color='#37474F')

        if not scores or len(scores) < 2:
            self.ax.set_xticks([])
            self.ax.set_yticks([0, 20, 40, 60, 80, 100])
            self.ax.set_ylim(0, 110)
            text = "历史数据不足，无法生成趋势图" if scores else "暂无历史记录"
            self.ax.text(0.5, 0.5, text, ha='center', va='center', transform=self.ax.transAxes, fontsize=14,
                         color='gray')
        else:
            # Take last 10 scores for clarity
            display_scores = scores[-10:]
            x_values = range(1, len(display_scores) + 1)

            # Plot the data
            self.ax.plot(x_values, display_scores, color='#1976D2', marker='o', linestyle='-', linewidth=2,
                         markersize=8, markerfacecolor='#42A5F5')

            # Add value labels on top of each point
            for i, score in enumerate(display_scores):
                self.ax.text(x_values[i], score + 3, str(score), ha='center', color='#0D47A1', fontsize=10,
                             fontweight='bold')

            self.ax.set_ylim(min(display_scores) - 10, 110)
            self.ax.set_xticks(x_values)
            self.ax.set_xticklabels([f"第{i}次" for i in x_values])

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()


# ======================================================================
#            最终、完整且经过验证的 TestPage 类 (请用此代码完整替换)
# ======================================================================
class TestPage(QWidget):
    """
    The main testing page, featuring a multi-step process with voice questions
    and score visualization. Refactored for better UI/UX and style consistency.
    """

    def __init__(self):
        super().__init__()       
        import yaml
        import random
        from yaml import FullLoader
        with open("questionnaire.yaml", encoding="utf-8") as f:
            all_questions = yaml.load(f, Loader=FullLoader)

        questions_num = 5
        self.questions = []
        for _ in range(questions_num):
            self.questions.append(
                all_questions.pop(random.randint(0, len(all_questions) - 1))
            )
        # TTS队列
        # 记录已经朗读的题目
        self.spoken_questions = set()

        # 初始化 TTS
        self.tts_queue = Queue()
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)
        self.tts_engine.setProperty('volume', 1.0)

        # 后台线程处理朗读
        self.tts_thread = Thread(target=self._tts_loop, daemon=True)
        self.tts_thread.start()

        self.current_question = 0
        self.current_step = 0

        self.setAutoFillBackground(True)

        self._setup_properties()
        self._init_ui()
        self._connect_signals()

        self._setup_mic_button_animation()
        self.update_step_ui()

        self._dot_animations = []  # 用于保留动画对象，防止 GC

        self.load_history_scores()
        logger.info("TestPage 初始化完成。")

    def _tts_loop(self):
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS朗读失败: {e}")
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
        self.cap = None
        self.camera_timer = QTimer(self)
        # 会话与录制文件管理
        self.session_timestamp = None
        self.session_dir = None
        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None
        # 视频录制相关（单段/题目段）
        self._video_writer = None
        self._video_fps = 30
        self._video_size = None
        self._video_filepath = None
        # 当前登录用户名（默认匿名）
        self.current_user = 'anonymous'

        # 数据库记录ID
        self.row_id = None

        # 多模态数据采集相关
        self.multimodal_collector = None

        # 舒特测试结果实例属性（用于信号穿透保存）
        self.schulte_elapsed = None  # 用时（秒）
        self.schulte_accuracy = None  # 准确率（百分比）

        # 环节时间戳记录
        self.part_timestamps = []

    def _init_ui(self):
        """初始化用户界面。"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(30, 30, 30, 30)
        self.main_layout.setSpacing(20)

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
        self.camera_timer.timeout.connect(self._update_camera_frame)
        self.audio_timer.timeout.connect(self._process_audio)
        self.btn_next.clicked.connect(self._next_step_or_question)
        self.btn_finish.clicked.connect(self._finish_test)
        self.btn_mic.clicked.connect(self._toggle_recording)

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

    # --- UI 创建辅助方法 ---
    def _create_step_navigator(self):
        container = QWidget()
        container.setObjectName("card")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(20, 15, 20, 15)
        self.step_labels = []
        self.step_opacity_effects = []
        for i, step_name in enumerate(self.steps):
            widget, number_label, text_label = QWidget(), QLabel(str(i + 1)), QLabel(step_name)
            h_layout = QHBoxLayout(widget)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(6)
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
        layout.setContentsMargins(20, 10, 20, 10)

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
        self.content_layout.setSpacing(20)
        self.camera_widget = self._create_camera_view()
        self.content_layout.addWidget(self.camera_widget, 2)
        self.answer_stack = QStackedWidget()
        self._create_answer_area_widgets()
        self.content_layout.addWidget(self.answer_stack, 3)

        self.score_page = ScorePage(username=self.current_user)
        self.answer_stack.addWidget(self.score_page)

        return container

    def _create_camera_view(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.camera_label = QLabel('摄像头画面加载中...')
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(480, 360)
        self.camera_label.setObjectName("cameraView")
        layout.addWidget(self.camera_label)

        return widget

    def _create_answer_area_widgets(self):
        # 语音答题页面
        page_qna = QWidget()
        layout_qna = QVBoxLayout(page_qna)
        layout_qna.setAlignment(Qt.AlignCenter)
        layout_qna.setSpacing(20)

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

        # 血压仪设备
        self.maibobo_device = None

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
        try:
            # 尝试导入 maibobo 相关模块
            try:
                from serial.tools.list_ports import comports
                import serial

                # 检测可用的串口设备
                available_ports = []
                for port in comports():
                    if "USB" in port.description and "Serial" in port.description:
                        available_ports.append(port.name)

                if available_ports:
                    # 尝试连接第一None个可用端口
                    test_port = available_ports[0]
                    try:
                        # 尝试打开串口连接
                        ser = serial.Serial(test_port, timeout=1)
                        ser.close()

                        self.bp_status_label.setText(f"血压仪器已连接 ✅ (端口: {test_port})")
                        self.bp_status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                        self.bp_start_button.setEnabled(True)  # 启用开始按钮

                        # 保存可用端口信息
                        self.bp_available_port = test_port
                        return
                    except Exception as e:
                        logger.warning(f"端口 {test_port} 连接失败: {e}")

                # 如果没有找到合适的设备，显示错误信息
                self.bp_status_label.setText("血压仪器未连接或正在测试，请确认设备连接状态 📥")
                self.bp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self.bp_start_button.setEnabled(False)
                self.bp_available_port = None

            except ImportError:
                # 如果没有安装相关库，显示错误信息
                self.bp_status_label.setText("血压仪器库未安装，请安装相关依赖 ❌")
                self.bp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
                self.bp_start_button.setEnabled(False)
                self.bp_available_port = None

        except Exception as e:
            logger.error(f"检测血压仪器出错: {e}")
            self.bp_status_label.setText("设备检测失败 ❌")
            self.bp_status_label.setStyleSheet("color: #F44336; font-weight: bold;")
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
            if not hasattr(self, 'bp_available_port') or not self.bp_available_port:
                QMessageBox.warning(self, "设备错误", "血压仪器未连接，无法开始测试")
                return

            self.bp_test_running = True
            self.bp_start_button.setText("停止测试")
            self.bp_start_button.setObjectName("finishButton")
            self.bp_start_button.style().unpolish(self.bp_start_button)
            self.bp_start_button.style().polish(self.bp_start_button)

            # 重置进度
            self.bp_test_progress = 0
            self.bp_progress_label.setText("测试进行中...")
            self.bp_progress_circle.setText("0%")

            # 隐藏结果区域
            self.result_container.setVisible(False)

            # 启动测试进度定时器
            self.bp_test_timer.start(100)  # 每100ms更新一次进度

            # 启动真实设备测试
            self._start_real_bp_test()

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

            # 停止进度定时器
            self.bp_test_timer.stop()

            # 停止并释放设备
            if hasattr(self, 'maibobo_device') and self.maibobo_device:
                try:
                    self.maibobo_device.stop()
                    logger.info("血压仪设备已停止")
                except Exception as e:
                    logger.warning(f"停止血压仪设备时出错: {e}")
                finally:
                    self.maibobo_device = None

            # 更新进度显示
            self.bp_progress_label.setText("测试已停止")
            self.bp_progress_circle.setText("停止")

            logger.info("血压测试已停止")

        except Exception as e:
            logger.error(f"停止血压测试失败: {e}")

    def _start_real_bp_test(self):
        """启动真实血压测试"""
        try:
            logger.info(f"正在连接真实血压仪设备，端口: {self.bp_available_port}")

            # 检查 MaiboboDevice 是否可用
            if not HAS_MAIBOBO_BACKEND:
                raise ImportError("MaiboboDevice 类不可用，请检查相关依赖")

            # 根据可用性选择导入路径
            if HAS_MAIBOBO_BACKEND:
                from backend.devices.maibobo import MaiboboDevice

            # 创建 MaiboboDevice 实例
            self.maibobo_device = MaiboboDevice(port=self.bp_available_port, timeout=1)

            # 启动设备
            self.maibobo_device.start()

            # 启动一个后台线程来读取设备数据
            import threading
            self.bp_device_thread = threading.Thread(target=self._read_device_data, daemon=True)
            self.bp_device_thread.start()

            logger.info("真实血压仪设备已启动，开始读取数据")

        except Exception as e:
            logger.error(f"启动真实血压测试失败: {e}")
            QMessageBox.critical(self, "设备错误", f"启动血压仪失败: {e}")
            self._stop_bp_test()

    def _read_device_data(self):
        """读取真实设备数据（在后台线程中运行）"""
        try:
            logger.info("开始读取血压仪数据...")

            # 等待设备稳定
            time.sleep(1)

            # 持续读取数据直到获得有效结果或超时
            start_time = time.time()
            timeout = 60  # 60秒超时

            while time.time() - start_time < timeout and self.bp_test_running:
                try:
                    # 读取设备数据
                    ret, frame = self.maibobo_device.read()

                    if ret and frame is not None:
                        logger.info(f"读取到血压仪数据: {frame}")

                        # 解析数据 - 根据 maibobo 数据格式
                        if hasattr(frame, '__getitem__'):
                            # 如果是数组或类似数组的对象
                            if len(frame) >= 11:  # 确保有足够的数据
                                systolic = frame[8]  # 收缩压
                                diastolic = frame[10]  # 舒张压
                                pulse = frame[2]  # 脉搏
                            else:
                                # 如果数据不足，尝试其他索引
                                systolic = getattr(frame, 'systolic', frame[0] if len(frame) > 0 else 120)
                                diastolic = getattr(frame, 'diastolic', frame[1] if len(frame) > 1 else 80)
                                pulse = getattr(frame, 'pulse', frame[2] if len(frame) > 2 else 75)
                        else:
                            # 如果是对象，尝试获取属性
                            systolic = getattr(frame, 'systolic', getattr(frame, 'value', 120))
                            diastolic = getattr(frame, 'diastolic', getattr(frame, 'value', 80))
                            pulse = getattr(frame, 'pulse', getattr(frame, 'value', 75))

                        # 验证数据有效性
                        if (isinstance(systolic, (int, float)) and 60 <= systolic <= 200 and
                                isinstance(diastolic, (int, float)) and 40 <= diastolic <= 120 and
                                isinstance(pulse, (int, float)) and 40 <= pulse <= 150):

                            # 数据有效，保存结果
                            self.bp_results = {
                                'systolic': int(systolic),
                                'diastolic': int(diastolic),
                                'pulse': int(pulse)
                            }

                            logger.info(f"血压测试完成: 收缩压={systolic}, 舒张压={diastolic}, 脉搏={pulse}")

                            # 在主线程中更新UI
                            QTimer.singleShot(0, self._complete_bp_test)
                            return
                        else:
                            logger.warning(f"血压仪数据无效: 收缩压={systolic}, 舒张压={diastolic}, 脉搏={pulse}")

                    # 等待一段时间再读取
                    time.sleep(0.5)

                except Exception as e:
                    logger.warning(f"读取血压仪数据时出错: {e}")
                    time.sleep(1)

            # 如果超时或测试被停止
            if self.bp_test_running:
                logger.warning("血压测试超时，未获得有效数据")
                QTimer.singleShot(0, self._complete_bp_test)

        except Exception as e:
            logger.error(f"读取血压仪数据失败: {e}")
            QTimer.singleShot(0, self._complete_bp_test)

    def _complete_bp_test(self):
        """完成血压测试，显示结果"""
        try:
            # 停止测试
            self._stop_bp_test()

            # 显示结果
            if (hasattr(self, 'bp_results') and
                    self.bp_results and
                    self.bp_results.get('systolic') is not None):

                self.systolic_label.setText(f"收缩压: {self.bp_results['systolic']} mmHg")
                self.diastolic_label.setText(f"舒张压: {self.bp_results['diastolic']} mmHg")
                self.pulse_label.setText(f"脉搏: {self.bp_results['pulse']} 次/分")

                # 根据血压值设置颜色
                systolic = self.bp_results['systolic']
                diastolic = self.bp_results['diastolic']

                if systolic < 120 and diastolic < 80:
                    # 正常血压
                    color = "#4CAF50"
                elif systolic < 130 and diastolic < 85:
                    # 正常高值
                    color = "#FF9800"
                else:
                    # 高血压
                    color = "#F44336"

                self.systolic_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
                self.diastolic_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")

                # 显示结果区域
                self.result_container.setVisible(True)

                # 更新进度显示
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

                # 保存结果到数据库
                self._save_bp_results_to_db()

            else:
                # 没有有效数据，显示失败信息
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

                # 显示错误信息
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
            # 更新进度
            self.bp_test_progress += 0.1  # 每100ms增加0.1秒
            progress_percent = min(100, int((self.bp_test_progress / self.bp_test_duration) * 100))

            # 更新进度显示
            self.bp_progress_circle.setText(f"{progress_percent}%")

            # 更新进度条样式
            if progress_percent < 30:
                color = "#FF9800"  # 橙色
            elif progress_percent < 70:
                color = "#2196F3"  # 蓝色
            else:
                color = "#4CAF50"  # 绿色

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

            # 检查是否超时
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

            # 构建血压数据字符串
            blood_data = f"{self.bp_results['systolic']}/{self.bp_results['diastolic']}/{self.bp_results['pulse']}"

            # 保存到数据库
            if hasattr(self, 'row_id') and self.row_id and TestTableStore:
                try:
                    store = TestTableStore(host="localhost", user="root", password="123456", database="test")
                    store.update_values(row_id=self.row_id, blood=blood_data)
                    logger.info(f"血压测试结果已保存到数据库，记录ID: {self.row_id}")
                    logger.info(f"血压数据: {blood_data}")
                except Exception as e:
                    logger.error(f"保存血压测试结果到数据库失败: {e}")
            else:
                logger.warning("无法保存血压测试结果：缺少数据库记录ID或TestTableStore")

        except Exception as e:
            logger.error(f"保存血压测试结果失败: {e}")

    # 在 TestPage 类中增加这个方法
    # 舒尔特方格测试后门
    def keyPressEvent(self, event):
        """全局监听键盘事件，用于测试调试后门"""
        # 语音问答步骤后门
        if self.current_step == 0:
            if event.key() == Qt.Key_Q:  # 按下 Q 键
                logger.info("测试后门触发：按下 Q，语音问答视为完成")
                # 直接跳到血压测试步骤
                self.current_step = 1
                self.update_step_ui()
                return
        # 血压测试步骤后门
        elif self.current_step == 1:
            if event.key() == Qt.Key_Q:  # 按下 Q 键
                logger.info("测试后门触发：按下 Q，血压测试视为完成")
                # 模拟血压测试结果
                self.bp_results = {
                    'systolic': 120,  # 收缩压
                    'diastolic': 80,  # 舒张压
                    'pulse': 75  # 脉搏
                }
                # 显示结果
                self._complete_bp_test()
                # 更新UI状态，确保"下一题"按钮可用
                self.update_step_ui()
                return
        # 舒尔特方格测试后门
        elif self.current_step == 2:
            if event.key() == Qt.Key_Q:  # 按下 Q 键
                logger.info("测试后门触发：按下 Q，舒尔特测试视为完成")
                # 使用默认值触发舒尔特结果处理
                self._on_schulte_result(30.0, 85.0)  # 默认30秒用时，85%准确率
                return
        # 保持原有键盘事件功能
        super().keyPressEvent(event)

    def _create_schulte_page(self):
        """创建舒特格测试页面"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        # 创建舒特格测试控件（注入当前用户名）
        self.schulte_widget = SchulteGridWidget(self.current_user)

        # 连接舒特格测试完成信号到下一步逻辑
        self.schulte_widget.test_completed.connect(self._on_schulte_completed)

        # 连接舒特结果信号至实例槽，保存用时与准确率
        self.schulte_widget.test_result_ready.connect(self._on_schulte_result)

        layout.addWidget(self.schulte_widget)

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
        self.btn_finish.setVisible(False)  # 初始时隐藏评估按钮
        layout.addWidget(self.btn_next)
        layout.addWidget(self.btn_finish)
        return container

    # --- UI 更新逻辑 ---
    def update_step_ui(self):
        for i, (num_label, text_label) in enumerate(self.step_labels):
            # 动画透明度
            target_opacity = 1.0 if i == self.current_step else 0.5
            anim = QPropertyAnimation(self.step_opacity_effects[i], b"opacity")
            anim.setDuration(400)
            anim.setStartValue(self.step_opacity_effects[i].opacity())
            anim.setEndValue(target_opacity)
            anim.setEasingCurve(QEasingCurve.InOutQuad)
            anim.start(QPropertyAnimation.DeleteWhenStopped)

            # 样式切换
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
                    # 已完成题目使用 mark_question_done 替换
                    self.mark_question_done(i)
                elif i == self.current_question:
                    # 当前题目黑点
                    icon = qta.icon('fa5s.circle', color='#212121')
                    dot.setPixmap(icon.pixmap(24, 24))
                    dot.setAlignment(Qt.AlignCenter)
                else:
                    # 未到题目黑点
                    icon = qta.icon('fa5s.circle', color='#212121')
                    dot.setPixmap(icon.pixmap(24, 24))
                    dot.setAlignment(Qt.AlignCenter)
        # 摄像头只在语音答题时显示
        self.camera_widget.setVisible(self.current_step == 0)

        # 音频电平只在语音答题时显示和更新
        if self.current_step == 0:
            # 确保音频定时器在语音答题步骤运行
            if not self.audio_timer.isActive():
                self.audio_timer.start(50)
        else:
            # 在其他步骤停止音频定时器
            if self.audio_timer.isActive():
                self.audio_timer.stop()
                self.audio_level.set_level(0)  # 重置音频电平显示

        # 根据当前步骤显示不同的页面
        if self.current_step == 0:  # 语音答题
            self.answer_stack.setCurrentIndex(0)
            self.lbl_question.setText(self.questions[self.current_question])
            self.btn_next.setText(
                "下一题" if self.current_question < len(self.questions) - 1 else "完成答题"
            )
            self.btn_next.setVisible(True)
            self.btn_next.setEnabled(False)
            self.btn_finish.setVisible(False)
            # **每次刷新题目时朗读**
            self._speak_current_question()

        elif self.current_step == 1:  # 血压测试
            self.answer_stack.setCurrentIndex(1)
            # 检查血压测试是否完成
            if hasattr(self, 'bp_results') and self.bp_results['systolic'] is not None:
                self.btn_next.setText("进入舒特格测试")
                self.btn_next.setEnabled(True)
            else:
                self.btn_next.setText("请先完成血压测试")
                self.btn_next.setEnabled(False)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
        elif self.current_step == 2:  # 舒特格测试
            self.answer_stack.setCurrentIndex(2)
            self.btn_next.setVisible(False)  # 隐藏下一步按钮，由舒特格测试控件自己管理
            self.btn_finish.setVisible(False)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
        elif self.current_step == 3:  # 分数展示
            # self.answer_stack.setCurrentIndex(4)  # 分数页面是第5个页面（索引4）
            self.answer_stack.setCurrentWidget(self.score_page)
            self.score_page._set_user(self.current_user)
            self.score_page._update_scores()  # 更新分数
            # 直接显示完成评估按钮，隐藏下一题按钮
            self.btn_next.setVisible(False)
            self.btn_finish.setVisible(True)
            if self.mic_anim.state() == QPropertyAnimation.Running:
                self.mic_anim.stop()
            self.score_value_label.setText(str(self.score) if self.score is not None else "计算中...")
            self.score_chart.update_chart(self.history_scores)

    # --- 核心功能方法 (之前遗漏的部分) ---
    def start_test(self):
        logger.info("TestPage.start_test() 已被调用。")
        # 预览刷新由 AV 采集器提供当前帧
        self.camera_timer.start(30)
        # 启动音频电平更新定时器
        self.audio_timer.start(50)
        self.current_step = 0
        self.current_question = 0
        self.btn_finish.setVisible(False)

        # 初始化已经朗读的题目集合
        self.spoken_questions = set()

        # 记录时间戳
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)

        # 初始化语音答题会话目录
        try:
            self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = 'recordings'
            user_dir = self.current_user or 'anonymous'
            if build_store_dir:
                self.session_dir = build_store_dir(base_dir, user_dir, self.session_timestamp)
            else:
                self.session_dir = os.path.join(base_dir, user_dir, self.session_timestamp)
                os.makedirs(self.session_dir, exist_ok=True)
            logger.info(f"语音答题会话目录: {self.session_dir}")
        except Exception as e:
            logger.error(f"创建会话目录失败: {e}")
            self.session_dir = 'recordings'
            os.makedirs(self.session_dir, exist_ok=True)

        # 重置路径收集
        self._audio_paths = []
        self._video_paths = []
        self._current_audio_target = None
        self._current_video_target = None

        self.update_step_ui()
        self._speak_current_question()  # 朗读第一题

        # 启动 AV 采集器（仅采集预览，不录制）
        try:
            av_start_collection(save_dir=self.session_dir, camera_index=1, video_fps=30.0, input_device_index=2)
        except Exception as e:
            logger.error(f"启动 AV 采集器失败: {e}")

        # 启动多模态数据采集
        if HAS_MULTIMODAL:
            try:
                # print('第一个保存路径:', self.session_dir)
                # 使用与音视频相同的会话目录，并启用实时显示窗口
                self.multimodal_collector = multidata_start_collection(
                    self.current_user,
                    part=1,
                    save_dir=self.session_dir,
                    enable_display=True,
                    display_title=f"多模态数据采集 - {self.current_user}",
                    display_width=320,
                    display_height=180
                )
                if self.multimodal_collector:
                    logger.info(f"多模态数据采集已启动，用户: {self.current_user}")
                    logger.info(f"多模态数据保存目录: {self.session_dir}")
                    logger.info("实时显示窗口已启用")
                    # 多模态小窗口定位到左上角（允许用户拖动）
                    QTimer.singleShot(300, self._position_multimodal_window)
                else:
                    logger.warning("多模态数据采集启动失败")
            except Exception as e:
                logger.error(f"启动多模态数据采集时出错: {e}")

        # 启动脑电EEG采集（与多模态统一目录）
        try:
            from get_eegdata import start as eeg_start
            eeg_start(save_dir=self.session_dir)
            logger.info(f"EEG采集已启动，保存目录: {self.session_dir}")
        except Exception as e:
            logger.error(f"启动EEG采集失败: {e}")

    def _update_camera_frame(self):
        try:
            frame = av_get_current_frame()
            if frame is None:
                return
            if self.camera_widget.isVisible():
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(self.camera_label.size(), Qt.KeepAspectRatio,
                                                            Qt.SmoothTransformation)
                self.camera_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"更新摄像头显示失败: {e}")

    def _start_video_recording(self, target_path: str = None):
        # 交给 AV 采集器统一开始音视频录制
        try:
            av_start_recording()
        except Exception as e:
            logger.error(f"开始音视频录制失败: {e}")

    def _stop_video_recording(self):
        # print(self._audio_paths)
        try:
            av_stop_recording()
            # 同步所有段落路径
            self._audio_paths = av_get_audio_paths()
            self._video_paths = av_get_video_paths()
             # 异步入队识别，不阻塞录音与下一题
            if HAS_SPEECH_RECOGNITION:
                try:
                    add_audio_for_recognition(
                        self._audio_paths[-1],
                        len(self._audio_paths),
                        self.questions[len(self._audio_paths)-1]
                    )
                except Exception as e:
                    print(f"加入语音识别队列失败: {e}")
        except Exception as e:
            logger.error(f"停止音视频录制失败: {e}")

    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        """Handles the logic for starting a recording."""
        if self.mic_anim.state() == QPropertyAnimation.Running:
            self.mic_anim.stop()
        self.mic_shadow.setEnabled(False)
        self.is_recording = True

        self.btn_mic.setObjectName("micButtonRecording")
        self.btn_mic.setIcon(qta.icon('fa5s.stop', color='white'))

        # 强制Qt重新应用样式表
        self.btn_mic.style().unpolish(self.btn_mic)
        self.btn_mic.style().polish(self.btn_mic)

        self.lbl_recording_status.setText("正在录音...")
        logger.info("开始音视频录制...")

        # 启动音频电平更新定时器
        self.audio_timer.start(50)  # 每50ms更新一次音频电平

        # 同时开始音视频录制
        self._start_video_recording()

    # (在 TestPage 类中, 替换此方法)
    def _stop_recording(self):
        """
        Handles the logic for stopping a recording, with a temporary red "stopped" feedback.
        """
        if not self.is_recording: return
        self.is_recording = False
        self.audio_timer.stop()
        # 停止当前题目的音视频录制
        self._stop_video_recording()

        # 立即将按钮变为 "已停止" 的红色状态
        self.btn_mic.setObjectName("micButtonStopped")  # 使用新的临时样式名
        self.btn_mic.setIcon(qta.icon('fa5s.check', color='white'))  # 显示对勾
        self.btn_mic.style().unpolish(self.btn_mic)
        self.btn_mic.style().polish(self.btn_mic)
        self.mic_shadow.setEnabled(False)

        self.btn_next.setEnabled(True)
        self.lbl_recording_status.setText("录制已完成，请进入下一题")
        logger.info("音视频录制完毕。")
        self.audio_level.set_level(0)

        # 延迟后, 将按钮恢复到初始的蓝色可点击状态
        def restore_button():
            self.mic_shadow.setEnabled(True)
            self.btn_mic.setObjectName("micButtonCallToAction")
            self.btn_mic.setIcon(qta.icon('fa5s.microphone-alt', color='white'))
            self.btn_mic.style().unpolish(self.btn_mic)
            self.btn_mic.style().polish(self.btn_mic)

            if self.mic_anim.state() != QPropertyAnimation.Running:
                self.mic_anim.start()

        QTimer.singleShot(1000, restore_button)  # 1秒延迟

    def _process_audio(self):
        """从AVCollector获取音频电平并更新UI显示"""
        try:
            # 从AVCollector获取实时音频电平
            level = av_get_current_audio_level()
            self.audio_level.set_level(level)
        except Exception as e:
            logger.warning(f"获取音频电平时发生错误: {e}")
            # 如果获取失败，将电平设为0
            self.audio_level.set_level(0)

    def _speak_current_question(self):
        if self.current_question not in self.spoken_questions:
            self.spoken_questions.add(self.current_question)
            self.tts_queue.put(self.questions[self.current_question])

    def _next_step_or_question(self):
        if self.current_step == 0:  # 语音答题
            if self.current_question < len(self.questions) - 1:
                self.current_question += 1
                self.update_step_ui()
                self._speak_current_question()
            else:
                self.current_step += 1
                # 记录时间戳
                call_timestamp = time.time()
                self.part_timestamps.append(call_timestamp)
                try:
                    self._close_camera()
                except Exception as e:
                    logger.warning(f"关闭摄像头失败: {e}")
                try:
                    multidata_stop_collection()
                except Exception as e:
                    logger.warning(f"停止多模态采集器失败: {e}")
                self.update_step_ui()
                self.row_id = self._persist_av_paths_to_db()
        elif self.current_step == 1: # 血压测试
            # 记录时间戳
            call_timestamp = time.time()
            self.part_timestamps.append(call_timestamp)
            # 启动多模态数据采集
            if HAS_MULTIMODAL:
                try:
                    # print('第二个保存路径:', self.session_dir)
                    # 使用与音视频相同的会话目录，并启用实时显示窗口
                    self.multimodal_collector = multidata_start_collection(
                        self.current_user,
                        part=2,
                        save_dir=self.session_dir,
                        enable_display=True,
                        display_title=f"多模态数据采集 - {self.current_user}",
                        display_width=320,
                        display_height=180
                    )
                    if self.multimodal_collector:
                        logger.info(f"多模态数据采集已启动，用户: {self.current_user}")
                        logger.info(f"多模态数据保存目录: {self.session_dir}")
                        logger.info("实时显示窗口已启用")
                        # 多模态小窗口定位到左上角（允许用户拖动）
                        QTimer.singleShot(300, self._position_multimodal_window)
                    else:
                        logger.warning("多模态数据采集启动失败")
                except Exception as e:
                    logger.error(f"启动多模态数据采集时出错: {e}")
            self.current_step += 1
            self.update_step_ui()

    def _on_schulte_completed(self):
        """舒特格测试完成，自动进入下一步"""
        logger.info("舒特格测试完成，自动进入分数展示页面")
        # 记录时间戳
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)
        try:
            multidata_stop_collection()
        except Exception as e:
            logger.warning(f"停止多模态采集器失败: {e}")
        self.current_step += 1
        if self.current_step == 3:
            self.save_score()
        self.update_step_ui()

    def _finish_test(self):
        if HAS_MULTIMODAL:
            try:
                multidata_stop_collection()
                self.multimodal_collector = None
                logger.info("多模态数据采集已停止")
                self._persist_multimodal_paths_to_db()
                from get_multidata import cleanup_collector
                cleanup_collector()
            except Exception as e:
                logger.error(f"停止多模态数据采集时出错: {e}")
        # 记录时间戳
        call_timestamp = time.time()
        self.part_timestamps.append(call_timestamp)
        # 保存调用时间戳数据
        if self.part_timestamps:
            import json
            # print(self.part_timestamps)
            # 同时保存为JSON格式，便于查看
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
        QTimer.singleShot(2000, self._auto_close_page)

    def _auto_close_page(self):
        """自动关闭页面"""
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

    def hideEvent(self, event):
        super().hideEvent(event)
        self.camera_timer.stop()
        self.audio_timer.stop()
        self._stop_video_recording()
        if self.is_recording: self._stop_recording()

        # 清理血压仪设备
        if hasattr(self, 'maibobo_device') and self.maibobo_device:
            try:
                self.maibobo_device.stop()
                self.maibobo_device = None
                logger.info("页面隐藏时已停止血压仪设备")
            except Exception as e:
                logger.warning(f"页面隐藏时停止血压仪设备失败: {e}")

        if hasattr(self, 'schulte_widget'):
            self.schulte_widget.reset_for_next_stage()

        if HAS_MULTIMODAL:
            try:
                multidata_stop_collection()
                self.multimodal_collector = None
                logger.info("页面隐藏时已停止多模态数据采集")
                self._persist_multimodal_paths_to_db()
                from get_multidata import cleanup_collector
                cleanup_collector()
            except Exception as e:
                logger.error(f"页面隐藏时停止多模态数据采集失败: {e}")

        # 停止脑电EEG采集并写库
        try:
            from get_eegdata import stop as eeg_stop, get_file_paths as eeg_paths
            eeg_stop()
            paths = eeg_paths()
            if paths:
                self._persist_eeg_paths_to_db(paths)
                logger.info("页面隐藏时已停止EEG采集并写入数据库")
        except Exception as e:
            logger.error(f"页面隐藏时停止EEG采集失败: {e}")

        # 舒尔特结果写库
        try:
            if self.schulte_accuracy and self.schulte_elapsed:
                self._on_schulte_result(self.schulte_elapsed, self.schulte_accuracy)
                logger.info("舒尔特结果写入数据库")
        except Exception as e:
            logger.error(f"舒尔特结果写入数据库失败: {e}")

    def _close_camera(self):
        try:
            self.camera_timer.stop()
            try:
                av_stop_recording()
            except Exception:
                pass
            try:
                av_stop_collection()
            except Exception:
                pass
            self.cap = None
            logger.info("语音答题环节结束，已停止 AV 采集器")
        except Exception as e:
            logger.warning(f"关闭摄像头时出现问题: {e}")

    def _persist_av_paths_to_db(self):
        """将音视频绝对路径和语音识别结果写入 test 表（video/audio/record 为 JSON 列表）。"""
        try:
            if not TestTableStore:
                logger.warning("TestTableStore 未可用，跳过数据库写入。")
                return None
            store = TestTableStore(host="localhost", user="root", password="123456", database="test")
            row_id = store.insert_row(
                name=self.current_user or 'anonymous',
                video=list(self._video_paths),
                audio=list(self._audio_paths),
            )
            logger.info(f"音视频路径已写入数据库 test 表，记录ID: {row_id}。")
            return row_id
        except Exception as e:
            logger.error(f"写入数据库失败: {e}")
            return None

    def _persist_multimodal_paths_to_db(self):
        """将多模态数据文件路径写入数据库"""
        try:
            if not HAS_MULTIMODAL or not TestTableStore:
                logger.warning("多模态数据采集或TestTableStore未可用，跳过数据库写入。")
                return

            from get_multidata import get_multimodal_file_paths
            file_paths = get_multimodal_file_paths()

            if not file_paths:
                logger.warning("未获取到多模态数据文件路径")
                return

            # logger.info(f"获取到多模态数据文件路径: {file_paths}")

            if not self.row_id:
                logger.warning("没有有效的数据库记录ID，无法更新多模态数据路径")
                return

            store = TestTableStore(host="localhost", user="root", password="123456", database="test")

            try:
                # 同时追加语音识别结果到 record 字段
                record_payload = None
                try:
                    record_payload = get_recognition_results()
                    logger.info(f"将 {len(record_payload or [])} 条语音识别结果写入 record 字段")
                    clear_recognition_results()
                except Exception as e:
                    logger.warning(f"获取语音识别结果失败: {e}")

                record_txt = os.path.join(self.session_dir,'emotion', "record.txt")
                with open(record_txt, 'w') as f:
                    f.write(str(record_payload))

                store.update_values(
                    row_id=self.row_id,
                    rgb=file_paths.get('rgb', ''),
                    depth=file_paths.get('depth', ''),
                    tobii=file_paths.get('eyetrack', ''),
                    record_text=record_payload
                )
                logger.info(f"多模态数据文件路径已更新到数据库记录ID: {self.row_id}")
            except Exception as e:
                logger.error(f"更新多模态数据路径到数据库记录ID: {self.row_id} 失败")

        except Exception as e:
            logger.error(f"写入多模态数据路径到数据库失败: {e}")

    def _persist_eeg_paths_to_db(self, eeg_paths: dict):
        """将EEG文件路径写入数据库的 brain 字段（JSON）。"""
        try:
            if not TestTableStore:
                logger.warning("TestTableStore 未可用，跳过EEG数据库写入。")
                return
            if not self.row_id:
                # 如果还未插入过记录，则先插入一条主记录以获取 row_id
                store = TestTableStore(host="localhost", user="root", password="123456", database="test")
                self.row_id = store.insert_row(
                    name=self.current_user or 'anonymous',
                    # score将在舒尔特测试完成后与accuracy和elapsed一起存储
                )
            store = TestTableStore(host="localhost", user="root", password="123456", database="test")
            # print(eeg_paths)
            store.update_values(row_id=self.row_id, eeg1=eeg_paths['ch1_txt'], eeg2=eeg_paths['ch2_txt'])
            logger.info(f"EEG路径已更新到数据库记录ID: {self.row_id}")
        except Exception as e:
            logger.error(f"写入EEG路径到数据库失败: {e}")

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
        if not os.path.exists(SCORES_CSV_FILE): return
        try:
            with open(SCORES_CSV_FILE, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) >= 2: self.history_scores.append(int(row[1]))
        except Exception as e:
            logger.error(f"读取历史分数时出错: {e}")

    def set_current_user(self, username: str):
        """由主窗口在登录后调用，设置当前用户，并同步到舒特格控件"""
        self.current_user = username or 'anonymous'
        if hasattr(self, 'schulte_widget') and self.schulte_widget:
            try:
                self.schulte_widget.set_username(self.current_user)
            except Exception as e:
                logger.warning(f"同步用户名到舒特格控件失败: {e}")

    # Edited by Wyy: 舒特结果槽函数，保存结果到实例属性，并按需后续入库
    def _on_schulte_result(self, elapsed_seconds: float, accuracy_percent: float):
        try:
            self.schulte_elapsed = float(elapsed_seconds)
            self.schulte_accuracy = float(accuracy_percent)

            # 基于准确率和用时计算score
            # 计算方法：准确率权重70%，用时权重30%
            # 用时越短分数越高，基准用时30秒
            time_score = max(0, min(100, 100 - (self.schulte_elapsed - 30) * 2))  # 30秒为满分，每多1秒扣2分
            accuracy_score = self.schulte_accuracy  # 准确率直接作为分数
            self.score = int(accuracy_score * 0.7 + time_score * 0.3)

            logger.info(f"舒特结果: 用时={self.schulte_elapsed:.2f}s, 准确率={self.schulte_accuracy:.1f}%, 计算得分={self.score}")

            ptime = os.path.abspath(self.session_dir)
            ptime = os.path.join(ptime, 'eeg', 'part_timestamps.txt')

            # 立即写库，判断 self.row_id 并调用 TestTableStore.update_values
            if hasattr(self, 'row_id') and self.row_id and TestTableStore:
                try:
                    store = TestTableStore(host="localhost", user="root", password="123456", database="test")
                    store.update_values(
                        row_id=self.row_id,
                        accuracy=self.schulte_accuracy,
                        elapsed=self.schulte_elapsed,
                        score=self.score,
                        ptime=ptime
                    )
                    logger.info(f"舒特结果和分数已保存到数据库记录ID: {self.row_id}")
                except Exception as e:
                    logger.error(f"保存舒特结果到数据库失败: {e}")
        except Exception as e:
            logger.warning(f"处理舒特结果信号失败: {e}")

    # Edited by Wyy: 定位多模态小窗口到左上角（通过窗口标题查找）
    def _position_multimodal_window(self):
        try:
            app = QApplication.instance()
            if not app:
                return
            title_prefix = "多模态数据采集 - "
            for w in app.topLevelWidgets():
                try:
                    # 仅移动目标小窗口，不改变其可拖动行为
                    if hasattr(w, 'windowTitle') and w.windowTitle() and w.windowTitle().startswith(title_prefix):
                        w.move(0, 0)
                        logger.info("已将多模态小窗口移动至左上角")
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"定位多模态窗口失败: {e}")


class MainWindow(QMainWindow):
    """应用程序主窗口，管理所有页面。"""

    def __init__(self):
        super().__init__()
        self._setup_main_window()
        self._create_pages()
        self._connect_signals()

        logger.info("应用程序主窗口初始化完成。")

    def _setup_main_window(self):
        """设置主窗口的标题、大小和图标。"""
        self.setWindowTitle('非接触人员状态评估系统')
        self.setGeometry(100, 100, 1280, 800)
        # 使用qtawesome的默认图标
        self.setWindowIcon(qta.icon('fa5s.robot', color='blue'))

    def _create_pages(self):
        """创建并添加所有页面到FadingStackedWidget，并添加脑负荷条与提示语"""
        self.stack = FadingStackedWidget()
        self.stack.set_animation_duration(400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 页面内容
        self.login_page = LoginPage(self.show_calibration_page)
        self.calibration_page = CalibrationPage()
        self.test_page = TestPage()

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.calibration_page)
        self.stack.addWidget(self.test_page)

        main_layout.addWidget(self.stack, 1)

        # 脑负荷条和提示语
        self.brain_load_bar = BrainLoadBar()
        self.brain_load_bar.setVisible(False)

        self.brain_load_tip = QLabel("本测试需要全程采集您的脑电信号来进行脑负荷测试")
        self.brain_load_tip.setAlignment(Qt.AlignCenter)
        self.brain_load_tip.setWordWrap(True)
        self.brain_load_tip.setStyleSheet("color: #666; font-size: 16px;")
        main_layout.addWidget(self.brain_load_tip, 0, Qt.AlignBottom)
        main_layout.addWidget(self.brain_load_bar, 0, Qt.AlignBottom)

        self.stack.setCurrentWidget(self.login_page)

    def _connect_signals(self):
        """连接页面之间的信号和槽。"""
        self.calibration_page.calibration_finished.connect(self.show_test_page)

    def show_calibration_page(self, username: str):
        """从登录页切换到校准页，并记录当前用户。"""
        logger.info("正在切换到校准页面...")
        self.current_user = username or 'anonymous'
        try:
            self.test_page.set_current_user(self.current_user)
        except Exception as e:
            logger.warning(f"同步用户名到测试页失败: {e}")
        self.stack.fade_to_index(1)

    def show_test_page(self):
        """从校准页切换到测试页，并显示脑负荷条"""
        logger.info("正在切换到测试页面...")
        self.stack.fade_to_index(2)

        self.brain_load_tip.setVisible(False)
        self.brain_load_bar.setVisible(True)

        self.test_page.start_test()

    def closeEvent(self, event):
        """关闭窗口前的清理工作。"""
        logger.info("应用程序正在关闭...")

        # 停止 AV 采集器
        try:
            av_stop_recording()
        except Exception:
            pass
        try:
            av_stop_collection()
        except Exception:
            pass

        if HAS_MULTIMODAL:
            try:
                multidata_stop_collection()
                logger.info("应用程序关闭时已停止多模态数据采集")
                from get_multidata import cleanup_collector
                cleanup_collector()
            except Exception as e:
                logger.error(f"应用程序关闭时停止多模态数据采集失败: {e}")

        # 清理血压仪设备
        if hasattr(self, 'test_page') and hasattr(self.test_page, 'maibobo_device'):
            try:
                if self.test_page.maibobo_device:
                    self.test_page.maibobo_device.stop()
                    logger.info("应用程序关闭时已停止血压仪设备")
            except Exception as e:
                logger.warning(f"应用程序关闭时停止血压仪设备失败: {e}")

        if hasattr(self, 'test_page') and hasattr(self.test_page, 'schulte_widget'):
            self.test_page.schulte_widget.reset_for_next_stage()

        SchulteGridWidget.cleanup_temp_files()
        super().closeEvent(event)


# --- 应用程序入口 ---
def main():
    """主函数，启动应用程序。"""
    app = QApplication(sys.argv)

    if HAS_MULTIMODAL:
        app.aboutToQuit.connect(lambda: multidata_stop_collection())
        from get_multidata import cleanup_collector
        app.aboutToQuit.connect(lambda: cleanup_collector())

    try:
        with open("style.qss", "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    except FileNotFoundError:
        logger.warning("样式表文件未找到，使用默认样式")

    logger.info("应用程序启动。")
    win = MainWindow()
    win.showFullScreen()
    # win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
