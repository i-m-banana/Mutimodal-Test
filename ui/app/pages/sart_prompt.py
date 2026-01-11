"""SART experiment prompt page - guides user to start SART test."""

import os
from pathlib import Path

from .. import config
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap

from ..utils.responsive import scale


class SARTPromptPage(QWidget):
    """SART实验提示页面（用整张图片替代中间内容）- 引导用户开始测试"""
    
    # 信号：点击开始按钮
    start_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        page_layout = QVBoxLayout(self)
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

        # 顶部标题（字体更大、单行显示）
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

        # 中间图片部分
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("border: none; background-color: transparent;")

        # 使用相对路径（相对于项目根目录）
        image_path = str(config.BASE_DIR / "assets" / "sart.png")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                # 按比例缩放，尽量填满中间区域（保留边距）
                scaled_pixmap = pixmap.scaled(
                    scale(1000), scale(600),  # 根据窗口大小调整
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
            else:
                image_label.setText("⚠️ 图片加载失败")
                image_label.setStyleSheet("font-size: 24px; color: #e74c3c;")
        else:
            image_label.setText("⚠️ 未找到图片: ui/assets/sart.png")
            image_label.setStyleSheet("font-size: 24px; color: #e74c3c;")

        # 添加大图并让它扩展空间
        layout.addStretch(1)
        layout.addWidget(image_label, 1, Qt.AlignCenter)
        layout.addStretch(1)

        # 大按钮部分（与其他页面统一）
        self.btn_start = QPushButton("我已了解规则，开始测试")
        self.btn_start.setObjectName("sartPrimaryButton")
        self.btn_start.setFixedSize(scale(500), scale(80))  # 大按钮
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
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
        self.btn_start.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn_start, 0, Qt.AlignCenter)
        layout.addStretch(1)

        page_layout.addWidget(content_frame)
