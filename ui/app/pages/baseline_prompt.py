"""Baseline calibration prompt page - guides user to start baseline."""

from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor

from ..utils.responsive import scale


class BaselinePromptPage(QWidget):
    """基线校准提示页面（带白色圆角外框）- 引导用户点击开始"""
    
    # 信号：点击开始按钮
    start_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(scale(20), scale(20), scale(20), scale(20))
        page_layout.setSpacing(scale(12))

        # 创建白色圆角矩形容器（外框 - 增加圆角和阴影）
        content_frame = QFrame()
        content_frame.setObjectName("baselinePromptFrame")
        content_frame.setStyleSheet("""
            QFrame#baselinePromptFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
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

        # 标题（更大字号，确保单行显示）
        title_label = QLabel("请注视屏幕中央的十字，进行30s静息基线采集。")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(False)  # 禁止换行
        title_label.setStyleSheet("""
            color: #2c3e50;
            font-size: 50px;
            font-weight: bold;
            padding: 20px;
        """)
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

        # 开始按钮（尺寸加大，与校准页面统一）
        self.btn_start = QPushButton("点击开始")
        self.btn_start.setObjectName("primaryButton")
        self.btn_start.setFixedSize(scale(280), scale(80))
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.setStyleSheet("""
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
        """)
        self.btn_start.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn_start, 0, Qt.AlignCenter)

        # 添加底部一点空间
        layout.addStretch(1)

        page_layout.addWidget(content_frame, 1)
