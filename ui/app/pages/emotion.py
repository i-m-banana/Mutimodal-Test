"""Emotion detection page with text reading and recording."""

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFrame,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QFont, QColor
import qtawesome as qta

from ..utils.widgets import AudioLevelMeter
from ..utils.responsive import scale


class EmotionDetectionPage(QWidget):
    
    recording_requested = pyqtSignal()
    next_clicked = pyqtSignal()
    
    def __init__(self, reading_text: str, parent=None):
        super().__init__(parent)
        self.reading_text_content = reading_text
        self._init_ui()
    
    def _init_ui(self):
        self.setStyleSheet("QWidget { background-color: transparent; }")
        layout_qna = QVBoxLayout(self)
        layout_qna.setAlignment(Qt.AlignCenter)
        layout_qna.setSpacing(scale(20))
        layout_qna.setContentsMargins(scale(20), scale(20), scale(20), scale(20))

        card_container = QFrame()
        card_container.setObjectName("emotionCardContainer")
        card_container.setMaximumWidth(scale(1200))
        card_container.setStyleSheet("""
            QFrame#emotionCardContainer {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                padding: 20px;
            }
        """)
        
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(15)
        card_shadow.setXOffset(0)
        card_shadow.setYOffset(5)
        card_shadow.setColor(QColor(0, 0, 0, 60))
        card_container.setGraphicsEffect(card_shadow)
        
        card_layout = QVBoxLayout(card_container)
        card_layout.setSpacing(scale(20))
        card_layout.setContentsMargins(scale(15), scale(15), scale(15), scale(15))

        title_label = QLabel("📖 请朗读以下文本")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(80)
        title_font.setBold(True)
        title_label.setFont(title_font)
        card_layout.addWidget(title_label)

        self.lbl_reading_text = QTextEdit()
        self.lbl_reading_text.setReadOnly(True)
        self.lbl_reading_text.setObjectName("readingTextDisplay")
        self.lbl_reading_text.setMinimumWidth(scale(900))
        self.lbl_reading_text.setMinimumHeight(scale(500))
        self.lbl_reading_text.setMaximumHeight(scale(500))
        
        text_font = QFont()
        text_font.setPointSize(28)
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
        
        self.lbl_reading_text.setPlainText(self.reading_text_content)
        self.lbl_reading_text.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        
        card_layout.addWidget(self.lbl_reading_text, 0, Qt.AlignCenter)

        control_container = QWidget()
        control_layout = QHBoxLayout(control_container)
        control_layout.setSpacing(scale(40))
        control_layout.setAlignment(Qt.AlignCenter)
        
        left_control = QWidget()
        left_layout = QVBoxLayout(left_control)
        left_layout.setSpacing(scale(15))
        left_layout.setAlignment(Qt.AlignCenter)

        self.btn_mic = QPushButton()
        self.btn_mic.setObjectName("micButtonCallToAction")
        self.btn_mic.setFixedSize(130, 130)
        self.btn_mic.setIconSize(QSize(60, 60))
        self.btn_mic.setCursor(Qt.PointingHandCursor)
        self.btn_mic.setIcon(qta.icon('fa5s.microphone-alt', color='white'))
        self.btn_mic.clicked.connect(self.recording_requested.emit)

        self.lbl_recording_status = QLabel("点击录音按钮开始朗读并录音")
        self.lbl_recording_status.setObjectName("statusLabel")
        self.lbl_recording_status.setAlignment(Qt.AlignCenter)
        status_font = QFont()
        status_font.setPointSize(16)
        self.lbl_recording_status.setFont(status_font)
        
        self.audio_level = AudioLevelMeter()
        self.audio_level.setFixedWidth(350)

        left_layout.addWidget(self.btn_mic, 0, Qt.AlignCenter)
        left_layout.addWidget(self.lbl_recording_status, 0, Qt.AlignCenter)
        left_layout.addWidget(self.audio_level, 0, Qt.AlignCenter)
        
        right_control = QWidget()
        right_layout = QVBoxLayout(right_control)
        right_layout.setSpacing(scale(10))
        right_layout.setAlignment(Qt.AlignCenter)
        
        self.btn_next = QPushButton("完成录音")
        self.btn_next.setObjectName("successButton")
        self.btn_next.setIcon(qta.icon('fa5s.arrow-right'))
        self.btn_next.setFixedSize(scale(280), scale(90))
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
        self.btn_next.clicked.connect(self.next_clicked.emit)
        self.btn_next.setEnabled(False)
        right_layout.addWidget(self.btn_next, 0, Qt.AlignCenter)

        control_layout.addWidget(left_control)
        control_layout.addWidget(right_control)
        
        card_layout.addWidget(control_container)
        
        layout_qna.addStretch(1)
        layout_qna.addWidget(card_container, 0, Qt.AlignCenter)
        layout_qna.addStretch(1)
