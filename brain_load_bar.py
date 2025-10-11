# brain_load_bar.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QPainter, QBrush, QLinearGradient, QColor
import random

class BrainLoadBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(50)
        self.value = 0  # 当前脑负荷值 1-100

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 20)

        # 模拟脑负荷更新
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_value)
        self.timer.start(500)

    def update_value(self):
        self.value = random.randint(1, 100)
        self.update()  # 触发重绘

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 背景条
        rect = QRectF(0, 0, self.width(), self.height()-20)
        radius = rect.height() / 2
        painter.setBrush(QColor("#eee"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, radius, radius)

        # 进度条填充
        fill_width = rect.width() * self.value / 100
        gradient = QLinearGradient(rect.topLeft(), rect.topRight())
        gradient.setColorAt(0, QColor("#4CAF50"))   # 绿色
        gradient.setColorAt(0.5, QColor("#FFC107")) # 黄色
        gradient.setColorAt(1, QColor("#F44336"))   # 红色
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(QRectF(rect.x(), rect.y(), fill_width, rect.height()), radius, radius)

        # 文字显示
        painter.setPen(Qt.black)
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, f"脑负荷: {self.value}")
