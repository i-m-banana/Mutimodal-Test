"""Brain load bar widget visualising fatigue score streamed from backend."""

from __future__ import annotations

import random

from PyQt5.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, pyqtProperty
from PyQt5.QtGui import QBrush, QColor, QLinearGradient, QPainter
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class BrainLoadBar(QWidget):
    """Gradient progress bar with optional simulated data."""

    def __init__(self, parent=None, *, min_value: int = 0, max_value: int = 100,
                 simulate: bool = False, update_interval_ms: int = 750) -> None:
        super().__init__(parent)
        # 减小高度以适应1080p全屏显示
        self.setFixedHeight(55)
        self._min = min_value
        self._max = max_value
        self._value = float(min_value)
        self._display_value = float(min_value)
        self._display_label = "待命"
        self._simulate = simulate

        layout = QVBoxLayout(self)
        # 进一步减小内边距以节省垂直空间
        layout.setContentsMargins(20, 3, 20, 6)
        self.caption_label = QLabel("脑负荷指数")
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setProperty("brainLoadCaption", True)
        layout.addWidget(self.caption_label)

        if self._simulate:
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._random_refresh)
            self._timer.start(update_interval_ms)
        else:
            self._timer = None

        self._animation = QPropertyAnimation(self, b"progress", self)
        self._animation.setDuration(450)
        self._animation.setEasingCurve(QEasingCurve.OutQuad)

    def value(self) -> float:
        return self._value

    def set_value(self, value: float, *, label: str | None = None) -> None:
        value = max(self._min, min(self._max, float(value)))
        if label is not None:
            self._display_label = label
        if self._timer:
            self._timer.stop()
            self._timer = None
        start = self._display_value
        self._animation.stop()
        self._animation.setStartValue(start)
        self._animation.setEndValue(value)
        if start == value:
            self._set_progress(value)
        else:
            self._animation.start()
        self._value = value

    def set_caption(self, caption: str) -> None:
        self.caption_label.setText(caption)

    def _random_refresh(self) -> None:
        new_value = random.uniform(self._min, self._max)
        self._display_label = "模拟数据"
        self._value = new_value
        self._display_value = new_value
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = QRectF(0, 0, self.width(), self.height() - 28)
        rect.moveTop(24)
        radius = rect.height() / 2

        painter.setBrush(QColor("#f0f2f5"))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, radius, radius)

        if self._max - self._min > 0:
            ratio = (self._display_value - self._min) / (self._max - self._min)
        else:
            ratio = 0
        fill_width = rect.width() * ratio
        if fill_width > 0:
            gradient = QLinearGradient(rect.topLeft(), rect.topRight())
            gradient.setColorAt(0.0, QColor("#2ecc71"))
            gradient.setColorAt(0.55, QColor("#f1c40f"))
            gradient.setColorAt(1.0, QColor("#e74c3c"))
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(QRectF(rect.x(), rect.y(), fill_width, rect.height()), radius, radius)

        painter.setPen(QColor("#1c2833"))
        painter.setFont(self.font())
        painter.drawText(rect, Qt.AlignCenter, f"脑负荷: {self._display_value:0.1f}")
        if self._display_label:
            painter.setPen(QColor("#5d6d7e"))
            painter.drawText(rect.adjusted(0, -22, 0, -22), Qt.AlignCenter, self._display_label)

    # Qt property accessors -------------------------------------------------
    def _get_progress(self) -> float:
        return self._display_value

    def _set_progress(self, value: float) -> None:
        self._display_value = value
        self.update()

    progress = pyqtProperty(float, fget=_get_progress, fset=_set_progress)
