"""Basic reusable UI widgets and visual helpers.

This module provides fundamental UI components used across the application:
- FadingStackedWidget: Page switcher with cross-fade animation
- AudioLevelMeter: Real-time audio level visualization
- create_shadow_effect: Standard drop shadow effect for UI elements
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QStackedWidget,
    QWidget,
)
from PyQt5.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QRectF,
    Qt,
)
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
)


def create_shadow_effect() -> QGraphicsDropShadowEffect:
    """Create a standard drop-shadow effect for UI cards and containers.
    
    Returns:
        QGraphicsDropShadowEffect with predefined blur, color, and offset
    """
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(25)
    shadow.setColor(QColor(0, 0, 0, 50))
    shadow.setOffset(0, 4)
    return shadow


class FadingStackedWidget(QStackedWidget):
    """QStackedWidget with smooth cross-fade animation between pages.
    
    This widget extends QStackedWidget to add opacity-based transitions
    when switching between different pages/widgets.
    
    Example:
        stack = FadingStackedWidget()
        stack.set_animation_duration(400)
        stack.fade_to_index(1)  # Fade to page 1
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 500
        self._current_index = 0
        self._next_index = 0
        self._animation = None

    def set_animation_duration(self, duration: int) -> None:
        """Set the duration of the fade animation in milliseconds.
        
        Args:
            duration: Animation duration in milliseconds (default: 500ms)
        """
        self._duration = duration

    def fade_to_index(self, index: int) -> None:
        """Switch to the specified page with a cross-fade animation.
        
        Args:
            index: Index of the target page to switch to
        """
        if self.currentIndex() == index:
            return

        self._current_index = self.currentIndex()
        self._next_index = index

        next_widget = self.widget(self._next_index)
        opacity_effect_next = QGraphicsOpacityEffect(self)
        next_widget.setGraphicsEffect(opacity_effect_next)

        animation = QPropertyAnimation(opacity_effect_next, b"opacity")
        animation.setDuration(self._duration)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutQuad)
        animation.finished.connect(self._on_animation_finished)

        self._animation = animation
        next_widget.show()
        next_widget.raise_()
        animation.start()

    def _on_animation_finished(self) -> None:
        """Clean up after animation completes."""
        current_widget = self.widget(self._current_index)
        current_widget.hide()
        current_widget.setGraphicsEffect(None)
        self.setCurrentIndex(self._next_index)
        next_widget = self.widget(self._next_index)
        next_widget.setGraphicsEffect(None)
        self._animation = None


class AudioLevelMeter(QWidget):
    """Real-time audio level meter with color-coded gradient visualization.
    
    Displays the current audio input level with a horizontal bar that changes
    color based on the volume:
    - Green (0-40%): Normal volume
    - Orange (40-75%): High volume
    - Red (75-100%): Very high volume
    
    Example:
        meter = AudioLevelMeter()
        meter.set_level(65)  # Set to 65% volume
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 30)
        self.level = 0
        self.setObjectName("audioLevelMeter")

    def set_level(self, level: int) -> None:
        """Update the audio level display.
        
        Args:
            level: Audio level from 0 (silent) to 100 (maximum)
        """
        new_level = min(100, max(0, level))
        if self.level != new_level:
            self.level = new_level
            self.update()

    def paintEvent(self, event):  # type: ignore[override]
        """Paint the audio level bar with appropriate color gradient."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.level <= 0:
            painter.setPen(QColor("#90A4AE"))
            painter.drawText(self.rect(), Qt.AlignCenter, "等待音频输入...")
            return

        bar_width = int(self.width() * (self.level / 100.0))
        bar_rect = QRect(0, 0, bar_width, self.height())

        gradient = QLinearGradient(0, 0, self.width(), 0)
        if self.level < 40:
            gradient.setColorAt(0, QColor("#66BB6A"))
            gradient.setColorAt(1, QColor("#43A047"))
        elif self.level < 75:
            gradient.setColorAt(0, QColor("#FFA726"))
            gradient.setColorAt(1, QColor("#FB8C00"))
        else:
            gradient.setColorAt(0, QColor("#EF5350"))
            gradient.setColorAt(1, QColor("#E53935"))

        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        path = QPainterPath()
        path.addRoundedRect(QRectF(bar_rect), self.height() / 2, self.height() / 2)
        painter.drawPath(path)


__all__ = [
    "AudioLevelMeter",
    "FadingStackedWidget",
    "create_shadow_effect",
]
