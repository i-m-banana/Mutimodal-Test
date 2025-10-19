"""Reusable UI widgets and visual helpers."""

from __future__ import annotations

import os
from pathlib import Path

from .. import config
from ..qt import (
    QBrush,
    QColor,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QEasingCurve,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPixmap,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSizePolicy,
    Qt,
    QVBoxLayout,
    QWidget,
    QStackedWidget,
)


def create_shadow_effect() -> QGraphicsDropShadowEffect:
    """Create a standard drop-shadow used across cards."""
    shadow = QGraphicsDropShadowEffect()
    shadow.setBlurRadius(25)
    shadow.setColor(QColor(0, 0, 0, 50))
    shadow.setOffset(0, 4)
    return shadow


class FadingStackedWidget(QStackedWidget):
    """QStackedWidget with cross-fade animation support."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._duration = 500
        self._current_index = 0
        self._next_index = 0
        self._animation = None

    def set_animation_duration(self, duration: int) -> None:
        self._duration = duration

    def fade_to_index(self, index: int) -> None:
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
        current_widget = self.widget(self._current_index)
        current_widget.hide()
        current_widget.setGraphicsEffect(None)
        self.setCurrentIndex(self._next_index)
        next_widget = self.widget(self._next_index)
        next_widget.setGraphicsEffect(None)
        self._animation = None


class AudioLevelMeter(QWidget):
    """Simple audio level meter with gradient visuals."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 30)
        self.level = 0
        self.setObjectName("audioLevelMeter")

    def set_level(self, level: int) -> None:
        new_level = min(100, max(0, level))
        if self.level != new_level:
            self.level = new_level
            self.update()

    def paintEvent(self, event):  # type: ignore[override]
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


class ScoreChartWidget(QWidget):
    """Matplotlib-backed chart to display history scores."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(200)
        config.plt.style.use('seaborn-v0_8-whitegrid')
        self.figure = config.Figure(figsize=(5, 3), dpi=100)
        self.figure.patch.set_facecolor('none')
        self.canvas = config.FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color:transparent;")
        self.ax = self.figure.add_subplot(111)

        self._font = None
        font_candidates = []
        if os.name == "nt":
            windows_dir = os.environ.get("WINDIR", "C:/Windows")
            font_candidates.extend([
                os.path.join(windows_dir, "Fonts", "msyh.ttc"),
                os.path.join(windows_dir, "Fonts", "msyh.ttf"),
            ])
        font_candidates.extend([
            os.path.join(str(Path.home()), ".fonts", "msyh.ttc"),
            os.path.join(str(Path.home()), ".fonts", "msyh.ttf"),
        ])
        for candidate in font_candidates:
            if os.path.exists(candidate):
                try:
                    self._font = config.font_manager.FontProperties(fname=candidate)
                    break
                except Exception:
                    continue
        if self._font is None:
            self._font = config.font_manager.FontProperties(family="Microsoft YaHei")

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def update_chart(self, scores: list[int]) -> None:
        self.ax.clear()
        self.ax.set_facecolor('#F5F5F5')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color('#B0BEC5')
        self.ax.spines['bottom'].set_color('#B0BEC5')
        self.ax.tick_params(axis='x', colors='#546E7A')
        self.ax.tick_params(axis='y', colors='#546E7A')
        self.ax.set_ylabel('评估分数', fontsize=12, color='#37474F', fontproperties=self._font)
        self.ax.set_xlabel('测试次数', fontsize=12, color='#37474F', fontproperties=self._font)

        if not scores or len(scores) < 2:
            self.ax.set_xticks([])
            self.ax.set_yticks([0, 20, 40, 60, 80, 100])
            self.ax.set_ylim(0, 110)
            text = "历史数据不足，无法生成趋势图" if scores else "暂无历史记录"
            self.ax.text(0.5, 0.5, text, ha='center', va='center', transform=self.ax.transAxes,
                         fontsize=14, color='gray', fontproperties=self._font)
        else:
            display_scores = scores[-10:]
            x_values = range(1, len(display_scores) + 1)
            self.ax.plot(
                x_values,
                display_scores,
                color='#1976D2',
                marker='o',
                linestyle='-',
                linewidth=2,
                markersize=8,
                markerfacecolor='#42A5F5',
            )
            for idx, score in enumerate(display_scores):
                self.ax.text(
                    list(x_values)[idx],
                    score + 3,
                    str(score),
                    ha='center',
                    color='#0D47A1',
                    fontsize=10,
                    fontweight='bold',
                    fontproperties=self._font,
                )
            self.ax.set_ylim(min(display_scores) - 10, 110)
            self.ax.set_xticks(list(x_values))
            self.ax.set_xticklabels([f"第{i}次" for i in x_values], fontproperties=self._font)

        self.figure.tight_layout(pad=2.0)
        self.canvas.draw()


__all__ = [
    "AudioLevelMeter",
    "FadingStackedWidget",
    "ScoreChartWidget",
    "create_shadow_effect",
]
