"""Widgets that display multimodal fatigue status feedback."""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class _BasePreview(QWidget):
    """Shared widget used by both embedded and floating fatigue indicators."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.status_label = QLabel("等待采集启动…", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #2c3e50; font-size: 13px;")
        layout.addWidget(self.status_label)

        self.fatigue_label = QLabel("疲劳度: --", self)
        self.fatigue_label.setAlignment(Qt.AlignCenter)
        self.fatigue_label.setStyleSheet("color: #34495e; font-size: 14px; font-weight: bold;")
        layout.addWidget(self.fatigue_label)

    # ------------------------------------------------------------------
    def set_status(self, text: str, *, level: str = "info") -> None:
        palette = {
            "info": "#2c3e50",
            "warn": "#e67e22",
            "error": "#c0392b",
        }.get(level, "#2c3e50")
        self.status_label.setStyleSheet(f"color: {palette}; font-size: 13px;")
        self.status_label.setText(text)

    def set_fatigue_score(self, value: Optional[float]) -> None:
        if value is None:
            self.fatigue_label.setText("疲劳度: --")
        else:
            self.fatigue_label.setText(f"疲劳度: {float(value):0.1f}")

    def clear_preview(self) -> None:
        """Maintain backwards compatibility for prior API usages."""

    def set_preview_base64(self, encoded: Optional[str]) -> None:  # pragma: no cover - legacy no-op
        if not encoded:
            return


class MultimodalPreviewPanel(_BasePreview):
    """Embedded preview panel for use inside stacked layouts."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("multimodalPreviewPanel")


class MultimodalPreviewWindow(_BasePreview):
    """Floating preview window kept for backwards compatibility."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("多模态实时预览")
        self.setAttribute(Qt.WA_AlwaysStackOnTop, False)
        self.setWindowFlag(Qt.Tool)
        self.setFixedSize(280, 200)

    def show_if_hidden(self) -> None:
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()


__all__ = ["MultimodalPreviewPanel", "MultimodalPreviewWindow"]
