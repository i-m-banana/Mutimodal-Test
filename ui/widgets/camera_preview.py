"""Reusable camera preview widget shared by calibration and test pages."""

from __future__ import annotations

from ui.app import config
from ui.app.qt import (
    QColor,
    QImage,
    QLabel,
    QPixmap,
    QTimer,
    Qt,
    QVBoxLayout,
    QWidget,
    QPainter,
    QFont,
    cv2,
    np,
)


class CameraPreviewWidget(QWidget):
    """Simple wrapper around ``config.av_get_current_frame`` with graceful fallbacks."""

    def __init__(self, width: int = 640, height: int = 480, *, placeholder_text: str | None = None,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._width = width
        self._height = height
        self._placeholder_text = placeholder_text or "摄像头画面加载中..."

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.label = QLabel(self._placeholder_text)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(self._width, self._height)
        layout.addWidget(self.label, alignment=Qt.AlignCenter)

        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._update_frame)
        self._frame_wait_count = 0
        self._frame_received = False

    def start_preview(self, interval_ms: int = 30) -> None:
        self._timer.setInterval(max(15, interval_ms))
        if not self._timer.isActive():
            self._timer.start()

    def stop_preview(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
        self._frame_wait_count = 0
        self._frame_received = False
        self.label.setText(self._placeholder_text)

    def _update_frame(self) -> None:
        if config.NO_CAMERA_MODE:
            self._render_placeholder()
            return

        try:
            frame = config.av_get_current_frame()
        except Exception as exc:  # noqa: BLE001
            config.logger.warning("获取摄像头帧失败: %s", exc)
            frame = None

        if frame is None:
            self._frame_wait_count += 1
            dots = "." * ((self._frame_wait_count // 30) % 4)
            self.label.setText(f"摄像头加载中{dots}")
            if self._frame_wait_count >= 300:
                self.label.setText("⚠️ 摄像头连接超时\n请检查设备")
            return

        self._frame_received = True
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.label.setPixmap(
                pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception as exc:  # noqa: BLE001
            config.logger.error("更新摄像头图像失败: %s", exc)
            self._render_placeholder()

    def _render_placeholder(self) -> None:
        try:
            w, h = self._width, self._height
            frame = np.full((h, w, 3), 204, dtype=np.uint8)
            box_w, box_h = int(w * 0.6), int(h * 0.7)
            x1, y1 = (w - box_w) // 2, (h - box_h) // 2
            x2, y2 = x1 + box_w, y1 + box_h

            overlay = frame.copy()
            overlay[:] = 0
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            face_area = np.full((box_h, box_w, 3), 204, dtype=np.uint8)
            frame[y1:y2, x1:x2] = face_area

            color = QColor("#4CAF50").getRgb()[:3]
            corner_len = 30
            thickness = 4
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

            qt_image = QImage(frame.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.label.setPixmap(
                pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        except Exception:  # noqa: BLE001 - fallback
            placeholder = QPixmap(self._width, self._height)
            placeholder.fill(QColor("#CCCCCC"))
            painter = QPainter(placeholder)
            painter.setPen(Qt.black)
            painter.setFont(QFont("Arial", 18))
            painter.drawText(placeholder.rect(), Qt.AlignCenter, self._placeholder_text)
            painter.end()
            self.label.setPixmap(placeholder)


__all__ = ["CameraPreviewWidget"]
