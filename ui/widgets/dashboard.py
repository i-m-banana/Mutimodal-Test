"""PyQt5 dashboard that visualises backend detector events via WebSocket."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from .brain_load_bar import BrainLoadBar
from ..services.backend_client import BackendClient, BackendEvent

try:  # pragma: no cover - optional dependency for runtime check
    import websockets  # type: ignore[import]
except ImportError:  # pragma: no cover
    websockets = None


class StatusBadge(QLabel):
    def __init__(self, text: str, *, color: str = "#95a5a6") -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignCenter)
        self.setProperty("statusBadge", True)
        self._color = color
        self._apply()

    def set_color(self, color: str) -> None:
        self._color = color
        self._apply()

    def _apply(self) -> None:
        self.setStyleSheet(
            f"padding: 6px 14px; border-radius: 14px; "
            f"color: white; background-color: {self._color};"
        )


class EventLog(QListWidget):
    def append_event(self, event: BackendEvent) -> None:
        payload_preview = event.payload
        if isinstance(payload_preview, dict):
            payload_preview = {k: payload_preview[k] for k in list(payload_preview)[:4]}
        item = QListWidgetItem(
            f"{event.timestamp:.2f} · {event.topic}\n{payload_preview}"
        )
        self.insertItem(0, item)
        while self.count() > 200:
            self.takeItem(self.count() - 1)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("多模态评估监控台")
        self.resize(960, 640)
        self._backend = BackendClient()
        self._setup_ui()
        self._wire_signals()
        self._backend.start()

    def _setup_ui(self) -> None:
        central = QWidget(self)
        layout = QGridLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setHorizontalSpacing(24)
        layout.setVerticalSpacing(24)

        status_box = QGroupBox("后端连接状态")
        status_layout = QHBoxLayout(status_box)
        self.status_badge = StatusBadge("离线", color="#e74c3c")
        status_layout.addWidget(self.status_badge)
        status_layout.addStretch()
        self.restart_button = QPushButton("重连")
        status_layout.addWidget(self.restart_button)
        layout.addWidget(status_box, 0, 0, 1, 2)

        brain_box = QGroupBox("脑负荷侦测")
        brain_layout = QHBoxLayout(brain_box)
        self.brain_bar = BrainLoadBar(simulate=False)
        brain_layout.addWidget(self.brain_bar)
        layout.addWidget(brain_box, 1, 0, 1, 2)

        self.result_title = QLabel("尚未收到检测结果")
        self.result_title.setWordWrap(True)
        font = QFont()
        font.setPointSize(12)
        font.setBold(True)
        self.result_title.setFont(font)
        self.result_details = QLabel("\u2022 等待后端推送数据")
        self.result_details.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_details.setWordWrap(True)
        self.result_details.setMinimumHeight(120)

        result_box = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_box)
        result_layout.addWidget(self.result_title)
        result_layout.addWidget(self.result_details)
        layout.addWidget(result_box, 2, 0, 2, 1)

        log_box = QGroupBox("事件流")
        log_layout = QVBoxLayout(log_box)
        self.event_log = EventLog()
        log_layout.addWidget(self.event_log)
        layout.addWidget(log_box, 2, 1, 2, 1)

        self.setCentralWidget(central)

    def _wire_signals(self) -> None:
        self._backend.connection_state_changed.connect(self._on_connection_state)
        self._backend.detection_result.connect(self._on_detection_result)
        self._backend.system_heartbeat.connect(self._on_heartbeat)
        self._backend.raw_event.connect(self._on_raw_event)
        self.restart_button.clicked.connect(self._backend.start)

    def _on_connection_state(self, connected: bool) -> None:
        self.status_badge.setText("在线" if connected else "离线")
        self.status_badge.set_color("#2ecc71" if connected else "#e74c3c")
        if not connected:
            self.brain_bar.set_caption("等待连接")
            self.brain_bar.set_value(0, label="--")

    def _on_detection_result(self, payload: Dict) -> None:
        score = payload.get("score") or payload.get("value")
        label = payload.get("label") or payload.get("state", "")
        self.brain_bar.set_caption(payload.get("detector", "检测器"))
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        self.brain_bar.set_value(score_value, label=label)
        details = [f"分数: {score_value:0.2f}"]
        for key, value in payload.items():
            if key in {"score", "value", "detector", "label", "state"}:
                continue
            details.append(f"{key}: {value}")
        self.result_title.setText(f"检测器 {payload.get('detector', '未知')} 返回")
        self.result_details.setText("\n".join(f"\u2022 {line}" for line in details))

    def _on_heartbeat(self, payload: Dict) -> None:
        cpu = payload.get("cpu", "--")
        mem = payload.get("memory", "--")
        hz = payload.get("interval", "--")
        self.status_badge.setToolTip(f"CPU: {cpu}%\nMemory: {mem}%\nHeartbeat: {hz}s")

    def _on_raw_event(self, event: Dict) -> None:
        backend_event = BackendEvent(
            topic=str(event.get("topic", "unknown")),
            payload=event.get("payload") or {},
            timestamp=float(event.get("timestamp", 0.0)),
        )
        self.event_log.append_event(backend_event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._backend.stop()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyleSheet(
        "QGroupBox { font-size: 14px; font-weight: 600; }"
        "QLabel[statusBadge='true'] { font-size: 15px; }"
        "QListWidget { background-color: #f8f9f9; border: 1px solid #d5d8dc; border-radius: 8px; }"
    )
    window = MainWindow()
    window.show()
    if websockets is None:
        QMessageBox.warning(window, "缺少依赖", "请先安装 websockets 库以接收后端事件。")
    return app.exec_()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
