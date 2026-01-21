"""Schulte grid test page component."""

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from ..utils.responsive import scale, scale_size
from ...widgets.schulte_grid import SchulteGridWidget


class SchultePage(QWidget):
    
    test_completed = pyqtSignal()
    test_result_ready = pyqtSignal(float, float)  # (elapsed_time, accuracy)
    
    def __init__(self, username: str, camera_widget: QWidget, parent=None):
        super().__init__(parent)
        self.username = username
        self.camera_widget = camera_widget
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(scale(8), 0, scale(8), 0)
        main_layout.setSpacing(scale(20))

        main_layout.addStretch(1)

        cam_width = scale_size(560, 420)[0]
        self.camera_widget.setMaximumWidth(cam_width)
        self.camera_widget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        main_layout.addWidget(self.camera_widget, 0, Qt.AlignRight)

        self.schulte_container = QWidget()
        schulte_layout = QVBoxLayout(self.schulte_container)
        schulte_layout.setContentsMargins(0, 0, 0, 0)
        schulte_layout.setSpacing(scale(10))

        self.schulte_widget = SchulteGridWidget(self.username)
        self.schulte_widget.test_completed.connect(self.test_completed.emit)
        self.schulte_widget.test_result_ready.connect(self.test_result_ready.emit)
        schulte_layout.addWidget(self.schulte_widget, 0, Qt.AlignCenter)

        main_layout.addWidget(self.schulte_container, 0, Qt.AlignLeft)

        main_layout.addStretch(1)
    
    def reinit_widget(self):
        layout = self.schulte_container.layout()
        if layout is None:
            return
        
        layout.removeWidget(self.schulte_widget)
        self.schulte_widget.deleteLater()
        
        self.schulte_widget = SchulteGridWidget(self.username)
        self.schulte_widget.test_completed.connect(self.test_completed.emit)
        self.schulte_widget.test_result_ready.connect(self.test_result_ready.emit)
        
        layout.addWidget(self.schulte_widget)
