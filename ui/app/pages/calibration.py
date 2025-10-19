"""Calibration page that prepares AV devices before testing."""

from __future__ import annotations

from .. import config
from ..qt import (
    QFont,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedLayout,
    QTimer,
    Qt,
    QVBoxLayout,
    QWidget,
    pyqtSignal,
    qta,
)
from ..utils.helpers import init_camera
from ..utils.responsive import scale, scale_font, scale_size
from ..widgets.camera_preview import CameraPreviewWidget


class CalibrationPage(QWidget):
    """校准页面，用于在测试前检查和准备摄像头。"""

    calibration_finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.camera_preview: CameraPreviewWidget | None = None

        self.stacked_layout = QStackedLayout()
        self.setLayout(self.stacked_layout)

        self._init_loading_widget()
        self._init_calibration_widget()
        self.stacked_layout.setCurrentIndex(0)

    # ----------------- Loading Widget -----------------
    def _init_loading_widget(self) -> None:
        loading_widget = QWidget()
        vbox = QVBoxLayout(loading_widget)
        vbox.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container_width, container_height = scale_size(500, 300)
        container.setFixedSize(container_width, container_height)
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(scale(30))

        self.loading_label = QLabel("正在初始化摄像头...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.loading_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        layout.addWidget(self.progress_bar)

        vbox.addWidget(container)
        self.stacked_layout.addWidget(loading_widget)

        self.loading_progress = 0
        self.loading_timer = QTimer(self)
        self.loading_timer.timeout.connect(self._update_loading_progress)

    def _update_loading_progress(self) -> None:
        if self.loading_progress < 95:
            self.loading_progress += 2
            self.progress_bar.setValue(self.loading_progress)

    # ----------------- Calibration Widget -----------------
    def _init_calibration_widget(self) -> None:
        calibration_widget = QWidget()
        layout = QVBoxLayout(calibration_widget)
        layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))
        layout.setSpacing(scale(15))

        title = QLabel("设备校准")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", scale_font(18)))
        layout.addWidget(title)

        cam_width, cam_height = scale_size(640, 480)
        self.camera_preview = CameraPreviewWidget(cam_width, cam_height)
        layout.addWidget(self.camera_preview, alignment=Qt.AlignCenter)

        self.finish_button = QPushButton("  校准完成")
        self.finish_button.setMinimumHeight(scale(50))
        self.finish_button.setFixedWidth(scale(220))
        self.finish_button.clicked.connect(self._on_finish_calibration)
        self.finish_button.setIcon(qta.icon("mdi.check-circle-outline"))
        layout.addWidget(self.finish_button, 0, Qt.AlignCenter)

        self.stacked_layout.addWidget(calibration_widget)

    # ----------------- Event Handlers -----------------
    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        config.logger.info("进入校准页面。")
        
        # 检查主窗口是否已预加载摄像头
        main_window = self.window()
        if hasattr(main_window, 'camera_preloaded') and main_window.camera_preloaded:
            config.logger.info("✅ 检测到摄像头已预加载，立即切换到校准视图")
            try:
                # 直接切换到校准视图，无需等待
                self._switch_to_calibration_view()
            except Exception as e:
                config.logger.error(f"使用预加载摄像头失败: {e}")
                # 失败则fallback到正常初始化流程
                self._start_camera_initialization()
            return
        
        # 未预加载，显示进度条并开始初始化
        config.logger.info("摄像头未预加载，开始异步初始化...")
        self._start_camera_initialization()
    
    def _start_camera_initialization(self) -> None:
        """启动摄像头初始化流程（显示进度条）"""
        self.loading_progress = 0
        self.progress_bar.setValue(0)
        self.loading_timer.start(30)
        
        # 异步初始化摄像头（非阻塞）
        try:
            init_camera(self._on_camera_init_finished)
        except Exception as e:
            config.logger.error(f"启动摄像头初始化失败: {e}")
            self.loading_timer.stop()
            self.loading_label.setText("❌ 摄像头初始化启动失败")
            QMessageBox.critical(
                self, 
                "错误", 
                f"无法启动摄像头初始化：{e}\n\n请检查：\n1. 摄像头是否连接\n2. 后端服务是否启动\n3. 其他程序是否占用摄像头"
            )

    def hideEvent(self, event):  # type: ignore[override]
        config.logger.info("离开校准页面。")
        try:
            if self.camera_preview:
                self.camera_preview.stop_preview()
        except Exception as e:
            config.logger.debug(f"停止摄像头预览时出错: {e}")
        super().hideEvent(event)

    # ----------------- Camera Logic -----------------
    def _on_camera_init_finished(self, success: bool) -> None:
        """摄像头初始化完成回调（在主线程执行）"""
        try:
            self.loading_timer.stop()
            self.progress_bar.setValue(100)
            
            if success:
                self.loading_label.setText("✅ 摄像头准备就绪")
                config.logger.info("摄像头初始化成功，切换到校准视图")
                QTimer.singleShot(500, self._switch_to_calibration_view)
            else:
                self.loading_label.setText("❌ 摄像头初始化失败")
                config.logger.error("摄像头初始化失败")
                
                # 提供更详细的错误信息
                error_msg = (
                    "无法打开摄像头。\n\n"
                    "可能的原因：\n"
                    "1. 摄像头未连接或已被其他程序占用\n"
                    "2. 后端服务未启动（请运行: python -m src.main --root .）\n"
                    "3. 权限不足或驱动问题\n\n"
                    "您可以：\n"
                    "• 关闭占用摄像头的程序后重试\n"
                    "• 启动后端服务后重试\n"
                    "• 使用调试模式：python -m ui.main --debug"
                )
                QMessageBox.critical(self, "摄像头错误", error_msg)
        except Exception as e:
            config.logger.error(f"处理摄像头初始化结果时出错: {e}")

    def _switch_to_calibration_view(self) -> None:
        """切换到校准视图并启动摄像头预览（安全，失败不崩溃）"""
        try:
            self.stacked_layout.setCurrentIndex(1)
            if self.camera_preview:
                self.camera_preview.start_preview()
                config.logger.info("摄像头预览已启动")
        except Exception as e:
            config.logger.error(f"切换到校准视图失败: {e}")
            QMessageBox.warning(
                self,
                "警告",
                f"摄像头预览启动失败：{e}\n\n画面将显示占位符，但不影响继续操作。"
            )

    def _on_finish_calibration(self) -> None:
        """完成校准（安全，失败不崩溃）"""
        try:
            config.logger.info("用户完成设备校准。")
            if self.camera_preview:
                self.camera_preview.stop_preview()
        except Exception as e:
            config.logger.debug(f"停止预览时出错: {e}")
        
        try:
            self.calibration_finished.emit()
        except Exception as e:
            config.logger.error(f"发送校准完成信号失败: {e}")


__all__ = ["CalibrationPage"]
