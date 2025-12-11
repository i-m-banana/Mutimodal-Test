"""Calibration page that prepares AV devices before testing."""

from __future__ import annotations

from .. import config
from ..qt import (
    QFont,
    QHBoxLayout,
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
    QFrame,
    QGraphicsDropShadowEffect,
    QColor,
)
from ..utils.helpers import init_camera
from ..utils.responsive import scale, scale_font, scale_size
from ...widgets.camera_preview import CameraPreviewWidget
from ...services.session_manager import SessionManager

try:
    from ui.services.backend_proxy import eeg_get_diagnostics
except ImportError:
    from ...services.backend_proxy import eeg_get_diagnostics  # type: ignore


class CalibrationPage(QWidget):
    """校准页面，用于在测试前检查和准备摄像头。"""

    calibration_finished = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.camera_preview: CameraPreviewWidget | None = None
        self.eeg_preconnect_started = False
        self.session_manager = SessionManager.get_instance()

        self.stacked_layout = QStackedLayout()
        self.setLayout(self.stacked_layout)

        self._init_loading_widget()
        self._init_calibration_widget()
        self.stacked_layout.setCurrentIndex(0)
        
        # EEG状态查询定时器
        self.eeg_status_timer = QTimer(self)
        self.eeg_status_timer.timeout.connect(self._update_eeg_status)
        self.eeg_status_timer.setInterval(1000)  # 每秒更新一次

        self.setStyleSheet("""
                QLabel {
                    font-family: "阿里健康体2.0 中文 45 R";
                    font-size: 18pt;
                }

                QLabel#titleLabel {
                    font-size: 28pt;
                    font-weight: bold;
                }

                QLabel#loadingLabel {
                    font-size: 28pt;
                }

                QLabel#eegLabel {
                    font-size: 28pt;
                }

                QLabel#statusIcon {
                    font-size: 22pt;
                }

                QLabel#statusText {
                    font-size: 28pt;
                }

                QPushButton {
                    font-family: "阿里健康体2.0 中文 45 R";
                    font-size: 24pt;
                }

                QPushButton#finishButton {
                    font-size: 24pt;
                    font-weight: bold;
                }
            """)

    # ----------------- Loading Widget -----------------
    def _init_loading_widget(self) -> None:
        loading_widget = QWidget()
        loading_widget.setObjectName("loadingWidget")
        # 设置loading页面的明显渐变背景（只应用到主容器，不影响子元素）
        loading_widget.setStyleSheet("""
            #loadingWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 1),
                    stop:0.5 rgba(200, 235, 245, 0.6),
                    stop:1 rgba(127, 219, 255, 0.5)
                );
            }
        """)
        vbox = QVBoxLayout(loading_widget)
        vbox.setAlignment(Qt.AlignCenter)

        container = QWidget()
        container_width, container_height = scale_size(500, 300)
        container.setFixedSize(container_width, container_height)
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(scale(30))

        self.loading_label = QLabel("正在初始化摄像头...")
        self.loading_label.setObjectName("loadingLabel")  # ✅ 添加这行
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
        """创建校准页面(卡片布局 - 所有元素在一个白色卡片中)"""
        calibration_widget = QWidget()
        calibration_widget.setObjectName("calibrationWidget")
        calibration_widget.setStyleSheet("""
            #calibrationWidget {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(255, 255, 255, 1),
                    stop:0.5 rgba(200, 235, 245, 0.6),
                    stop:1 rgba(127, 219, 255, 0.5)
                );
            }
        """)

        # 页面主布局（用于居中卡片）
        page_layout = QVBoxLayout(calibration_widget)
        page_layout.setAlignment(Qt.AlignCenter)
        page_layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))

        # 创建白色卡片容器（加大尺寸）
        card_container = QFrame()
        card_container.setObjectName("calibrationCardContainer")
        card_container.setFixedSize(scale(1400), scale(950))  # ✅ 从1200×800增加到1400×950
        card_container.setStyleSheet("""
            QFrame#calibrationCardContainer {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                padding: 30px;
            }
        """)

        # 添加底部阴影效果
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(15)
        card_shadow.setXOffset(0)
        card_shadow.setYOffset(5)
        card_shadow.setColor(QColor(0, 0, 0, 60))
        card_container.setGraphicsEffect(card_shadow)

        # 卡片内的垂直布局
        layout = QVBoxLayout(card_container)
        layout.setContentsMargins(scale(40), scale(40), scale(40), scale(40))  # ✅ 从20增加到40
        layout.setSpacing(scale(30))  # ✅ 从25增加到30
        layout.setAlignment(Qt.AlignCenter)

        # 标题（字体加大）
        title = QLabel("设备校准")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("titleLabel")
        title.setStyleSheet("font-size: 48px; font-weight: bold; color: #333;")  # ✅ 从40px增加到48px
        layout.addWidget(title)

        # 摄像头预览（尺寸加大）
        cam_width, cam_height = scale_size(700, 525)  # ✅ 从560×420增加到700×525
        self.camera_preview = CameraPreviewWidget(cam_width, cam_height)
        layout.addWidget(self.camera_preview, alignment=Qt.AlignCenter)

        # EEG设备状态显示（字体加大）
        eeg_status_container = QWidget()
        eeg_status_layout = QHBoxLayout(eeg_status_container)
        eeg_status_layout.setContentsMargins(0, 0, 0, 0)
        eeg_status_layout.setAlignment(Qt.AlignCenter)
        eeg_status_layout.setSpacing(scale(10))  # ✅ 从8增加到10

        eeg_label = QLabel("脑电设备:")
        eeg_label.setObjectName("eegLabel")
        eeg_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #333;")  # ✅ 从28px增加到32px
        eeg_status_layout.addWidget(eeg_label)

        self.eeg_status_icon = QLabel("●")
        self.eeg_status_icon.setObjectName("statusIcon")
        self.eeg_status_icon.setStyleSheet("color: gray; font-size: 32px;")  # ✅ 从28px增加到32px
        eeg_status_layout.addWidget(self.eeg_status_icon)

        self.eeg_status_text = QLabel("检测中...")
        self.eeg_status_text.setObjectName("statusText")
        self.eeg_status_text.setStyleSheet("font-size: 32px; color: #666;")  # ✅ 从28px增加到32px
        eeg_status_layout.addWidget(self.eeg_status_text)

        layout.addWidget(eeg_status_container, 0, Qt.AlignCenter)

        # 校准完成按钮（尺寸加大）
        self.finish_button = QPushButton("✓ 校准完成")
        self.finish_button.setObjectName("primaryButton")
        self.finish_button.setFixedSize(scale(280), scale(80))  # ✅ 从240×70增加到280×80
        self.finish_button.setCursor(Qt.PointingHandCursor)
        self.finish_button.setStyleSheet("""
            QPushButton#primaryButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 32px;
                font-weight: bold;
            }
            QPushButton#primaryButton:hover {
                background-color: #3498DB;
            }
            QPushButton#primaryButton:pressed {
                background-color: #2E86C1;
            }
        """)
        self.finish_button.clicked.connect(self._on_finish_calibration)
        layout.addWidget(self.finish_button, 0, Qt.AlignCenter)

        # 将卡片添加到页面（居中显示）
        page_layout.addStretch(1)
        page_layout.addWidget(card_container, 0, Qt.AlignCenter)
        page_layout.addStretch(1)

        self.stacked_layout.addWidget(calibration_widget)

    # ----------------- Event Handlers -----------------
    def showEvent(self, event):  # type: ignore[override]
        super().showEvent(event)
        config.logger.info("进入校准页面。")
        
        # 启动EEG状态查询
        self.eeg_status_timer.start()
        self._update_eeg_status()  # 立即查询一次
        
        # 🔥 启动EEG预连接（仅启动一次）
        if not self.eeg_preconnect_started:
            self._start_eeg_preconnect()
            self.eeg_preconnect_started = True
        
        # 检查主窗口是否已预加载摄像头
        main_window = self.window()
        if hasattr(main_window, 'camera_preloaded') and main_window.camera_preloaded:
            config.logger.debug("✅ 检测到摄像头已预加载，立即切换到校准视图")
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
        
        try:
            session_dir = self.session_manager.get_session_dir()
            config.logger.info(f"🎥 校准页面初始化摄像头: {session_dir}")
            init_camera(self._on_camera_init_finished, session_dir=session_dir)
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
        
        # 停止EEG状态查询
        self.eeg_status_timer.stop()
        
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
    
    def _update_eeg_status(self) -> None:
        """更新EEG设备连接状态（异步，不阻塞UI）"""
        try:
            diag = eeg_get_diagnostics(timeout=1.0)
            
            hardware_available = diag.get("hardware_driver_available", False)
            simulation_mode = diag.get("simulation_mode")
            force_simulation = diag.get("force_simulation", False)
            is_running = diag.get("running", False)
            device_connected = diag.get("device_connected", False)

            # 向后兼容：后端老版本没有 simulation_mode 字段时，回退到 force_simulation
            if simulation_mode is None:
                simulation_mode = force_simulation or not hardware_available

            if simulation_mode:
                # 模拟数据模式
                self.eeg_status_icon.setStyleSheet("color: #2980b9;")
                self.eeg_status_text.setText("模拟数据")
            elif is_running and device_connected:
                # 采集线程运行且设备已连接
                self.eeg_status_icon.setStyleSheet("color: #27ae60;")
                self.eeg_status_text.setText("已连接")
            elif is_running:
                # 采集线程已启动但尚未建立连接
                self.eeg_status_icon.setStyleSheet("color: #f39c12;")
                self.eeg_status_text.setText("连接中...")
            else:
                # 其他情况统一视为连接失败，避免显示额外状态
                self.eeg_status_icon.setStyleSheet("color: #e74c3c;")
                self.eeg_status_text.setText("连接失败")
                
        except ConnectionError:
            # 后端未连接
            self.eeg_status_icon.setStyleSheet("color: #e74c3c;")
            self.eeg_status_text.setText("连接失败")
        except TimeoutError:
            # 查询超时
            self.eeg_status_icon.setStyleSheet("color: #e74c3c;")
            self.eeg_status_text.setText("连接失败")
        except Exception as e:
            # 其他错误
            config.logger.debug(f"查询EEG状态失败: {e}")
            self.eeg_status_icon.setStyleSheet("color: #e74c3c;")
            self.eeg_status_text.setText("连接失败")
    
    def _start_eeg_preconnect(self) -> None:
        """启动EEG预连接（在校准页面就开始连接设备）"""
        try:
            from ...utils_common.thread_process_manager import get_thread_manager
            from ...services.backend_proxy import eeg_start_collection
            from datetime import datetime
            from pathlib import Path
            import os
            
            thread_manager = get_thread_manager()
            
            def preconnect_eeg():
                try:
                    main_window = self.window()
                    current_user = getattr(main_window, 'current_user', 'anonymous')
                    session_dir = self.session_manager.get_session_dir()
                    config.logger.debug(f"🔗 EEG预连接使用会话目录: {session_dir}")
                    
                    result = eeg_start_collection(
                        username=current_user,
                        save_dir=session_dir,
                        part=1
                    )
                    status = (result or {}).get("status", "").lower()
                    if status in {"started", "already-running"}:
                        config.logger.info(f"✅ EEG预连接已启动，保存目录: {session_dir}")
                    else:
                        config.logger.warning(f"⚠️ EEG预连接启动失败: {result}")
                    
                    # # ✅ 第二步：启动脑负荷推理（多模态数据采集）
                    # try:
                    #     # `backend_proxy` 提供的接口名为 multimodal_start_collection（不是 multidata_start_collection）
                    #     from ...services.backend_proxy import multimodal_start_collection

                    #     # 使用显式关键字参数以避免参数顺序或命名歧义
                    #     multidata_result = multimodal_start_collection(
                    #         username=current_user,
                    #         save_dir=session_dir,
                    #         part=0,
                    #     )
                    #     multidata_status = (multidata_result or {}).get("status", "").lower()
                    #     if multidata_status in {"started", "running", "already-running"}:
                    #         config.logger.debug(f"🧠 脑负荷推理已在校准阶段启动，保存目录: {session_dir}")
                    #     else:
                    #         config.logger.warning(f"⚠️ 脑负荷推理启动失败: {multidata_result}")
                    # except Exception as e:
                    #     config.logger.error(f"❌ 启动脑负荷推理失败: {e}", exc_info=True)
                        
                except Exception as e:
                    config.logger.error(f"❌ EEG预连接失败: {e}", exc_info=True)
            
            thread_manager.submit_data_task(
                preconnect_eeg,
                task_name="EEG设备预连接"
            )
        except Exception as e:
            config.logger.error(f"❌ 启动EEG预连接失败: {e}")


__all__ = ["CalibrationPage"]
