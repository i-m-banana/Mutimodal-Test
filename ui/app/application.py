"""Application bootstrap and main window orchestration for the modular UI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import yaml

from . import config
from .config import (
    DEBUG_MODE,
    HAS_MULTIMODAL,
    av_stop_collection,
    av_stop_recording,
    bp_stop_measurement,
    logger,
    multidata_stop_collection,
    stop_recognition,
)
from .qt import (
    QApplication,
    QLabel,
    QKeySequence,
    QMainWindow,
    QShortcut,
    Qt,
    QVBoxLayout,
    QWidget,
    qta,
)
from .utils.widgets import FadingStackedWidget
from .utils.responsive import get_scaler, scale, scale_size
from .pages.calibration import CalibrationPage
from .pages.login import LoginPage
from .pages.test import TestPage
from ..widgets.brain_load_bar import BrainLoadBar
from ..widgets.schulte_grid import SchulteGridWidget
from ..utils_common.thread_process_manager import (
    get_lifecycle_manager,
    get_thread_manager,
    shutdown_all_managers,
)
from ..services.backend_launcher import get_backend_launcher

STYLE_PATH = config.BASE_DIR / "style.qss"


class MainWindow(QMainWindow):
    """Primary UI shell controlling page flow and cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.current_user = "debug"
        self._debug_shortcuts: list[QShortcut] = []
        self.camera_preloaded = False

        self._setup_main_window()
        self._create_pages()
        self._connect_signals()
        self._setup_debug_shortcuts()
        
        # 延迟预加载摄像头（不阻塞UI启动）
        from .qt import QTimer
        QTimer.singleShot(800, self._preload_camera)

        logger.info("应用程序主窗口初始化完成。")

    def _setup_main_window(self) -> None:
        self.setWindowTitle('非接触人员状态评估系统')
        
        # 使用响应式缩放
        scaler = get_scaler()
        window_width, window_height = scale_size(1280, 800)
        self.setGeometry(100, 100, window_width, window_height)
        
        # 在小屏幕上允许最大化
        if scaler.is_small_screen:
            logger.info(f"检测到小屏幕，窗口尺寸调整为: {window_width}x{window_height}")
        
        self.setWindowIcon(qta.icon('fa5s.robot', color='blue'))

    def _create_pages(self) -> None:
        self.stack = FadingStackedWidget()
        self.stack.set_animation_duration(400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 使用响应式边距，减小上下边距以适配1080p显示器
        side_margin = scale(20)
        top_margin = scale(10)  # 顶部边距减小
        spacing = scale(12)  # 间距也减小
        bottom_margin = scale(8)  # 底部边距进一步减小
        main_layout.setContentsMargins(side_margin, top_margin, side_margin, bottom_margin)
        main_layout.setSpacing(spacing)

        self.login_page = LoginPage(self.show_calibration_page)
        self.calibration_page = CalibrationPage()
        self.test_page = TestPage()

        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.calibration_page)
        self.stack.addWidget(self.test_page)

        main_layout.addWidget(self.stack, 1)

        self.brain_load_bar = BrainLoadBar()
        self.brain_load_bar.setVisible(False)

        self.brain_load_tip = QLabel("本测试需要全程采集您的脑电信号来进行脑负荷测试")
        self.brain_load_tip.setAlignment(Qt.AlignCenter)
        self.brain_load_tip.setWordWrap(True)
        # 减小字体和上下边距，适配小屏幕
        self.brain_load_tip.setStyleSheet("color: #666; font-size: 14px; padding: 2px 0;")
        self.brain_load_tip.setMaximumHeight(scale(26))  # 进一步减小最大高度

        main_layout.addWidget(self.brain_load_tip, 0, Qt.AlignBottom)
        main_layout.addWidget(self.brain_load_bar, 0, Qt.AlignBottom)

        self.stack.setCurrentWidget(self.login_page)

    def _connect_signals(self) -> None:
        self.calibration_page.calibration_finished.connect(self.show_test_page)

    def _setup_debug_shortcuts(self) -> None:
        def bind(sequence: str, handler, description: str) -> None:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(handler)
            self._debug_shortcuts.append(shortcut)
            logger.debug("注册调试快捷键 %s -> %s", sequence, description)

        bind("Ctrl+Alt+1", self._debug_show_login, "切换登录页")
        bind("Ctrl+Alt+2", self._debug_show_calibration, "切换校准页")
        bind("Ctrl+Alt+3", self._debug_show_test, "切换测试页")

    def _debug_show_login(self) -> None:
        logger.info("调试快捷键：跳转到登录页面")
        self.stack.fade_to_index(0)
        self.brain_load_tip.setVisible(True)
        self.brain_load_bar.setVisible(False)

    def _debug_show_calibration(self) -> None:
        logger.info("调试快捷键：跳转到校准页面")
        if not getattr(self, "current_user", None):
            self.current_user = "debug"
        try:
            self.test_page.set_current_user(self.current_user)
        except Exception as exc:  # noqa: BLE001
            logger.debug("同步调试用户名失败: %s", exc)
        self.stack.fade_to_index(1)
        self.brain_load_tip.setVisible(True)
        self.brain_load_bar.setVisible(False)

    def _debug_show_test(self) -> None:
        logger.info("调试快捷键：跳转到测试页面")
        if not getattr(self, "current_user", None):
            self.current_user = "debug"
        try:
            self.test_page.set_current_user(self.current_user)
        except Exception as exc:  # noqa: BLE001
            logger.debug("同步调试用户名失败: %s", exc)
        self.stack.fade_to_index(2)
        self.brain_load_tip.setVisible(False)
        self.brain_load_bar.setVisible(True)
        try:
            self.test_page.start_test()
        except Exception as exc:  # noqa: BLE001
            logger.warning("调试快捷键启动测试失败: %s", exc)

    def show_calibration_page(self, username: str) -> None:
        logger.info("正在切换到校准页面...")
        self.current_user = username or 'anonymous'
        try:
            self.test_page.set_current_user(self.current_user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("同步用户名到测试页失败: %s", exc)
        self.stack.fade_to_index(1)

    def show_test_page(self) -> None:
        logger.info("正在切换到测试页面...")
        self.stack.fade_to_index(2)
        self.brain_load_tip.setVisible(False)
        self.brain_load_bar.setVisible(True)
        self.test_page.start_test()

    def _preload_camera(self) -> None:
        """在后台预加载摄像头，不阻塞UI"""
        from .utils.helpers import init_camera
        
        logger.info("🎥 开始预加载摄像头（后台异步）...")
        
        def on_preload_finished(success: bool) -> None:
            if success:
                logger.info("✅ 摄像头预加载成功，校准页面将立即就绪")
                self.camera_preloaded = True
            else:
                logger.warning("⚠️ 摄像头预加载失败，将在校准页重试")
                self.camera_preloaded = False
        
        try:
            init_camera(on_preload_finished)
        except Exception as e:
            logger.error(f"启动摄像头预加载失败: {e}")
            self.camera_preloaded = False

    def closeEvent(self, event) -> None:  # noqa: N802
        logger.info("应用程序正在关闭...")

        try:
            av_stop_recording()
        except Exception:  # noqa: BLE001
            pass
        try:
            av_stop_collection()
        except Exception:  # noqa: BLE001
            pass

        if HAS_MULTIMODAL:
            try:
                multidata_stop_collection()
                logger.info("应用程序关闭时已停止多模态数据采集")
                from ..services.backend_proxy import cleanup_collector

                cleanup_collector()
            except Exception as exc:  # noqa: BLE001
                logger.error("应用程序关闭时停止多模态数据采集失败: %s", exc)
            finally:
                try:
                    if hasattr(self, "test_page"):
                        self.test_page._stop_multimodal_monitoring()
                except Exception:  # noqa: BLE001
                    pass

        if hasattr(self, 'test_page') and hasattr(self.test_page, 'tts_task_id'):
            try:
                thread_manager = get_thread_manager()
                if self.test_page.tts_task_id:
                    thread_manager.cancel_task(self.test_page.tts_task_id)
                    self.test_page.tts_queue.put(None)
                    logger.info("应用程序关闭时已停止TTS任务")
            except Exception as exc:  # noqa: BLE001
                logger.warning("应用程序关闭时停止TTS任务失败: %s", exc)

        try:
            bp_stop_measurement()
        except Exception as exc:  # noqa: BLE001
            logger.debug("关闭应用时停止血压测量失败: %s", exc)

        if hasattr(self, 'test_page') and hasattr(self.test_page, 'schulte_widget'):
            self.test_page.schulte_widget.reset_for_next_stage()

        stop_recognition()

        try:
            shutdown_all_managers()
            logger.info("所有线程进程管理器已关闭")
        except Exception as exc:  # noqa: BLE001
            logger.error("关闭管理器失败: %s", exc)

        SchulteGridWidget.cleanup_temp_files()
        super().closeEvent(event)


def _apply_style(app: QApplication) -> None:
    if STYLE_PATH.exists():
        try:
            app.setStyleSheet(STYLE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("加载样式表失败: %s", exc)
    else:
        logger.warning("样式表文件未找到，使用默认样式")


def create_application(argv: Sequence[str] | None = None) -> tuple[QApplication, MainWindow]:
    args = list(argv) if argv is not None else sys.argv
    app = QApplication(args)

    # 读取系统配置,决定是否自动启动后端
    project_root = Path(__file__).resolve().parents[2]
    system_config_path = project_root / "config" / "system.yaml"
    auto_start_backend = True  # 默认自动启动
    
    if system_config_path.exists():
        try:
            with system_config_path.open("r", encoding="utf-8") as f:
                system_config = yaml.safe_load(f) or {}
                auto_start_backend = system_config.get("ui", {}).get("auto_start_backend", True)
        except Exception as exc:
            logger.warning("读取系统配置失败,将使用默认值: %s", exc)
    
    # 根据配置决定是否启动后端服务器
    backend_launcher = get_backend_launcher()
    if auto_start_backend:
        logger.info("自动启动后端服务器模式 (可在 config/system.yaml 中配置)")
        if not backend_launcher.start():
            logger.warning("后端服务器启动失败，某些功能可能不可用")
        # 确保应用退出时停止后端
        app.aboutToQuit.connect(backend_launcher.stop)
    else:
        logger.info("手动启动后端服务器模式 - 请确保后端已在独立窗口运行")
        logger.info("如需自动启动,请修改 config/system.yaml 中的 ui.auto_start_backend 为 true")

    lifecycle_manager = get_lifecycle_manager()
    status = lifecycle_manager.get_all_status()
    if not status.get('is_initialized'):
        lifecycle_manager.start_all()
    app.aboutToQuit.connect(lambda: lifecycle_manager.shutdown_all())
    app.setProperty("lifecycle_manager", lifecycle_manager)

    _apply_style(app)

    window = MainWindow()
    return app, window


def main(argv: Sequence[str] | None = None) -> int:
    app, window = create_application(argv)
    if DEBUG_MODE:
        window.show()
    else:
        window.showFullScreen()
    logger.info("应用程序启动（模式：%s）。", "调试" if DEBUG_MODE else "正常")
    return app.exec_()


__all__ = ["MainWindow", "create_application", "main"]
