"""Application bootstrap and main window orchestration for the modular UI."""

from __future__ import annotations

import os
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
    QTimer,
    Qt,
    QVBoxLayout,
    QWidget,
    qta,
)
from .utils.widgets import FadingStackedWidget
from .utils.responsive import get_scaler, scale, scale_size
from .pages.calibration import CalibrationPage
from .pages.login import LoginPage
from .pages.baseline import BaselineCalibrationPage
from .pages.sart import SARTPage
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
        
        # SART 模式配置
        self.sart_mode = "short"  # 默认短时模式
        self.sart_duration = 300  # 默认5分钟（300秒）

        self._setup_main_window()
        self._create_pages()
        self._connect_signals()
        self._setup_debug_shortcuts()
        
        # ✅ 恢复预加载：应用启动时预加载摄像头(使用临时目录)
        # 在校准页面时会用正确的session_dir重新初始化
        from .qt import QTimer
        QTimer.singleShot(800, self._preload_camera)

        logger.info("应用程序主窗口初始化完成。")
    
    def set_sart_mode(self, mode: str) -> None:
        """
        设置 SART 实验模式
        
        Args:
            mode: "short" (5分钟) 或 "long" (25分钟)
        """
        if mode not in ["short", "long"]:
            logger.warning(f"无效的 SART 模式: {mode}，使用默认短时模式")
            mode = "short"
        
        self.sart_mode = mode
        self.sart_duration = 60 if mode == "short" else 1500
        
        logger.info(f"✅ 已设置 SART 模式为: {mode} (时长: {self.sart_duration}秒)")
        
        # 如果 SART 页面已创建，更新其配置
        if hasattr(self, 'sart_page'):
            self.sart_page.set_mode(mode, self.sart_duration)

    def _setup_main_window(self) -> None:
        self.setWindowTitle('非接触人员状态评估系统')
        self.setStyleSheet("""
                   QStackedWidget {
                       background: qlineargradient(
                           x1:0, y1:0, x2:1, y2:1,
                           stop:0 #E5F7F9,
                           stop:0.5 #F2FBFC,
                           stop:1 white
                       );
                   }
               """)
        
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
        # ⭐ 设置从左上到右下的渐变背景
        self.stack.setStyleSheet("""
                         QStackedWidget {
                             background: qlineargradient(
                                 x1:0, y1:0, x2:1, y2:1,
                                 stop:0 #E5F7F9,
                                 stop:1 white
                             );
                         }
                     """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 使用响应式边距，减小上下边距以适配1080p显示器
        side_margin = scale(20)
        top_margin = scale(10)  # 顶部边距减小
        spacing = scale(12)  # 间距也减小
        bottom_margin = scale(30)  # 底部边距进一步减小
        main_layout.setContentsMargins(side_margin, top_margin, side_margin, bottom_margin)
        main_layout.setSpacing(spacing)

        # 创建所有页面
        self.login_page = LoginPage(self.show_calibration_page)
        self.calibration_page = CalibrationPage()
        self.baseline_page = BaselineCalibrationPage()
        self.sart_page = SARTPage(mode=self.sart_mode, duration=self.sart_duration)
        self.test_page = TestPage()

        # 按顺序添加到堆栈
        self.stack.addWidget(self.login_page)           # 0: 登录
        self.stack.addWidget(self.calibration_page)     # 1: 设备校准
        self.stack.addWidget(self.baseline_page)        # 2: 基线校准
        self.stack.addWidget(self.sart_page)            # 3: SART实验
        self.stack.addWidget(self.test_page)            # 4: 测试主流程

        main_layout.addWidget(self.stack, 1)

        # brain_load_tip 占位符（不显示文字，仅保持布局）
        self.brain_load_tip = QLabel("")  # 空文本占位
        self.brain_load_tip.setAlignment(Qt.AlignCenter)
        self.brain_load_tip.setStyleSheet("""
            QLabel {
                background: transparent;
                color: transparent;
                font-size: 0px;
                padding: 0px;
                margin: 0px;
            }
        """)
        self.brain_load_tip.setVisible(False)  # 默认隐藏
        main_layout.addWidget(self.brain_load_tip)

        self.stack.setCurrentWidget(self.login_page)

    def _connect_signals(self) -> None:
        self.calibration_page.calibration_finished.connect(self.show_baseline_prompt_in_test)  # 修改：显示基线提示而不是直接进入基线
        self.baseline_page.baseline_finished.connect(self.show_sart_prompt_in_test)  # 修改：显示SART提示而不是直接进入SART
        self.sart_page.sart_finished.connect(self.show_test_page)

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
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(0)

    def _debug_show_calibration(self) -> None:
        logger.info("调试快捷键：跳转到校准页面")
        if not getattr(self, "current_user", None):
            self.current_user = "debug"
        try:
            self.test_page.set_current_user(self.current_user)
        except Exception as exc:  # noqa: BLE001
            logger.debug("同步调试用户名失败: %s", exc)
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(1)

    def _debug_show_test(self) -> None:
        logger.info("调试快捷键：跳转到测试页面（显示基线提示）")
        if not getattr(self, "current_user", None):
            self.current_user = "debug"
        try:
            self.test_page.set_current_user(self.current_user)
            self.sart_page.set_session_dir(self.test_page.session_dir if hasattr(self.test_page, 'session_dir') else 'recordings')
        except Exception as exc:  # noqa: BLE001
            logger.debug("同步调试用户名失败: %s", exc)
        
        # 后门跳过时也要停止多模态数据采集
        logger.info("⏭️ 调试后门跳过SART，停止多模态数据采集...")
        try:
            multidata_stop_collection()
            logger.info("✅ 多模态数据采集已停止（调试后门）")
        except Exception as exc:
            logger.error(f"❌ 停止多模态数据采集失败（调试后门）: {exc}")
        
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(4)  # 跳到测试页面（索引4）
        
        # 显示基线提示页面（索引0），而不是情绪检测
        if hasattr(self.test_page, 'answer_stack'):
            self.test_page.answer_stack.setCurrentIndex(0)
            logger.info("✅ 已显示基线校准提示页面")

    def show_calibration_page(self, username: str) -> None:
        logger.info("正在切换到校准页面...")
        self.current_user = username or 'anonymous'
        try:
            self.test_page.set_current_user(self.current_user)
        except Exception as exc:  # noqa: BLE001
            logger.warning("同步用户名到测试页失败: %s", exc)
        
        # ⚠️ 修复：在切换到校准页面时就创建session_dir，避免EEG预连接和后续数据使用不同目录
        # 这样确保整个测试流程（校准→基线→文本QA→血压→舒尔特）使用同一个session_dir
        if not hasattr(self.test_page, 'session_dir') or not self.test_page.session_dir:
            from datetime import datetime
            import os
            session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            user_dir = self.current_user or 'anonymous'
            
            # 使用项目根目录的绝对路径
            project_root = Path(__file__).parent.parent.parent
            session_dir = os.path.join(project_root, "recordings", user_dir, session_timestamp)
            
            # 保存到 test_page，供所有后续阶段使用
            self.test_page.session_timestamp = session_timestamp
            self.test_page.session_dir = session_dir
            
            logger.info(f"🆕 创建整个测试会话的session目录: {session_dir}")
            
            # ✅ 更新session_dir：即使摄像头已预加载，也要用正确的session_dir重新初始化
            # 这样可以确保后续录制的文件保存到正确的目录
            logger.info(f"🎥 用正确的session_dir更新摄像头服务: {session_dir}")
            from .utils.helpers import init_camera
            
            def on_camera_update_finished(success: bool) -> None:
                if success:
                    logger.info("✅ 摄像头session_dir更新成功")
                    self.camera_preloaded = True
                else:
                    logger.warning("⚠️ 摄像头session_dir更新失败")
                    self.camera_preloaded = False
            
            try:
                # 传递正确的session_dir，更新后端AV服务的保存路径
                init_camera(on_camera_update_finished, session_dir=session_dir)
            except Exception as e:
                logger.error(f"更新摄像头session_dir失败: {e}")
                # 即使更新失败也标记为已加载，校准页面会fallback到自己初始化
                self.camera_preloaded = False
        else:
            logger.info(f"✅ 已有session目录: {self.test_page.session_dir}")
        
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(1)

    def show_baseline_page(self) -> None:
        """切换到基线校准页面"""
        logger.info("正在切换到基线校准页面...")
        
        # ✅ session_dir应该已经在校准页面创建了，这里只是防御性检查
        if not hasattr(self.test_page, 'session_dir') or not self.test_page.session_dir:
            logger.warning("⚠️ session_dir未在校准阶段创建，现在创建（这不应该发生）")
            from datetime import datetime
            import os
            session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            user_dir = self.current_user or 'anonymous'
            
            project_root = Path(__file__).parent.parent.parent
            session_dir = os.path.join(project_root, "recordings", user_dir, session_timestamp)
            
            self.test_page.session_timestamp = session_timestamp
            self.test_page.session_dir = session_dir
            
            logger.info(f"⚠️ 补救：创建会话目录: {session_dir}")
        else:
            logger.info(f"✅ 使用校准阶段创建的session目录: {self.test_page.session_dir}")
        
        # 传递时间戳列表和会话信息（包括保存回调函数）
        if hasattr(self.test_page, 'part_timestamps'):
            save_callback = getattr(self.test_page, '_save_timestamp_immediately', None)
            self.baseline_page.set_part_timestamps(self.test_page.part_timestamps, save_callback)
        
        # 传递会话信息（用于EEG采集）
        logger.info(f"📂 传递会话信息到基线页面:")
        logger.info(f"   - session_dir = {self.test_page.session_dir}")
        logger.info(f"   - current_user = {self.current_user}")
        self.baseline_page.set_session_info(
            self.test_page.session_dir,
            self.current_user
        )
        
        # 🔥 关键修改：在进入基线页面前启动EEG采集，保持整个流程连续
        self._start_eeg_collection_for_session()
        
        # ✅ 重置基线页面状态（防止上次的完成提示残留）
        self.baseline_page.reset()
        logger.info("✅ 已重置基线校准页面状态")
        
        self.brain_load_tip.setVisible(True)
        self.stack.fade_to_index(2)
        # 基线页面会在showEvent()中自动开始
    
    def _start_eeg_collection_for_session(self) -> None:
        """为整个测试会话启动EEG采集（异步，非阻塞）
        
        ⚠️ 注意：如果EEG已经在校准阶段启动（calibration页面的预连接），
        后端会返回 'already-running' 状态，这是正常的。重要的是确保
        使用的是当前的 session_dir。
        """
        from ..utils_common.thread_process_manager import get_thread_manager
        thread_manager = get_thread_manager()
        
        def start_eeg():
            try:
                from ..services.backend_proxy import eeg_start
                result = eeg_start(
                    username=self.current_user or 'anonymous',
                    save_dir=self.test_page.session_dir,
                    part=1
                )
                status = result.get('status', '').lower()
                
                if status == 'already-running':
                    # EEG已在运行（可能在校准阶段启动）
                    old_dir = result.get('save_dir', 'unknown')
                    if old_dir != os.path.join(self.test_page.session_dir, 'eeg'):
                        logger.warning(f"⚠️ EEG已在运行但目录不匹配！")
                        logger.warning(f"   当前EEG目录: {old_dir}")
                        logger.warning(f"   期望session目录: {self.test_page.session_dir}/eeg")
                        logger.info("💡 建议：在校准页面时应该已经设置了正确的session_dir")
                    else:
                        logger.info(f"✅ EEG采集已在运行（校准阶段启动），继续使用: {old_dir}")
                elif status == 'started':
                    logger.info(f"🧠 整个测试会话的EEG采集已启动: {result}")
                    logger.info(f"📂 EEG数据保存到: {self.test_page.session_dir}/eeg/")
                else:
                    logger.warning(f"⚠️ EEG启动返回未知状态: {status}")
                    
            except Exception as e:
                logger.error(f"❌ 启动测试会话EEG采集失败: {e}")
                logger.info("测试将继续运行，但不会记录EEG数据")
        
        thread_manager.submit_data_task(
            start_eeg,
            task_name="测试会话EEG采集启动"
        )
    
    def show_baseline_prompt_in_test(self) -> None:
        """设备校准完成后，跳转到test页面显示基线提示"""
        logger.info("设备校准完成，显示基线校准提示页面...")
        
        # 切换到test页面
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(4)
        self.test_page.start_eeg_collection()  # 启动SART阶段的EEG采集
        
        # 显示基线提示页面（answer_stack中的第0个widget）
        if hasattr(self.test_page, 'answer_stack'):
            self.test_page.answer_stack.setCurrentIndex(0)
            # 隐藏摄像头和按钮
            if hasattr(self.test_page, '_hide_camera_and_buttons'):
                self.test_page._hide_camera_and_buttons()
            # 更新导航栏状态
            if hasattr(self.test_page, '_update_stage_nav_status'):
                self.test_page._update_stage_nav_status()
            logger.info("✅ 已显示基线校准提示页面")
    
    def show_sart_prompt_in_test(self) -> None:
        """基线校准完成后，返回test页面显示SART提示"""
        logger.info("基线校准完成，显示SART实验提示页面...")
        
        # 切换到test页面
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(4)
        
        # 显示SART提示页面（answer_stack中的第1个widget）
        if hasattr(self.test_page, 'answer_stack'):
            self.test_page.answer_stack.setCurrentIndex(1)
            # 隐藏摄像头和按钮
            if hasattr(self.test_page, '_hide_camera_and_buttons'):
                self.test_page._hide_camera_and_buttons()
            # 更新导航栏状态
            if hasattr(self.test_page, '_update_stage_nav_status'):
                self.test_page._update_stage_nav_status()
            logger.info("✅ 已显示SART实验提示页面")
    
    def show_sart_page(self) -> None:
        """切换到SART实验页面"""
        logger.info("正在切换到SART实验页面...")
        
        # 同步时间戳列表（包括保存回调函数）
        if hasattr(self.test_page, 'part_timestamps'):
            save_callback = getattr(self.test_page, '_save_timestamp_immediately', None)
            self.sart_page.set_part_timestamps(self.test_page.part_timestamps, save_callback)
        
        # 确保会话目录已创建（可能已在基线阶段创建）
        if not hasattr(self.test_page, 'session_dir') or not self.test_page.session_dir:
            from datetime import datetime
            import os
            session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            user_dir = self.current_user or 'anonymous'
            
            # 🐛 修复：使用项目根目录的绝对路径，而不是相对路径
            project_root = Path(__file__).parent.parent.parent
            session_dir = os.path.join(project_root, "recordings", user_dir, session_timestamp)
            
            # 保存到 test_page 以便后续使用
            self.test_page.session_timestamp = session_timestamp
            self.test_page.session_dir = session_dir
            
            logger.info(f"为 SART 创建会话目录: {session_dir}")
        
        # 传递会话信息（用于EEG采集和结果保存）
        logger.info(f"📂 传递会话信息到SART页面:")
        logger.info(f"   - session_dir = {self.test_page.session_dir}")
        logger.info(f"   - current_user = {self.current_user}")
        self.sart_page.set_session_info(
            self.test_page.session_dir,
            self.current_user
        )
        
        # ✅ 重置SART页面状态（防止上次的完成提示残留）
        self.sart_page.reset()
        logger.info("✅ 已重置SART实验页面状态")
        
        self.brain_load_tip.setVisible(True)
        self.stack.fade_to_index(3)
        
        # SART会在showEvent()中自动开始测试
    
    def show_test_page(self) -> None:
        """切换到测试主流程页面"""
        logger.info("正在切换到测试页面（文本问答、血压、舒尔特）...")
        
        # SART测试结束，停止多模态数据采集（EEG、RGB等）
        logger.info("📊 SART测试已完成，停止多模态数据采集...")
        try:
            multidata_stop_collection()
            logger.info("✅ 多模态数据采集已停止，疲劳度评估将自动开始")
        except Exception as exc:
            logger.error(f"❌ 停止多模态数据采集失败: {exc}")
        
        self.brain_load_tip.setVisible(False)
        self.stack.fade_to_index(4)
        self.test_page.start_test()

    def _preload_camera(self) -> None:
        """在后台预加载摄像头，不阻塞UI"""
        from .utils.helpers import init_camera
        
        logger.info("🎥 开始预加载摄像头（后台异步，使用临时目录）...")
        
        def on_preload_finished(success: bool) -> None:
            if success:
                logger.info("✅ 摄像头预加载成功，校准页面将更新为正确的session_dir")
                self.camera_preloaded = True
            else:
                logger.warning("⚠️ 摄像头预加载失败，将在校准页重试")
                self.camera_preloaded = False
        
        try:
            # 预加载时不传session_dir，使用默认的'recordings'
            # 在校准页面会用正确的session_dir重新初始化
            init_camera(on_preload_finished, session_dir=None)
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
        try:
            stop_recognition()
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
            try:
                self.test_page.schulte_widget.reset_for_next_stage()
            except Exception as exc:  # noqa: BLE001
                logger.debug("关闭应用时重置舒尔特widget失败: %s", exc)
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

    lifecycle_manager = get_lifecycle_manager()
    status = lifecycle_manager.get_all_status()
    if not status.get('is_initialized'):
        lifecycle_manager.start_all()
    app.aboutToQuit.connect(lambda: lifecycle_manager.shutdown_all())
    app.setProperty("lifecycle_manager", lifecycle_manager)

    _apply_style(app)

    window = MainWindow()
    
    # 📍 解析命令行参数以设置 SART 模式
    import argparse
    parser = argparse.ArgumentParser(description='非接触人员状态评估系统')
    parser.add_argument('--sart-mode', 
                       choices=['short', 'long'], 
                       default='short',
                       help='SART实验模式: short(5分钟/低负荷) 或 long(25分钟/疲劳诱发), 默认short')
    
    # 解析已有的参数（去掉Qt自己的参数）
    known_args, _ = parser.parse_known_args(args[1:])  # 跳过程序名
    
    # 设置 SART 模式
    if known_args.sart_mode:
        window.set_sart_mode(known_args.sart_mode)
        logger.info(f"✅ 从命令行参数设置 SART 模式: {known_args.sart_mode}")
    
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
