"""
多模态数据采集器 (Multi-Modal Data Collector)

功能：
- 采集多种传感器数据：
  - RGB可见光 (RealSense)
  - 深度数据 (RealSense)
  - 眼动追踪 (Tobii)

主要特性：
1. 保持5秒滑动窗口队列（线程安全）
2. 眼动采样频率为RGB/深度的5倍（单线程、计数器实现）
3. 深度视频以灰度外观保存（3通道BGR容器，便于播放与后处理）
4. 支持真实设备与模拟数据两种模式
5. UI与采集解耦：采集器仅负责采集与保存；
   - `RealTimeDisplayWindow` 经由全局接口获取数据
   - 疲劳度评分在窗口中每5秒异步计算与更新
6. 提供全局轻量接口：启动/停止采集、获取数据与文件路径、显示窗口管理

简要使用：
    # 启动采集
    from get_multidata import start_collection
    start_collection("username", enable_display=True)

    # 拉取数据
    from get_multidata import get_latest_data, get_current_data
    latest = get_latest_data()
    current = get_current_data()

    # 停止采集
    from get_multidata import stop_collection
    stop_collection()
"""

import os
import time
import json
import logging
from datetime import datetime
from collections import deque
from threading import Thread, Lock, Event
from typing import Optional, Dict, Any, Tuple
import numpy as np


# 尝试导入GUI相关库
try:
    from PyQt5.QtWidgets import (
        QWidget, QVBoxLayout, QHBoxLayout, QLabel, QApplication,
        QMainWindow, QPushButton, QFrame
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QImage, QPixmap, QFont
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print("警告：PyQt5不可用，将无法显示实时影像窗口")

# 尝试导入传感器库
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("警告：缺少opencv-python库，将使用模拟RGB数据")

try:
    import pyrealsense2 as rs
    HAS_REALSENSE = True
except ImportError:
    HAS_REALSENSE = False
    print("警告：缺少pyrealsense2库，将使用模拟深度数据")

try:
    from tobii import TobiiEngine
    HAS_TOBII = True
except ImportError:
    HAS_TOBII = False
    print("警告：缺少tobii库，将使用模拟眼动数据")


# 模型推理导入（带兜底占位实现）
try:
    from model_inference import infer_fatigue_score
except Exception:
    def infer_fatigue_score(rgb_frames, depth_frames, eyetrack_samples):  # type: ignore
        return 0.0

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealTimeDisplayWindow(QWidget):
    """实时影像显示窗口"""
    # 从后台线程更新分数与图像的信号
    scoreUpdated = pyqtSignal(float)
    imageUpdated = pyqtSignal(object)  # 传递numpy数组

    # 复用的样式常量，减少重复创建
    STYLE_STATUS_DEFAULT = """
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 6px;
                background-color: #f5f5f5;
                border-radius: 4px;
                margin: 3px;
                border: 1px solid #ddd;
            }
        """
    STYLE_STATUS_RUNNING = """
            QLabel {{
                color: {color};
                font-size: 12px;
                padding: 6px;
                background-color: #f5f5f5;
                border-radius: 4px;
                margin: 3px;
                border: 2px solid {color};
            }}
        """
    STYLE_STATUS_STOPPED = """
            QLabel {
                color: #999;
                font-size: 12px;
                padding: 6px;
                background-color: #f5f5f5;
                border-radius: 4px;
                margin: 3px;
                border: 1px solid #ccc;
            }
        """
    STYLE_STATUS_ERROR = """
            QLabel {
                color: #F44336;
                font-size: 12px;
                padding: 6px;
                background-color: #f5f5f5;
                border-radius: 4px;
                margin: 3px;
                border: 1px solid #F44336;
            }
        """

    def __init__(self, title="多模态数据采集实时显示", width=640, height=480):
        super().__init__()
        self.title = title
        self.width = width
        self.height = height
        self.fatigue_score = 0.0  # 疲劳度分数，外部可设置
        self._target_display_size = (self.width, self.height)

        self._init_ui()
        self._setup_window()

        # 启动状态更新定时器
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self._update_collection_status)
        self.status_timer.start(1000)  # 每秒更新一次状态

        # 立即更新一次状态，避免显示"初始化中"
        self._update_collection_status()

        # 连接分数更新信号
        self.scoreUpdated.connect(self.set_fatigue_score)
        # 连接图像更新信号，确保跨线程安全更新
        self.imageUpdated.connect(self._apply_image)

        # 启动每5秒的评分调度器
        self.score_timer = QTimer(self)
        self.score_timer.timeout.connect(self._schedule_score_update)
        self.score_timer.start(5000)

    def _init_ui(self):
        """初始化用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # 标题
        title_label = QLabel(self.title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        title_label.setStyleSheet("color: #1976D2; margin-bottom: 1px;")
        layout.addWidget(title_label)

        # 多模态数据采集状态显示
        self.collection_status_label = QLabel("数据采集状态: 初始化中...")
        self.collection_status_label.setAlignment(Qt.AlignCenter)
        self.collection_status_label.setStyleSheet("""
            QLabel {
                color: #666;
                font-size: 12px;
                padding: 6px;
                background-color: #f5f5f5;
                border-radius: 4px;
                margin: 3px;
                border: 1px solid #ddd;
            }
        """)
        layout.addWidget(self.collection_status_label)

        # 影像显示区域
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.width, self.height)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                border: 2px solid #ddd;
                border-radius: 8px;
            }
        """)
        self.image_label.setText("等待影像数据...")
        layout.addWidget(self.image_label, 0, Qt.AlignCenter)

        # 疲劳度分数显示
        fatigue_layout = QHBoxLayout()
        fatigue_layout.setAlignment(Qt.AlignCenter)

        fatigue_label = QLabel("疲劳度分数:")
        fatigue_label.setFont(QFont("Microsoft YaHei", 6))
        fatigue_label.setStyleSheet("""
            QLabel {
                color: #1976D2;
                background-color: #e3f2fd;
                padding: 1px 2px;
                border-radius: 5px;
                border: 1px solid #1976D2;
            }
        """)

        self.score_text = QLabel("0.00")
        self.score_text.setFont(QFont("Microsoft YaHei", 6, QFont.Bold))
        self.score_text.setStyleSheet("""
            QLabel {
                color: #1976D2;
                background-color: #e3f2fd;
                padding: 1px 2px;
                border-radius: 5px;
                border: 1px solid #1976D2;
            }
        """)

        fatigue_layout.addWidget(fatigue_label)
        fatigue_layout.addWidget(self.score_text)
        layout.addLayout(fatigue_layout)

        # 状态信息
        self.status_label = QLabel("状态: 等待数据...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self.status_label)

    def _setup_window(self):
        """设置窗口属性"""
        self.setWindowTitle(self.title)
        self.setFixedSize(self.width + 30, self.height + 90)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        screen = QApplication.desktop().screenGeometry()
        self.move(screen.width() - self.width - 50, 50)

        # 设置窗口样式
        self.setStyleSheet("""
            QWidget {
                background-color: white;
                font-family: "Microsoft YaHei";
            }
        """)

    def update_image(self, image_array):
        """线程安全的图像更新入口：发射信号，由主线程应用。"""
        self.imageUpdated.emit(image_array)

    def _apply_image(self, image_array):
        """在主线程中应用图像到UI。"""
        if image_array is None:
            self.image_label.setText("无影像数据")
            return

        try:
            if len(image_array.shape) == 3:
                height, width, channel = image_array.shape
                bytes_per_line = 3 * width

                rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)
                target_w, target_h = self._target_display_size
                scaled_pixmap = pixmap.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                self.image_label.setPixmap(scaled_pixmap)
                self.status_label.setText("状态: 影像更新成功")
            else:
                self.image_label.setText("无效的影像数据格式")
        except Exception as e:
            logger.warning(f"更新影像显示失败: {e}")
            self.image_label.setText("影像更新失败")
            self.status_label.setText("状态: 影像更新失败")

    def set_fatigue_score(self, score: float):
        """设置疲劳度分数"""
        self.fatigue_score = score
        self.score_text.setText(f"{score:.2f}")

        # 根据分数设置颜色
        if score >= 80:
            color = "#4CAF50"  # 绿色 - 正常
        elif score >= 60:
            color = "#FF9800"  # 橙色 - 轻度疲劳
        else:
            color = "#F44336"  # 红色 - 重度疲劳

        self.score_text.setStyleSheet(f"""
            QLabel {{
                color: {color};
                background-color: #f5f5f5;
                padding: 5px 10px;
                border-radius: 4px;
                border: 2px solid {color};
                font-weight: bold;
            }}
        """)

        self.status_label.setText(f"状态: 疲劳度分数已更新 - {score:.2f}")

    def _schedule_score_update(self):
        """在后台线程中计算疲劳度分数，完成后用信号更新UI。"""
        try:
            worker = Thread(target=self._compute_score_background, daemon=True)
            worker.start()
        except Exception as e:
            logger.debug(f"调度评分任务失败: {e}")

    def _compute_score_background(self):
        """后台线程：获取最近窗口数据并进行推理。"""
        try:
            # 通过全局接口取数据快照，避免直接依赖内部实现
            if 'get_current_data' in globals():
                data = get_current_data()
            else:
                data = {}
            rgb_list = data.get('rgb', [])
            depth_list = data.get('depth', [])
            eye_list = data.get('eyetrack', [])
            score = infer_fatigue_score(rgb_list, depth_list, eye_list)
            self.scoreUpdated.emit(float(score))
        except Exception as e:
            logger.debug(f"后台评分失败: {e}")

    def _update_collection_status(self):
        """定时更新数据采集状态"""
        try:
            # 检查全局采集器状态
            if '_global_collector' in globals() and globals()['_global_collector']:
                collector = globals()['_global_collector']
                if collector.running:
                    # 获取数据摘要
                    summary = collector.get_data_summary()
                    if summary:
                        fill_percentage = summary.get('fill_percentage', 0)
                        total_samples = summary.get('total_samples', 0)
                        queue_capacity = summary.get('queue_capacity', 0)

                        # 根据填充率设置颜色和状态
                        if fill_percentage >= 80:
                            color = "#4CAF50"  # 绿色
                            status_text = "数据采集正常"
                        elif fill_percentage >= 50:
                            color = "#FF9800"  # 橙色
                            status_text = "数据采集中"
                        else:
                            color = "#2196F3"  # 蓝色
                            status_text = "数据采集启动中"

                        status_text = f"{status_text} ({total_samples}/{queue_capacity})"
                        self.collection_status_label.setText(status_text)
                        self.collection_status_label.setStyleSheet(self.STYLE_STATUS_RUNNING.format(color=color))
                    else:
                        self.collection_status_label.setText("数据采集状态: 运行中")
                        self.collection_status_label.setStyleSheet(self.STYLE_STATUS_DEFAULT)
                else:
                    self.collection_status_label.setText("数据采集状态: 已停止")
                    self.collection_status_label.setStyleSheet(self.STYLE_STATUS_STOPPED)
            else:
                # 检查是否有启动采集的函数被调用过
                if 'start_collection' in globals():
                    self.collection_status_label.setText("数据采集状态: 等待启动")
                    self.collection_status_label.setStyleSheet(self.STYLE_STATUS_RUNNING.format(color="#FF9800"))
                else:
                    self.collection_status_label.setText("数据采集状态: 未初始化")
                    self.collection_status_label.setStyleSheet(self.STYLE_STATUS_STOPPED)
        except Exception as e:
            # 记录错误信息，帮助调试
            logger.error(f"更新数据采集状态时发生错误: {e}")
            self.collection_status_label.setText("数据采集状态: 错误")
            self.collection_status_label.setStyleSheet(self.STYLE_STATUS_ERROR)


# =============================================================================
# 数据采集类: MultiModalDataCollector
# 采集、保存、维护数据队列
# =============================================================================

class MultiModalDataCollector:
    """多模态数据采集器"""

    def __init__(self, username: str = "anonymous", part: int = 1, queue_duration: float = 5.0, save_dir: str = None):
        self.username = username
        self.part = part
        self.queue_duration = queue_duration
        self.sample_rate = 30 # 采样率 30Hz
        self.save_dir = save_dir  # 保存路径 - 优先使用外部传入的路径
        # RGB/深度按每5次循环采样一次，眼动每次循环采样（眼动频率为RGB/深度的5倍）
        self.rgb_depth_interval = 5
        self._cycle_count = 0

        # 计算队列长度
        self.queue_length = int(self.queue_duration * 15)

        # 数据队列（5秒滑动窗口）
        self.rgb_queue = deque(maxlen=self.queue_length)
        self.depth_queue = deque(maxlen=self.queue_length)
        self.eyetrack_queue = deque(maxlen=self.queue_length * self.rgb_depth_interval)
        self.timestamp_queue = deque(maxlen=self.queue_length)

        # 调用时间戳记录
        self.call_timestamps = []

        # 线程控制
        self.running = False
        self.collection_thread = None
        self.stop_event = Event()

        # 线程锁
        self.data_lock = Lock()

        # 设备实例
        self.rs_pipeline = None
        self.tobii_engine = None

        # 初始化设备
        self._init_devices()

        logger.info(f"多模态数据采集器初始化完成，队列长度: {self.queue_length}")
        if self.save_dir:
            logger.info(f"使用外部指定的保存目录: {self.save_dir}")

    def _init_devices(self):
        """初始化各种传感器设备"""
        # 尝试初始化RealSense设备（RGB + 深度）
        if HAS_REALSENSE:
            try:
                self._init_realsense()
            except Exception as e:
                logger.error(f"RealSense设备初始化失败: {e}")
        else:
            logger.info("RealSense库不可用，将使用模拟深度数据")

        # 尝试初始化眼动追踪设备
        if HAS_TOBII:
            try:
                self._init_tobii()
            except Exception as e:
                logger.error(f"Tobii眼动设备初始化失败: {e}")
        else:
            logger.info("Tobii库不可用，将使用模拟眼动数据")

    def _init_realsense(self):
        """初始化RealSense设备"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if not devices:
                logger.warning("未检测到RealSense设备")
                return
            logger.info(f"检测到 {len(devices)} 个RealSense设备")

            # 创建pipeline
            self.rs_pipeline = rs.pipeline()
            self.rs_config = rs.config()

            # 配置RGB和深度流
            self.rs_config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            # 启动pipeline
            self.rs_profile = self.rs_pipeline.start(self.rs_config)

            # 获取深度传感器
            depth_sensor = self.rs_profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()

            # 创建对齐器
            self.rs_align = rs.align(rs.stream.color)

            logger.info("RealSense设备初始化成功")

        except Exception as e:
            logger.error(f"RealSense初始化失败: {e}")
            self.rs_pipeline = None

    def _init_tobii(self):
        """初始化Tobii眼动追踪设备"""
        try:
            self.tobii_engine = TobiiEngine()
            self.tobii_engine.__enter__()
            logger.info("Tobii眼动设备初始化成功")
        except Exception as e:
            logger.error(f"Tobii初始化失败: {e}")
            self.tobii_engine = None

    def _create_save_directory(self):
        """创建数据保存目录"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            base_dir = './recordings'
            user_dir = os.path.join(base_dir, self.username)
            timestamp_dir = os.path.join(user_dir, timestamp, 'fatigue')
            os.makedirs(timestamp_dir, exist_ok=True)
            self.save_dir = timestamp_dir
            logger.info(f"创建多模态数据保存目录: {self.save_dir}")

            # 创建数据文件
            self._create_data_files()

        except Exception as e:
            logger.error(f"创建保存目录失败: {e}")

    def _create_data_files(self):
        """创建数据保存文件"""
        self.save_dir = os.path.join(self.save_dir, 'fatigue')
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            # RGB视频文件
            if HAS_CV2:
                rgb_path = os.path.join(self.save_dir, f'rgb{self.part}.avi')
                self.rgb_path = rgb_path  # 保存路径
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.rgb_writer = cv2.VideoWriter(rgb_path, fourcc, 15.0, (1920, 1080))
                logger.info(f"RGB视频文件创建: {rgb_path}")

            # 深度视频文件（以灰度形式保存，便于后续算法处理与查看）
            if HAS_CV2:
                depth_video_path = os.path.join(self.save_dir,  f'depth{self.part}.avi')
                self.depth_video_path = depth_video_path
                depth_fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.depth_writer = cv2.VideoWriter(depth_video_path, depth_fourcc, 15.0, (1920, 1080))
                logger.info(f"深度视频文件创建: {depth_video_path}")

            # 眼动数据文件
            eyetrack_path = os.path.join(self.save_dir, f'eyetrack{self.part}.json')
            self.eyetrack_path = eyetrack_path
            logger.info(f"眼动数据文件路径: {eyetrack_path}")

            # 元数据文件
            metadata = {
                'username': self.username,
                'start_time': datetime.now().isoformat(),
                'sample_rate': self.sample_rate,
                'queue_duration': self.queue_duration,
                'queue_length': self.queue_length,
                'available_devices': {
                    'realsense': HAS_REALSENSE,
                    'tobii': HAS_TOBII,
                    'opencv': HAS_CV2
                }
            }

            metadata_path = os.path.join(self.save_dir, f'metadata{self.part}.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            logger.info("数据文件创建完成")

        except Exception as e:
            logger.error(f"创建数据文件失败: {e}")

    def start_collection(self):
        """开始数据采集"""
        if self.running:
            logger.warning("数据采集已在运行中")
            return

        logger.info("开始多模态数据采集...")

        # 如果没有指定保存目录，则自动创建
        if not self.save_dir:
            self._create_save_directory()
        else:
            # 使用外部指定的目录，创建数据文件
            self._create_data_files()

        # 启动采集线程
        self.running = True
        self.stop_event.clear()
        self.collection_thread = Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()

        logger.info("数据采集线程已启动")

    def stop_collection(self):
        """停止数据采集"""
        if not self.running:
            return

        logger.info("停止多模态数据采集...")

        self.running = False
        self.stop_event.set()

        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=2.0)

        # 保存剩余数据
        self._save_remaining_data()

        # 关闭设备
        self._cleanup_devices()

        # 关闭实时显示窗口（通过全局接口）
        close_realtime_display()

        logger.info("数据采集已停止")

    def _collection_loop(self):
        """数据采集主循环"""
        last_sample_time = time.time()

        while self.running and not self.stop_event.is_set():
            try:
                current_time = time.time()

                # 控制采样率
                if current_time - last_sample_time >= 1.0 / (self.sample_rate * self.rgb_depth_interval * 2):
                    self._collect_sample()
                    last_sample_time = current_time

                time.sleep(0.001)  # 短暂休眠避免CPU占用过高

            except Exception as e:
                logger.error(f"数据采集循环出错: {e}")
                time.sleep(0.1)

    def _collect_sample(self):
        """采集一个数据样本"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._cycle_count += 1

        # 采集眼动数据（每次循环都采集）
        eyetrack_data = self._collect_eyetrack()

        # 采集RGB/深度（每5次循环采集一次）
        rgb_data = None
        depth_data = None
        if self._cycle_count % self.rgb_depth_interval == 0:
            rgb_data, depth_data = self._collect_aligned_images()

        # 更新队列
        with self.data_lock:
            # 每次循环都记录眼动数据
            eyetrack_features = self._extract_eyetrack_features(eyetrack_data)
            self.eyetrack_queue.append(eyetrack_features)
            # 仅在采集到RGB/深度样本时，更新对应队列与时间戳
            if rgb_data is not None:
                self.rgb_queue.append(rgb_data)
            if depth_data is not None:
                self.depth_queue.append(depth_data)
            if rgb_data is not None or depth_data is not None:
                self.timestamp_queue.append(timestamp)

        # 保存原始数据
        self._save_raw_data(rgb_data, depth_data, eyetrack_data, timestamp)

        # 更新实时显示窗口（通过全局接口）
        if rgb_data is not None:
            update_display_image(rgb_data)
    
    def _collect_aligned_images(self):
        if self.rs_pipeline and HAS_REALSENSE:
            try:
                frames = self.rs_pipeline.wait_for_frames(timeout_ms=100)
                aligned_frames = self.rs_align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                if color_frame:
                    rgb_image = np.asanyarray(color_frame.get_data())
                if depth_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                return rgb_image.copy(), depth_image.copy()
            except Exception as e:
                logger.debug(f"对齐数据采集失败: {e}")

    def _depth_to_bgr(self, depth_image: np.ndarray) -> np.ndarray:
        """将16位深度图转换为适合视频写入的灰度外观BGR图。

        说明：为确保视频写入器兼容性，这里返回的是将灰度复制到三个通道后的BGR图像，
        显示效果为灰度，但底层依然是3通道，便于通用播放器解码。
        """
        try:
            # 将16位深度缩放为8位灰度。alpha可根据具体相机量程调整。
            depth_8u = cv2.convertScaleAbs(depth_image, alpha=0.03)
            # 复制为三通道BGR以适配常见编码器/播放器
            gray_bgr = cv2.cvtColor(depth_8u, cv2.COLOR_GRAY2BGR)
            return gray_bgr
        except Exception as e:
            # 兜底：若转换失败，则返回全黑帧，避免中断写入
            logger.debug(f"深度可视化转换失败: {e}")
            return np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

    def _collect_eyetrack(self):
        """采集眼动数据"""
        if self.tobii_engine and HAS_TOBII:
            try:
                data = self.tobii_engine.read()
                return data
            except Exception as e:
                logger.debug(f"眼动数据采集失败: {e}")

        # 返回模拟数据
        return self._generate_mock_eyetrack()

    def _generate_mock_eyetrack(self):
        """生成模拟眼动数据"""
        return {
            'gaze_point': [float(np.random.uniform(0, 640)), float(np.random.uniform(0, 480))],
            'head_pose': [
                float(np.random.uniform(-30, 30)),  # roll
                float(np.random.uniform(-30, 30)),  # pitch
                float(np.random.uniform(-30, 30)),  # yaw
                float(np.random.uniform(-0.3, 0.3)),  # tx (m)
                float(np.random.uniform(-0.3, 0.3)),  # ty (m)
                float(np.random.uniform(0.3, 1.0))   # tz (m)
            ],
            'timestamp': time.time(),
            'valid': bool(np.random.choice([True, False], p=[0.8, 0.2]))
        }

    def _extract_eyetrack_features(self, eyetrack_data):
        """从眼动原始数据中提取 gaze_point(2) + head_pose(6) 共8个数值。
        若缺失或格式异常，用0补齐到长度8。
        """
        try:
            gaze_point_raw = None
            head_pose_raw = None

            if isinstance(eyetrack_data, dict):
                gaze_point_raw = eyetrack_data.get('gaze_point', eyetrack_data.get('gaze'))
                head_pose_raw = eyetrack_data.get('head_pose', eyetrack_data.get('head'))
            
            features = gaze_point_raw + head_pose_raw

            # 最终确保长度为8
            if len(features) < 8:
                features += [0.0] * (8 - len(features))
            elif len(features) > 8:
                features = features[:8]
            return features
        except Exception:
            return [0.0] * 8

    def _save_raw_data(self, rgb_data, depth_data, eyetrack_data, timestamp):
        """保存原始数据到文件"""
        try:
            # 保存RGB帧到视频
            if self.rgb_writer and rgb_data is not None and HAS_CV2:
                self.rgb_writer.write(rgb_data)

            # 保存深度帧到视频（以彩色可视化形式写入）
            if hasattr(self, 'depth_writer') and self.depth_writer and depth_data is not None and HAS_CV2:
                depth_bgr = self._depth_to_bgr(depth_data)
                self.depth_writer.write(depth_bgr)

            # 眼动数据保存到JSON
            if eyetrack_data is not None:
                # 确保数据类型兼容JSON序列化
                serializable_eyetrack = self._prepare_eyetrack_for_json(eyetrack_data)
                serializable_eyetrack['timestamp'] = timestamp

                with open(self.eyetrack_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(serializable_eyetrack, ensure_ascii=False) + '\n')

        except Exception as e:
            # 只在调试模式下记录详细错误，避免日志刷屏
            if logger.isEnabledFor(logging.DEBUG):
                logger.error(f"保存原始数据失败: {e}")
            else:
                # 在INFO级别下，只记录一次错误类型
                if not hasattr(self, '_last_save_error') or self._last_save_error != type(e).__name__:
                    logger.warning(f"数据保存遇到问题: {type(e).__name__}")
                    self._last_save_error = type(e).__name__

    def _prepare_eyetrack_for_json(self, eyetrack_data):
        """准备眼动数据以兼容JSON序列化"""
        try:
            # 创建可序列化的副本
            serializable = {}
            for key, value in eyetrack_data.items():
                if isinstance(value, np.ndarray):
                    serializable[key] = value.tolist()
                elif isinstance(value, np.integer):
                    serializable[key] = int(value)
                elif isinstance(value, np.floating):
                    serializable[key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable[key] = bool(value)
                else:
                    serializable[key] = value
            return serializable
        except Exception as e:
            logger.error(f"准备眼动数据序列化失败: {e}")
            # 返回简化的数据结构
            return {
                'gaze_point': [0.0, 0.0],
                'head_pose': [0.0, 0.0, 0.0],
                'timestamp': time.time(),
                'valid': False,
                'error': 'serialization_failed'
            }

    def _save_remaining_data(self):
        """保存剩余数据"""
        try:
            # 关闭RGB视频写入器
            if self.rgb_writer and HAS_CV2:
                self.rgb_writer.release()
                logger.info("RGB视频写入器已关闭")

            # 关闭深度视频写入器
            if hasattr(self, 'depth_writer') and self.depth_writer and HAS_CV2:
                self.depth_writer.release()
                logger.info("深度视频写入器已关闭")

            # 保存调用时间戳数据
            if self.call_timestamps:
                call_timestamp_path = os.path.join(self.save_dir, f'call_timestamps{self.part}.npy')
                call_timestamp_array = np.array(self.call_timestamps)
                np.save(call_timestamp_path, call_timestamp_array)
                logger.info(f"调用时间戳数据已保存: {call_timestamp_path}")

                # 同时保存为JSON格式，便于查看
                call_timestamp_json_path = os.path.join(self.save_dir, f'call_timestamps{self.part}.json')
                call_timestamps_formatted = [
                    {
                        'timestamp': ts,
                        'datetime': datetime.fromtimestamp(ts).isoformat(),
                        'call_index': i
                    }
                    for i, ts in enumerate(self.call_timestamps)
                ]
                with open(call_timestamp_json_path, 'w', encoding='utf-8') as f:
                    json.dump(call_timestamps_formatted, f, ensure_ascii=False, indent=2)
                logger.info(f"调用时间戳JSON数据已保存: {call_timestamp_json_path}")

        except Exception as e:
            logger.error(f"保存剩余数据失败: {e}")
    
    def _cleanup_devices(self):
        """清理设备资源"""
        try:
            if self.rs_pipeline:
                self.rs_pipeline.stop()

            if self.tobii_engine:
                self.tobii_engine.__exit__(None, None, None)

        except Exception as e:
            logger.error(f"清理设备资源失败: {e}")

    def get_current_data(self) -> Dict[str, Any]:
        """获取当前队列中的数据"""
        # 记录调用时间戳
        call_timestamp = time.time()
        self.call_timestamps.append(call_timestamp)

        with self.data_lock:
            return {
                'rgb': list(self.rgb_queue),
                'depth': list(self.depth_queue),
                'eyetrack': list(self.eyetrack_queue),
                'timestamps': list(self.timestamp_queue),
                'queue_length': len(self.rgb_queue),
                'is_full': len(self.rgb_queue) >= self.queue_length
            }

    def get_latest_data(self) -> Dict[str, Any]:
        """获取最新的数据样本"""
        with self.data_lock:
            if not self.rgb_queue:
                return {}

            return {
                'rgb': self.rgb_queue[-1] if self.rgb_queue else None,
                'depth': self.depth_queue[-1] if self.depth_queue else None,
                'eyetrack': self.eyetrack_queue[-1] if self.eyetrack_queue else None,
                'timestamp': self.timestamp_queue[-1] if self.timestamp_queue else None
            }

    def get_file_paths(self) -> Dict[str, str]:
        """获取多模态数据文件的绝对路径"""
        if not self.save_dir:
            return {}

        paths = {}

        # RGB视频文件路径
        if hasattr(self, 'rgb_path'):
            if os.path.exists(self.rgb_path):
                paths['rgb'] = os.path.abspath(self.rgb_path)

        # 深度视频文件路径
        if hasattr(self, 'depth_video_path'):
            if os.path.exists(self.depth_video_path):
                paths['depth'] = os.path.abspath(self.depth_video_path)

        # 眼动数据文件路径
        if hasattr(self, 'eyetrack_path'):
            if os.path.exists(self.eyetrack_path):
                paths['eyetrack'] = os.path.abspath(self.eyetrack_path)

        # 调用时间戳数据文件路径
        call_timestamp_path = os.path.join(self.save_dir, f'call_timestamps{self.part}.npy')
        if os.path.exists(call_timestamp_path):
            paths['call_timestamps'] = os.path.abspath(call_timestamp_path)

        # 调用时间戳JSON文件路径
        call_timestamp_json_path = os.path.join(self.save_dir, f'call_timestamps{self.part}.json')
        if os.path.exists(call_timestamp_json_path):
            paths['call_timestamps_json'] = os.path.abspath(call_timestamp_json_path)

        # 元数据文件路径
        metadata_path = os.path.join(self.save_dir, f'metadata{self.part}.json')
        if os.path.exists(metadata_path):
            paths['metadata'] = os.path.abspath(metadata_path)

        return paths

    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据统计摘要"""
        try:
            with self.data_lock:
                total_samples = len(self.rgb_queue)
                queue_capacity = self.queue_length
                fill_percentage = total_samples / queue_capacity * 100 if queue_capacity > 0 else 0

                return {
                    'total_samples': total_samples,
                    'queue_capacity': queue_capacity,
                    'fill_percentage': fill_percentage,
                    'is_running': self.running,
                    'save_directory': self.save_dir
                }
        except Exception as e:
            logger.error(f"获取数据摘要失败: {e}")
            return {}


# =============================================================================
# 全局实例管理
# =============================================================================
_global_collector = None
_global_display_window = None


# =============================================================================
# 数据采集管理接口
# =============================================================================

def start_collection(username: str = "anonymous", save_dir: str = None, enable_display: bool = False,
                              part: int = 1, display_title: str = None, display_width: int = 640, display_height: int = 480):
    """
    为测试页面启动数据采集
    这个函数应该在TestPage.start_test()时调用
    
    Args:
        username: 用户名
        save_dir: 外部指定的保存目录，如果为None则自动创建
        enable_display: 是否启用实时显示窗口
        part: 采集环节（血压前后）
        display_title: 显示窗口标题
        display_width: 显示窗口宽度
        display_height: 显示窗口高度
    """
    global _global_collector

    try:
        if _global_collector and _global_collector.running:
            logger.warning("全局采集器已在运行中")
            return _global_collector

        _global_collector = MultiModalDataCollector(username, save_dir=save_dir, part=part)
        _global_collector.start_collection()
        logger.info(f"为测试页面启动数据采集，用户: {username}")
        if save_dir:
            logger.info(f"使用外部指定的保存目录: {save_dir}")

        # 如果启用实时显示，创建显示窗口
        if enable_display:
            enable_realtime_display(True, display_title, display_width, display_height)

        return _global_collector
    except Exception as e:
        logger.error(f"启动测试数据采集失败: {e}")
        return None


def stop_collection():
    """
    为测试页面停止数据采集
    这个函数应该在TestPage结束时调用
    """
    global _global_collector

    try:
        if _global_collector:
            _global_collector.stop_collection()
            # 不立即设置为None，保留文件路径信息
            logger.info("测试页面数据采集已停止")
    except Exception as e:
        logger.error(f"停止测试数据采集失败: {e}")


def cleanup_collector():
    """
    清理全局采集器实例
    这个函数应该在完全不需要采集器时调用
    """
    global _global_collector

    try:
        if _global_collector:
            _global_collector = None
            logger.info("全局采集器已清理")
    except Exception as e:
        logger.error(f"清理全局采集器失败: {e}")


# =============================================================================
# 数据访问接口
# =============================================================================

def get_current_data() -> Dict[str, Any]:
    """获取当前数据的便捷接口"""
    if _global_collector:
        return _global_collector.get_current_data()
    return {}


def get_latest_data() -> Dict[str, Any]:
    """获取最新数据的便捷接口"""
    if _global_collector:
        return _global_collector.get_latest_data()
    return {}


def get_multimodal_file_paths() -> Dict[str, str]:
    """获取多模态数据文件的绝对路径的便捷接口"""
    if _global_collector:
        return _global_collector.get_file_paths()
    return {}


# =============================================================================
# 实时显示窗口管理接口
# =============================================================================

def enable_realtime_display(enable: bool = True, title: str = None, width: int = 640, height: int = 480) -> bool:
    """启用或禁用实时显示窗口的全局接口"""
    global _global_display_window

    if not HAS_GUI:
        logger.warning("PyQt5不可用，无法启用实时显示窗口")
        return False

    if enable:
        if _global_display_window is None:
            try:
                window_title = title or "多模态数据采集实时显示"
                _global_display_window = RealTimeDisplayWindow(window_title, width, height)
                _global_display_window.show()
                logger.info("实时显示窗口已启用")
                return True
            except Exception as e:
                logger.error(f"创建实时显示窗口失败: {e}")
                return False
        else:
            _global_display_window.show()
            logger.info("实时显示窗口已显示")
            return True
    else:
        if _global_display_window:
            _global_display_window.hide()
            logger.info("实时显示窗口已隐藏")
        return True


def update_display_image(image_array):
    """更新显示窗口的影像的全局接口"""
    global _global_display_window
    if _global_display_window:
        try:
            _global_display_window.update_image(image_array)
        except Exception as e:
            logger.debug(f"更新显示影像失败: {e}")


def close_realtime_display():
    """关闭实时显示窗口的全局接口"""
    global _global_display_window
    if _global_display_window:
        try:
            _global_display_window.close()
            _global_display_window = None
            logger.info("实时显示窗口已关闭")
        except Exception as e:
            logger.error(f"关闭实时显示窗口失败: {e}")


# 测试代码
if __name__ == "__main__":
    print("=== 多模态数据采集器测试 ===")

    # 简化的实例化测试
    print("启动数据采集器...")
    collector = MultiModalDataCollector("test_user")

    try:
        # 启动采集
        collector.start_collection()
        print("数据采集已启动")

        # 运行3秒
        time.sleep(3)

        # # 获取最新数据
        # latest = collector.get_latest_data()
        # print(f"最新数据: {type(latest.get('rgb'))}")

        # 获取当前数据
        current = collector.get_current_data()
        print(f"当前数据rgb: {len(current.get('rgb'))}") 
        print(f"当前数据depth: {len(current.get('depth'))}") 
        print(f"当前数据eyetrack: {len(current.get('eyetrack'))}") 

        # 停止采集
        collector.stop_collection()
        print("数据采集已停止")

    except Exception as e:
        print(f"测试失败: {e}")

    print("=== 测试完成 ===")
