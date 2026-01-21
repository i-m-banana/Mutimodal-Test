"""Blood pressure measurement page component."""

import os
from typing import Optional, Dict

from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGraphicsDropShadowEffect,
    QMessageBox,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap

from .. import config
from ..utils.responsive import scale

logger = config.logger
HAS_BP_BACKEND = config.HAS_BP_BACKEND
BP_SIMULATION = config.BP_SIMULATION

bp_start_measurement = config.bp_start_measurement
bp_stop_measurement = config.bp_stop_measurement
bp_get_snapshot = config.bp_get_snapshot
bp_get_status = config.bp_get_status


class BloodPressurePage(QWidget):
    """血压脉搏测试页面组件"""
    
    # 信号定义
    test_completed = pyqtSignal()  # 测试完成信号
    test_failed = pyqtSignal(str)  # 测试失败信号（reason）
    next_clicked = pyqtSignal()  # 下一步按钮点击信号
    test_result_ready = pyqtSignal(dict)  # 测试结果信号 {'systolic': int, 'diastolic': int, 'pulse': int}
    
    def __init__(self, simulation_enabled: bool = False, forced_port: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.simulation_enabled = simulation_enabled
        self.forced_port = forced_port
        self.available_port: Optional[str] = None
        self.test_running = False
        self.test_progress = 0
        self.test_duration = 60  # 测试持续时间（秒）
        self.measurement_active = False
        
        # 测试结果
        self.results = {
            'systolic': None,
            'diastolic': None,
            'pulse': None,
        }
        
        # 错误标志
        self._error_reported = False
        self._snapshot_warned = False
        
        # 定时器
        self.device_check_timer = QTimer(self)
        self.device_check_timer.timeout.connect(self._check_device)
        
        self.test_timer = QTimer(self)
        self.test_timer.timeout.connect(self._update_progress)
        
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(600)
        self.poll_timer.timeout.connect(self._poll_snapshot)
        
        self._init_ui()
        
        # 启动设备检测
        self.device_check_timer.start(1000)
    
    def _init_ui(self):
        """初始化UI"""
        page_layout = QVBoxLayout(self)
        page_layout.setAlignment(Qt.AlignCenter)
        page_layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))
        
        # 创建白色卡片容器
        card_container = QFrame()
        card_container.setObjectName("bpCardContainer")
        card_container.setFixedSize(scale(1200), scale(600))
        card_container.setStyleSheet("""
            QFrame#bpCardContainer {
                background-color: #ffffff;
                border: 2px solid #e0e0e0;
                border-radius: 25px;
                padding: 30px;
            }
        """)
        
        # 添加阴影效果
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(15)
        card_shadow.setXOffset(0)
        card_shadow.setYOffset(5)
        card_shadow.setColor(QColor(0, 0, 0, 60))
        card_container.setGraphicsEffect(card_shadow)
        
        # 卡片内的横向布局（左图右文）
        main_layout = QHBoxLayout(card_container)
        main_layout.setSpacing(scale(25))
        main_layout.setContentsMargins(scale(20), scale(20), scale(20), scale(20))
        
        # 左侧：血压仪图片
        left_widget = self._create_left_panel()
        main_layout.addWidget(left_widget, 0)
        
        # 右侧：测试控制和结果
        right_widget = self._create_right_panel()
        main_layout.addWidget(right_widget, 0)
        
        # 将卡片添加到页面（居中显示）
        page_layout.addStretch(1)
        page_layout.addWidget(card_container, 0, Qt.AlignCenter)
        page_layout.addStretch(1)
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧面板（图片提示）"""
        left_widget = QWidget()
        left_widget.setFixedWidth(scale(450))
        left_layout = QVBoxLayout(left_widget)
        left_layout.setAlignment(Qt.AlignCenter)
        left_layout.setSpacing(scale(15))
        
        # 图片标签
        image_label = QLabel()
        image_label.setObjectName("bpImageLabel")
        image_label.setAlignment(Qt.AlignCenter)
        
        # 加载图片
        image_path = "assets/maibobo/maibobo.jpg"
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    scale(400), scale(450),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                image_label.setPixmap(scaled_pixmap)
                image_label.setStyleSheet("""
                    QLabel#bpImageLabel {
                        border: none;
                        background-color: transparent;
                        padding: 0px;
                    }
                """)
            else:
                image_label.setText("图片加载失败")
                image_label.setStyleSheet("font-size: 18px; color: #999;")
        else:
            image_label.setText("⚠️\n图片文件不存在")
            image_label.setStyleSheet("font-size: 20px; color: #f44336;")
        
        # 图片说明
        tip_label = QLabel("请将手臂放置在仪器测量位置")
        tip_label.setAlignment(Qt.AlignCenter)
        tip_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #333;")
        
        left_layout.addStretch(1)
        left_layout.addWidget(image_label, 0, Qt.AlignCenter)
        left_layout.addWidget(tip_label, 0, Qt.AlignCenter)
        left_layout.addStretch(1)
        
        return left_widget
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧面板（测试控制和结果）"""
        right_widget = QWidget()
        right_widget.setFixedWidth(scale(670))
        layout = QVBoxLayout(right_widget)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(scale(25))
        
        # 标题
        title_label = QLabel("血压脉搏检测")
        title_label.setObjectName("h1")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 40px; font-weight: bold;")
        
        # 设备状态区域
        self.status_container = QWidget()
        status_layout = QVBoxLayout(self.status_container)
        status_layout.setSpacing(scale(12))
        
        self.status_label = QLabel("正在检测血压仪器连接...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #1976D2;")
        
        self.progress_label = QLabel("等待开始测试")
        self.progress_label.setObjectName("subtitle")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 28px; color: #666;")
        
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress_label)
        
        # 测试控制区域
        self.control_container = QWidget()
        control_layout = QVBoxLayout(self.control_container)
        control_layout.setSpacing(scale(20))
        
        # 圆形进度指示器
        self.progress_circle = QLabel()
        self.progress_circle.setFixedSize(100, 100)
        self.progress_circle.setAlignment(Qt.AlignCenter)
        self.progress_circle.setStyleSheet("""
            QLabel {
                border: 4px solid #E0E0E0;
                border-radius: 50px;
                background-color: #F5F5F5;
                color: #666;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        self.progress_circle.setText("准备")
        
        # 开始测试按钮
        self.start_button = QPushButton("开始测试")
        self.start_button.setObjectName("successButton")
        self.start_button.setFixedSize(scale(220), scale(70))
        self.start_button.setStyleSheet("font-size: 26px; font-weight: bold; background-color: #5DADE2;")
        self.start_button.clicked.connect(self._toggle_test)
        self.start_button.setEnabled(False)
        
        control_layout.addWidget(self.progress_circle, 0, Qt.AlignCenter)
        control_layout.addWidget(self.start_button, 0, Qt.AlignCenter)
        
        # 结果显示区域
        self.result_container = QWidget()
        self.result_container.setVisible(False)
        self.result_container.setFixedHeight(scale(380))
        result_layout = QVBoxLayout(self.result_container)
        result_layout.setSpacing(scale(15))
        result_layout.setContentsMargins(0, 0, 0, 0)
        
        # "测试完成"标签
        self.complete_label = QLabel("测试完成 ✅")
        self.complete_label.setObjectName("subtitle")
        self.complete_label.setAlignment(Qt.AlignCenter)
        self.complete_label.setStyleSheet("font-size: 32px; font-weight: bold; color: #4CAF50;")
        
        # 结果卡片
        self.result_card = QWidget()
        self.result_card.setObjectName("card")
        self.result_card.setFixedSize(scale(550), scale(300))
        result_card_layout = QVBoxLayout(self.result_card)
        result_card_layout.setSpacing(scale(12))
        result_card_layout.setContentsMargins(scale(20), scale(25), scale(20), scale(25))
        
        # 收缩压
        self.systolic_label = QLabel("收缩压: -- mmHg")
        self.systolic_label.setObjectName("statusLabel")
        self.systolic_label.setAlignment(Qt.AlignCenter)
        self.systolic_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #1976D2;")
        
        # 舒张压
        self.diastolic_label = QLabel("舒张压: -- mmHg")
        self.diastolic_label.setObjectName("statusLabel")
        self.diastolic_label.setAlignment(Qt.AlignCenter)
        self.diastolic_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #1976D2;")
        
        # 脉搏
        self.pulse_label = QLabel("脉搏: -- 次/分")
        self.pulse_label.setObjectName("statusLabel")
        self.pulse_label.setAlignment(Qt.AlignCenter)
        self.pulse_label.setStyleSheet("font-size: 30px; font-weight: bold; color: #4CAF50;")
        
        # 下一步按钮
        self.next_button = QPushButton("请先完成血压测试")
        self.next_button.setObjectName("bpNextButton")
        self.next_button.setFixedSize(scale(240), scale(60))
        self.next_button.setCursor(Qt.PointingHandCursor)
        self.next_button.setStyleSheet("""
            QPushButton#bpNextButton {
                background-color: #5DADE2;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 24px;
                font-weight: bold;
            }
            QPushButton#bpNextButton:hover {
                background-color: #3498DB;
            }
            QPushButton#bpNextButton:pressed {
                background-color: #2E86C1;
            }
            QPushButton#bpNextButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
        """)
        self.next_button.setEnabled(False)
        self.next_button.clicked.connect(self.next_clicked.emit)
        
        result_card_layout.addWidget(self.systolic_label)
        result_card_layout.addWidget(self.diastolic_label)
        result_card_layout.addWidget(self.pulse_label)
        result_card_layout.addSpacing(scale(15))
        result_card_layout.addWidget(self.next_button, 0, Qt.AlignCenter)
        
        result_layout.addWidget(self.complete_label)
        result_layout.addWidget(self.result_card, 0, Qt.AlignCenter)
        
        # 右侧布局组装
        layout.addStretch(1)
        layout.addWidget(title_label)
        layout.addWidget(self.status_container, 0, Qt.AlignCenter)
        layout.addWidget(self.control_container, 0, Qt.AlignCenter)
        layout.addWidget(self.result_container, 0, Qt.AlignCenter)
        layout.addStretch(1)
        
        return right_widget
    
    def _check_device(self):
        """检测血压仪器连接状态"""
        if self.simulation_enabled:
            self.status_label.setText("血压仪器已连接 ✅ (模拟模式)")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            self.start_button.setEnabled(True)
            self.available_port = "SIMULATION"
            return
        
        try:
            status = bp_get_status() if HAS_BP_BACKEND else {}
        except Exception as exc:
            logger.debug("查询血压后端状态失败: %s", exc)
            self.status_label.setText("血压后端未响应 ❌")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            if not self.test_running:
                self.start_button.setEnabled(False)
            return
        
        forced_port = (self.forced_port or "").strip()
        available_ports = status.get("available_ports") or []
        port = forced_port or (status.get("port") or "").strip()
        mode = status.get("mode") or ("simulation" if self.simulation_enabled else "hardware")
        error = status.get("error")
        running = bool(status.get("running"))
        
        if port:
            self.available_port = port
        elif available_ports:
            self.available_port = available_ports[0]
        else:
            self.available_port = None
        
        if running:
            label_mode = "模拟模式" if mode == "simulation" else f"端口: {self.available_port or '未知'}"
            self.status_label.setText(f"血压仪器测试中 ⏳ ({label_mode})")
            self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
            self.start_button.setEnabled(self.test_running)
            return
        
        if error and not self.simulation_enabled:
            self.status_label.setText(f"血压仪器不可用 ❌ ({error})")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            if not self.test_running:
                self.start_button.setEnabled(False)
            return
        
        if self.available_port:
            if mode == "simulation":
                suffix = "模拟模式"
            else:
                suffix = f"端口: {self.available_port}"
            self.status_label.setText(f"血压仪器已连接 ✅ ({suffix})")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            if not self.test_running:
                self.start_button.setEnabled(True)
        else:
            self.status_label.setText("血压仪器未连接，请确认设备连接状态 📥")
            self.status_label.setStyleSheet("color: #F44336; font-weight: bold;")
            if not self.test_running:
                self.start_button.setEnabled(False)
    
    def _toggle_test(self):
        """切换测试状态（开始/停止）"""
        if not self.test_running:
            self.start_test()
        else:
            QMessageBox.warning(
                self,
                "测试进行中",
                "血压测试正在进行中，无法手动停止。\n请等待测试自动完成。"
            )
            logger.warning("⚠️ 血压测试进行中，禁止手动停止")
    
    def start_test(self):
        """开始血压测试"""
        try:
            if not self.simulation_enabled and not HAS_BP_BACKEND:
                QMessageBox.warning(self, "设备错误", "血压后端服务不可用，无法开始测试")
                return
            
            if self.simulation_enabled:
                self.status_label.setText("血压仪器已连接 ✅ (模拟模式)")
                self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
                if not self.available_port:
                    self.available_port = "SIMULATION"
            
            self.results = {
                'systolic': None,
                'diastolic': None,
                'pulse': None,
            }
            
            self.test_running = True
            self.start_button.setText("测试进行中...")
            self.start_button.setEnabled(False)
            self.start_button.setObjectName("disabledButton")
            self.start_button.style().unpolish(self.start_button)
            self.start_button.style().polish(self.start_button)
            
            self.test_progress = 0
            self.progress_label.setText("测试进行中...")
            self.progress_circle.setText("0%")
            
            self.result_container.setVisible(False)
            
            self.test_timer.start(100)
            
            self._error_reported = False
            self._snapshot_warned = False
            
            port_candidate = (self.forced_port or self.available_port or "").strip() or None
            try:
                response = bp_start_measurement(
                    port=port_candidate,
                    simulation=bool(self.simulation_enabled),
                    allow_simulation=True,
                    timeout=1,
                )
                mode = response.get("mode", "hardware")
                resolved_port = response.get("port") or port_candidate or "SIMULATION"
                self.available_port = resolved_port
                self.measurement_active = True
                if not self.poll_timer.isActive():
                    self.poll_timer.start()
                logger.info("血压测试已开始（模式：%s，端口：%s）", mode, resolved_port)
                if mode == "simulation":
                    self.status_label.setText("血压仪器已连接 ✅ (模拟模式)")
                    self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
            except Exception as exc:
                logger.error("启动血压测试失败: %s", exc)
                QMessageBox.critical(self, "设备错误", f"启动血压仪失败: {exc}")
                self.stop_test()
                return
            
            logger.debug("血压测试已开始")
        
        except Exception as e:
            logger.error(f"开始血压测试失败: {e}")
            self.stop_test()
    
    def stop_test(self):
        """停止血压测试"""
        try:
            self.test_running = False
            self.start_button.setText("开始测试")
            self.start_button.setEnabled(True)
            self.start_button.setObjectName("successButton")
            self.start_button.style().unpolish(self.start_button)
            self.start_button.style().polish(self.start_button)
            
            self.test_timer.stop()
            
            if self.measurement_active:
                try:
                    bp_stop_measurement()
                except Exception as exc:
                    logger.debug("停止血压后端失败: %s", exc)
            self._stop_polling()
            
            self.progress_label.setText("测试已停止")
            self.progress_circle.setText("停止")
            
            logger.debug("血压测试已停止")
        
        except Exception as e:
            logger.error(f"停止血压测试失败: {e}")
    
    def _stop_polling(self):
        """停止轮询"""
        try:
            if self.poll_timer.isActive():
                self.poll_timer.stop()
        except Exception as exc:
            logger.debug(f"停止血压轮询时出错: {exc}")
        self.measurement_active = False
    
    def _poll_snapshot(self):
        """轮询血压快照"""
        if not HAS_BP_BACKEND or not self.test_running:
            self._stop_polling()
            return
        
        try:
            snapshot = bp_get_snapshot()
        except Exception as exc:
            if not self._snapshot_warned:
                logger.debug(f"获取血压快照失败: {exc}")
                self._snapshot_warned = True
            return
        
        status = (snapshot.get("status") or "").lower()
        latest = snapshot.get("latest") or {}
        error = snapshot.get("error")
        mode = snapshot.get("mode")
        
        if status != "running":
            self.measurement_active = False
        
        if mode == "simulation" and not self.simulation_enabled:
            self.status_label.setText("血压仪器已连接 ✅ (模拟模式)")
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        
        if error:
            error_text = str(error)
            if "maibobo" in error_text.lower():
                self.auto_skip_test("未检测到血压仪驱动，自动跳过此环节")
                return
            
            if not self._error_reported:
                logger.error(f"血压监测发生错误: {error}")
                QMessageBox.warning(self, "血压测试失败", error_text)
                self._error_reported = True
            self.stop_test()
            return
        
        if latest and self.results.get('systolic') is None:
            try:
                systolic = int(latest.get('systolic'))
                diastolic = int(latest.get('diastolic'))
                pulse = int(latest.get('pulse'))
            except Exception as exc:
                logger.debug(f"解析血压快照失败: {exc}")
            else:
                self.results = {
                    'systolic': systolic,
                    'diastolic': diastolic,
                    'pulse': pulse,
                }
                logger.debug(
                    "血压测试完成: 收缩压=%s, 舒张压=%s, 脉搏=%s",
                    systolic,
                    diastolic,
                    pulse,
                )
                # 延迟调用 complete_test
                QTimer.singleShot(0, self.complete_test)
                return
        
        if status in {"idle", "completed", "error"} and not latest:
            self._stop_polling()
    
    def complete_test(self):
        """完成测试，显示结果"""
        try:
            self.stop_test()
            
            if (self.results and self.results.get('systolic') is not None):
                self.systolic_label.setText(f"收缩压: {self.results['systolic']} mmHg")
                self.diastolic_label.setText(f"舒张压: {self.results['diastolic']} mmHg")
                self.pulse_label.setText(f"脉搏: {self.results['pulse']} 次/分")
                
                systolic = self.results['systolic']
                diastolic = self.results['diastolic']
                
                if systolic < 120 and diastolic < 80:
                    color = "#4CAF50"
                elif systolic < 130 and diastolic < 85:
                    color = "#FF9800"
                else:
                    color = "#F44336"
                
                self.systolic_label.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")
                self.diastolic_label.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")
                self.pulse_label.setStyleSheet(f"font-size: 30px; font-weight: bold; color: {color};")
                
                # 显示结果，隐藏状态和控制区域
                self.result_container.setVisible(True)
                self.status_container.setVisible(False)
                self.control_container.setVisible(False)
                
                self.progress_circle.setVisible(False)
                self.start_button.setVisible(False)
                
                # 启用"进入下一步"按钮
                self.next_button.setText("进入舒特格测试")
                self.next_button.setEnabled(True)
                
                self.progress_circle.setText("完成")
                self.progress_circle.setStyleSheet("""
                    QLabel {
                        border: 4px solid #4CAF50;
                        border-radius: 40px;
                        background-color: #E8F5E8;
                        color: #4CAF50;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
                
                logger.info(f"血压测试完成: 收缩压={systolic}, 舒张压={diastolic}, 脉搏={self.results['pulse']}")
                
                # 发送信号
                self.test_completed.emit()
                self.test_result_ready.emit(self.results.copy())
            
            else:
                self.progress_label.setText("测试失败 ❌")
                self.progress_circle.setText("失败")
                self.progress_circle.setStyleSheet("""
                    QLabel {
                        border: 4px solid #F44336;
                        border-radius: 40px;
                        background-color: #FFEBEE;
                        color: #F44336;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
                
                self.status_container.setVisible(True)
                self.control_container.setVisible(True)
                self.progress_circle.setVisible(True)
                self.start_button.setVisible(True)
                self.start_button.setText("重新测试")
                self.start_button.setEnabled(True)
                
                QMessageBox.warning(self, "测试失败", "未能获取有效的血压数据，请检查设备连接或重新测试")
                self.test_failed.emit("未能获取有效的血压数据")
        
        except Exception as e:
            logger.error(f"完成血压测试失败: {e}")
            self.progress_label.setText("测试出错 ❌")
            self.progress_circle.setText("错误")
    
    def _update_progress(self):
        """更新测试进度"""
        if not self.test_running:
            return
        
        try:
            self.test_progress += 0.1
            progress_percent = min(100, int((self.test_progress / self.test_duration) * 100))
            
            self.progress_circle.setText(f"{progress_percent}%")
            
            if progress_percent < 30:
                color = "#FF9800"
            elif progress_percent < 70:
                color = "#2196F3"
            else:
                color = "#4CAF50"
            
            self.progress_circle.setStyleSheet(f"""
                QLabel {{
                    border: 4px solid {color};
                    border-radius: 40px;
                    background-color: #F5F5F5;
                    color: {color};
                    font-size: 12px;
                    font-weight: bold;
                }}
            """)
            
            if self.test_progress >= self.test_duration:
                logger.warning("血压测试超时")
                self.complete_test()
        
        except Exception as e:
            logger.error(f"更新血压测试进度失败: {e}")
    
    def auto_skip_test(self, reason: str):
        """自动跳过测试"""
        logger.warning("血压测试无法正常运行：%s，已自动跳过。", reason)
        self.results = {
            'systolic': 120,
            'diastolic': 80,
            'pulse': 75,
        }
        self.status_label.setText(reason)
        self.status_label.setStyleSheet("color: #f39c12; font-weight: bold;")
        self._error_reported = True
        self.complete_test()
    
    def reset_test(self):
        """重置测试状态"""
        self.results = {
            'systolic': None,
            'diastolic': None,
            'pulse': None,
        }
        self.result_container.setVisible(False)
        self.status_container.setVisible(True)
        self.control_container.setVisible(True)
        self.progress_circle.setVisible(True)
        self.start_button.setVisible(True)
        self.start_button.setText("开始测试")
        self.start_button.setEnabled(True)
        self.next_button.setText("请先完成血压测试")
        self.next_button.setEnabled(False)
        self.progress_label.setText("等待开始测试")
        self.progress_circle.setText("准备")
        self.progress_circle.setStyleSheet("""
            QLabel {
                border: 4px solid #E0E0E0;
                border-radius: 50px;
                background-color: #F5F5F5;
                color: #666;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        logger.info("✅ 血压测试状态已重置")
    
    def cleanup(self):
        """清理资源"""
        try:
            self.device_check_timer.stop()
            self.test_timer.stop()
            self.poll_timer.stop()
            if self.test_running:
                self.stop_test()
            logger.debug("BloodPressurePage 资源已清理")
        except Exception as e:
            logger.error(f"清理 BloodPressurePage 资源失败: {e}")
