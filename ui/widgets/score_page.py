# score_page_enhanced.py
import sys, os, random
from datetime import datetime, timedelta
from typing import Any, Dict
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDialog, QGraphicsDropShadowEffect, QGridLayout, QFrame, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt5.QtGui import QPainter, QPen, QColor, QFont, QBrush, QConicalGradient, QPainterPath, QLinearGradient, \
    QRadialGradient
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import logging

from services.backend_client import get_backend_client

logger = logging.getLogger()

# 强制使用 Qt5Agg 后端
plt.switch_backend('Qt5Agg')

# 系统字体路径（Windows 微软雅黑）
font_path = "C:/Windows/Fonts/msyh.ttc"
if not os.path.exists(font_path):
    print("警告: 微软雅黑字体未找到，中文可能显示方块")
zh_font = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


SKIP_DATABASE = _env_flag("UI_SKIP_DATABASE")

DEFAULT_METRICS = ["疲劳检测", "情绪", "血压脉搏", "脑负荷", "舒尔特准确率"]
ALL_SCORE_KEYS = DEFAULT_METRICS + ["舒尔特综合得分", "收缩压", "舒张压", "脉搏"]


class ModernGaugeWidget(QWidget):
    """现代化仪表盘控件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(350, 350)
        self.setMaximumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.value = 0
        self.max_value = 100

    def setValue(self, value):
        self.value = min(max(0, value), self.max_value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 获取绘制区域
        rect = self.rect()
        side = min(rect.width(), rect.height())
        painter.translate(rect.center())
        scale = side / 350.0
        painter.scale(scale, scale)

        # 绘制外圆阴影
        shadow_gradient = QRadialGradient(0, 0, 140)
        shadow_gradient.setColorAt(0, QColor(0, 0, 0, 30))
        shadow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(shadow_gradient))
        painter.drawEllipse(-140, -140, 280, 280)

        # 绘制背景圆环
        painter.setPen(QPen(QColor(240, 240, 240), 25))
        painter.setBrush(Qt.NoBrush)
        painter.drawArc(-120, -120, 240, 240, 225 * 16, -270 * 16)

        # 绘制进度圆环
        gradient = QConicalGradient(0, 0, 225)
        if self.value < 60:
            gradient.setColorAt(0, QColor(255, 120, 120))
            gradient.setColorAt(0.5, QColor(255, 80, 80))
            gradient.setColorAt(1, QColor(255, 60, 60))
            main_color = QColor(255, 80, 80)
        elif self.value < 80:
            gradient.setColorAt(0, QColor(255, 200, 0))
            gradient.setColorAt(0.5, QColor(255, 170, 0))
            gradient.setColorAt(1, QColor(255, 140, 0))
            main_color = QColor(255, 170, 0)
        else:
            gradient.setColorAt(0, QColor(100, 255, 100))
            gradient.setColorAt(0.5, QColor(50, 220, 50))
            gradient.setColorAt(1, QColor(0, 200, 0))
            main_color = QColor(50, 220, 50)

        pen = QPen(QBrush(gradient), 25)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        span_angle = int(self.value / self.max_value * 270 * 16)
        painter.drawArc(-120, -120, 240, 240, 225 * 16, -span_angle)

        # 绘制内圆
        inner_gradient = QRadialGradient(0, 0, 85)
        inner_gradient.setColorAt(0, QColor(255, 255, 255))
        inner_gradient.setColorAt(1, QColor(245, 245, 245))
        painter.setPen(QPen(QColor(230, 230, 230), 1))
        painter.setBrush(QBrush(inner_gradient))
        painter.drawEllipse(-85, -85, 170, 170)

        # 绘制分数
        painter.setPen(QPen(main_color, 3))
        painter.setFont(QFont("Arial", 56, QFont.Bold))
        score_text = str(int(self.value))
        painter.drawText(-60, -20, 120, 60, Qt.AlignCenter, score_text)

        # 绘制"分"字
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setFont(QFont("Microsoft YaHei", 18))
        painter.drawText(-25, 30, 50, 30, Qt.AlignCenter, "分")

        # 绘制刻度
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(QFont("Arial", 10))
        for i in range(0, 101, 20):
            angle = 225 - (i / 100.0 * 270)
            angle_rad = angle * np.pi / 180

            # 刻度线
            x1 = 135 * np.cos(angle_rad)
            y1 = -135 * np.sin(angle_rad)
            x2 = 145 * np.cos(angle_rad)
            y2 = -145 * np.sin(angle_rad)
            painter.drawLine(int(x1), int(y1), int(x2), int(y2))

            # 刻度数字
            text_x = 155 * np.cos(angle_rad) - 12
            text_y = -155 * np.sin(angle_rad) + 5
            painter.drawText(int(text_x), int(text_y), str(i))


class HistoryDialog(QDialog):
    """历史数据展示对话框"""

    def __init__(self, data_interface, use_mock_on_empty=True, source_hint: str | None = None):
        super().__init__()
        self.setWindowTitle("历史数据分析")
        self.setMinimumSize(1000, 800)
        self.data_interface = data_interface
        self.use_mock_on_empty = use_mock_on_empty  # 当没有有效数据时是否使用模拟数据
        # 数据来源提示："真实" | "模拟" | "自动"
        self.source_hint = source_hint or "自动"
        self.current_metric = None
        self.zh_font = font_manager.FontProperties(family="Microsoft YaHei")

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(20)

        # 标题
        title = QLabel("历史数据趋势分析")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size:28px; font-weight:bold; margin-bottom:20px;")
        layout.addWidget(title)

        # 数据来源提示条（右上角）
        source_row = QHBoxLayout()
        source_row.addStretch(1)
        self.source_label = QLabel("")
        self._update_source_label(self.source_hint)
        self.source_label.setAlignment(Qt.AlignRight)
        self.source_label.setStyleSheet("color:#777; font-size:12px; margin-top:-10px;")
        source_row.addWidget(self.source_label)
        layout.addLayout(source_row)

        # 指标选择按钮
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)
        self.metric_buttons = {}
        metrics = ["疲劳检测", "情绪", "收缩压", "舒张压", "脉搏", "脑负荷", "舒尔特准确率"]

        for m in metrics:
            btn = QPushButton(m, self)
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    font-size: 16px;
                    font-weight: bold;
                    background-color: #f0f0f0;
                    border: 2px solid #ddd;
                    border-radius: 22px;
                    padding: 0 30px;
                }
                QPushButton:hover {
                    background-color: #e0e0e0;
                    border-color: #bbb;
                }
                QPushButton:pressed {
                    background-color: #4CAF50;
                    color: white;
                }
            """)
            btn.clicked.connect(lambda checked, metric=m: self._draw_chart(metric))
            self.metric_buttons[m] = btn
            btn_layout.addWidget(btn)

        btn_container = QWidget()
        btn_container.setLayout(btn_layout)
        layout.addWidget(btn_container, alignment=Qt.AlignCenter)

        # matplotlib图表容器
        chart_container = QWidget()
        chart_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #ddd;
                border-radius: 15px;
            }
        """)
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(20, 20, 20, 20)

        # matplotlib图
        self.figure = Figure(facecolor='white', figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: transparent;")
        chart_layout.addWidget(self.canvas)

        layout.addWidget(chart_container, 1)

        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 50))
        shadow.setOffset(0, 3)
        chart_container.setGraphicsEffect(shadow)

    def _draw_chart(self, metric):
        self.current_metric = metric

        # 更新按钮样式
        for m, btn in self.metric_buttons.items():
            if m == metric:
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 16px;
                        font-weight: bold;
                        background-color: #4CAF50;
                        color: white;
                        border: 2px solid #4CAF50;
                        border-radius: 22px;
                        padding: 0 30px;
                    }
                """)
            else:
                btn.setStyleSheet("""
                    QPushButton {
                        font-size: 16px;
                        font-weight: bold;
                        background-color: #f0f0f0;
                        border: 2px solid #ddd;
                        border-radius: 22px;
                        padding: 0 30px;
                    }
                    QPushButton:hover {
                        background-color: #e0e0e0;
                        border-color: #bbb;
                    }
                """)

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # 设置背景色
        ax.set_facecolor('#fafafa')

        data = self.data_interface()
        history_data = data.get("历史", {})
        history_values = history_data.get(metric, [])
        history_dates = data.get("历史日期", [])
        data_validity = data.get("数据有效性", {})

        if not history_values or not history_dates:
            ax.text(0.5, 0.5, '暂无历史数据', ha='center', va='center',
                    transform=ax.transAxes, fontproperties=self.zh_font, fontsize=24,
                    color='#666')
        else:
            # 确保数据长度一致（取最短的）
            min_len = min(len(history_dates), len(history_values))
            history_dates = history_dates[:min_len]
            history_values = history_values[:min_len]

            # 过滤空值数据点(值为0或None的点)
            valid_pairs = [(v, d) for v, d in zip(history_values, history_dates) 
                          if v is not None and v > 0]
            
            # 检查是否有有效数据
            valid_count = data_validity.get(metric, 0)
            if not valid_pairs or valid_count == 0:
                # 如果启用了模拟数据模式,则生成并显示模拟数据
                if self.use_mock_on_empty:
                    # 动态提示：本次使用模拟数据
                    try:
                        self._update_source_label("模拟（无有效历史，已生成示例）")
                    except Exception:
                        pass
                    logger.info(f"指标 '{metric}' 无有效数据,使用模拟数据显示")
                    # 生成模拟历史数据
                    num_records = random.randint(10, 25)
                    base_date = datetime.now()
                    mock_dates = []
                    mock_values = []
                    
                    for i in range(num_records):
                        days_ago = random.randint(0, 30)
                        hours_ago = random.randint(0, 23)
                        test_date = base_date - timedelta(days=days_ago, hours=hours_ago)
                        mock_dates.append(test_date)
                        
                        # 根据不同指标生成合理范围的值
                        if metric in ["收缩压"]:
                            mock_values.append(random.randint(100, 140))
                        elif metric in ["舒张压"]:
                            mock_values.append(random.randint(60, 90))
                        elif metric in ["脉搏"]:
                            mock_values.append(random.randint(60, 100))
                        elif metric in ["舒尔特准确率"]:
                            mock_values.append(random.randint(80, 100))
                        else:  # 疲劳、情绪、脑负荷等分数类指标
                            mock_values.append(random.randint(40, 100))
                    
                    # 按时间排序
                    sorted_pairs = sorted(zip(mock_dates, mock_values))
                    history_dates = [d for d, v in sorted_pairs]
                    history_values = [v for d, v in sorted_pairs]
                    
                    # 继续绘制图表(使用模拟数据)
                else:
                    # 不使用模拟数据,显示"暂无有效数据"提示
                    ax.text(0.5, 0.5, f'{metric}\n暂无有效数据\n（所有值为0或空）', 
                           ha='center', va='center',
                           transform=ax.transAxes, fontproperties=self.zh_font, 
                           fontsize=20, color='#999')
                    ax.text(0.5, 0.3, f'总记录数: {len(history_values)} | 有效数据: 0',
                           ha='center', va='center',
                           transform=ax.transAxes, fontproperties=self.zh_font,
                           fontsize=14, color='#bbb')
                    self.figure.tight_layout()
                    self.canvas.draw()
                    return
            else:
                # 分离有效值和日期
                history_values, history_dates = zip(*valid_pairs)
                history_values = list(history_values)
                history_dates = list(history_dates)
            
            # 转换日期为datetime对象(如果还不是的话)
            dates = []
            for d in history_dates:
                if isinstance(d, str):
                    dates.append(datetime.strptime(d, "%Y-%m-%d %H:%M:%S"))
                elif isinstance(d, datetime):
                    dates.append(d)
                else:
                    dates.append(datetime.now())  # 兜底值

            # 绘制折线图
            line = ax.plot(dates, history_values, 'o-', linewidth=3, markersize=10,
                           color='#2196F3', markerfacecolor='white',
                           markeredgecolor='#2196F3', markeredgewidth=3)[0]

            # 填充区域
            ax.fill_between(dates, history_values, alpha=0.3, color='#2196F3')

            # 添加平均线
            avg = sum(history_values) / len(history_values)
            ax.axhline(y=avg, color='#FF5722', linestyle='--', linewidth=2.5, alpha=0.7)
            # 平均值显示为整数（更简洁）
            ax.text(dates[-1], avg, f'平均值: {int(round(avg))}',
                    fontproperties=self.zh_font, fontsize=14, color='#FF5722',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                              edgecolor='#FF5722', alpha=0.8))

            # 设置坐标轴
            ax.set_xlabel("测试时间", fontproperties=self.zh_font, fontsize=16)

            # 根据不同指标设置Y轴标签和单位
            if metric in ["收缩压", "舒张压"]:
                ax.set_ylabel("血压 (mmHg)", fontproperties=self.zh_font, fontsize=16)
            elif metric == "脉搏":
                ax.set_ylabel("脉搏 (次/分)", fontproperties=self.zh_font, fontsize=16)
            elif metric == "舒尔特准确率":
                ax.set_ylabel("准确率 (%)", fontproperties=self.zh_font, fontsize=16)
            else:
                ax.set_ylabel("分数", fontproperties=self.zh_font, fontsize=16)

            # 标题包含数据统计
            total_records = min_len  # 总记录数
            valid_records = len(history_values)  # 有效数据数
            title_text = f"{metric} 历史趋势"
            if valid_records < total_records:
                title_text += f" (显示 {valid_records}/{total_records} 条有效数据)"
            ax.set_title(title_text, fontproperties=self.zh_font,
                         fontsize=20, fontweight='bold', pad=20)

            # 设置x轴日期格式
            if len(dates) <= 7:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            elif len(dates) <= 15:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))

            # 旋转日期标签
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

            # 设置网格
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_axisbelow(True)

            # 设置y轴范围
            y_margin = (max(history_values) - min(history_values)) * 0.1
            ax.set_ylim(min(history_values) - y_margin, max(history_values) + y_margin)

            # 添加鼠标悬停显示数值的功能
            self.annotation = ax.annotate('', xy=(0, 0), xytext=(20, 20),
                                          textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                                          arrowprops=dict(arrowstyle="->", color='black'),
                                          fontsize=14, fontproperties=self.zh_font)
            self.annotation.set_visible(False)

            def on_hover(event):
                if event.inaxes == ax:
                    cont, ind = line.contains(event)
                    if cont:
                        idx = ind["ind"][0]
                        date_val = dates[idx]
                        y_val = history_values[idx]
                        self.annotation.xy = (mdates.date2num(date_val), y_val)

                        # 格式化时间显示
                        time_str = date_val.strftime("%Y-%m-%d %H:%M")

                        # 格式化数值：取整显示（更清晰）
                        try:
                            y_display = int(round(float(y_val)))
                        except (ValueError, TypeError):
                            y_display = y_val

                        # 根据不同指标显示不同单位
                        if metric in ["收缩压", "舒张压"]:
                            text = f'{time_str}\n{metric}: {y_display} mmHg'
                        elif metric == "脉搏":
                            text = f'{time_str}\n脉搏: {y_display} 次/分'
                        elif metric == "舒尔特准确率":
                            text = f'{time_str}\n准确率: {y_display}%'
                        else:
                            text = f'{time_str}\n分数: {y_display}'

                        self.annotation.set_text(text)
                        self.annotation.set_visible(True)
                        self.canvas.draw_idle()
                    else:
                        self.annotation.set_visible(False)
                        self.canvas.draw_idle()

            self.canvas.mpl_connect('motion_notify_event', on_hover)

        # 调整布局
        self.figure.tight_layout()
        self.canvas.draw()

    def _update_source_label(self, mode: str):
        mode_text = str(mode).strip()
        if mode_text in ("真实", "模拟", "自动") or mode_text.startswith("模拟"):
            text = f"数据来源：{mode_text}"
        else:
            text = f"数据来源：{mode_text}"
        self.source_label.setText(text)


class BloodPressureWidget(QWidget):
    """血压脉搏显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.systolic = 120  # 收缩压
        self.diastolic = 80  # 舒张压
        self.pulse = 75  # 脉搏
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # 收缩压
        systolic_layout = QHBoxLayout()
        systolic_label = QLabel("收缩压:")
        systolic_label.setStyleSheet("font-size:16px; color:#FF6B35; font-weight:bold;")
        self.systolic_value = QLabel("120 mmHg")
        self.systolic_value.setStyleSheet("font-size:18px; color:#FF6B35; font-weight:bold;")
        systolic_layout.addWidget(systolic_label)
        systolic_layout.addStretch()
        systolic_layout.addWidget(self.systolic_value)
        layout.addLayout(systolic_layout)

        # 舒张压
        diastolic_layout = QHBoxLayout()
        diastolic_label = QLabel("舒张压:")
        diastolic_label.setStyleSheet("font-size:16px; color:#FFA500; font-weight:bold;")
        self.diastolic_value = QLabel("80 mmHg")
        self.diastolic_value.setStyleSheet("font-size:18px; color:#FFA500; font-weight:bold;")
        diastolic_layout.addWidget(diastolic_label)
        diastolic_layout.addStretch()
        diastolic_layout.addWidget(self.diastolic_value)
        layout.addLayout(diastolic_layout)

        # 脉搏
        pulse_layout = QHBoxLayout()
        pulse_label = QLabel("脉搏:")
        pulse_label.setStyleSheet("font-size:16px; color:#4CAF50; font-weight:bold;")
        self.pulse_value = QLabel("75 次/分")
        self.pulse_value.setStyleSheet("font-size:18px; color:#4CAF50; font-weight:bold;")
        pulse_layout.addWidget(pulse_label)
        pulse_layout.addStretch()
        pulse_layout.addWidget(self.pulse_value)
        layout.addLayout(pulse_layout)

    def set_values(self, systolic, diastolic, pulse):
        """设置血压脉搏数值"""
        self.systolic = systolic
        self.diastolic = diastolic
        self.pulse = pulse

        self.systolic_value.setText(f"{systolic} mmHg")
        self.diastolic_value.setText(f"{diastolic} mmHg")
        self.pulse_value.setText(f"{pulse} 次/分")


class ScorePage(QWidget):
    def __init__(self, username=None, data_interface=None):
        super().__init__()
        self.setWindowTitle("测试结果")
        self.setMinimumSize(1200, 800)
        # 当前用户名
        self.username = username or 'anonymous'

        # 数据接口，可替换为真实接口
        self._external_data_interface = data_interface
        self._use_mock_data = SKIP_DATABASE
        self._db_error_logged = False
        self._history_future = None
        self._pending_history_user = None
        self._current_data = self._mock_data_interface() if self._use_mock_data else self._blank_data()
        self.data_interface = self._fetch_data
        
        # 测试结果数据（从test.py传入）
        self._test_results = None

        # 加载动画定时器
        self._loading_angle = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._update_loading_animation)
        self._is_loading = False

        self._init_ui()
        self._update_scores()
        QTimer.singleShot(0, self._refresh_data)

    def _init_ui(self):
        # 设置窗口背景
        self.setStyleSheet("QWidget { background-color: #f5f5f5; }")

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(30)

        # 内容区域
        content_layout = QHBoxLayout()
        content_layout.setSpacing(30)

        # 左侧分数卡片容器
        left_container = QWidget()
        left_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 20px;
            }
        """)
        left_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        left_shadow = QGraphicsDropShadowEffect()
        left_shadow.setBlurRadius(20)
        left_shadow.setColor(QColor(0, 0, 0, 40))
        left_shadow.setOffset(0, 5)
        left_container.setGraphicsEffect(left_shadow)

        self.score_labels = {}
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(30, 30, 30, 30)
        left_layout.setSpacing(20)

        # 左侧标题
        left_title = QLabel("各项指标得分")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("font-size:28px; font-weight:bold; color:#333;")
        left_layout.addWidget(left_title)

        # 添加分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setStyleSheet("background-color: #e0e0e0;")
        left_layout.addWidget(line1)

        # 指标卡片 - 简化设计
        metrics = ["疲劳检测", "情绪", "脑负荷", "舒尔特准确率"]
        icons = ["👁", "😊", "🧠", "🎯"]  # 简单的图标

        for m, icon in zip(metrics, icons):
            # 创建水平布局
            h_layout = QHBoxLayout()
            h_layout.setSpacing(15)

            # 图标
            icon_label = QLabel(icon)
            icon_label.setFixedSize(30, 30)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("font-size:20px;")
            h_layout.addWidget(icon_label)

            # 指标名称
            name_label = QLabel(m)
            name_label.setStyleSheet("font-size:18px; color:#333; font-weight:500;")
            h_layout.addWidget(name_label)

            h_layout.addStretch()

            # 数值
            value_label = QLabel("0")
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setStyleSheet("font-size:24px; font-weight:bold;")
            h_layout.addWidget(value_label)

            # 添加到布局
            left_layout.addLayout(h_layout)
            self.score_labels[m] = value_label

            # 添加分隔线（除了最后一个）
            if m != metrics[-1]:
                separator = QFrame()
                separator.setFrameShape(QFrame.HLine)
                separator.setFrameShadow(QFrame.Plain)
                separator.setStyleSheet("background-color: #f0f0f0;")
                left_layout.addWidget(separator)

        # 血压脉搏区域
        bp_separator = QFrame()
        bp_separator.setFrameShape(QFrame.HLine)
        bp_separator.setFrameShadow(QFrame.Plain)
        bp_separator.setStyleSheet("background-color: #f0f0f0;")
        left_layout.addWidget(bp_separator)

        # 血压脉搏标题
        bp_title_layout = QHBoxLayout()
        bp_icon = QLabel("❤")
        bp_icon.setFixedSize(30, 30)
        bp_icon.setAlignment(Qt.AlignCenter)
        bp_icon.setStyleSheet("font-size:20px;")
        bp_title_layout.addWidget(bp_icon)

        bp_title = QLabel("血压脉搏")
        bp_title.setStyleSheet("font-size:18px; color:#333; font-weight:500;")
        bp_title_layout.addWidget(bp_title)
        bp_title_layout.addStretch()

        left_layout.addLayout(bp_title_layout)

        # 血压脉搏组件
        self.bp_widget = BloodPressureWidget()
        self.bp_widget.setStyleSheet("""
            BloodPressureWidget {
                background-color: #f8f9fa;
                border-radius: 10px;
                border: 1px solid #e9ecef;
            }
        """)
        left_layout.addWidget(self.bp_widget)

        # 添加空间
        left_layout.addSpacing(20)

        # 底部说明
        info_label = QLabel("* 数值越高表示状态越好")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size:14px; color:#999;")
        left_layout.addWidget(info_label)

        content_layout.addWidget(left_container, 5)

        # 右侧综合得分容器
        right_container = QWidget()
        right_container.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 20px;
            }
        """)
        right_container.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        right_shadow = QGraphicsDropShadowEffect()
        right_shadow.setBlurRadius(20)
        right_shadow.setColor(QColor(0, 0, 0, 40))
        right_shadow.setOffset(0, 5)
        right_container.setGraphicsEffect(right_shadow)

        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(30, 30, 30, 30)
        right_layout.setSpacing(20)

        # 右侧标题
        right_title = QLabel("综合评估")
        right_title.setAlignment(Qt.AlignCenter)
        right_title.setStyleSheet("font-size:28px; font-weight:bold; color:#333;")
        right_layout.addWidget(right_title)

        # 添加分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setFrameShadow(QFrame.Sunken)
        line2.setStyleSheet("background-color: #e0e0e0;")
        right_layout.addWidget(line2)

        # 添加上部空间
        right_layout.addSpacing(20)

        # 仪表盘
        self.gauge = ModernGaugeWidget()
        right_layout.addWidget(self.gauge, alignment=Qt.AlignCenter)

        # 添加中间空间
        right_layout.addSpacing(20)

        # 等级评价
        self.lbl_level = QLabel("")
        self.lbl_level.setAlignment(Qt.AlignCenter)
        self.lbl_level.setStyleSheet("font-size:36px; font-weight:bold;")
        right_layout.addWidget(self.lbl_level)

        # 评语
        self.lbl_comment = QLabel("")
        self.lbl_comment.setAlignment(Qt.AlignCenter)
        self.lbl_comment.setWordWrap(True)
        self.lbl_comment.setStyleSheet("font-size:16px; color:#666; line-height:1.5;")
        right_layout.addWidget(self.lbl_comment)

        # 添加下部空间
        right_layout.addSpacing(30)

        # 历史数据按钮
        self.btn_history = QPushButton("查看历史数据")
        self.btn_history.setFixedSize(220, 50)
        self.btn_history.setStyleSheet("""
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 25px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.btn_history.clicked.connect(self._show_history)
        right_layout.addWidget(self.btn_history, alignment=Qt.AlignCenter)

        content_layout.addWidget(right_container, 4)

        main_layout.addLayout(content_layout)

        # 添加加载覆盖层（初始隐藏）
        self.loading_overlay = QWidget(self)
        self.loading_overlay.setStyleSheet("background-color: rgba(255, 255, 255, 180);")
        self.loading_overlay.setVisible(False)

        loading_layout = QVBoxLayout(self.loading_overlay)
        loading_layout.setAlignment(Qt.AlignCenter)

        self.loading_label = QLabel("正在加载历史数据...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setStyleSheet("font-size:24px; font-weight:bold; color:#333; background:transparent;")
        loading_layout.addWidget(self.loading_label)

        # 加载动画标签
        self.loading_spinner = QLabel("⏳")
        self.loading_spinner.setAlignment(Qt.AlignCenter)
        self.loading_spinner.setStyleSheet("font-size:48px; background:transparent;")
        loading_layout.addWidget(self.loading_spinner)

    def _get_score_level(self, score):
        """根据分数返回等级评价和评语"""
        if score >= 90:
            return "优秀", "#00aa00", "各项指标表现出色，请继续保持！"
        elif score >= 80:
            return "良好", "#88aa00", "整体状态不错，仍有提升空间。"
        elif score >= 70:
            return "中等", "#aaaa00", "状态一般，建议适当调整。"
        elif score >= 60:
            return "及格", "#aa8800", "需要注意休息和调整状态。"
        else:
            return "需要改进", "#aa0000", "建议充分休息，调整作息。"

    def _update_loading_animation(self):
        """更新加载动画"""
        spinners = ["⏳", "⌛", "⏳", "⌛"]
        self._loading_angle = (self._loading_angle + 1) % len(spinners)
        self.loading_spinner.setText(spinners[self._loading_angle])

    def _show_loading(self):
        """显示加载动画"""
        if not self._is_loading:
            self._is_loading = True
            self.loading_overlay.setGeometry(self.rect())
            self.loading_overlay.setVisible(True)
            self.loading_overlay.raise_()
            self._loading_timer.start(250)

    def _hide_loading(self):
        """隐藏加载动画"""
        if self._is_loading:
            self._is_loading = False
            self.loading_overlay.setVisible(False)
            self._loading_timer.stop()

    def resizeEvent(self, event):
        """窗口大小改变时调整加载覆盖层大小"""
        super().resizeEvent(event)
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.setGeometry(self.rect())

    def _blank_data(self) -> Dict[str, Any]:
        base = {metric: 0 for metric in ALL_SCORE_KEYS}
        base["历史"] = {metric: [] for metric in ALL_SCORE_KEYS}
        base["历史日期"] = []
        return base

    def _refresh_data(self) -> None:
        if self._external_data_interface:
            try:
                dataset = self._external_data_interface(self.username)
            except Exception as exc:
                self._handle_db_error(exc, "外部数据接口调用失败")
                return
            self._apply_real_data(dataset or {})
            return

        if self._use_mock_data:
            self._current_data = self._mock_data_interface()
            self._update_scores()
            return

        # 显示加载动画
        self._show_loading()
        self._request_history_from_backend()

    def _request_history_from_backend(self) -> None:
        if self._history_future and not self._history_future.done():
            if self._pending_history_user == (self.username or "anonymous"):
                return

        try:
            client = get_backend_client()
        except Exception as exc:
            self._handle_db_error(exc, "连接后端失败")
            return

        requested_user = self.username or "anonymous"
        # 默认获取最近30条记录,足够绘制平滑曲线
        # 可设置 limit=0 获取所有历史,或 limit=50 获取更多数据
        payload = {"name": requested_user, "limit": 30}
        future = client.send_command_future("db.get_user_history", payload)
        self._history_future = future
        self._pending_history_user = requested_user

        def _dispatch_result(fut):
            def _apply():
                if requested_user != (self.username or "anonymous"):
                    return
                if fut is self._history_future:
                    self._history_future = None
                    self._pending_history_user = None
                try:
                    result = fut.result()
                except Exception as exc:
                    self._handle_db_error(exc, "获取历史数据失败")
                    return
                self._handle_history_result(result or {})

            QTimer.singleShot(0, _apply)

        future.add_done_callback(_dispatch_result)

    def _handle_history_result(self, response: Dict[str, Any]) -> None:
        history = response.get("history") if isinstance(response, dict) else None
        if history is None:
            history = {}
        self._apply_real_data(history)
        # 隐藏加载动画
        self._hide_loading()

    def _apply_real_data(self, data: Dict[str, Any]) -> None:
        normalized = self._merge_real_data(data)
        self._current_data = normalized
        self._use_mock_data = False
        self._update_scores()

    def _merge_real_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        base = self._blank_data()
        if isinstance(self._current_data, dict) and self._current_data:
            for key in ALL_SCORE_KEYS:
                if key in self._current_data:
                    base[key] = self._current_data[key]
            existing_history = self._current_data.get("历史")
            if isinstance(existing_history, dict):
                for metric, series in existing_history.items():
                    if isinstance(series, list):
                        base["历史"][metric] = list(series)
            existing_dates = self._current_data.get("历史日期")
            if isinstance(existing_dates, list):
                base["历史日期"] = list(existing_dates)

        for key in ALL_SCORE_KEYS:
            value = data.get(key)
            if value is not None:
                base[key] = value

        history_section = data.get("历史") if isinstance(data, dict) else None
        if isinstance(history_section, dict):
            for metric, series in history_section.items():
                if isinstance(series, list):
                    base["历史"][metric] = series

        history_dates = data.get("历史日期") if isinstance(data, dict) else None
        if isinstance(history_dates, list):
            base["历史日期"] = history_dates

        return base

    def _handle_db_error(self, exc: Exception, context: str) -> None:
        if not self._db_error_logged:
            logger.error(f"{context}: {exc}")
            logger.warning("分数页将切换为模拟数据以避免界面卡顿。")
        self._db_error_logged = True
        self._use_mock_data = True
        self._history_future = None
        self._pending_history_user = None
        self._current_data = self._mock_data_interface()
        self._update_scores()
        # 隐藏加载动画
        self._hide_loading()

    def _mock_data_interface(self):
        # 生成最多30条历史数据
        num_records = random.randint(5, 30)

        # 生成历史日期（从30天前到现在）
        base_date = datetime.now()
        history_dates = []
        for i in range(num_records):
            days_ago = random.randint(0, 30)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            test_date = base_date - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
            history_dates.append(test_date.strftime("%Y-%m-%d %H:%M:%S"))

        # 按时间排序
        history_dates.sort()

        # 生成历史数据(确保有有效的非零值)
        history_data = {
            "疲劳检测": [random.randint(40, 100) for _ in range(num_records)],
            "情绪": [random.randint(40, 100) for _ in range(num_records)],
            "血压脉搏": [random.randint(60, 120) for _ in range(num_records)],  # 保留原有字段用于兼容
            "收缩压": [random.randint(100, 140) for _ in range(num_records)],
            "舒张压": [random.randint(60, 90) for _ in range(num_records)],
            "脉搏": [random.randint(60, 100) for _ in range(num_records)],
            "脑负荷": [random.randint(40, 100) for _ in range(num_records)],
            "舒尔特准确率": [random.randint(80, 100) for _ in range(num_records)],
            "舒尔特综合得分": [random.randint(40, 100) for _ in range(num_records)],
        }
        
        # 计算每个指标的有效数据数量(非零非空值的数量)
        data_validity = {}
        for metric, values in history_data.items():
            valid_count = sum(1 for v in values if v is not None and v > 0)
            data_validity[metric] = valid_count

        # 模拟接口数据
        return {
            "疲劳检测": random.randint(40, 100),
            "情绪": random.randint(40, 100),
            "血压脉搏": random.randint(60, 120),  # 保留原有字段用于兼容
            "收缩压": random.randint(100, 140),
            "舒张压": random.randint(60, 90),
            "脉搏": random.randint(60, 100),
            "脑负荷": random.randint(40, 100),
            "舒尔特准确率": random.randint(80, 100),
            "舒尔特综合得分": random.randint(40, 100),
            "历史": history_data,
            "历史日期": history_dates,
            "数据有效性": data_validity  # 添加数据有效性统计
        }

    def _fetch_data(self):
        # 优先返回真实测试结果(即使在模拟模式下)
        if self._test_results:
            # 确保 _current_data 包含最新的测试结果
            if not self._current_data:
                self._current_data = {}
            # 合并测试结果到当前数据
            for key in ["疲劳检测", "情绪", "脑负荷", "舒尔特准确率", 
                       "收缩压", "舒张压", "脉搏", "舒尔特综合得分"]:
                if key in self._test_results:
                    self._current_data[key] = self._test_results[key]
            return self._current_data
        
        # 没有测试结果时,使用模拟数据或空数据
        if self._use_mock_data:
            self._current_data = self._mock_data_interface()
        elif not self._current_data:
            self._current_data = self._blank_data()
        return self._current_data

    def set_force_mock(self, enabled: bool):
        enabled = bool(enabled)
        previous = self._use_mock_data
        if enabled:
            if not previous:
                logger.info("分数页数据切换为模拟模式")
            self._use_mock_data = True
            self._db_error_logged = True
            self._current_data = self._mock_data_interface()
            self._update_scores()
            return

        self._use_mock_data = SKIP_DATABASE
        if self._use_mock_data:
            if not previous:
                logger.info("环境变量 UI_SKIP_DATABASE 生效，分数页继续使用模拟数据")
            self._current_data = self._mock_data_interface()
            self._update_scores()
            return

        if previous:
            logger.info("分数页恢复使用真实数据接口")
        self._db_error_logged = False
        self._current_data = self._blank_data()
        self._update_scores()
        self._refresh_data()

    def _set_user(self, username):
        """设置用户名"""
        self.username = username or 'anonymous'
        self._history_future = None
        self._pending_history_user = None
        if self._use_mock_data:
            self._refresh_data()
        else:
            # 如果还没有测试结果数据，才初始化为空白数据
            # 如果已经有测试结果，保留它
            if not self._test_results:
                self._current_data = self._blank_data()
            self._refresh_data()
    
    def set_test_results(self, results_data: dict):
        """
        接收从test.py传入的测试结果数据
        
        Args:
            results_data: 包含所有测试结果的字典,格式:
                {
                    "疲劳检测": float,  # 平均疲劳度分数
                    "情绪": float,  # 情绪分数
                    "脑负荷": float,  # 平均脑负荷分数
                    "舒尔特准确率": float,  # 舒尔特准确率
                    "收缩压": int,
                    "舒张压": int,
                    "脉搏": int,
                    "舒尔特综合得分": int,
                    "_metadata": dict  # 元数据
                }
        """
        try:
            logger.info(f"📥 ScorePage接收到测试结果数据")
            logger.info(f"  疲劳检测={results_data.get('疲劳检测', 'N/A')}, 情绪={results_data.get('情绪', 'N/A')}, 脑负荷={results_data.get('脑负荷', 'N/A')}")
            
            # 保存测试结果(会在 _fetch_data() 中自动合并)
            self._test_results = results_data
            
            # 记录元数据
            if "_metadata" in results_data:
                metadata = results_data["_metadata"]
                logger.info(
                    f"  数据统计: 疲劳样本={metadata.get('fatigue_sample_count', 0)}, "
                    f"脑负荷样本={metadata.get('brain_load_sample_count', 0)}, "
                    f"有情绪={metadata.get('has_emotion_score', False)}, "
                    f"有舒尔特={metadata.get('has_schulte_result', False)}, "
                    f"有血压={metadata.get('has_bp_result', False)}"
                )
            
            logger.info("✅ 测试结果已保存,将在下次更新时显示")
                
        except Exception as e:
            logger.error(f"处理测试结果数据失败: {e}", exc_info=True)

    def _update_scores(self):
        """更新分数显示"""
        data = self.data_interface()

        # 更新左侧指标卡片（除血压脉搏外的其他指标）
        for key, lbl in self.score_labels.items():
            raw_value = data.get(key, 0)
            
            # 确保数值为数字类型
            try:
                value = float(raw_value) if raw_value is not None else 0
            except (ValueError, TypeError):
                value = 0
            
            # 取整显示（所有指标都显示为整数）
            value_int = int(round(value))

            # 根据不同指标显示不同单位
            if key == "舒尔特准确率":
                lbl.setText(f"{value_int}%")
            else:
                lbl.setText(f"{value_int} 分")

            # 根据分数设置颜色
            if value_int >= 80:
                color = "#00aa00"  # 绿色
            elif value_int >= 60:
                color = "#FFA500"  # 橙色
            else:
                color = "#aa0000"  # 红色
            lbl.setStyleSheet(f"font-size:24px; font-weight:bold; color:{color};")

        # 更新血压脉搏组件
        systolic = data.get("收缩压", 120)
        diastolic = data.get("舒张压", 80)
        pulse = data.get("脉搏", 75)
        self.bp_widget.set_values(systolic, diastolic, pulse)

        # 更新仪表盘（取整显示）
        raw_total = data.get("舒尔特综合得分", 0)
        try:
            total_score = int(round(float(raw_total))) if raw_total is not None else 0
        except (ValueError, TypeError):
            total_score = 0
        self.gauge.setValue(total_score)

        # 更新等级评价和评语
        level, color, comment = self._get_score_level(total_score)
        self.lbl_level.setText(f"{level}")
        self.lbl_level.setStyleSheet(f"font-size:36px; font-weight:bold; color:{color};")
        self.lbl_comment.setText(comment)

    def _show_history(self):
        """显示历史数据对话框"""
        # 优先尝试同步获取一次真实历史数据（不依赖全局模拟开关），
        # 若获取失败或无有效数据，则交给弹窗在空数据时自动使用模拟数据。
        def _try_fetch_history_once():
            try:
                client = get_backend_client()
                payload = {"name": self.username or "anonymous", "limit": 30}
                resp = client.send_command_sync("db.get_user_history", payload, timeout=5.0)
                history = resp.get("history") if isinstance(resp, dict) else None
                # 返回符合 HistoryDialog 期望的数据结构（包含“历史”“历史日期”等键）
                if isinstance(history, dict) and history:
                    return history
            except Exception as e:
                logger.debug(f"同步获取历史数据失败，将使用现有数据/模拟数据: {e}")
            return None

        history_for_dialog = _try_fetch_history_once()

        # 为弹窗提供数据接口：优先使用实时获取的历史数据；否则回落到当前页面的数据接口
        data_provider = (lambda: history_for_dialog) if history_for_dialog else self.data_interface

        source_hint = "真实" if history_for_dialog else "自动"
        dlg = HistoryDialog(data_provider, use_mock_on_empty=True, source_hint=source_hint)
        dlg.exec_()


# 调试用 main
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 可以选择使用真实数据或模拟数据
    # win = ScorePage(username="test_user")  # 使用真实数据库
    win = ScorePage()  # 使用模拟数据

    win.show()
    sys.exit(app.exec_())