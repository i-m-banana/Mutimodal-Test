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
        painter.setFont(QFont("阿里健康体2.0 中文 45 R", 56, QFont.Bold))
        score_text = str(int(self.value))
        painter.drawText(-60, -20, 120, 60, Qt.AlignCenter, score_text)

        # 绘制"分"字
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.setFont(QFont("阿里健康体2.0 中文 45 R", 18))
        painter.drawText(-25, 30, 50, 30, Qt.AlignCenter, "分")

        # 绘制刻度
        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.setFont(QFont("阿里健康体2.0 中文 45 R", 10))
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
            
            # 检查是否有有效数据（直接使用 valid_pairs，不依赖 data_validity 字段）
            if not valid_pairs:
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
                ax.set_ylabel("专注度 (%)", fontproperties=self.zh_font, fontsize=16)
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
        self._current_data = self._mock_data_interface() if self._use_mock_data else self._blank_data()
        self.data_interface = self._fetch_data
        
        # 测试结果数据（从test.py传入）
        self._test_results = None
        
        # 基准值数据（用户最佳状态）
        self._baseline_data = {}

        # 加载动画定时器
        self._loading_angle = 0
        self._loading_timer = QTimer(self)
        self._loading_timer.timeout.connect(self._update_loading_animation)
        self._is_loading = False
        
        # 中文字体
        self.zh_font = font_manager.FontProperties(family="Microsoft YaHei")

        self._init_ui()
        # 不在初始化时更新分数，等待数据加载完成后再更新
        # self._update_scores()  # ← 删除此行，避免重复调用
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

        # 左侧雷达图容器
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

        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(30, 30, 30, 30)
        left_layout.setSpacing(20)

        # 左侧标题
        left_title = QLabel("各项指标对比")
        left_title.setAlignment(Qt.AlignCenter)
        left_title.setStyleSheet("font-size:28px; font-weight:bold; color:#333;")
        left_layout.addWidget(left_title)

        # 添加分隔线
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setFrameShadow(QFrame.Sunken)
        line1.setStyleSheet("background-color: #e0e0e0;")
        left_layout.addWidget(line1)

        # 雷达图容器
        radar_widget = QWidget()
        radar_widget.setStyleSheet("background-color: white;")
        radar_layout = QVBoxLayout(radar_widget)
        radar_layout.setContentsMargins(10, 10, 10, 10)

        # matplotlib雷达图
        self.radar_figure = Figure(facecolor='white', figsize=(7, 7))
        self.radar_canvas = FigureCanvas(self.radar_figure)
        self.radar_canvas.setStyleSheet("background-color: transparent;")
        radar_layout.addWidget(self.radar_canvas)

        left_layout.addWidget(radar_widget, 1)

        # 底部说明
        info_label = QLabel("🔵 蓝色=本次测试 | 🔴 红色虚线=理想基准 (80分)")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("font-size:13px; color:#666; font-weight:500;")
        left_layout.addWidget(info_label)
        
        # 指标说明
        metric_info = QLabel("※ 得分相对于首次测试换算，蓝色区域越接近红色正七边形表示状态越好")
        metric_info.setAlignment(Qt.AlignCenter)
        metric_info.setStyleSheet("font-size:11px; color:#999;")
        left_layout.addWidget(metric_info)

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
        """刷新数据 - 统一使用同步获取逻辑（与历史数据弹窗保持一致）"""
        if self._external_data_interface:
            try:
                dataset = self._external_data_interface(self.username)
            except Exception as exc:
                logger.error(f"外部数据接口调用失败: {exc}")
                dataset = {}
            self._apply_real_data(dataset or {})
            return

        if self._use_mock_data:
            self._current_data = self._mock_data_interface()
            self._update_scores()
            return

        # 显示加载动画
        self._show_loading()
        
        # 使用与 _show_history() 完全相同的同步获取逻辑
        try:
            client = get_backend_client()
            payload = {"name": self.username or "anonymous", "limit": 30}
            logger.debug(f"同步获取用户 '{payload['name']}' 的历史数据...")
            resp = client.send_command_sync("db.get_user_history", payload, timeout=5.0)
            history = resp.get("history") if isinstance(resp, dict) else None
            
            if isinstance(history, dict) and history:
                logger.info(f"✅ 成功获取用户 '{payload['name']}' 的历史数据，记录数: {len(history.get('历史日期', []))}")
                self._apply_real_data(history)
            else:
                logger.warning(f"用户 '{payload['name']}' 无历史数据或数据格式错误")
                self._apply_real_data({})
                
        except Exception as e:
            logger.error(f"同步获取历史数据失败: {e}")
            # 失败时使用空数据，不生成模拟数据
            self._apply_real_data({})
        finally:
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

    def set_user(self, username):
        """
        设置当前用户名（公开方法）
        当用户切换时，会清空之前用户的历史数据缓存，重新获取新用户的数据
        """
        old_username = self.username
        self.username = username or 'anonymous'
        
        # 如果用户名确实改变了，清空所有缓存数据
        if old_username != self.username:
            logger.info(f"用户切换: {old_username} → {self.username}，清空历史数据缓存")
            self._current_data = self._blank_data()
            # 注意：不清空 _test_results，因为那是本次测试的实时结果
        
        # 重新获取新用户的历史数据
        self._refresh_data()
    
    # 保留私有方法作为别名，向后兼容
    def _set_user(self, username):
        """向后兼容的私有方法，调用公开方法"""
        self.set_user(username)
    
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

        # 更新雷达图
        self._draw_radar_chart(data)

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
    
    def _calculate_baseline(self):
        """
        从历史数据中计算基准值（个人最佳状态）
        策略：使用用户的第一次测试作为基准（假设第一次测试在最佳状态下进行）
        
        逻辑：
        1. 优先使用【第一次测试】的完整数据（最早的历史记录）
        2. 如果第一次数据不完整，使用【最近一次7维度都有效的测试】
        3. 完全没有历史数据时，使用健康标准默认值
        """
        data = self.data_interface()
        history = data.get("历史", {})
        history_dates = data.get("历史日期", [])
        
        logger.debug(f"计算基准值 - 当前用户: {self.username}, 历史记录数: {len(history_dates)}")
        
        # 需要计算基准的指标
        metrics = ["疲劳检测", "情绪", "脑负荷", "舒尔特准确率", "收缩压", "舒张压", "脉搏"]
        baseline = {}
        
        # 如果没有历史数据，返回健康标准默认值
        if not history_dates or len(history_dates) == 0:
            logger.info(f"用户 '{self.username}' 无历史数据，使用健康标准默认值作为基准")
            return {
                "疲劳检测": 30,      # 疲劳度低表示状态好
                "情绪": 30,          # 情绪压力低表示状态好
                "脑负荷": 30,        # 脑负荷低表示状态好
                "舒尔特准确率": 95,  # 准确率高表示状态好
                "收缩压": 120,
                "舒张压": 80,
                "脉搏": 75
            }
        
        # 尝试获取第一次测试的数据（按时间排序，取最早的）
        def get_record_at_index(idx):
            """获取指定索引的测试记录（7个维度的值）"""
            record = {}
            for metric in metrics:
                values = history.get(metric, [])
                if idx < len(values):
                    val = values[idx]
                    # 检查是否为有效值
                    if val is not None and val > 0:
                        record[metric] = val
            return record
        
        # 策略1: 尝试使用第一次测试（索引0）
        first_record = get_record_at_index(0)
        if len(first_record) == len(metrics):
            logger.info(f"用户 '{self.username}' 使用第一次测试作为基准: {first_record}")
            return first_record
        else:
            logger.info(f"用户 '{self.username}' 第一次测试数据不完整（仅{len(first_record)}/{len(metrics)}维度有效），搜索其他完整记录")
        
        # 策略2: 搜索最近一次7维度都完整的测试
        for idx in range(len(history_dates) - 1, -1, -1):  # 从最新往前搜索
            record = get_record_at_index(idx)
            if len(record) == len(metrics):
                logger.info(f"使用第{idx}次测试（最近完整记录）作为基准: {record}")
                return record
        
        # 策略3: 如果所有记录都不完整，尝试拼凑（每个维度取最早的有效值）
        logger.warning("所有测试记录都不完整，尝试拼凑基准值")
        for metric in metrics:
            values = history.get(metric, [])
            # 找到该维度最早的有效值
            for val in values:
                if val is not None and val > 0:
                    baseline[metric] = val
                    break
        
        # 策略4: 如果某些维度仍然没有有效值，用默认值补齐
        defaults = {
            "疲劳检测": 30,
            "情绪": 30,
            "脑负荷": 30,
            "舒尔特准确率": 95,
            "收缩压": 120,
            "舒张压": 80,
            "脉搏": 75
        }
        for metric in metrics:
            if metric not in baseline:
                baseline[metric] = defaults[metric]
                logger.warning(f"指标 '{metric}' 无历史数据，使用默认值 {defaults[metric]}")
        
        logger.info(f"最终拼凑的基准值: {baseline}")
        return baseline
    
    def _draw_radar_chart(self, data):
        """绘制雷达图对比本次测试值和基准值
        
        新设计：
        - 基准线固定为 80 分（正七边形红色虚线）
        - 本次测试值相对于首次测试表现进行换算
        - 通常本次测试会在正七边形内（视觉上更美观）
        """
        self.radar_figure.clear()
        
        # 定义要展示的指标（不包括综合得分）
        metrics = ["疲劳检测", "情绪", "脑负荷", "舒尔特准确率", "收缩压", "舒张压", "脉搏"]
        metric_labels = ["疲劳", "情绪", "脑负荷", "专注度", "收缩压", "舒张压", "脉搏"]
        
        # 获取本次测试值
        current_values = []
        for metric in metrics:
            raw_value = data.get(metric, 0)
            try:
                value = float(raw_value) if raw_value is not None else 0
            except (ValueError, TypeError):
                value = 0
            current_values.append(value)
        
        # 计算首次测试基准值（用于相对比较）
        baseline = self._calculate_baseline()
        baseline_values = [baseline.get(m, 0) for m in metrics]
        
        # 新换算逻辑：基准固定为 80 分，本次测试相对换算
        BASELINE_SCORE = 80  # 基准线固定为 80 分
        normalized_baseline = [BASELINE_SCORE] * len(metrics)  # 正七边形
        normalized_current = []
        
        for i, metric in enumerate(metrics):
            curr = current_values[i]
            base = baseline_values[i]
            
            # 避免除以零
            if base == 0:
                base = 1
            
            if metric == "疲劳检测":
                # 疲劳分数越低越好，需要进行反转。
                ratio = curr / base
                score = BASELINE_SCORE * (2 - ratio)
                score = max(0, min(100, score))
                
            elif metric in ["情绪", "脑负荷", "舒尔特准确率"]:
                # 准确率越高越好：原始值越大，表现越好
                # 换算逻辑：
                # - 如果本次 >= 首次：表现更好或持平，得分 >= 80
                # - 如果本次 < 首次：表现下降，得分 < 80
                # 公式：80 * (curr/base)，限制在 [0, 100]
                ratio = curr / base
                score = BASELINE_SCORE * ratio
                score = max(0, min(100, score))
                
            elif metric in ["收缩压", "舒张压", "脉搏"]:
                # 血压/脉搏：越接近理想值越好，直接用偏离度计算得分
                # 不相对于首次测试，而是直接评估当前值的健康程度
                ideal_values = {"收缩压": 120, "舒张压": 80, "脉搏": 75}
                max_deviations = {"收缩压": 40, "舒张压": 20, "脉搏": 25}  # 最大可接受偏离
                
                ideal = ideal_values[metric]
                max_dev = max_deviations[metric]
                
                # 计算本次测试的偏离度
                deviation = abs(curr - ideal)
                
                # 换算逻辑：
                # - 偏离度 = 0（完美）: 得分 = 80
                # - 偏离度 = max_dev（极差）: 得分 = 0
                # 公式：80 * (1 - deviation/max_dev)，限制在 [0, 80]
                score = BASELINE_SCORE * (1 - deviation / max_dev)
                score = max(0, min(BASELINE_SCORE, score))  # 最高不超过 80 分
            else:
                # 其他未定义指标，默认按比例换算
                ratio = curr / base
                score = BASELINE_SCORE * ratio
                score = max(0, min(100, score))
            
            normalized_current.append(score)
        
        # 闭合雷达图
        normalized_current += normalized_current[:1]
        normalized_baseline += normalized_baseline[:1]
        
        # 计算角度
        num_vars = len(metric_labels)
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]
        
        # 创建极坐标子图
        ax = self.radar_figure.add_subplot(111, projection='polar')
        ax.set_facecolor('#fafafa')
        
        # 绘制基准线（红色虚线正七边形，固定为 80 分）
        ax.plot(angles, normalized_baseline, 'r--', linewidth=2.5, label='理想基准线 (80分)', alpha=0.7)
        ax.fill(angles, normalized_baseline, 'r', alpha=0.1)
        
        # 绘制本次测试值（蓝色实线）
        ax.plot(angles, normalized_current, 'b-', linewidth=3, label='本次测试', marker='o', 
                markersize=8, markerfacecolor='white', markeredgecolor='b', markeredgewidth=2)
        ax.fill(angles, normalized_current, 'b', alpha=0.25)
        
        # 设置刻度标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontproperties=self.zh_font, fontsize=12)
        
        # 设置Y轴范围和刻度
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=10, color='#666')
        
        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                prop=self.zh_font, fontsize=11, framealpha=0.9)
        
        # 在每个数据点旁边显示实际数值
        for angle, curr_val, metric in zip(angles[:-1], current_values, metrics):
            # 计算文本位置
            x = angle
            y = normalized_current[angles.index(angle)] + 8
            
            # 格式化显示文本
            if metric in ["收缩压", "舒张压"]:
                text = f"{int(curr_val)}mmHg"
            elif metric == "脉搏":
                text = f"{int(curr_val)}次/分"
            elif metric == "舒尔特准确率":
                text = f"{int(curr_val)}%"
            elif metric in ["疲劳检测", "情绪", "脑负荷"]:
                # 这些指标越低越好，但雷达图上显示的是反转后的值（越大越好）
                # 标签显示原始值，便于理解
                text = f"{int(curr_val)}分"
            else:
                text = f"{int(curr_val)}分"
            
            ax.text(x, y, text, ha='center', va='center', 
                fontproperties=self.zh_font, fontsize=9, 
                color='#2196F3', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            edgecolor='#2196F3', alpha=0.8))
        
        self.radar_figure.tight_layout()
        self.radar_canvas.draw()
        
        # 保存zh_font供雷达图使用
        if not hasattr(self, 'zh_font'):
            self.zh_font = font_manager.FontProperties(family="Microsoft YaHei")

    def _show_history(self):
        """显示历史数据对话框 - 直接使用已缓存的历史数据（与主页面数据一致）"""
        # 直接使用当前页面已加载的数据，确保与雷达图显示的数据完全一致
        dlg = HistoryDialog(self.data_interface, use_mock_on_empty=True, source_hint="当前")
        dlg.exec_()


# 调试用 main
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 可以选择使用真实数据或模拟数据
    # win = ScorePage(username="test_user")  # 使用真实数据库
    win = ScorePage()  # 使用模拟数据

    win.show()
    sys.exit(app.exec_())