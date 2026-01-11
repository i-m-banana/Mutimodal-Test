"""Stage navigation bar component for test workflow."""

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGraphicsDropShadowEffect,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QRegion

from ..app.utils.responsive import scale


class CircleLabel(QLabel):
    """完美圆形的数字标签"""
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
    
    def resizeEvent(self, event):
        """在尺寸变化时重新应用圆形遮罩"""
        super().resizeEvent(event)
        size = min(self.width(), self.height())
        # 确保是正方形
        self.setFixedSize(size, size)
        # 应用圆形遮罩
        region = QRegion(0, 0, size, size, QRegion.Ellipse)
        self.setMask(region)


class StageNavigationBar(QWidget):
    """阶段导航栏组件"""
    
    # 信号：点击导航按钮
    stage_clicked = pyqtSignal(str)  # 参数：stage_name
    
    def __init__(self, stages: list, parent=None):
        """
        Args:
            stages: 阶段名称列表，例如 ['多模态疲劳检测', '情绪检测', '血压脉搏检测', '舒尔特专注度检测', '分数展示']
        """
        super().__init__(parent)
        self.stages = stages
        self.stage_buttons = {}  # 保存每个阶段的组件引用
        self.stage_completed = {stage: False for stage in stages}  # 记录完成状态
        self._init_ui()
    
    def _init_ui(self):
        """初始化UI"""
        self.setObjectName("stepNavigator")
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(scale(12), scale(8), scale(12), scale(8))
        layout.setSpacing(scale(6))
        
        # 设置渐变背景和圆角
        self.setStyleSheet("""
            QWidget#stepNavigator {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #7FDBFF,
                    stop:0.5 #A0E7FF,
                    stop:1 #C0F0FF
                );
                border-radius: 30px;
                border: none;
            }
        """)
        
        # 添加右下黑色阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(3)
        shadow.setYOffset(3)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)
        
        # 创建所有阶段的导航按钮
        for i, stage_name in enumerate(self.stages):
            # 创建横向布局容器：数字在左，文字在右
            stage_widget = QWidget()
            stage_widget.setStyleSheet("background: transparent;")
            stage_layout = QHBoxLayout(stage_widget)
            stage_layout.setContentsMargins(scale(10), scale(4), scale(10), scale(4))
            stage_layout.setSpacing(scale(8))
            stage_layout.setAlignment(Qt.AlignCenter)
            
            # 数字标签（使用自定义圆形标签）
            size = scale(45)
            number_label = CircleLabel(str(i + 1))
            number_label.setObjectName("stageNumber")
            number_label.setFixedSize(size, size)
            number_label.setStyleSheet("""
                QLabel#stageNumber {
                    background-color: rgb(227, 247, 253);
                    border: none;
                    font-size: 30px;
                    font-weight: bold;
                    color: rgb(77, 171, 201);
                }
            """)
            
            # 阶段名称标签
            name_label = QLabel(stage_name)
            name_label.setObjectName("stageName")
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            name_label.setStyleSheet("""
                QLabel#stageName {
                    background: transparent;
                    font-size: 32px;
                    font-weight: bold;
                    color: rgb(77, 171, 201);
                }
            """)
            
            stage_layout.addWidget(number_label, 0, Qt.AlignCenter)
            stage_layout.addWidget(name_label, 0, Qt.AlignLeft | Qt.AlignVCenter)
            
            # 创建可点击按钮（透明覆盖层）
            stage_btn = QPushButton(stage_widget)
            stage_btn.setObjectName("stageNavButton")
            stage_btn.setCursor(Qt.PointingHandCursor)
            stage_btn.setStyleSheet("""
                QPushButton#stageNavButton {
                    background: transparent;
                    border: none;
                }
            """)
            stage_btn.setGeometry(0, 0, stage_widget.width(), stage_widget.height())
            
            # 绑定点击事件
            stage_btn.clicked.connect(lambda checked, s=stage_name: self.stage_clicked.emit(s))
            
            # 保存引用（用于后续更新状态）
            self.stage_buttons[stage_name] = {
                'widget': stage_widget,
                'number': number_label,
                'name': name_label,
                'button': stage_btn
            }
            
            layout.addWidget(stage_widget, 1)
            
            # 添加分隔线(最后一个不加)
            if i < len(self.stages) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.VLine)
                line.setFixedWidth(2)
                line.setFixedHeight(scale(45))
                line.setStyleSheet("background-color: rgba(255, 255, 255, 0.5); border: none;")
                layout.addWidget(line, 0, Qt.AlignCenter)
    
    def update_stage_status(self, current_stage: str):
        """
        更新导航栏的视觉状态
        
        Args:
            current_stage: 当前激活的阶段名称
        """
        for stage_name, components in self.stage_buttons.items():
            number_label = components['number']
            name_label = components['name']

            is_current = (stage_name == current_stage)
            is_completed = self.stage_completed.get(stage_name, False)

            # 根据状态设置样式
            if is_current:
                # 当前阶段：水绿色填充圆圈，白色数字
                number_label.setStyleSheet("""
                    QLabel#stageNumber {
                        background-color: rgb(77, 171, 201);
                        border: none;
                        font-size: 24px;
                        font-weight: bold;
                        color: white;
                    }
                """)
                name_label.setStyleSheet("""
                    QLabel#stageName {
                        background: transparent;
                        font-size: 24px;
                        font-weight: bold;
                        color: rgb(77, 171, 201);
                    }
                """)
            else:
                # 其他阶段：浅青绿色圆圈，水绿色数字
                number_label.setStyleSheet("""
                    QLabel#stageNumber {
                        background-color: rgb(227, 247, 253);
                        border: none;
                        font-size: 24px;
                        font-weight: bold;
                        color: rgb(77, 171, 201);
                    }
                """)

                # 文字颜色：已完成的变绿色，未开始的保持水绿色
                if is_completed:
                    name_label.setStyleSheet("""
                        QLabel#stageName {
                            background: transparent;
                            font-size: 24px;
                            font-weight: bold;
                            color: #4CAF50;
                        }
                    """)
                else:
                    name_label.setStyleSheet("""
                        QLabel#stageName {
                            background: transparent;
                            font-size: 24px;
                            font-weight: bold;
                            color: rgb(77, 171, 201);
                        }
                    """)
    
    def mark_stage_completed(self, stage_name: str):
        """标记某个阶段为已完成"""
        if stage_name in self.stage_completed:
            self.stage_completed[stage_name] = True
