"""Login page UI definition."""

from __future__ import annotations

from .. import config
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpacerItem, QSizePolicy, QFrame, QDialog,
    QMessageBox, QApplication
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QLinearGradient, QColor, QPixmap
import qtawesome as qta
from ..utils.widgets import create_shadow_effect
from ..utils.responsive import scale
import csv
from pathlib import Path


class LoginPage(QWidget):
    """登录页面UI和逻辑。"""

    def __init__(self, on_login_success_callback) -> None:
        super().__init__()
        self.on_login_success = on_login_success_callback
        self._init_ui()

    def _init_ui(self) -> None:
        self.setAutoFillBackground(True)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ========== 🎯 顶部标题区域 ==========
        title_h_layout = QHBoxLayout()
        title_h_layout.setContentsMargins(0, 0, 0, 0)
        title_h_layout.addStretch()  # 左侧自动填充

        # 创建独立的标题容器
        title_container = self._create_title_container()
        title_h_layout.addWidget(title_container)

        title_h_layout.addSpacing(100)  # 距离右边100px，与登录框对齐

        # ========== 登录表单区域 ==========
        form_h_layout = QHBoxLayout()
        form_h_layout.setContentsMargins(0, 0, 0, 0)
        form_h_layout.addStretch()  # 左侧自动填充

        form_container = self._create_login_form()
        form_h_layout.addWidget(form_container)
        form_h_layout.addSpacing(100)  # 距离右边100px

        # ========== 组合到主布局 ==========
        main_layout.addStretch()  # 顶部自动填充
        main_layout.addLayout(title_h_layout)  # 添加标题
        main_layout.addSpacing(20)  # 标题和登录框之间的间距
        main_layout.addLayout(form_h_layout)  # 添加登录表单
        main_layout.addStretch()  # 底部自动填充

    def _create_title_container(self) -> QWidget:
        """创建独立的系统标题容器（透明背景）"""
        container = QFrame()
        container.setObjectName("titleFrame")
        container.setStyleSheet("""
            QFrame#titleFrame {
                background-color: transparent;  /* 透明背景 */
                border: none;
            }
        """)

        # 设置固定宽度（与登录框保持一致）
        container.setFixedWidth(700)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(scale(30), scale(20), scale(30), scale(20))
        layout.setSpacing(0)

        # ========== 标题行（图标 + 文字横向排列）==========
        title_row = QHBoxLayout()
        title_row.setSpacing(scale(15))
        title_row.setAlignment(Qt.AlignCenter)

        # 图标
        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.user-shield', color='#00BCFE').pixmap(60, 60))
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setFixedSize(60, 60)
        title_row.addWidget(icon_label)

        # 标题文字
        title = QLabel('多模态状态评估系统')
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setFont(QFont("阿里健康体2.0 中文 45 R", 36, 50))
        title.setStyleSheet("""
            QLabel {
                color: #00BCFE;
                letter-spacing: 3px;
                font-weight: bold; 
                background: transparent;
                font-size: 60px;
            }
        """)

        title_row.addWidget(title)
        layout.addLayout(title_row)

        return container

    def _create_login_form(self) -> QWidget:
        container = QFrame()
        container.setObjectName("loginFrame")
        container.setStyleSheet("""
            QFrame#loginFrame {
               background-color: rgba(12, 58, 98, 178);
               border: 3px solid #1794E3;
               border-radius: 20px;
            }
        """)

        container.setFixedSize(700, 600)
        container.setGraphicsEffect(create_shadow_effect())

        layout = QVBoxLayout(container)
        layout.setContentsMargins(scale(40), scale(50), scale(40), scale(50))  # 增加上下边距
        layout.setSpacing(0)  # 手动控制间距

        # ========== 🎯 新增：用户登录小标题 ==========
        login_header = QHBoxLayout()
        login_header.setSpacing(scale(10))

        # 用户图标（使用系统内置图标）
        user_icon_label = QLabel()
        # 使用 QtAwesome 图标
        user_icon_label.setPixmap(qta.icon('fa5s.user-circle', color='#00BCFE').pixmap(28, 28))
        # 或者使用 Qt 内置图标（备选方案）
        # from PyQt5.QtWidgets import QStyle
        # user_icon = self.style().standardIcon(QStyle.SP_UserIcon)
        # user_icon_label.setPixmap(user_icon.pixmap(28, 28))

        # "用户登录" 文字
        login_title = QLabel('用户登录')
        login_title.setFont(QFont("阿里健康体2.0 中文 45 R", 18, 75))
        login_title.setStyleSheet("""
                QLabel {
                    color: #00BCFE;
                    letter-spacing: 2px;
                    font-size: 44px;
                }
            """)

        # 组合图标和文字
        login_header.addWidget(user_icon_label)
        login_header.addWidget(login_title)
        login_header.addStretch()

        layout.addLayout(login_header)
        layout.addSpacing(scale(40))  # 标题和用户名之间
        # ========== 用户登录标题结束 ==========

        # ========== 用户名输入框 ==========
        self.username_input = QLineEdit('admin')
        self.username_input.setPlaceholderText('👤 用户名')
        self.username_input.setFont(QFont("阿里健康体2.0 中文 45 R", 22))  # 增大字体从14到22
        self.username_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 200);
                height: 80px;
                width: 150px;
                border: 2px solid #BBDEFB;
                border-radius: 10px;
                padding: 0 15px;
                font-size: 28px;
                color: #0D47A1;
            }
            QLineEdit:focus {
                border: 2px solid #1565C0;
                background-color: rgba(255, 255, 255, 240);
            }
            QLineEdit::placeholder {
                color: #64B5F6;
                font-size: 24px;
            }
        """)
        layout.addWidget(self.username_input)
        layout.addSpacing(scale(20))  # 用户名和密码之间

        # ========== 密码输入框 ==========
        self.password_input = QLineEdit('123456')
        self.password_input.setPlaceholderText('🔒 密码')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setFont(QFont("阿里健康体2.0 中文 45 R", 22))  # 增大字体从14到22
        self.password_input.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 200);
                height: 80px;
                width: 150px;
                border: 2px solid #BBDEFB;
                border-radius: 10px;
                padding: 0 15px 0 15px;
                font-size: 28px;
                color: #0D47A1;
            }
            QLineEdit:focus {
                border: 2px solid #1565C0;
                background-color: rgba(255, 255, 255, 240);
            }
            QLineEdit::placeholder {
                color: #64B5F6;
                font-size: 24px;
            }
        """)

        # 密码显示/隐藏按钮
        password_layout = QHBoxLayout(self.password_input)
        password_layout.setContentsMargins(0, 0, 5, 0)
        password_layout.addStretch()
        self.toggle_password_button = QPushButton()
        self.toggle_password_button.setIcon(qta.icon('fa5s.eye-slash', color='#1976D2'))
        self.toggle_password_button.setCursor(Qt.PointingHandCursor)
        self.toggle_password_button.setFlat(True)
        self.toggle_password_button.setCheckable(True)
        self.toggle_password_button.clicked.connect(self._toggle_password_visibility)
        password_layout.addWidget(self.toggle_password_button)

        layout.addWidget(self.password_input)
        layout.addSpacing(scale(20))  # 密码和注册之间

        # ========== 注册按钮 ==========
        # 1. 创建一个水平布局容器
        register_layout = QHBoxLayout()

        # 2. 添加弹簧（把按钮推到右边）
        register_layout.addStretch()

        # 3. 创建缩小版的注册按钮
        register_button = QPushButton('注 册')
        register_button.setFixedHeight(55)  # 增大高度从45到55
        register_button.setMinimumWidth(100)  # 增大宽度从80到100
        register_button.setCursor(Qt.PointingHandCursor)
        register_button.setFont(QFont("阿里健康体2.0 中文 45 R", 20, 75))  # 增大字体从15到20
        register_button.clicked.connect(self._show_register_dialog)
        register_button.setStyleSheet("""
                  QPushButton {
                      background: transparent;
                      color: white;
                      border-radius: 8px;
                      font-size: 24px;
                      font-weight: bold;
                      letter-spacing: 1px;
                  }
                  QPushButton:hover {
                      background-color: rgba(21, 101, 192, 0.1);
                      border: 2px solid #1E88E5;
                      color: #1E88E5;
                  }
                  QPushButton:pressed {
                      background-color: rgba(13, 71, 161, 0.2);
                      border: 2px solid #0D47A1;
                      color: #0D47A1;
                  }
              """)

        # 4. 把按钮添加到水平布局
        register_layout.addWidget(register_button)

        # 5. 把水平布局添加到主布局
        layout.addLayout(register_layout)
        layout.addSpacing(scale(40))  # 注册和登录之间

        # ========== 登录按钮 ==========
        login_button = QPushButton('登 录')
        login_button.setFixedHeight(150)
        login_button.setMinimumWidth(10)  # ✅ 设置最小宽度为600
        login_button.setCursor(Qt.PointingHandCursor)
        login_button.setFont(QFont("阿里健康体2.0 中文 45 R", 15, 75))
        login_button.clicked.connect(self._perform_login)
        login_button.setStyleSheet("""
            QPushButton {
                 background-color: rgba(0, 188, 254, 0.9);
                height: 80px;
                width: 150px;
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 40px;
                font-weight: bold;
                letter-spacing: 5px;
            }
            QPushButton:hover {
                 background: qlineargradient(x1:0, y1:0, x4:1, y4:0,
                    stop:0 #0D47A1, stop:1 #1565C0);
            }
            QPushButton:pressed {
                 background: qlineargradient(x1:0, y1:0, x4:1, y4:0,
                    stop:0 #0D47A1, stop:1 #1565C0);
            }
        """)
        layout.addWidget(login_button)
        layout.addSpacing(scale(30))  # 底部留白

        layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # # ========== 版权信息 ==========
        # copyright_label = QLabel("© 2025 多模态状态评估系统. All Rights Reserved.")
        # copyright_label.setAlignment(Qt.AlignCenter)
        # copyright_label.setFont(QFont("阿里健康体2.0 中文 45 R", 10))
        # copyright_label.setStyleSheet("""
        #     QLabel {
        #         color: #00B5FE;
        #         font-size: 36px;
        #         margin-top: 10px;
        #     }
        # """)
        # layout.addWidget(copyright_label)

        return container

    def paintEvent(self, event):  # type: ignore[override]
        from PyQt5.QtGui import QPainter, QPixmap
        import os
        bg_path = str(config.BASE_DIR / "assets" / "login.png")

        painter = QPainter(self)
        pixmap = QPixmap(bg_path)
        if not pixmap.isNull():
            painter.drawPixmap(self.rect(), pixmap)
        else:
            from PyQt5.QtGui import QLinearGradient, QColor
            gradient = QLinearGradient(0, 0, 0, self.height())
            gradient.setColorAt(0, QColor("#F4F6F7"))
            gradient.setColorAt(1, QColor("#EAECEE"))
            painter.fillRect(self.rect(), gradient)
        super().paintEvent(event)

    def _toggle_password_visibility(self, checked: bool) -> None:
        if checked:
            self.password_input.setEchoMode(QLineEdit.Normal)
            self.toggle_password_button.setIcon(qta.icon('fa5s.eye', color='grey'))
        else:
            self.password_input.setEchoMode(QLineEdit.Password)
            self.toggle_password_button.setIcon(qta.icon('fa5s.eye-slash', color='grey'))

    def _perform_login(self) -> None:
        users = config.load_users_from_csv()
        user = self.username_input.text()
        pwd = self.password_input.text()
        if user in users and users[user] == pwd:
            config.logger.info("用户 '%s' 登录成功。", user)
            self.on_login_success(user)
        else:
            config.logger.warning("用户 '%s' 尝试登录失败。", user)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("登录失败")
            msg.setInformativeText("用户名或密码错误，请重试。")
            msg.setWindowTitle("错误")
            msg.exec_()
    
    def _show_register_dialog(self) -> None:
        """显示注册对话框"""
        dialog = RegisterDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 注册成功后自动填充用户名
            username = dialog.username_input.text()
            self.username_input.setText(username)
            self.password_input.setText("")
            self.password_input.setFocus()


class RegisterDialog(QDialog):
    """注册对话框"""
    
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("用户注册")
        self.setModal(True)
        self.setFixedSize(450, 520)  # 增大尺寸以适应更大的字体（从400x450增大到450x520）
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))
        layout.setSpacing(scale(20))
        
        # 标题
        title = QLabel("创建新账户")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("阿里健康体2.0 中文 45 R", 22, 75))
        title.setStyleSheet("color: #1565C0; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # 提示文字
        hint = QLabel("请填写以下信息完成注册")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #666; font-size: 18px; margin-bottom: 10px;")  # 从15px增大到18px
        layout.addWidget(hint)
        
        layout.addSpacing(10)
        
        # 用户名输入
        username_label = QLabel("用户名:")
        username_label.setStyleSheet("color: #333; font-size: 20px; font-weight: bold;")  # 从16px增大到20px
        layout.addWidget(username_label)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名 (3-20个字符)")
        self.username_input.setObjectName("loginInput")
        self.username_input.setFixedHeight(50)  # 从45增大到50
        self.username_input.setStyleSheet("font-size: 18px;")  # 添加字体大小
        layout.addWidget(self.username_input)
        
        # 密码输入
        password_label = QLabel("密码:")
        password_label.setStyleSheet("color: #333; font-size: 20px; font-weight: bold;")  # 从16px增大到20px
        layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码 (6-20个字符)")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("loginInput")
        self.password_input.setFixedHeight(50)  # 从45增大到50
        self.password_input.setStyleSheet("font-size: 18px;")  # 添加字体大小
        layout.addWidget(self.password_input)
        
        # 确认密码输入
        confirm_label = QLabel("确认密码:")
        confirm_label.setStyleSheet("color: #333; font-size: 20px; font-weight: bold;")  # 从16px增大到20px
        layout.addWidget(confirm_label)
        
        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("请再次输入密码")
        self.confirm_input.setEchoMode(QLineEdit.Password)
        self.confirm_input.setObjectName("loginInput")
        self.confirm_input.setFixedHeight(50)  # 从45增大到50
        self.confirm_input.setStyleSheet("font-size: 18px;")  # 添加字体大小
        layout.addWidget(self.confirm_input)
        
        layout.addSpacing(10)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(scale(15))
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.setObjectName("finishButton")
        cancel_button.setFixedHeight(50)  # 从45增大到50
        cancel_button.setCursor(Qt.PointingHandCursor)
        cancel_button.setStyleSheet("font-size: 20px; font-weight: bold;")  # 添加字体样式
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # 注册按钮
        register_button = QPushButton("注册")
        register_button.setObjectName("successButton")
        register_button.setFixedHeight(50)  # 从45增大到50
        register_button.setCursor(Qt.PointingHandCursor)
        register_button.setStyleSheet("font-size: 20px; font-weight: bold;")  # 添加字体样式
        register_button.clicked.connect(self._perform_register)
        button_layout.addWidget(register_button)
        
        layout.addLayout(button_layout)
    
    def _perform_register(self) -> None:
        """执行注册操作"""
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm = self.confirm_input.text()
        
        # 验证输入
        if not username:
            QMessageBox.warning(self, "注册失败", "用户名不能为空！")
            return
        
        if len(username) < 3 or len(username) > 20:
            QMessageBox.warning(self, "注册失败", "用户名长度必须在3-20个字符之间！")
            return
        
        if not username.isalnum():
            QMessageBox.warning(self, "注册失败", "用户名只能包含字母和数字！")
            return
        
        if not password:
            QMessageBox.warning(self, "注册失败", "密码不能为空！")
            return
        
        if len(password) < 6 or len(password) > 20:
            QMessageBox.warning(self, "注册失败", "密码长度必须在6-20个字符之间！")
            return
        
        if password != confirm:
            QMessageBox.warning(self, "注册失败", "两次输入的密码不一致！")
            return
        
        # 检查用户名是否已存在
        users = config.load_users_from_csv()
        if username in users:
            QMessageBox.warning(self, "注册失败", f"用户名 '{username}' 已存在，请选择其他用户名！")
            return
        
        # 保存到CSV文件
        try:
            csv_path = config.BASE_DIR / "data" / "users" / "users.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查文件是否存在且不为空，确保最后一行有换行符
            if csv_path.exists() and csv_path.stat().st_size > 0:
                with open(csv_path, 'rb') as f:
                    f.seek(-1, 2)  # 移到文件最后一个字节
                    last_char = f.read(1)
                    needs_newline = last_char not in (b'\n', b'\r')
                
                if needs_newline:
                    # 如果最后没有换行符，先添加一个
                    with open(csv_path, 'a', encoding='utf-8') as f:
                        f.write('\n')
            
            # 追加新用户到CSV文件
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([username, password])
            
            config.logger.info(f"新用户 '{username}' 注册成功")
            
            # 显示成功消息
            QMessageBox.information(
                self, 
                "注册成功", 
                f"用户 '{username}' 注册成功！\n\n请使用新账户登录。"
            )
            
            self.accept()
            
        except Exception as e:
            config.logger.error(f"注册失败: {e}")
            QMessageBox.critical(
                self, 
                "注册失败", 
                f"保存用户信息时发生错误：{e}\n\n请联系管理员。"
            )


__all__ = ["LoginPage", "RegisterDialog"]
