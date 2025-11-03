"""Login page UI definition."""

from __future__ import annotations

from .. import config
from ..qt import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpacerItem, QSizePolicy, QFrame, QDialog,
    Qt, QFont, qta, QPainter, QLinearGradient, QColor, QMessageBox
)
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
        main_layout.setAlignment(Qt.AlignCenter)
        form_container = self._create_login_form()
        main_layout.addWidget(form_container)

    def _create_login_form(self) -> QWidget:
        container = QFrame()
        container.setObjectName("loginFrame")
        # 使用固定尺寸，不进行响应式缩放
        container.setFixedSize(450, 550)
        container.setGraphicsEffect(create_shadow_effect())

        layout = QVBoxLayout(container)
        # 使用响应式边距和间距
        layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))
        layout.setSpacing(scale(15))

        icon_label = QLabel()
        icon_label.setPixmap(qta.icon('fa5s.user-shield', color='#1565C0').pixmap(60, 60))
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        title = QLabel('非接触人员状态评估系统')
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("loginTitle")
        layout.addWidget(title)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.username_input = QLineEdit('')
        self.username_input.setPlaceholderText('用户名')
        self.username_input.setObjectName("loginInput")
        # 不设置 textMargins，让 CSS 的 padding 来控制
        self.username_input.setFixedHeight(50)

        self.password_input = QLineEdit('123456')
        self.password_input.setPlaceholderText('密码')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("loginInput")
        self.password_input.setFixedHeight(50)

        password_layout = QHBoxLayout(self.password_input)
        password_layout.setContentsMargins(0, 0, 5, 0)
        password_layout.addStretch()
        self.toggle_password_button = QPushButton()
        self.toggle_password_button.setIcon(qta.icon('fa5s.eye-slash', color='grey'))
        self.toggle_password_button.setCursor(Qt.PointingHandCursor)
        self.toggle_password_button.setFlat(True)
        self.toggle_password_button.setCheckable(True)
        self.toggle_password_button.clicked.connect(self._toggle_password_visibility)
        password_layout.addWidget(self.toggle_password_button)

        layout.addWidget(self.username_input)
        layout.addWidget(self.password_input)
        layout.addSpacing(20)

        login_button = QPushButton('登 录')
        login_button.setObjectName("loginButton")
        login_button.setFixedHeight(50)
        login_button.setCursor(Qt.PointingHandCursor)
        login_button.clicked.connect(self._perform_login)
        layout.addWidget(login_button)

        # 添加注册按钮
        register_button = QPushButton('注 册')
        register_button.setObjectName("successButton")
        register_button.setFixedHeight(50)
        register_button.setCursor(Qt.PointingHandCursor)
        register_button.clicked.connect(self._show_register_dialog)
        layout.addWidget(register_button)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        copyright_label = QLabel("© 2025 智能评估系统. All Rights Reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setObjectName("copyrightLabel")
        layout.addWidget(copyright_label)

        return container

    def paintEvent(self, event):  # type: ignore[override]
        painter = QPainter(self)
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
        self.setFixedSize(500, 500)  # ✅ 增大对话框尺寸
        self._init_ui()
    
    def _init_ui(self) -> None:
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(scale(30), scale(30), scale(30), scale(30))
        layout.setSpacing(scale(20))
        
        # 标题
        title = QLabel("创建新账户")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("阿里健康体2.0 中文 45 R", 28, QFont.Bold))  # ✅ 增大标题字号
        title.setStyleSheet("color: #1565C0; margin-bottom: 15px;")
        layout.addWidget(title)
        
        # ❌ 删除提示小字，不占位置
        
        layout.addSpacing(20)  # ✅ 增加顶部间距
        
        # 用户名输入
        username_label = QLabel("用户名:")
        username_label.setStyleSheet("color: #333; font-size: 18px; font-weight: bold;")  # ✅ 增大标签字号
        layout.addWidget(username_label)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("请输入用户名 (3-20个字符)")
        self.username_input.setObjectName("loginInput")
        self.username_input.setFixedHeight(50)  # ✅ 增大输入框高度
        layout.addWidget(self.username_input)
        
        # 密码输入
        password_label = QLabel("密码:")
        password_label.setStyleSheet("color: #333; font-size: 18px; font-weight: bold;")  # ✅ 增大标签字号
        layout.addWidget(password_label)
        
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("请输入密码 (6-20个字符)")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setObjectName("loginInput")
        self.password_input.setFixedHeight(50)  # ✅ 增大输入框高度
        layout.addWidget(self.password_input)
        
        # 确认密码输入
        confirm_label = QLabel("确认密码:")
        confirm_label.setStyleSheet("color: #333; font-size: 18px; font-weight: bold;")  # ✅ 增大标签字号
        layout.addWidget(confirm_label)
        
        self.confirm_input = QLineEdit()
        self.confirm_input.setPlaceholderText("请再次输入密码")
        self.confirm_input.setEchoMode(QLineEdit.Password)
        self.confirm_input.setObjectName("loginInput")
        self.confirm_input.setFixedHeight(50)  # ✅ 增大输入框高度
        layout.addWidget(self.confirm_input)
        
        layout.addSpacing(20)  # ✅ 增加底部间距
        
        # 按钮布局
        button_layout = QHBoxLayout()
        button_layout.setSpacing(scale(15))
        
        # 取消按钮
        cancel_button = QPushButton("取消")
        cancel_button.setObjectName("finishButton")
        cancel_button.setFixedHeight(50)  # ✅ 增大按钮高度
        cancel_button.setCursor(Qt.PointingHandCursor)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        # 注册按钮
        register_button = QPushButton("注册")
        register_button.setObjectName("successButton")
        register_button.setFixedHeight(50)  # ✅ 增大按钮高度
        register_button.setCursor(Qt.PointingHandCursor)
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
