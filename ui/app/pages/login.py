"""Login page UI definition."""

from __future__ import annotations

from .. import config
from ..qt import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpacerItem, QSizePolicy, QFrame,
    Qt, QFont, qta, QPainter, QLinearGradient, QColor, QMessageBox
)
from ..utils.widgets import create_shadow_effect
from ..utils.responsive import scale


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

        self.username_input = QLineEdit('admin')
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


__all__ = ["LoginPage"]
