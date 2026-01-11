"""Service layer for backend communication.

UI层服务模块,负责与后端WebSocket通信。所有硬件操作都在后端完成。

Architecture:
- backend_client.py: WebSocket客户端,处理双向通信
- backend_proxy.py: 统一的后端命令代理,封装所有WebSocket命令
- av_service.py: 音视频服务代理(包含本地模拟逻辑)
- session_manager.py: 会话目录管理
- database_service.py: 数据库操作服务

所有服务命令通过backend_proxy发送到后端,由后端的ui_command_router路由处理。
"""

from __future__ import annotations

__all__ = []
