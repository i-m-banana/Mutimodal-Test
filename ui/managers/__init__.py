"""业务逻辑管理器模块

本目录包含应用程序的业务逻辑管理类，负责协调各种服务和数据处理。
与 services 目录的区别：
- services: 底层网络服务、后端通信
- managers: 高层业务逻辑、数据管理和协调
"""

from .test_db_manager import TestDBManager
from .session_manager import SessionManager
from .score_calculator import ScoreCalculator
from .speech_recognition_manager import (
    add_audio_for_recognition,
    get_recognition_results,
    clear_recognition_results,
    stop_recognition,
)

__all__ = [
    "TestDBManager",
    "SessionManager",
    "ScoreCalculator",
    "add_audio_for_recognition",
    "get_recognition_results",
    "clear_recognition_results",
    "stop_recognition",
]
