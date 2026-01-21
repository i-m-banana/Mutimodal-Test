"""会话目录统一管理器

单例模式,负责会话目录的创建和路径管理。
整个应用生命周期中,每次登录创建一个会话目录,所有模块共享此目录。
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..app.config import logger

# 项目根目录
project_root = Path(__file__).resolve().parent.parent.parent


class SessionManager:
    """会话目录单例管理器
    
    职责:
    - 在用户登录后创建唯一的会话目录
    - 为各模块提供标准化的子目录路径
    - 保证整个会话期间目录路径不变
    
    使用模式:
        manager = SessionManager.get_instance()
        session_dir = manager.start_session(username="user01")
        eeg_dir = manager.get_eeg_dir()
    """
    
    _instance: Optional['SessionManager'] = None
    
    def __init__(self):
        if SessionManager._instance is not None:
            raise RuntimeError("SessionManager 是单例,请使用 get_instance() 获取实例")
        
        self._session_dir: Optional[str] = None
        self._username: Optional[str] = None
        self._session_timestamp: Optional[str] = None
        self._is_active = False
    
    @classmethod
    def get_instance(cls) -> 'SessionManager':
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def start_session(self, username: str, base_dir: str = "recordings") -> str:
        """启动新会话,创建会话目录
        
        Args:
            username: 用户名
            base_dir: 基础录制目录,默认 "recordings"
        
        Returns:
            会话目录的绝对路径
        
        Raises:
            RuntimeError: 如果已有活动会话
        """
        if self._is_active:
            logger.warning(f"会话已激活,当前用户: {self._username}, 将重新创建会话")
        
        self._session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._username = username
        self._session_dir = str(project_root / base_dir / username / self._session_timestamp)
        
        os.makedirs(self._session_dir, exist_ok=True)
        self._is_active = True
        
        logger.info(f"✅ 会话已启动: {self._session_dir}")
        return self._session_dir
    
    def end_session(self) -> None:
        """结束当前会话"""
        if self._is_active:
            logger.info(f"🔚 会话已结束: {self._session_dir}")
            self._session_dir = None
            self._username = None
            self._session_timestamp = None
            self._is_active = False
    
    def get_session_dir(self) -> str:
        """获取会话根目录
        
        Returns:
            会话目录绝对路径
        
        Raises:
            RuntimeError: 如果会话未启动
        """
        if not self._is_active or not self._session_dir:
            raise RuntimeError("会话未启动,请先调用 start_session()")
        return self._session_dir
    
    def get_eeg_dir(self) -> str:
        """获取EEG数据子目录"""
        return os.path.join(self.get_session_dir(), "eeg")
    
    def get_emotion_dir(self) -> str:
        """获取情绪识别子目录"""
        return os.path.join(self.get_session_dir(), "emotion")
    
    def get_sart_dir(self) -> str:
        """获取SART实验子目录"""
        return os.path.join(self.get_session_dir(), "sart")
    
    def get_baseline_dir(self) -> str:
        """获取基线校准子目录"""
        return os.path.join(self.get_session_dir(), "baseline")
    
    def get_fatigue_dir(self) -> str:
        """获取疲劳评估子目录"""
        return os.path.join(self.get_session_dir(), "fatigue")
    
    def get_bp_dir(self) -> str:
        """获取血压测量子目录"""
        return os.path.join(self.get_session_dir(), "bp")
    
    def get_schulte_dir(self) -> str:
        """获取舒尔特方格子目录"""
        return os.path.join(self.get_session_dir(), "schulte")
    
    def is_active(self) -> bool:
        """检查会话是否激活"""
        return self._is_active
    
    def get_username(self) -> Optional[str]:
        """获取当前会话用户名"""
        return self._username
    
    def get_timestamp(self) -> Optional[str]:
        """获取会话时间戳"""
        return self._session_timestamp
