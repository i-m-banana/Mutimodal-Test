"""统一的推理模型接口基类

支持两种模式：
1. integrated: 模型直接集成在后端进程中
2. remote: 通过WebSocket代理到独立进程
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseInferenceModel(ABC):
    """推理模型基类
    
    子类需要实现：
    - initialize(): 初始化模型资源
    - infer(): 执行推理
    - cleanup(): 清理资源
    """
    
    def __init__(
        self,
        name: str,
        *,
        logger: Optional[logging.Logger] = None,
        **options
    ):
        """初始化推理模型
        
        Args:
            name: 模型名称
            logger: 日志记录器
            **options: 模型选项
        """
        self.name = name
        self.logger = logger or logging.getLogger(f"model.{name}")
        self.options = options
        self._initialized = False
    
    def load(self) -> None:
        """加载模型（统一入口）"""
        if self._initialized:
            self.logger.warning(f"模型 {self.name} 已经初始化")
            return
        
        self.logger.info(f"加载模型: {self.name}")
        self.initialize()
        self._initialized = True
        self.logger.info(f"✅ 模型 {self.name} 加载完成")
    
    def unload(self) -> None:
        """卸载模型（统一入口）"""
        if not self._initialized:
            return
        
        self.logger.info(f"卸载模型: {self.name}")
        self.cleanup()
        self._initialized = False
        self.logger.info(f"✅ 模型 {self.name} 已卸载")
    
    @abstractmethod
    def initialize(self) -> None:
        """初始化模型资源（子类实现）"""
        pass
    
    @abstractmethod
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理（子类实现）
        
        Args:
            data: 输入数据
            
        Returns:
            推理结果
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """清理模型资源（子类实现）"""
        pass
    
    @property
    def is_initialized(self) -> bool:
        """模型是否已初始化"""
        return self._initialized


__all__ = ["BaseInferenceModel"]
