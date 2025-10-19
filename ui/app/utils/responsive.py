"""响应式布局工具 - 支持不同分辨率自动缩放"""

from __future__ import annotations

from typing import Tuple
from ..qt import QApplication, QScreen


class ResponsiveScaler:
    """响应式缩放工具类"""
    
    # 设计基准分辨率(调整为1080p以适配更多显示器)
    BASE_WIDTH = 1920
    BASE_HEIGHT = 1080
    
    # 最小支持分辨率(1080p)
    MIN_WIDTH = 1920
    MIN_HEIGHT = 1080
    
    def __init__(self):
        self._scale_factor = 1.0
        self._detect_screen_size()
    
    def _detect_screen_size(self):
        """检测屏幕尺寸并计算缩放因子"""
        app = QApplication.instance()
        if app is None:
            return
        
        # 获取主屏幕
        screen: QScreen = app.primaryScreen()
        if screen is None:
            return
        
        # 获取屏幕几何信息
        geometry = screen.availableGeometry()
        width = geometry.width()
        height = geometry.height()
        
        # 计算缩放因子（使用宽度和高度中较小的比例）
        width_scale = width / self.BASE_WIDTH
        height_scale = height / self.BASE_HEIGHT
        
        self._scale_factor = min(width_scale, height_scale)
        
        # 确保缩放因子不会太小
        if self._scale_factor < 0.6:
            self._scale_factor = 0.6
        elif self._scale_factor > 1.0:
            self._scale_factor = 1.0
        
        print(f"[ResponsiveScaler] 屏幕分辨率: {width}x{height}")
        print(f"[ResponsiveScaler] 缩放因子: {self._scale_factor:.2f}")
    
    def scale(self, value: int) -> int:
        """缩放单个数值"""
        return int(value * self._scale_factor)
    
    def scale_size(self, width: int, height: int) -> Tuple[int, int]:
        """缩放尺寸（宽度，高度）"""
        return self.scale(width), self.scale(height)
    
    def scale_font(self, point_size: int) -> int:
        """缩放字体大小"""
        scaled = int(point_size * self._scale_factor)
        # 字体最小不小于8pt
        return max(8, scaled)
    
    @property
    def factor(self) -> float:
        """获取当前缩放因子"""
        return self._scale_factor
    
    def is_small_screen(self) -> bool:
        """判断是否是小屏幕(低于基准分辨率)"""
        return self._scale_factor < 1.0  # 低于基准1920x1080即为小屏幕


# 全局单例
_scaler: ResponsiveScaler | None = None


def get_scaler() -> ResponsiveScaler:
    """获取全局缩放器实例"""
    global _scaler
    if _scaler is None:
        _scaler = ResponsiveScaler()
    return _scaler


def scale(value: int) -> int:
    """快捷函数：缩放单个数值"""
    return get_scaler().scale(value)


def scale_size(width: int, height: int) -> Tuple[int, int]:
    """快捷函数：缩放尺寸"""
    return get_scaler().scale_size(width, height)


def scale_font(point_size: int) -> int:
    """快捷函数：缩放字体"""
    return get_scaler().scale_font(point_size)


def is_small_screen() -> bool:
    """快捷函数：判断是否为小屏幕"""
    return get_scaler().is_small_screen()
