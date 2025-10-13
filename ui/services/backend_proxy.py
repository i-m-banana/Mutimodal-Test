"""统一的后端命令代理服务.

UI层通过此模块向后端发送WebSocket命令,实现UI与后端的完全分离。
所有硬件操作都在后端完成,UI只负责发送命令和接收结果。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
    from .backend_client import get_backend_client
except ImportError:  # pragma: no cover
    from ui.services.backend_client import get_backend_client  # type: ignore

_logger = logging.getLogger("ui.backend_proxy")


def send_command(action: str, payload: Optional[Dict[str, Any]] = None,
                 *, timeout: float = 10.0) -> Dict[str, Any]:
    """发送命令到后端并等待响应.
    
    Args:
        action: 命令动作,格式: "service.action" (例如: "multimodal.start")
        payload: 命令参数字典
        timeout: 超时时间(秒)
        
    Returns:
        后端返回的结果字典
        
    Raises:
        ConnectionError: 无法连接到后端
        TimeoutError: 命令执行超时
    """
    client = get_backend_client()
    
    # 等待后端连接建立(最多5秒)
    if not client.wait_for_connection(timeout=5.0):
        raise ConnectionError("Failed to establish backend connection within 5 seconds")
    
    return client.send_command_sync(action, payload, timeout=timeout)


# ============================================================================
# AV (音视频) 服务代理
# ============================================================================

def av_start_preview(**kwargs) -> Dict[str, Any]:
    """启动音视频预览."""
    return send_command("av.start_preview", kwargs)


def av_stop_preview() -> Dict[str, Any]:
    """停止音视频预览."""
    return send_command("av.stop_preview")


def av_start_recording(**kwargs) -> Dict[str, Any]:
    """启动音视频录制."""
    return send_command("av.start_recording", kwargs)


def av_stop_recording() -> Dict[str, Any]:
    """停止音视频录制."""
    return send_command("av.stop_recording")


def av_get_frame(timeout: float = 1.0) -> Dict[str, Any]:
    """获取当前音视频帧."""
    return send_command("av.get_frame", timeout=timeout)


# ============================================================================
# EEG 服务代理
# ============================================================================

def eeg_start_collection(username: str, save_dir: str, part: int = 1) -> Dict[str, Any]:
    """启动EEG采集."""
    payload = {
        "username": username,
        "save_dir": save_dir,
        "part": part,
    }
    return send_command("eeg.start", payload)


# 别名,兼容旧代码
eeg_start = eeg_start_collection


def eeg_stop_collection() -> Dict[str, Any]:
    """停止EEG采集."""
    return send_command("eeg.stop")


# 别名,兼容旧代码  
eeg_stop = eeg_stop_collection


def eeg_get_file_paths() -> Dict[str, Any]:
    """获取EEG文件路径."""
    return send_command("eeg.file_paths")


# 别名,兼容旧代码
eeg_paths = eeg_get_file_paths


def eeg_get_snapshot(timeout: float = 2.0) -> Dict[str, Any]:
    """获取EEG快照数据.
    
    使用较短超时(2秒)避免阻塞其他命令。
    """
    return send_command("eeg.snapshot", timeout=timeout)


# ============================================================================
# 多模态服务代理
# ============================================================================

def multimodal_start_collection(username: str, save_dir: str, 
                                 part: int = 1, queue_duration: float = 5.0) -> Dict[str, Any]:
    """启动多模态数据采集."""
    payload = {
        "username": username,
        "save_dir": save_dir,
        "part": part,
        "queue_duration": queue_duration,
    }
    return send_command("multimodal.start", payload)


def multimodal_stop_collection() -> Dict[str, Any]:
    """停止多模态数据采集."""
    return send_command("multimodal.stop")


def multimodal_cleanup() -> Dict[str, Any]:
    """清理多模态采集器."""
    return send_command("multimodal.cleanup")


# 别名,兼容旧代码
cleanup_collector = multimodal_cleanup


def multimodal_get_snapshot(timeout: float = 2.0) -> Dict[str, Any]:
    """获取多模态快照数据.
    
    使用较短超时(2秒)避免阻塞其他命令。
    快照请求应该快速返回，避免影响定时器轮询。
    """
    return send_command("multimodal.snapshot", timeout=timeout)


def multimodal_get_paths() -> Dict[str, Any]:
    """获取多模态文件路径."""
    return send_command("multimodal.paths")


# 别名,兼容旧代码
get_multimodal_file_paths = multimodal_get_paths


# ============================================================================
# 血压服务代理
# ============================================================================

def bp_start_measurement(port: Optional[str] = None, 
                         simulation: bool = False) -> Dict[str, Any]:
    """启动血压测量."""
    payload = {"port": port, "simulation": simulation}
    return send_command("bp.start", payload)


def bp_stop_measurement() -> Dict[str, Any]:
    """停止血压测量."""
    return send_command("bp.stop")


def bp_get_snapshot(timeout: float = 2.0) -> Dict[str, Any]:
    """获取血压快照数据.
    
    使用较短超时(2秒)避免阻塞其他命令。
    """
    return send_command("bp.snapshot", timeout=timeout)


def bp_get_status() -> Dict[str, Any]:
    """获取血压服务状态."""
    return send_command("bp.status")


# ============================================================================
# TTS (语音合成) 服务代理
# ============================================================================

def tts_speak(text: str, **kwargs) -> Dict[str, Any]:
    """文本转语音."""
    payload = {"text": text, **kwargs}
    return send_command("tts.speak", payload)


# ============================================================================
# Emotion (情绪分析) 服务代理
# ============================================================================

def emotion_analyze(
    audio_paths: list,
    video_paths: list,
    text_data: list,
    timeout: float = 15.0
) -> Dict[str, Any]:
    """
    执行情绪分析.
    
    Args:
        audio_paths: 音频文件路径列表
        video_paths: 视频文件路径列表
        text_data: 文本识别结果列表
        timeout: 超时时间(秒)，默认15秒
        
    Returns:
        {
            "emotion_score": 0.72,
            "emotion_label": "neutral",
            "audio_score": 0.68,
            "video_score": 0.75,
            "text_score": 0.73,
            "confidence": 0.88,
            "inference_time_ms": 156
        }
    """
    payload = {
        "audio_paths": audio_paths,
        "video_paths": video_paths,
        "text_data": text_data,
        "timeout": timeout
    }
    return send_command("emotion.analyze", payload, timeout=timeout)


__all__ = [
    "send_command",
    # AV
    "av_start_preview",
    "av_stop_preview",
    "av_start_recording",
    "av_stop_recording",
    "av_get_frame",
    # EEG
    "eeg_start_collection",
    "eeg_start",  # 别名
    "eeg_stop_collection",
    "eeg_stop",  # 别名
    "eeg_get_file_paths",
    "eeg_paths",  # 别名
    "eeg_get_snapshot",
    # Multimodal
    "multimodal_start_collection",
    "multimodal_stop_collection",
    "multimodal_cleanup",
    "cleanup_collector",  # 别名
    "multimodal_get_snapshot",
    "multimodal_get_paths",
    "get_multimodal_file_paths",  # 别名
    # BP
    "bp_start_measurement",
    "bp_stop_measurement",
    "bp_get_snapshot",
    "bp_get_status",
    # TTS
    "tts_speak",
    # Emotion
    "emotion_analyze",
]
