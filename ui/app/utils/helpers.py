"""Helper functions shared across UI pages."""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from .. import config


# 全局辅助对象，用于从后台线程安全地调用主线程回调
_callback_helper = None


def _get_callback_helper():
    """获取或创建回调辅助对象（必须在主线程中调用）"""
    global _callback_helper
    if _callback_helper is None:
        from ..qt import QTimer
        from PyQt5.QtCore import QObject, pyqtSignal
        
        class CallbackHelper(QObject):
            """辅助类，用于从后台线程发送信号到主线程"""
            callback_signal = pyqtSignal(bool)
        
        _callback_helper = CallbackHelper()
    return _callback_helper


def init_camera(callback: Callable[[bool], None]) -> None:
    """
    异步初始化 AV 预览管道（非阻塞）。
    使用后台线程执行初始化，完成后在主线程回调。
    """
    from ...utils_common.thread_process_manager import get_thread_manager
    
    # 在主线程中获取辅助对象并连接信号
    helper = _get_callback_helper()
    
    # 断开之前的所有连接，避免重复调用
    try:
        helper.callback_signal.disconnect()
    except TypeError:
        pass  # 没有连接时会抛出 TypeError
    
    # 连接新的回调
    helper.callback_signal.connect(callback)
    
    def _init_async():
        """在后台线程中执行摄像头初始化"""
        try:
            if config.NO_CAMERA_MODE:
                time.sleep(0.5)
                # 通过信号发送到主线程
                helper.callback_signal.emit(True)
                config.logger.info("模拟摄像头初始化成功。")
                return

            preview_dir = os.path.join('recordings')
            os.makedirs(preview_dir, exist_ok=True)

            primary_index = config.ACTIVE_CAMERA_INDEX if config.ACTIVE_CAMERA_INDEX is not None else 0
            camera_candidates = [primary_index]
            for fallback_idx in (0, 1):
                if fallback_idx not in camera_candidates:
                    camera_candidates.append(fallback_idx)

            audio_index = config.ACTIVE_AUDIO_DEVICE_INDEX
            audio_candidates: list[Optional[int]] = [audio_index] if audio_index is not None else [None]
            if None not in audio_candidates:
                audio_candidates.append(None)

            last_error: Exception | None = None
            success = False

            for index in camera_candidates:
                for audio_candidate in audio_candidates:
                    try:
                        config.av_start_collection(
                            save_dir=preview_dir,
                            camera_index=index,
                            video_fps=30.0,
                            input_device_index=audio_candidate,
                        )
                        config.ACTIVE_CAMERA_INDEX = index
                        config.ACTIVE_AUDIO_DEVICE_INDEX = audio_candidate
                        audio_desc = "默认" if audio_candidate is None else str(audio_candidate)
                        
                        config.logger.info(
                            "摄像头初始化成功，使用设备索引 %s，音频设备 %s。",
                            index,
                            audio_desc,
                        )
                        success = True
                        break
                    except Exception as exc:
                        last_error = exc
                        config.logger.warning(
                            "摄像头设备索引 %s / 音频设备 %s 初始化失败: %s",
                            index,
                            audio_candidate if audio_candidate is not None else "默认",
                            exc,
                        )
                        try:
                            config.av_stop_collection()
                        except Exception:
                            pass
                
                if success:
                    break

            if success:
                helper.callback_signal.emit(True)
            else:
                message = str(last_error) if last_error is not None else "未知错误"
                config.logger.error("摄像头初始化失败: %s", message)
                helper.callback_signal.emit(False)
                
        except Exception as e:
            config.logger.error(f"摄像头异步初始化过程出错: {e}")
            helper.callback_signal.emit(False)
    
    # 提交到后台线程执行，不阻塞UI
    thread_manager = get_thread_manager()
    thread_manager.submit_data_task(_init_async, task_name="初始化摄像头")


__all__ = ["init_camera"]



__all__ = ["init_camera"]
