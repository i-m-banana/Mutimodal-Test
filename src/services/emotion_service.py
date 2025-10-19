"""情绪分析服务 - 处理问卷答题的情绪推理

通过 EventBus 将数据发送到情绪模型进行推理
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio
import time

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus


class EmotionService:
    """情绪分析服务
    
    职责:
    1. 接收答题结束请求
    2. 整理音视频和文本数据
    3. 通过模型代理发送推理请求
    4. 返回情绪分数
    """
    
    def __init__(
        self,
        bus: EventBus,
        logger: Optional[logging.Logger] = None
    ):
        self.bus = bus
        self.logger = logger or logging.getLogger("service.emotion")
        self._pending_requests = {}  # 存储待处理的请求
        
        # 订阅推理结果
        self.bus.subscribe(EventTopic.DETECTION_RESULT, self._on_detection_result)
        
        self.logger.info("情绪分析服务已初始化")
    
    def _on_detection_result(self, event: Event) -> None:
        """处理模型推理结果"""
        payload = event.payload or {}
        detector = payload.get("detector", "")
        request_id = payload.get("request_id")
        
        # 只记录情绪相关的检测结果
        if detector == "model_emotion":
            self.logger.debug(f"📥 收到情绪检测结果: request_id={request_id}, pending={list(self._pending_requests.keys())}")
            
            if request_id and request_id in self._pending_requests:
                self.logger.info(f"✅ 情绪分析完成: request_id={request_id}")
                future = self._pending_requests.pop(request_id)
                future.set_result(payload)
            else:
                self.logger.warning(f"⚠️  未找到对应请求: request_id={request_id}")
        # 其他检测器结果静默忽略
    
    async def analyze_emotion_async(
        self,
        audio_paths: List[str],
        video_paths: List[str],
        text_data: List[Dict],
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """
        📍 情绪分析接口 - 异步版本
        
        参数:
            audio_paths: 音频文件路径列表
            video_paths: 视频文件路径列表
            text_data: 文本识别结果列表
            timeout: 超时时间(秒)
        
        返回: 
            {
                "emotion_score": 0.72,
                "emotion_label": "neutral",
                "confidence": 0.88,
                "inference_time_ms": 156
            }
        """
        try:
            self.logger.info(
                f"开始情绪分析: {len(audio_paths)} 个音频, "
                f"{len(video_paths)} 个视频, {len(text_data)} 个文本"
            )
            
            # 生成请求ID
            request_id = f"emotion_{int(time.time() * 1000)}"
            
            # 创建Future用于接收结果
            import asyncio
            future = asyncio.Future()
            self._pending_requests[request_id] = future
            
            self.logger.info(f"📤 发布情绪请求: request_id={request_id}, 等待结果...")
            
            # 发布情绪分析请求事件
            self.bus.publish(Event(
                topic=EventTopic.EMOTION_REQUEST,
                payload={
                    "request_id": request_id,
                    "audio_paths": audio_paths,
                    "video_paths": video_paths,
                    "text_data": text_data
                }
            ))
            
            # 等待结果(带超时)
            try:
                result = await asyncio.wait_for(future, timeout=timeout)
                
                predictions = result.get("predictions", {})
                emotion_score = predictions.get("emotion_score", 0.0)
                
                self.logger.info(
                    f"情绪分析完成: score={emotion_score:.2f}"
                )
                
                return {
                    "emotion_score": emotion_score,
                    "emotion_label": "positive" if emotion_score > 50 else "negative",
                    "confidence": 1.0,
                    **predictions
                }
                
            except asyncio.TimeoutError:
                self._pending_requests.pop(request_id, None)
                self.logger.error(f"情绪分析超时 (>{timeout}s)")
                return {
                    "emotion_score": 0.0,
                    "emotion_label": "timeout",
                    "confidence": 0.0,
                    "error": f"分析超时 (>{timeout}s)"
                }
            
        except Exception as exc:
            self.logger.error(f"情绪分析失败: {exc}", exc_info=True)
            return {
                "emotion_score": 0.0,
                "emotion_label": "error",
                "confidence": 0.0,
                "error": str(exc)
            }
    
    def analyze_emotion(
        self,
        audio_paths: List[str],
        video_paths: List[str],
        text_data: List[Dict],
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """
        📍 情绪分析接口 - 同步版本
        
        参数:
            audio_paths: 音频文件路径列表
            video_paths: 视频文件路径列表
            text_data: 文本识别结果列表
            timeout: 超时时间(秒)
        
        返回: 
            {
                "emotion_score": 0.72,
                "emotion_label": "neutral",
                "confidence": 0.88,
                ...
            }
        """
        try:
            # 创建事件循环并运行异步版本
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    asyncio.wait_for(
                        self.analyze_emotion_async(audio_paths, video_paths, text_data),
                        timeout=timeout
                    )
                )
                return result
            finally:
                loop.close()
        except asyncio.TimeoutError:
            self.logger.error(f"情绪分析超时 (>{timeout}s)")
            return {
                "emotion_score": 0.0,
                "emotion_label": "timeout",
                "confidence": 0.0,
                "error": f"分析超时 (>{timeout}s)"
            }
        except Exception as exc:
            self.logger.error(f"情绪分析失败: {exc}", exc_info=True)
            return {
                "emotion_score": 0.0,
                "emotion_label": "error",
                "confidence": 0.0,
                "error": str(exc)
            }
    
    def shutdown(self):
        """关闭服务"""
        self.logger.info("情绪分析服务正在关闭...")
        # 清理资源
