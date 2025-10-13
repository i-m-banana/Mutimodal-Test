"""情绪分析服务 - 处理问卷答题的情绪推理

通过 model_proxy_service 将数据发送到情绪模型后端进行推理
"""

import logging
from typing import Dict, List, Optional, Any
import asyncio


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
        model_proxy=None,
        logger: Optional[logging.Logger] = None
    ):
        self.model_proxy = model_proxy
        self.logger = logger or logging.getLogger("service.emotion")
        self.logger.info("情绪分析服务已初始化")
    
    async def analyze_emotion_async(
        self,
        audio_paths: List[str],
        video_paths: List[str],
        text_data: List[Dict]
    ) -> Dict[str, Any]:
        """
        📍 情绪分析接口 - 异步版本
        
        参数:
            audio_paths: 音频文件路径列表
            video_paths: 视频文件路径列表
            text_data: 文本识别结果列表
        
        返回: 
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
        try:
            if not self.model_proxy:
                self.logger.warning("模型代理服务未启用，返回默认值")
                return {
                    "emotion_score": 0.5,
                    "emotion_label": "unknown",
                    "confidence": 0.0,
                    "error": "模型代理服务未启用"
                }
            
            self.logger.info(
                f"开始情绪分析: {len(audio_paths)} 个音频, "
                f"{len(video_paths)} 个视频, {len(text_data)} 个文本"
            )
            
            # 通过模型代理发送请求
            result = await self.model_proxy.request_inference(
                model_type="emotion",
                data={
                    "audio_paths": audio_paths,
                    "video_paths": video_paths,
                    "text_data": text_data
                }
            )
            
            self.logger.info(
                f"情绪分析完成: {result.get('emotion_label', 'unknown')} "
                f"(score={result.get('emotion_score', 0)})"
            )
            
            return result
            
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
