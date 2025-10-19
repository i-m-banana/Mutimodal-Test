"""模型代理服务 - 管理所有模型后端连接

此服务作为主后端与多个独立模型后端之间的桥梁:
1. 建立与各模型后端的WebSocket连接
2. 将采集的数据分发到对应模型后端
3. 接收模型推理结果并发布到事件总线

使用方法:
在 Orchestrator 中初始化并启动此服务。
"""

import logging
from typing import Any, Dict, List, Optional

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus

try:
    from ..interfaces.model_ws_client import ModelBackendClient
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False


class ModelProxyService:
    """模型代理服务 - 管理与多个模型后端的连接和通信
    
    职责:
    1. 建立与各模型后端的WebSocket连接
    2. 订阅多模态数据快照事件
    3. 将数据分发到对应的模型后端
    4. 接收推理结果并发布到事件总线
    
    Attributes:
        bus: 事件总线实例
        clients: 模型后端客户端字典 {model_type: client}
    """
    
    def __init__(
        self,
        bus: EventBus,
        model_configs: List[Dict[str, Any]],
        *,
        logger: Optional[logging.Logger] = None
    ):
        """初始化模型代理服务
        
        Args:
            bus: 事件总线实例
            model_configs: 模型后端配置列表
            logger: 日志记录器
        """
        self.bus = bus
        self.model_configs = model_configs
        self.logger = logger or logging.getLogger("service.model_proxy")
        self.clients: Dict[str, "ModelBackendClient"] = {}
        self._running = False
        
        if not HAS_CLIENT:
            self.logger.warning("ModelBackendClient 不可用,模型代理服务将被禁用")
    
    def start(self) -> None:
        """启动所有模型后端客户端"""
        if self._running:
            self.logger.warning("模型代理服务已在运行")
            return
        
        if not HAS_CLIENT:
            self.logger.warning("模型代理服务未启用 (缺少依赖)")
            return
        
        self.logger.info("启动模型代理服务...")
        
        # 连接到所有启用的模型后端
        enabled_count = 0
        for config in self.model_configs:
            if not config.get("enabled", True):
                self.logger.info(f"跳过禁用的模型后端: {config.get('name')}")
                continue
            
            model_type = config["type"]
            conn_cfg = config.get("connection", {})
            
            host = conn_cfg.get("host", "127.0.0.1")
            port = conn_cfg.get("port", 8766)
            url = f"ws://{host}:{port}"
            
            reconnect = conn_cfg.get("reconnect", True)
            reconnect_interval = conn_cfg.get("reconnect_interval", 5.0)
            
            try:
                client = ModelBackendClient(
                    model_type,
                    url,
                    reconnect=reconnect,
                    reconnect_interval=reconnect_interval
                )
                client.start()
                self.clients[model_type] = client
                enabled_count += 1
                self.logger.info(f"✅ 已启动模型客户端: {model_type} -> {url}")
            except Exception as e:
                self.logger.error(f"启动模型客户端失败 ({model_type}): {e}")
        
        if enabled_count == 0:
            self.logger.warning("没有启用的模型后端")
            return
        
        # 订阅需要模型推理的事件
        self.bus.subscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
        self.logger.info(f"已订阅事件: {EventTopic.MULTIMODAL_SNAPSHOT.value}")
        
        self._running = True
        self.logger.info(f"✅ 模型代理服务已启动 (共 {enabled_count} 个模型后端)")
    
    def stop(self) -> None:
        """停止所有模型后端客户端"""
        if not self._running:
            return
        
        self.logger.info("停止模型代理服务...")
        
        # 取消订阅
        try:
            self.bus.unsubscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
        except Exception as e:
            self.logger.error(f"取消订阅失败: {e}")
        
        # 停止所有客户端
        for model_type, client in self.clients.items():
            try:
                client.stop()
                self.logger.info(f"已停止模型客户端: {model_type}")
            except Exception as e:
                self.logger.error(f"停止模型客户端失败 ({model_type}): {e}")
        
        self.clients.clear()
        self._running = False
        self.logger.info("✅ 模型代理服务已停止")
    
    def _on_multimodal_data(self, event: Event) -> None:
        """处理多模态数据快照,发送到各模型后端
        
        当 MultimodalService 发布 MULTIMODAL_SNAPSHOT 事件时,
        此方法会被调用,将数据分发到各个模型后端进行推理。
        
        Args:
            event: 多模态数据快照事件
        """
        payload = event.payload or {}
        
        # 提取数据
        rgb_b64 = payload.get("rgb_b64")
        depth_b64 = payload.get("depth_b64")
        timestamp = payload.get("timestamp")
        frame_count = payload.get("frame_count")
        
        if not rgb_b64:
            self.logger.warning("多模态数据缺少RGB图像,跳过推理")
            return
        
        metadata = {
            "timestamp": timestamp,
            "frame_count": frame_count
        }
        
        # 发送到多模态模型后端
        if "multimodal" in self.clients:
            self._send_to_multimodal(rgb_b64, depth_b64, metadata)
        
        # 发送到情绪模型后端
        if "emotion" in self.clients:
            self._send_to_emotion(rgb_b64, metadata)
        
        # TODO: 添加其他模型后端的分发逻辑
    
    def _send_to_multimodal(
        self,
        rgb_b64: str,
        depth_b64: Optional[str],
        metadata: Dict[str, Any]
    ) -> None:
        """发送数据到多模态模型后端"""
        client = self.clients["multimodal"]
        
        if not client.is_healthy():
            self.logger.warning("多模态模型后端未连接,跳过推理")
            return
        
        try:
            data = {
                "rgb_frame": rgb_b64,
                "depth_frame": depth_b64,
                "metadata": metadata
            }
            
            future = client.send_inference_request(data, timeout=5.0)
            
            # 异步处理结果
            def on_result(f):
                try:
                    result = f.result(timeout=5.0)
                    self._publish_inference_result("multimodal", result, metadata)
                except TimeoutError:
                    self.logger.error("多模态推理超时")
                except Exception as e:
                    self.logger.error(f"多模态推理失败: {e}")
            
            future.add_done_callback(on_result)
            
        except Exception as e:
            self.logger.error(f"发送多模态推理请求失败: {e}")
    
    def _send_to_emotion(
        self,
        rgb_b64: str,
        metadata: Dict[str, Any]
    ) -> None:
        """发送数据到情绪模型后端"""
        client = self.clients["emotion"]
        
        if not client.is_healthy():
            self.logger.warning("情绪模型后端未连接,跳过推理")
            return
        
        try:
            data = {
                "rgb_frame": rgb_b64,
                "metadata": metadata
            }
            
            future = client.send_inference_request(data, timeout=5.0)
            
            def on_result(f):
                try:
                    result = f.result(timeout=5.0)
                    self._publish_inference_result("emotion", result, metadata)
                except TimeoutError:
                    self.logger.error("情绪识别超时")
                except Exception as e:
                    self.logger.error(f"情绪识别失败: {e}")
            
            future.add_done_callback(on_result)
            
        except Exception as e:
            self.logger.error(f"发送情绪推理请求失败: {e}")
    
    def _publish_inference_result(
        self,
        model_type: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """将模型推理结果发布到事件总线
        
        Args:
            model_type: 模型类型
            result: 推理结果 (来自模型后端)
            metadata: 元数据
        """
        if result.get("status") != "success":
            error = result.get("error", "Unknown error")
            self.logger.error(f"{model_type} 推理失败: {error}")
            return
        
        predictions = result.get("predictions", {})
        latency_ms = result.get("latency_ms", 0)
        
        # 发布检测结果事件
        self.bus.publish(Event(
            topic=EventTopic.DETECTION_RESULT,
            payload={
                "detector": f"model_{model_type}",
                "status": "detected",
                "label": model_type,
                "predictions": predictions,
                "latency_ms": latency_ms,
                "timestamp": metadata.get("timestamp"),
                "frame_count": metadata.get("frame_count")
            }
        ))
        
        self.logger.debug(
            f"✅ {model_type} 推理完成: "
            f"耗时={latency_ms:.1f}ms, "
            f"结果={list(predictions.keys())}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """获取所有模型后端的状态
        
        Returns:
            状态字典,包含每个模型后端的连接状态
        """
        status = {
            "running": self._running,
            "total_backends": len(self.clients),
            "backends": {}
        }
        
        for model_type, client in self.clients.items():
            status["backends"][model_type] = client.get_status()
        
        return status
    
    def get_healthy_backends(self) -> List[str]:
        """获取健康的模型后端列表
        
        Returns:
            健康的模型类型列表
        """
        return [
            model_type
            for model_type, client in self.clients.items()
            if client.is_healthy()
        ]
