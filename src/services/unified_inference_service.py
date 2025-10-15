"""统一推理服务 - 支持集成模式和远程代理模式

根据配置决定是直接调用集成模型，还是通过WebSocket代理到远程进程
"""

import importlib
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import Future, ThreadPoolExecutor

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..models.base_inference_model import BaseInferenceModel

try:
    from ..interfaces.model_ws_client import ModelBackendClient
    HAS_CLIENT = True
except ImportError:
    HAS_CLIENT = False


class UnifiedInferenceService:
    """统一推理服务
    
    支持两种推理模式：
    1. integrated: 模型直接集成在进程中
    2. remote: 通过WebSocket代理到独立进程
    
    可以在配置文件中灵活切换模式，无需修改代码
    """
    
    def __init__(
        self,
        bus: EventBus,
        model_configs: List[Dict[str, Any]],
        *,
        logger: Optional[logging.Logger] = None
    ):
        """初始化统一推理服务
        
        Args:
            bus: 事件总线
            model_configs: 模型配置列表
            logger: 日志记录器
        """
        self.bus = bus
        self.model_configs = model_configs
        self.logger = logger or logging.getLogger("service.inference")
        
        # 集成模式的模型实例
        self.integrated_models: Dict[str, BaseInferenceModel] = {}
        
        # 远程模式的客户端实例
        self.remote_clients: Dict[str, "ModelBackendClient"] = {}
        
        # 线程池用于异步处理
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inference")
        
        self._running = False
    
    def start(self) -> None:
        """启动推理服务"""
        if self._running:
            self.logger.warning("推理服务已在运行")
            return
        
        self.logger.info("启动统一推理服务...")
        
        enabled_count = 0
        for config in self.model_configs:
            if not config.get("enabled", True):
                self.logger.info(f"跳过禁用的模型: {config.get('name')}")
                continue
            
            model_name = config["name"]
            model_type = config["type"]
            mode = config.get("mode", "remote")
            
            if mode == "integrated":
                # 集成模式：直接加载模型
                enabled_count += self._start_integrated_model(model_name, model_type, config)
            elif mode == "remote":
                # 远程模式：连接到模型后端
                enabled_count += self._start_remote_client(model_name, model_type, config)
            else:
                self.logger.error(f"未知的模型模式: {mode} (模型: {model_name})")
        
        if enabled_count == 0:
            self.logger.warning("没有启用的模型")
            return
        
        # 订阅需要推理的事件
        self.bus.subscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
        self.logger.info(f"已订阅事件: {EventTopic.MULTIMODAL_SNAPSHOT.value}")
        
        self._running = True
        self.logger.info(f"✅ 统一推理服务已启动 (共 {enabled_count} 个模型)")
    
    def _start_integrated_model(
        self,
        model_name: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> int:
        """启动集成模式的模型
        
        Returns:
            1 if success, 0 if failed
        """
        integrated_config = config.get("integrated", {})
        class_path = integrated_config.get("class")
        options = integrated_config.get("options", {})
        
        if not class_path:
            self.logger.error(f"集成模型缺少 class 配置: {model_name}")
            return 0
        
        try:
            # 动态导入模型类
            module_name, _, class_name = class_path.rpartition(".")
            if not module_name.startswith("src."):
                module_name = f"src.{module_name}"
            
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            
            # 实例化并加载模型
            model = model_class(model_name, logger=self.logger, **options)
            model.load()
            
            self.integrated_models[model_type] = model
            self.logger.info(f"✅ 集成模型已加载: {model_name} ({model_type})")
            return 1
            
        except Exception as e:
            self.logger.error(f"加载集成模型失败 ({model_name}): {e}", exc_info=True)
            return 0
    
    def _start_remote_client(
        self,
        model_name: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> int:
        """启动远程模式的客户端
        
        Returns:
            1 if success, 0 if failed
        """
        if not HAS_CLIENT:
            self.logger.error(f"WebSocket客户端不可用，无法启动远程模型: {model_name}")
            return 0
        
        remote_config = config.get("remote", {})
        host = remote_config.get("host", "127.0.0.1")
        port = remote_config.get("port", 8766)
        url = f"ws://{host}:{port}"
        
        reconnect = remote_config.get("reconnect", True)
        reconnect_interval = remote_config.get("reconnect_interval", 5.0)
        
        try:
            client = ModelBackendClient(
                model_type,
                url,
                reconnect=reconnect,
                reconnect_interval=reconnect_interval
            )
            client.start()
            
            self.remote_clients[model_type] = client
            self.logger.info(f"✅ 远程模型客户端已启动: {model_name} ({model_type}) -> {url}")
            return 1
            
        except Exception as e:
            self.logger.error(f"启动远程模型客户端失败 ({model_name}): {e}", exc_info=True)
            return 0
    
    def stop(self) -> None:
        """停止推理服务"""
        if not self._running:
            return
        
        self.logger.info("停止统一推理服务...")
        
        # 取消订阅
        try:
            self.bus.unsubscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
        except Exception as e:
            self.logger.error(f"取消订阅失败: {e}")
        
        # 卸载集成模型
        for model_type, model in self.integrated_models.items():
            try:
                model.unload()
                self.logger.info(f"已卸载集成模型: {model_type}")
            except Exception as e:
                self.logger.error(f"卸载集成模型失败 ({model_type}): {e}")
        self.integrated_models.clear()
        
        # 停止远程客户端
        for model_type, client in self.remote_clients.items():
            try:
                client.stop()
                self.logger.info(f"已停止远程客户端: {model_type}")
            except Exception as e:
                self.logger.error(f"停止远程客户端失败 ({model_type}): {e}")
        self.remote_clients.clear()
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        self._running = False
        self.logger.info("✅ 统一推理服务已停止")
    
    def _on_multimodal_data(self, event: Event) -> None:
        """处理多模态数据，分发到各模型"""
        payload = event.payload or {}
        
        # 提取数据
        rgb_b64 = payload.get("rgb_b64")
        depth_b64 = payload.get("depth_b64")
        timestamp = payload.get("timestamp")
        frame_count = payload.get("frame_count")
        
        if not rgb_b64:
            self.logger.warning("多模态数据缺少RGB图像")
            return
        
        metadata = {
            "timestamp": timestamp,
            "frame_count": frame_count
        }
        
        # 分发到疲劳度模型
        if "fatigue" in self.integrated_models or "fatigue" in self.remote_clients:
            self._submit_inference("fatigue", {
                "rgb_frames": [rgb_b64],
                "depth_frames": [depth_b64] if depth_b64 else [],
                "eyetrack_samples": [],
                "elapsed_time": 5.0
            }, metadata)
        
        # 分发到情绪模型
        # TODO: 需要完整的样本数据（视频、音频、文本）
        
        # TODO: 其他模型
    
    def _submit_inference(
        self,
        model_type: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """提交推理任务（异步）"""
        def _infer():
            try:
                # 选择推理方式
                if model_type in self.integrated_models:
                    result = self._infer_integrated(model_type, data)
                elif model_type in self.remote_clients:
                    result = self._infer_remote(model_type, data)
                else:
                    self.logger.warning(f"模型未加载: {model_type}")
                    return
                
                # 发布结果
                if result:
                    self._publish_result(model_type, result, metadata)
                    
            except Exception as e:
                self.logger.error(f"推理任务失败 ({model_type}): {e}", exc_info=True)
        
        self._executor.submit(_infer)
    
    def _infer_integrated(
        self,
        model_type: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """使用集成模型推理"""
        model = self.integrated_models[model_type]
        
        try:
            self.logger.info(f"🔄 开始集成模型推理: {model_type}")
            result = model.infer(data)
            
            # 输出推理结果关键信息
            if result and result.get("status") == "success":
                predictions = result.get("predictions", result)
                if model_type == "fatigue":
                    fatigue_score = predictions.get("fatigue_score", 0)
                    prediction_class = predictions.get("prediction_class", 0)
                    self.logger.info(f"✅ 疲劳度推理完成: score={fatigue_score:.2f}, class={prediction_class}")
                elif model_type == "emotion":
                    emotion_score = predictions.get("emotion_score", 0)
                    self.logger.info(f"✅ 情绪推理完成: score={emotion_score:.2f}")
                else:
                    self.logger.info(f"✅ {model_type} 推理完成: {predictions}")
            else:
                self.logger.warning(f"⚠️  {model_type} 推理返回异常: {result}")
            
            return result
        except Exception as e:
            self.logger.error(f"集成模型推理失败 ({model_type}): {e}", exc_info=True)
            return None
    
    def _infer_remote(
        self,
        model_type: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """使用远程客户端推理"""
        client = self.remote_clients[model_type]
        
        if not client.is_healthy():
            self.logger.warning(f"远程模型未连接: {model_type}")
            return None
        
        try:
            self.logger.info(f"🌐 发送远程推理请求: {model_type}")
            future = client.send_inference_request(data, timeout=10.0)
            result = future.result(timeout=10.0)
            
            # 输出推理结果关键信息
            if result and result.get("status") == "success":
                predictions = result.get("predictions", result)
                if model_type == "fatigue":
                    fatigue_score = predictions.get("fatigue_score", 0)
                    self.logger.info(f"✅ 远程疲劳度推理完成: score={fatigue_score:.2f}")
                elif model_type == "emotion":
                    emotion_score = predictions.get("emotion_score", 0)
                    self.logger.info(f"✅ 远程情绪推理完成: score={emotion_score:.2f}")
                else:
                    self.logger.info(f"✅ 远程 {model_type} 推理完成")
            else:
                self.logger.warning(f"⚠️  远程 {model_type} 推理返回异常")
            
            return result
        except Exception as e:
            self.logger.error(f"远程模型推理失败 ({model_type}): {e}", exc_info=True)
            return None
    
    def _publish_result(
        self,
        model_type: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """发布推理结果到事件总线"""
        if result.get("status") != "success":
            error = result.get("error", "Unknown error")
            self.logger.error(f"{model_type} 推理失败: {error}")
            return
        
        # 提取预测结果（根据模型类型）
        predictions = {}
        if model_type == "fatigue":
            predictions = {
                "fatigue_score": result.get("fatigue_score", 0),
                "prediction_class": result.get("prediction_class", 0)
            }
        elif model_type == "emotion":
            predictions = {
                "emotion_score": result.get("emotion_score", 0)
            }
        
        # 发布到事件总线
        self.bus.publish(Event(
            topic=EventTopic.DETECTION_RESULT,
            payload={
                "detector": f"model_{model_type}",
                "status": "detected",
                "label": model_type,
                "predictions": predictions,
                "timestamp": metadata.get("timestamp"),
                "frame_count": metadata.get("frame_count")
            }
        ))
        
        self.logger.debug(f"✅ {model_type} 推理完成: {predictions}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        return {
            "running": self._running,
            "integrated_models": list(self.integrated_models.keys()),
            "remote_clients": list(self.remote_clients.keys()),
            "total": len(self.integrated_models) + len(self.remote_clients)
        }


__all__ = ["UnifiedInferenceService"]
