"""统一推理服务 - 集成模式

直接调用集成模型进行推理
"""

import importlib
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..models.base_inference_model import BaseInferenceModel


class UnifiedInferenceService:
    """统一推理服务 - 集成模式
    
    模型直接在后端进程中运行
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
            mode = config.get("mode", "integrated")
            
            if mode != "integrated":
                self.logger.warning(f"模型 {model_name} 配置为 {mode} 模式,但仅支持集成模式,跳过")
                continue
            
            # 集成模式:直接加载模型
            enabled_count += self._start_integrated_model(model_name, model_type, config)
        
        if enabled_count == 0:
            self.logger.warning("没有启用的模型")
            return
        
        # 订阅需要推理的事件
        self.bus.subscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
        self.bus.subscribe(EventTopic.EMOTION_REQUEST, self._on_emotion_request)
        self.bus.subscribe(EventTopic.EEG_REQUEST, self._on_eeg_request)
        self.logger.info(f"已订阅事件: {EventTopic.MULTIMODAL_SNAPSHOT.value}, {EventTopic.EMOTION_REQUEST.value}, {EventTopic.EEG_REQUEST.value}")
        
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
    
    def stop(self) -> None:
        """停止推理服务"""
        if not self._running:
            return
        
        self.logger.info("停止统一推理服务...")
        
        # 取消订阅
        try:
            self.bus.unsubscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
            self.bus.unsubscribe(EventTopic.EMOTION_REQUEST, self._on_emotion_request)
            self.bus.unsubscribe(EventTopic.EEG_REQUEST, self._on_eeg_request)
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
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        self._running = False
        self.logger.info("✅ 统一推理服务已停止")
    
    def _on_multimodal_data(self, event: Event) -> None:
        """处理多模态数据,分发到各模型"""
        payload = event.payload or {}
        
        # 提取数据
        status = payload.get("status", "idle")
        timestamp = payload.get("timestamp")
        frame_count = payload.get("frame_count", 0)
        elapsed_time = payload.get("elapsed_time", 0.0)
        
        # 优先使用内存模式(避免重复I/O)
        memory_mode = payload.get("memory_mode", False)
        file_mode = payload.get("file_mode", False)
        
        # 验证数据有效性
        if memory_mode:
            # 内存模式: 直接使用numpy数组
            rgb_frames_memory = payload.get("rgb_frames_memory", [])
            depth_frames_memory = payload.get("depth_frames_memory", [])
            eyetrack_memory = payload.get("eyetrack_memory", [])
            
            if not rgb_frames_memory:
                # 没有RGB帧数据时,静默跳过
                return
            
            # 内存模式不需要文件路径
            rgb_video_path = None
            depth_video_path = None
            eyetrack_json_path = None
            rgb_frames_b64 = []
            depth_frames_b64 = []
            eyetrack_samples = []
            
        elif file_mode:
            # 文件模式:检查文件路径是否存在
            rgb_video_path = payload.get("rgb_video_path")
            depth_video_path = payload.get("depth_video_path")
            eyetrack_json_path = payload.get("eyetrack_json_path")
            
            if not rgb_video_path:
                # 没有RGB视频文件时,静默跳过
                return
                
            # 使用文件路径进行推理
            rgb_frames_memory = []
            depth_frames_memory = []
            eyetrack_memory = []
            rgb_frames_b64 = []
            depth_frames_b64 = []
            eyetrack_samples = []
        else:
            # Base64模式:提取多帧序列数据
            rgb_frames_b64 = payload.get("rgb_frames_b64", [])
            depth_frames_b64 = payload.get("depth_frames_b64", [])
            eyetrack_samples = payload.get("eyetrack_samples", [])
            rgb_video_path = None
            depth_video_path = None
            eyetrack_json_path = None
            rgb_frames_memory = []
            depth_frames_memory = []
            eyetrack_memory = []
            
            if not rgb_frames_b64:
                # 没有RGB帧序列数据时,静默跳过
                return
        
        # 检查采集状态
        if status != "running":
            return
        
        # 验证帧数量是否足够(避免模型推理失败)
        # 疲劳度模型需要足够的帧序列(至少30帧)
        MIN_FRAMES_FOR_FATIGUE = 30
        
        # 检查帧数是否足够
        if memory_mode:
            # 内存模式: 检查数组长度
            if len(rgb_frames_memory) < MIN_FRAMES_FOR_FATIGUE:
                # 数据不足时静默跳过
                return
        elif file_mode:
            # 文件模式: 检查 frame_count 元数据
            if frame_count < MIN_FRAMES_FOR_FATIGUE:
                # 数据不足时静默跳过,避免日志刷屏
                return
        else:
            # Base64模式: 检查数组长度
            if len(rgb_frames_b64) < MIN_FRAMES_FOR_FATIGUE:
                # 数据不足时静默跳过
                return
        
        metadata = {
            "timestamp": timestamp,
            "frame_count": frame_count
        }
        
        # 分发到疲劳度模型(使用完整的多模态数据)
        if "fatigue" in self.integrated_models:
            inference_data = {
                "elapsed_time": elapsed_time
            }
            
            # 根据模式选择数据格式(优先使用内存模式)
            if memory_mode:
                inference_data.update({
                    "memory_mode": True,
                    "rgb_frames_memory": rgb_frames_memory,
                    "depth_frames_memory": depth_frames_memory,
                    "eyetrack_memory": eyetrack_memory,
                })
            elif file_mode:
                inference_data.update({
                    "file_mode": True,
                    "rgb_video_path": rgb_video_path,
                    "depth_video_path": depth_video_path,
                    "eyetrack_json_path": eyetrack_json_path,
                })
            else:
                inference_data.update({
                    "rgb_frames": rgb_frames_b64,
                    "depth_frames": depth_frames_b64,
                    "eyetrack_samples": eyetrack_samples,
                })
            
            self._submit_inference("fatigue", inference_data, metadata)
    
    def _on_emotion_request(self, event: Event) -> None:
        """处理情绪分析请求"""
        payload = event.payload or {}
        request_id = payload.get("request_id")
        audio_paths = payload.get("audio_paths", [])
        video_paths = payload.get("video_paths", [])
        text_data = payload.get("text_data", [])
        
        if not audio_paths and not video_paths:
            self.logger.warning("情绪分析请求缺少音视频数据")
            return
        
        # 分发到情绪模型
        if "emotion" in self.integrated_models:
            # 提取文本数据（字段名是 recognized_text）
            text_list = []
            for item in text_data:
                if isinstance(item, dict):
                    text = item.get("recognized_text", "")
                    text_list.append(text)
            
            # 记录提取的文本数据
            if text_data:
                total_chars = sum(len(t) for t in text_list)
                self.logger.info(f"\n{'='*60}")
                self.logger.info(f"📝 语音识别文本提取")
                self.logger.info(f"{'='*60}")
                self.logger.info(f"样本数量: {len(text_list)}")
                self.logger.info(f"总字符数: {total_chars}")
                self.logger.info(f"-" * 60)
                for i, text in enumerate(text_list, 1):
                    self.logger.info(f"第{i}题: {text}")
                self.logger.info(f"{'='*60}\n")
            else:
                self.logger.warning("⚠️  未提取到语音识别文本")
            
            # 使用多样本模式进行推理
            num_samples = min(len(video_paths), len(audio_paths))
            if num_samples == 0:
                self.logger.warning("没有可用的音视频文件")
                return
            
            # 构建多样本推理数据
            inference_data = {
                "multi_sample_mode": True,  # 新增多样本模式
                "video_paths": video_paths[:num_samples],
                "audio_paths": audio_paths[:num_samples],
                "text_list": text_list[:num_samples]  # 按样本顺序的文本列表
            }
            
            metadata = {
                "request_id": request_id,
                "timestamp": payload.get("timestamp")
            }
            
            self._submit_inference("emotion", inference_data, metadata)
    
    def _on_eeg_request(self, event: Event) -> None:
        """处理EEG脑负荷分析请求"""
        payload = event.payload or {}
        request_id = payload.get("request_id")
        eeg_signal = payload.get("eeg_signal")
        sampling_rate = payload.get("sampling_rate", 250)
        subject_id = payload.get("subject_id", "unknown")
        memory_mode = payload.get("memory_mode", True)
        
        if eeg_signal is None:
            self.logger.warning("EEG分析请求缺少信号数据")
            return
        
        # 分发到EEG模型
        if "eeg" in self.integrated_models:
            # 构建推理数据
            inference_data = {
                "memory_mode": memory_mode,
                "eeg_signal": eeg_signal,
                "sampling_rate": sampling_rate,
                "subject_id": subject_id
            }
            
            metadata = {
                "request_id": request_id,
                "timestamp": payload.get("timestamp")
            }
            
            self._submit_inference("eeg", inference_data, metadata)
    
    def _submit_inference(
        self,
        model_type: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """提交推理任务（异步）"""
        def _infer():
            try:
                # 集成模式推理
                if model_type in self.integrated_models:
                    result = self._infer_integrated(model_type, data)
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
                inference_mode = result.get("inference_mode", "unknown")
                inference_time = result.get("inference_time_ms", 0)
                if model_type == "fatigue":
                    fatigue_score = predictions.get("fatigue_score", 0)
                    prediction_class = predictions.get("prediction_class", 0)
                    # 压缩输出: 疲劳度单行显示
                    self.logger.info(f"✅疲劳度 {fatigue_score:.1f} [C{prediction_class}] {inference_time:.0f}ms")
                elif model_type == "emotion":
                    emotion_score = predictions.get("emotion_score", 0)
                    inference_time = result.get("inference_time_ms", 0)
                    self.logger.info(f"✅情绪 {emotion_score:.1f} {inference_time:.0f}ms")
                elif model_type == "eeg":
                    brain_load_score = predictions.get("brain_load_score", 0)
                    state = predictions.get("state", "unknown")
                    num_windows = predictions.get("num_windows", 0)
                    # 压缩输出: 脑负荷单行显示
                    self.logger.info(f"✅脑负荷 {brain_load_score:.1f} [{state[:3]}] {num_windows}win {inference_time:.0f}ms")
                else:
                    self.logger.info(f"✅ {model_type} 推理完成")
            else:
                # 非 success 情况：no-data 视为正常缺数据（信息级日志），其他按失败处理
                if result and result.get("status") == "no-data":
                    msg = result.get("error", "no-data")
                    self.logger.info(f"ℹ️  {model_type} 暂无有效数据，跳过发布: {msg}")
                    return None
                error_msg = result.get("error", "未知错误") if result else "返回结果为空"
                self.logger.warning(f"⚠️  {model_type} 推理失败: {error_msg}")
                # 推理失败时返回None，不发布结果
                return None
            
            return result
        except Exception as e:
            # 捕获所有异常，避免影响系统运行
            error_msg = str(e)
            self.logger.error(f"集成模型推理异常 ({model_type}): {error_msg}")
            
            # 对于特定的错误，提供更友好的提示
            if "Invalid computed output size" in error_msg:
                self.logger.debug(
                    f"提示: {model_type}模型输入数据不足，"
                    "可能是因为摄像头未开启或采集帧数过少"
                )
            
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
                "prediction_class": result.get("prediction_class", 0),
                "inference_time_ms": result.get("inference_time_ms", 0),
                "inference_mode": result.get("inference_mode", "unknown")
            }
        elif model_type == "emotion":
            predictions = {
                "emotion_score": result.get("emotion_score", 0),
                "prediction": result.get("prediction", 0),
                "probabilities": result.get("probabilities", []),
                "inference_time_ms": result.get("inference_time_ms", 0),
                "inference_mode": result.get("inference_mode", "file")
            }
        elif model_type == "eeg":
            predictions = {
                "brain_load_score": result.get("brain_load_score", 0),
                "state": result.get("state", "unknown"),
                "num_windows": result.get("num_windows", 0),
                "inference_time_ms": result.get("inference_time_ms", 0),
                "inference_mode": result.get("inference_mode", "memory")
            }
        
        # 发布到事件总线
        self.bus.publish(Event(
            topic=EventTopic.DETECTION_RESULT,
            payload={
                "detector": f"model_{model_type}",
                "status": "detected",
                "label": model_type,
                "predictions": predictions,
                "request_id": metadata.get("request_id"),  # 用于情绪分析请求匹配
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
            "total": len(self.integrated_models)
        }


__all__ = ["UnifiedInferenceService"]
