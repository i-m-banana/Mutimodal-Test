"""统一推理服务

注意事项：
1. RGB疲劳度和EEG疲劳度已从实时推理中移除
2. 所有疲劳度评估现在由FatigueAssessmentService在SART测试结束后统一处理
3. 使用保存的视频和EEG数据文件进行离线推理
4. 仅保留EEG脑负荷和情绪识别的实时推理
"""

import importlib
import logging
from typing import Any, Dict, List, Optional

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..core.thread_pool import get_thread_pool
from ..models.base_inference_model import BaseInferenceModel


class UnifiedInferenceService:
    """统一推理服务"""
    
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
        
        # 模型实例
        self.models: Dict[str, BaseInferenceModel] = {}
        
        # 使用统一线程池（CPU密集型任务）
        self._thread_pool = get_thread_pool()
        
        # EEG服务引用（通过EventBus获取，延迟初始化）
        self._eeg_service = None
        
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
            
            enabled_count += self._load_model(model_name, model_type, config)
        
        if enabled_count == 0:
            self.logger.warning("没有启用的模型")
            return
        
        # 订阅需要推理的事件
        self.bus.subscribe(EventTopic.MULTIMODAL_SNAPSHOT, self._on_multimodal_data)
        self.bus.subscribe(EventTopic.EMOTION_REQUEST, self._on_emotion_request)
        self.bus.subscribe(EventTopic.EEG_REQUEST, self._on_eeg_request)
        self.logger.debug(f"已订阅事件: {EventTopic.MULTIMODAL_SNAPSHOT.value}, {EventTopic.EMOTION_REQUEST.value}, {EventTopic.EEG_REQUEST.value}")
        
        self._running = True
        self.logger.info(f"✅ 统一推理服务已启动 (共 {enabled_count} 个模型)")
    
    def _load_model(
        self,
        model_name: str,
        model_type: str,
        config: Dict[str, Any]
    ) -> int:
        """加载模型
        
        Returns:
            1 if success, 0 if failed
        """
        integrated_config = config.get("integrated", {})
        class_path = integrated_config.get("class")
        options = integrated_config.get("options", {})
        
        if not class_path:
            self.logger.error(f"模型缺少 class 配置: {model_name}")
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
            
            self.models[model_type] = model
            self.logger.debug(f"✅ 模型已加载: {model_name} ({model_type})")
            return 1
            
        except Exception as e:
            self.logger.error(f"加载模型失败 ({model_name}): {e}", exc_info=True)
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
        
        self.logger.info("推理任务已停止提交到线程池")
        
        # 卸载模型
        for model_type, model in self.models.items():
            try:
                model.unload()
                self.logger.debug(f"已卸载模型: {model_type}")
            except Exception as e:
                self.logger.error(f"卸载模型失败 ({model_type}): {e}")
        self.models.clear()
        
        self._running = False
        self.logger.info("✅ 统一推理服务已停止")
    
    def _on_multimodal_data(self, event: Event) -> None:
        """处理多模态数据,分发到情绪识别模型
        
        注意：RGB疲劳度推理已移至 FatigueAssessmentService，
        在录制完成后使用文件路径进行离线推理，不在此处实时处理
        """
        payload = event.payload or {}
        
        # 提取数据
        status = payload.get("status", "idle")
        timestamp = payload.get("timestamp")
        frame_count = payload.get("frame_count", 0)
        elapsed_time = payload.get("elapsed_time", 0.0)
        
        # 检查采集状态
        if status != "running":
            return
        
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
            
            # 只取最后30帧用于推理(避免内存累积)
            max_frames_for_inference = 30
            rgb_frames_memory = rgb_frames_memory[-max_frames_for_inference:]
            depth_frames_memory = depth_frames_memory[-max_frames_for_inference:]
            eyetrack_memory = eyetrack_memory[-max_frames_for_inference:]
            
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
        
        metadata = {
            "timestamp": timestamp,
            "frame_count": frame_count
        }
        
        # ===== 注意:RGB疲劳度推理已从此处移除 =====
        # RGB疲劳度现在在 FatigueAssessmentService 中处理
        # 前端录制完成后,发送 FATIGUE_ASSESSMENT_REQUEST 事件
        # 使用保存的视频文件进行离线推理
        
        # ===== 情绪识别推理策略调整 =====
        # 为避免在录制期间每次 snapshot 都触发情绪推理（导致频繁的文件打开错误），
        # 现在仅在采集停止后（status != "running"）触发情绪推理
        # 这样可以确保视频文件已完全写入并关闭，避免 OpenCV 报错
        # 如果需要实时推理，可以在 payload 中添加 "trigger_emotion": True 标志
        
        # 情绪识别: 仅在录制停止后触发（避免录制期间频繁推理）
        if "emotion" in self.models and status != "running":
            self.logger.debug(f"📊 准备触发情绪推理 (status={status}, 采集已停止)")
            
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
            
            self._submit_inference("emotion", inference_data, metadata)
        elif "emotion" in self.models and status == "running":
            # 录制期间跳过情绪推理，避免频繁触发
            self.logger.debug("⏭️  跳过情绪推理 (status=running, 等待录制完成)")
    
    def _on_emotion_request(self, event: Event) -> None:
        """处理情绪分析请求"""
        payload = event.payload or {}
        request_id = payload.get("request_id")
        audio_paths = payload.get("audio_paths", [])
        video_paths = payload.get("video_paths", [])
        # text_data = payload.get("text_data", [])
        
        if not audio_paths and not video_paths:
            self.logger.warning("情绪分析请求缺少音视频数据")
            return
        
        # 分发到情绪模型（V2 架构：不再依赖文本模态）
        if "emotion" in self.models:
            # # 提取文本数据（字段名是 recognized_text）
            # text_list = []
            # for item in text_data:
            #     if isinstance(item, dict):
            #         text = item.get("recognized_text", "")
            #         text_list.append(text)
            
            # # 记录提取的文本数据
            # if text_data:
            #     total_chars = sum(len(t) for t in text_list)
            #     self.logger.info(f"\n{'='*60}")
            #     self.logger.info(f"📝 语音识别文本提取")
            #     self.logger.info(f"{'='*60}")
            #     self.logger.info(f"样本数量: {len(text_list)}")
            #     self.logger.info(f"总字符数: {total_chars}")
            #     self.logger.info(f"-" * 60)
            #     for i, text in enumerate(text_list, 1):
            #         self.logger.info(f"第{i}题: {text}")
            #     self.logger.info(f"{'='*60}\n")
            # else:
            #     self.logger.warning("⚠️  未提取到语音识别文本")
            
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
                # "text_list": text_list[:num_samples]  # 按样本顺序的文本列表
            }
            
            metadata = {
                "request_id": request_id,
                "timestamp": payload.get("timestamp")
            }
            
            self._submit_inference("emotion", inference_data, metadata)
    
    def _on_eeg_request(self, event: Event) -> None:
        """处理EEG脑负荷分析请求（优化：异步获取数据，避免阻塞EventBus）"""
        payload = event.payload or {}
        request_id = payload.get("request_id")
        eeg_signal = payload.get("eeg_signal")  # 可能为None（内存模式）
        sampling_rate = payload.get("sampling_rate", 250)
        subject_id = payload.get("subject_id", "unknown")
        memory_mode = payload.get("memory_mode", True)
        
        # 如果eeg_signal为None，需要从EEGService获取（在线程池中异步执行）
        if eeg_signal is None and memory_mode:
            # 延迟获取EEG服务引用
            if self._eeg_service is None:
                self._eeg_service = getattr(self.bus, '_eeg_service', None)
            
            if self._eeg_service is None:
                self.logger.warning("EEG服务未注册，无法获取数据")
                return
            
            # 异步获取数据并推理（避免阻塞EventBus的publish）
            window_seconds = payload.get("window_seconds", 2.0)
            
            def _fetch_and_infer():
                try:
                    self.logger.debug(f"📥 开始获取EEG数据窗口 ({window_seconds}秒)")
                    
                    # 在线程池中获取数据
                    import numpy as np
                    window_data = self._eeg_service.get_recent_window(
                        seconds=window_seconds, 
                        sample_rate=500.0
                    )
                    
                    ch1_data = window_data.get("ch1", [])
                    ch2_data = window_data.get("ch2", [])
                    
                    self.logger.debug(f"📥 获取到EEG数据: ch1={len(ch1_data)}, ch2={len(ch2_data)}")
                    
                    if len(ch1_data) < 500:
                        self.logger.debug(f"EEG数据不足500样本 (仅{len(ch1_data)})，跳过推理")
                        return  # 静默跳过
                    
                    if len(ch1_data) != len(ch2_data):
                        self.logger.warning(f"EEG通道长度不匹配 {len(ch1_data)}≠{len(ch2_data)}")
                        return
                    
                    # 转换为 [n_samples, 2] 格式（numpy数组，避免大list的内存开销）
                    eeg_signal = np.column_stack([ch1_data, ch2_data])

                    simulation_mode = False
                    try:
                        diagnostics = self._eeg_service.diagnostics()
                        simulation_mode = bool(diagnostics.get("simulation_mode"))
                    except Exception as diag_exc:
                        # 诊断信息获取失败时，保持默认值
                        self.logger.debug(f"无法获取EEG诊断信息: {diag_exc}")
                    
                        # 执行推理 - EEG脑负荷模型
                    if "eeg" in self.models:
                        self.logger.debug(f"🧠 开始EEG脑负荷推理 ({len(ch1_data)}样本)")
                        # 创建数据副本以避免共享引用
                        inference_data = {
                            "memory_mode": memory_mode,
                            "eeg_signal": eeg_signal.copy(),  # 创建副本
                            "sampling_rate": sampling_rate,
                            "subject_id": subject_id,
                            "simulation_mode": simulation_mode,
                        }
                        metadata = {
                            "request_id": request_id,
                            "timestamp": payload.get("timestamp")
                        }
                        # 直接调用推理（已经在线程池中）
                        result = self._infer("eeg", inference_data)
                        if result:
                            self._publish_result("eeg", result, metadata)
                        # 清理推理数据
                        inference_data.clear()
                    
                    # ===== 注意：EEG疲劳度推理已从此处移除 =====
                    # EEG疲劳度现在在 FatigueAssessmentService 中处理
                    # SART测试结束后，使用保存的EEG数据文件进行离线推理
                    # 不再进行实时推理，避免重复计算和资源浪费
                    
                    if "eeg" not in self.models:
                        self.logger.warning("EEG脑负荷模型未加载")
                    
                except Exception as exc:
                    self.logger.error(f"EEG推理失败: {exc}", exc_info=True)
                finally:
                    # 清理EEG信号数据
                    eeg_signal = None
                    ch1_data = None
                    ch2_data = None
                    window_data = None
                    import gc
                    gc.collect()
            
            # 提交到CPU线程池异步执行
            self._thread_pool.submit_cpu_task(_fetch_and_infer)
            return
        
        # 如果已经有eeg_signal，直接推理
        if eeg_signal is None:
            self.logger.warning("EEG分析请求缺少信号数据")
            return
        
        # 转换为numpy数组以便复制
        import numpy as np
        if not isinstance(eeg_signal, np.ndarray):
            eeg_signal = np.array(eeg_signal)
        
        # EEG脑负荷模型
        if "eeg" in self.models:
            simulation_mode = payload.get("simulation_mode")

            if simulation_mode is None and self._eeg_service is None:
                self._eeg_service = getattr(self.bus, '_eeg_service', None)

            if simulation_mode is None and self._eeg_service is not None:
                try:
                    diagnostics = self._eeg_service.diagnostics()
                    simulation_mode = bool(diagnostics.get("simulation_mode"))
                except Exception as diag_exc:
                    self.logger.debug(f"无法获取EEG诊断信息: {diag_exc}")

            # 创建数据副本，避免多个模型共享同一数据引用
            inference_data = {
                "memory_mode": memory_mode,
                "eeg_signal": eeg_signal.copy(),  # 创建副本
                "sampling_rate": sampling_rate,
                "subject_id": subject_id,
            }

            if simulation_mode is not None:
                inference_data["simulation_mode"] = bool(simulation_mode)

            metadata = {
                "request_id": request_id,
                "timestamp": payload.get("timestamp")
            }
            self._submit_inference("eeg", inference_data, metadata)
        
        # EEG疲劳度模型
        if "eeg_fatigue" in self.models:
            # 创建数据副本，避免多个模型共享同一数据引用
            inference_data_fatigue = {
                "memory_mode": memory_mode,
                "eeg_signal": eeg_signal.copy(),  # 创建副本
                "sampling_rate": sampling_rate,
                "subject_id": subject_id,
            }

            metadata_fatigue = {
                "request_id": request_id,
                "timestamp": payload.get("timestamp")
            }
            self._submit_inference("eeg_fatigue", inference_data_fatigue, metadata_fatigue)
    
    def _submit_inference(
        self,
        model_type: str,
        data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """提交推理任务到线程池（异步执行，避免阻塞主线程）
        
        Args:
            model_type: 模型类型 (rgb_fatigue, eeg_fatigue, eeg, emotion等)
            data: 推理数据字典（包含模型输入数据）
            metadata: 元数据字典（用于结果发布）
        
        注意：
            - 所有推理任务在独立线程中执行
            - 推理完成后自动清理数据引用
            - 支持numpy数组和普通数据的混合清理
        """
        def _do_infer():
            try:
                # 推理
                if model_type in self.models:
                    result = self._infer(model_type, data)
                else:
                    self.logger.warning(f"模型未加载: {model_type}")
                    return
                
                # 发布结果
                if result:
                    self._publish_result(model_type, result, metadata)
                    
            except Exception as e:
                self.logger.error(f"推理任务失败 ({model_type}): {e}", exc_info=True)
            finally:
                # 显式清理推理数据,释放内存
                # 清理可能包含numpy数组的数据引用
                for key in list(data.keys()):
                    data[key] = None
                data.clear()
                # 触发垃圾回收
                import gc
                gc.collect()
        
        # 提交到CPU线程池异步执行
        self._thread_pool.submit_cpu_task(_do_infer)
    
    def _infer(
        self,
        model_type: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """执行模型推理"""
        model = self.models[model_type]
        
        try:
            self.logger.debug(f"🔄 开始模型推理: {model_type}")
            result = model.infer(data)
            
            # 输出推理结果关键信息
            if result and result.get("status") == "success":
                predictions = result.get("predictions", result)
                inference_mode = result.get("inference_mode", "unknown")
                inference_time = result.get("inference_time_ms", 0)
                if model_type == "rgb_fatigue":
                    rgb_fatigue_score = predictions.get("rgb_fatigue_score", 0)
                    prediction_class = predictions.get("prediction_class", 0)
                    # 压缩输出: RGB疲劳度单行显示
                    self.logger.debug(f"✅RGB疲劳度 {rgb_fatigue_score:.1f} [C{prediction_class}] {inference_time:.0f}ms")
                elif model_type == "eeg_fatigue":
                    eeg_fatigue_score = predictions.get("eeg_fatigue_score", 0)
                    num_windows = predictions.get("num_windows", 0)
                    # 压缩输出: EEG疲劳度单行显示
                    self.logger.debug(f"✅EEG疲劳度 {eeg_fatigue_score:.1f} {num_windows}窗口 {inference_time:.0f}ms")
                elif model_type == "emotion":
                    emotion_score = predictions.get("emotion_score", 0)
                    inference_time = result.get("inference_time_ms", 0)
                    self.logger.debug(f"✅情绪 {emotion_score:.1f} {inference_time:.0f}ms")
                elif model_type == "eeg":
                    brain_load_score = predictions.get("brain_load_score", 0)
                    state = predictions.get("state", "unknown")
                    num_windows = predictions.get("num_windows", 0)
                    simulation_mode = bool(data.get("simulation_mode"))
                    # 压缩输出: 脑负荷单行显示
                    sim_suffix = "(模拟)" if simulation_mode else ""
                    self.logger.debug(
                        f"✅脑负荷{sim_suffix} {brain_load_score:.1f} [{state[:3]}] {num_windows}win {inference_time:.0f}ms"
                    )
                else:
                    self.logger.debug(f"✅ {model_type} 推理完成")
            else:
                # 非 success 情况：no-data 视为正常缺数据（信息级日志），其他按失败处理
                if result and result.get("status") == "no-data":
                    msg = result.get("error", "no-data")
                    self.logger.debug(f"ℹ️  {model_type} 暂无有效数据，跳过发布: {msg}")
                    return None
                error_msg = result.get("error", "未知错误") if result else "返回结果为空"
                self.logger.debug(f"⚠️  {model_type} 推理失败: {error_msg}")
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
        # 注意：rgb_fatigue 和 eeg_fatigue 已不再在此处实时推理
        # 保留代码结构以兼容旧版本或特殊场景
        predictions = {}
        if model_type == "rgb_fatigue":
            predictions = {
                "rgb_fatigue_score": result.get("rgb_fatigue_score", 0),
                "prediction_class": result.get("prediction_class", 0),
                "inference_time_ms": result.get("inference_time_ms", 0),
                "inference_mode": result.get("inference_mode", "unknown")
            }
        elif model_type == "eeg_fatigue":
            # EEG疲劳度已移至FatigueAssessmentService，不再实时推理
            predictions = {
                "eeg_fatigue_score": result.get("eeg_fatigue_score", 0),
                "num_windows": result.get("num_windows", 0),
                "window_results": result.get("window_results", []),
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
            "models": list(self.models.keys()),
            "total": len(self.models)
        }


__all__ = ["UnifiedInferenceService"]
