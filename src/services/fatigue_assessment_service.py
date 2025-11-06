"""
疲劳度评估服务

提供统一的疲劳度评估接口，融合EEG和RGB两种模态的疲劳度检测结果

功能：
1. 接收录制完成的会话目录路径
2. 使用RGB视频和EEG数据进行疲劳度推理
3. 加权融合两个模型的结果（EEG权重更大）
4. 返回综合疲劳度分数给前端

优化：
- 使用统一线程池管理，避免阻塞事件总线
- RGB视频推理在独立线程中异步执行（30-60秒高耗时任务）
- 支持并发处理多个评估请求（最多2个同时进行）
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..core.thread_pool import get_thread_pool


class FatigueAssessmentService:
    """疲劳度评估服务
    
    集成RGB疲劳度和EEG疲劳度模型，提供统一的评估接口
    """
    
    # 默认权重配置
    DEFAULT_EEG_WEIGHT = 0.7  # EEG疲劳度权重（较大）
    DEFAULT_RGB_WEIGHT = 0.3  # RGB疲劳度权重（较小）
    
    def __init__(
        self,
        bus: EventBus,
        *,
        eeg_weight: float = DEFAULT_EEG_WEIGHT,
        rgb_weight: float = DEFAULT_RGB_WEIGHT,
        logger: Optional[logging.Logger] = None
    ):
        """初始化疲劳度评估服务
        
        Args:
            bus: 事件总线
            eeg_weight: EEG疲劳度权重（默认0.7）
            rgb_weight: RGB疲劳度权重（默认0.3）
            logger: 日志记录器
        """
        self.bus = bus
        self.eeg_weight = eeg_weight
        self.rgb_weight = rgb_weight
        self.logger = logger or logging.getLogger("service.fatigue_assessment")
        
        # 模型实例（延迟加载）
        self._rgb_model = None
        self._eeg_model = None
        
        # 使用统一线程池管理
        self._thread_pool = get_thread_pool()
        
        self._running = False
    
    def start(self) -> None:
        """启动服务"""
        if self._running:
            self.logger.warning("疲劳度评估服务已在运行")
            return
        
        self.logger.info("启动疲劳度评估服务...")
        self.logger.info(f"权重配置: EEG={self.eeg_weight}, RGB={self.rgb_weight}")
        
        # 订阅疲劳度评估请求事件
        self.bus.subscribe(EventTopic.FATIGUE_ASSESSMENT_REQUEST, self._on_assessment_request)
        
        self._running = True
        self.logger.info("✅ 疲劳度评估服务已启动")
    
    def stop(self) -> None:
        """停止服务"""
        if not self._running:
            return
        
        self.logger.info("停止疲劳度评估服务...")
        
        # 取消订阅
        try:
            self.bus.unsubscribe(EventTopic.FATIGUE_ASSESSMENT_REQUEST, self._on_assessment_request)
        except Exception as e:
            self.logger.error(f"取消订阅失败: {e}")
        
        # 卸载模型
        if self._rgb_model:
            try:
                self._rgb_model.unload()
                self._rgb_model = None
            except Exception as e:
                self.logger.error(f"卸载RGB模型失败: {e}")
        
        if self._eeg_model:
            try:
                self._eeg_model.unload()
                self._eeg_model = None
            except Exception as e:
                self.logger.error(f"卸载EEG模型失败: {e}")
        
        self._running = False
        self.logger.info("✅ 疲劳度评估服务已停止")
    
    def _load_models(self) -> None:
        """延迟加载模型"""
        if self._rgb_model is None:
            try:
                from ..models.rgb_fatigue_model import RGBFatigueModel
                self._rgb_model = RGBFatigueModel("rgb_fatigue", logger=self.logger)
                self._rgb_model.load()
                self.logger.info("✓ RGB疲劳度模型加载完成")
            except Exception as e:
                self.logger.error(f"加载RGB疲劳度模型失败: {e}")
        
        if self._eeg_model is None:
            try:
                from ..models.eeg_fatigue_model import EEGFatigueModel
                self._eeg_model = EEGFatigueModel("eeg_fatigue", logger=self.logger)
                self._eeg_model.load()
                self.logger.info("✓ EEG疲劳度模型加载完成")
            except Exception as e:
                self.logger.error(f"加载EEG疲劳度模型失败: {e}")
    
    def _on_assessment_request(self, event: Event) -> None:
        """处理疲劳度评估请求（快速验证后提交到线程池）
        
        事件payload格式：
        {
            "request_id": str,  # 请求ID
            "session_dir": str,  # 录制会话目录路径
            "subject_id": str,  # 被试ID
            "timestamp": float  # 请求时间戳
        }
        
        优化：
        - 快速验证参数后立即返回，不阻塞事件总线
        - 实际推理任务提交到专用推理线程池
        - 支持并发处理多个评估请求
        """
        payload = event.payload or {}
        request_id = payload.get("request_id")
        session_dir = payload.get("session_dir")
        subject_id = payload.get("subject_id", "unknown")
        
        # 快速验证参数
        if not session_dir:
            self.logger.error("疲劳度评估请求缺少会话目录")
            self._publish_error(request_id, "缺少会话目录参数")
            return
        
        session_path = Path(session_dir)
        if not session_path.exists():
            self.logger.error(f"会话目录不存在: {session_dir}")
            self._publish_error(request_id, f"会话目录不存在: {session_dir}")
            return
        
        # 提交到推理线程池异步执行（不阻塞事件总线）
        self.logger.info(f"📤 提交疲劳度评估任务到线程池: {request_id}")
        future = self._thread_pool.submit_inference_task(
            self._do_assessment,
            request_id,
            session_dir,
            subject_id
        )
        
        # 添加异常处理回调
        def _on_done(f):
            try:
                f.result()  # 触发异常（如果有）
            except Exception as e:
                self.logger.error(f"❌ 疲劳度评估任务异常: {request_id}, 错误: {e}", exc_info=True)
                self._publish_error(request_id, f"评估任务执行失败: {str(e)}")
        
        future.add_done_callback(_on_done)
    
    def _do_assessment(
        self,
        request_id: str,
        session_dir: str,
        subject_id: str
    ) -> None:
        """在独立线程中执行完整的疲劳度评估流程
        
        Args:
            request_id: 请求ID
            session_dir: 会话目录路径
            subject_id: 被试ID
        
        注意：此方法在推理线程池中执行，耗时30-60秒
        """
        session_path = Path(session_dir)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"📊 [线程池] 开始疲劳度评估")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"请求ID: {request_id}")
        self.logger.info(f"会话目录: {session_dir}")
        self.logger.info(f"被试ID: {subject_id}")
        
        try:
            # 确保模型已加载
            self._load_models()
            
            # 查找数据文件
            files = self._find_session_files(session_path)
            
            # 执行RGB疲劳度推理（耗时操作）
            rgb_result = self._assess_rgb_fatigue(files, session_path)
            
            # 执行EEG疲劳度推理（传入session_path以支持基线更新）
            eeg_result = self._assess_eeg_fatigue(files, subject_id, session_dir=session_path)
            
            # 融合结果
            final_result = self._compute_weighted_score(rgb_result, eeg_result)
            final_result["request_id"] = request_id
            final_result["subject_id"] = subject_id
            final_result["session_dir"] = session_dir
            
            # 添加基线更新信息到最终结果
            if "baseline_updated" in eeg_result:
                final_result["baseline_updated"] = eeg_result["baseline_updated"]
                final_result["baseline_reason"] = eeg_result.get("baseline_reason", "")
            
            # 发布结果
            self._publish_result(final_result)
            
        except Exception as e:
            self.logger.error(f"❌ 疲劳度评估执行失败: {e}", exc_info=True)
            self._publish_error(request_id, f"评估执行失败: {str(e)}")
    
    def _find_session_files(self, session_dir: Path) -> Dict[str, Optional[Path]]:
        """查找会话目录中的数据文件
        
        注意：新版RGB疲劳模型(MediaPipe)只需要RGB视频，不再需要深度视频和眼动数据
        """
        files = {
            "rgb_video": None,
            "eeg_csv": None,
        }
        
        # RGB视频 - 新模型只需要RGB视频
        for pattern in ["rgb_video.mp4", "rgb.mp4", "color.mp4", "rgb0.avi"]:
            path = session_dir / pattern
            if path.exists():
                files["rgb_video"] = path
                self.logger.info(f"  ✓ RGB视频: {path.name}")
                break
        
        # 也在fatigue子目录中查找
        if not files["rgb_video"]:
            fatigue_dir = session_dir / "fatigue"
            if fatigue_dir.exists():
                for pattern in ["rgb0.avi", "rgb_video.mp4", "rgb.mp4"]:
                    path = fatigue_dir / pattern
                    if path.exists():
                        files["rgb_video"] = path
                        self.logger.info(f"  ✓ RGB视频: fatigue/{path.name}")
                        break
        
        # EEG数据
        eeg_dir = session_dir / "eeg"
        if eeg_dir.exists():
            eeg_csvs = [f for f in eeg_dir.glob("*.csv") if "kss" not in f.name.lower()]
            if eeg_csvs:
                files["eeg_csv"] = sorted(eeg_csvs)[-1]
                self.logger.info(f"  ✓ EEG数据: eeg/{files['eeg_csv'].name}")
        
        return files
    
    def _assess_rgb_fatigue(
        self,
        files: Dict[str, Optional[Path]],
        session_dir: Path
    ) -> Dict[str, Any]:
        """评估RGB疲劳度 (新版本使用MediaPipe，只需要RGB视频)"""
        if not files["rgb_video"]:
            self.logger.warning("跳过RGB疲劳度评估：缺少RGB视频文件")
            return {"status": "skipped", "rgb_fatigue_score": 0.0}
        
        if not self._rgb_model:
            self.logger.error("RGB疲劳度模型未加载")
            return {"status": "error", "rgb_fatigue_score": 0.0}
        
        try:
            self.logger.info("\n📷 RGB疲劳度推理 (MediaPipe方法)...")
            
            inference_data = {
                "file_mode": True,
                "rgb_video_path": str(files["rgb_video"]),
                "start_sec": 0.0,
                "duration_sec": None  # 处理整个视频
            }
            
            start_time = time.time()
            result = self._rgb_model.infer(inference_data)
            inference_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"  RGB疲劳度: {result.get('rgb_fatigue_score', 0):.2f}")
            self.logger.info(f"  推理耗时: {inference_time:.1f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"RGB疲劳度推理失败: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "rgb_fatigue_score": 0.0}
    
    def _assess_eeg_fatigue(
        self,
        files: Dict[str, Optional[Path]],
        subject_id: str,
        session_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """评估EEG疲劳度
        
        Args:
            files: 数据文件字典
            subject_id: 被试ID（如 "shh0", "zyp1"）
            session_dir: 会话目录（用于基线更新）
        """
        if not files["eeg_csv"]:
            self.logger.warning("跳过EEG疲劳度评估：缺少EEG数据文件")
            return {"status": "skipped", "eeg_fatigue_score": 0.0}
        
        if not self._eeg_model:
            self.logger.error("EEG疲劳度模型未加载")
            return {"status": "error", "eeg_fatigue_score": 0.0}
        
        try:
            self.logger.info("\n🧠 EEG疲劳度推理...")
            
            # 提取被试基础标识（去掉末尾数字）
            subject_base = ''.join(c for c in subject_id if not c.isdigit())
            
            # 优先使用会话模式（支持基线更新）
            if session_dir and session_dir.exists():
                inference_data = {
                    "session_dir": str(session_dir),
                    "subject_base": subject_base,
                    "qc_is_lowload": True,  # 假设低负荷状态
                    "update_baseline": True  # 启用基线更新
                }
                
                start_time = time.time()
                result = self._eeg_model.infer(inference_data)
                print(result,"---------------------------inferfatigueeeg---------------")
                inference_time = (time.time() - start_time) * 1000
                
                self.logger.info(f"  EEG疲劳度: {result.get('eeg_fatigue_score', 0):.2f}")
                self.logger.info(f"  窗口数量: {result.get('num_windows', 0)}")
                
                # 显示基线更新信息
                if result.get('baseline_updated'):
                    self.logger.info(f"  ✅ 基线已更新: {result.get('baseline_reason', '')}")
                elif 'baseline_reason' in result:
                    self.logger.info(f"  ℹ️ 基线未更新: {result.get('baseline_reason', '')}")
                
                self.logger.info(f"  推理耗时: {inference_time:.1f}ms")
                
            else:
                # 回退到文件模式（不更新基线）
                inference_data = {
                    "file_mode": True,
                    "eeg_file_path": str(files["eeg_csv"]),
                    "sampling_rate": 500.0,
                    "subject_id": subject_id
                }
                
                start_time = time.time()
                result = self._eeg_model.infer(inference_data)
                inference_time = (time.time() - start_time) * 1000
                
                self.logger.info(f"  EEG疲劳度: {result.get('eeg_fatigue_score', 0):.2f}")
                self.logger.info(f"  窗口数量: {result.get('num_windows', 0)}")
                self.logger.info(f"  推理耗时: {inference_time:.1f}ms")
            
            return result
            
        except Exception as e:
            self.logger.error(f"EEG疲劳度推理失败: {e}", exc_info=True)
            return {"status": "error", "error": str(e), "eeg_fatigue_score": 0.0}
    
    def _compute_weighted_score(
        self,
        rgb_result: Dict[str, Any],
        eeg_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """计算智能融合疲劳度分数
        
        改进融合策略：
        1. 基础加权融合 (EEG 70% + RGB 30%)
        2. 一致性检测 - 两个模型结果是否一致
        3. 冲突处理 - EEG敏感度更高，优先采信EEG
        4. 动态阈值调整 - 考虑数据质量
        """
        self.logger.info(f"\n🔀 智能融合疲劳度分数")
        
        # 提取分数
        rgb_score = rgb_result.get('rgb_fatigue_score', 0.0) if rgb_result.get('status') == 'success' else 0.0
        eeg_score = eeg_result.get('eeg_fatigue_score', 0.0) if eeg_result.get('status') == 'success' else 0.0
        
        # 检查有效性
        rgb_valid = rgb_result.get('status') == 'success'
        eeg_valid = eeg_result.get('status') == 'success'
        
        # 初始化融合参数
        final_score = 0.0
        used_eeg_weight = 0.0
        used_rgb_weight = 0.0
        fusion_method = "unknown"
        confidence = 0.0
        
        # === 情况1: 两个模型都失败 ===
        if not rgb_valid and not eeg_valid:
            final_score = 0.0
            status = "error"
            message = "RGB和EEG推理都失败"
            confidence = 0.0
            fusion_method = "none"
        
        # === 情况2: 仅EEG有效 ===
        elif not rgb_valid:
            # EEG是0-100范围，需要转换到50-90范围
            final_score = 10 + (1 - (eeg_score / 100)) * 90  # 0-100 -> 50-90
            used_eeg_weight = 1.0
            status = "partial"
            message = f"仅使用EEG疲劳度 (原始={eeg_score:.1f}/100, 转换={final_score:.1f}/90)"
            confidence = 0.7  # EEG单独置信度
            fusion_method = "eeg_only"
        
        # === 情况3: 仅RGB有效 ===
        elif not eeg_valid:
            # RGB已经是50-90范围，直接使用
            final_score = rgb_score
            used_rgb_weight = 1.0
            status = "partial"
            message = "仅使用RGB疲劳度"
            confidence = 0.5  # RGB单独置信度较低
            fusion_method = "rgb_only"
        
        # === 情况4: 两个模型都有效 - 智能融合 ===
        else:
            # 注意：EEG是0-100范围，RGB已反转为50-90范围（分数越高越疲劳）
            # 将EEG也转换到50-90范围，然后直接融合
            eeg_score_50_90 = 50 + (1 - (eeg_score / 100)) * 40  # 0-100 -> 50-90
            
            # 1. 一致性检测（基于50-90范围的差异）
            score_diff = abs(eeg_score_50_90 - rgb_score)
            
            # 定义一致性等级（基于50-90范围的差异）
            if score_diff < 5:
                consistency = "high"  # 高度一致
                consistency_factor = 1.0
                confidence = 0.95
            elif score_diff < 10:
                consistency = "medium"  # 中度一致
                consistency_factor = 0.85
                confidence = 0.80
            else:
                consistency = "low"  # 低一致性（冲突）
                consistency_factor = 0.7
                confidence = 0.65
            
            # 2. 冲突处理 - EEG更敏感，在高疲劳时优先采信EEG
            if consistency == "low":
                # 冲突情况下的策略
                if eeg_score_50_90 > 70:  # EEG检测到重度疲劳（>74/90，即>60/100）
                    # 更信任EEG，提高EEG权重
                    adjusted_eeg_weight = 0.85
                    adjusted_rgb_weight = 0.15
                    final_score = adjusted_eeg_weight * eeg_score_50_90 + adjusted_rgb_weight * rgb_score
                    fusion_method = "eeg_priority"
                    message = f"检测到冲突(EEG={eeg_score_50_90:.1f}/90重度疲劳, RGB={rgb_score:.1f}/90正常)，优先采信EEG"
                    self.logger.warning(f"⚠️ 模型结果冲突: EEG={eeg_score:.1f}/100→{eeg_score_50_90:.1f}/90重度疲劳, RGB={rgb_score:.1f}/90正常")
                    self.logger.info(f"  → 采用EEG优先策略 (权重: EEG 85%, RGB 15%)")
                    used_eeg_weight = 0.85
                    used_rgb_weight = 0.15
                else:
                    # 低疲劳冲突，使用标准权重但降低置信度
                    final_score = self.eeg_weight * eeg_score_50_90 + self.rgb_weight * rgb_score
                    fusion_method = "weighted_low_confidence"
                    message = f"检测到轻微冲突(EEG={eeg_score_50_90:.1f}/90, RGB={rgb_score:.1f}/90)"
                    used_eeg_weight = self.eeg_weight
                    used_rgb_weight = self.rgb_weight
            else:
                # 一致性好，使用标准加权
                final_score = self.eeg_weight * eeg_score_50_90 + self.rgb_weight * rgb_score
                fusion_method = f"weighted_{consistency}_consistency"
                message = f"成功融合EEG和RGB疲劳度 (一致性: {consistency}, 差异={score_diff:.1f}/90)"
                used_eeg_weight = self.eeg_weight
                used_rgb_weight = self.rgb_weight
            
            status = "success"
        
        # === 评估疲劳等级（基于50-90分数范围，分数越高越疲劳）===
        # 根据融合方法和置信度调整阈值
        if fusion_method == "eeg_only":
            # EEG主导时（0-100范围），需要先转换到50-90
            # EEG: <30为正常, 30-60为轻度, >60为重度
            # 映射到50-90: 0->50, 100->90
            # <30/100 -> <62, 30-60/100 -> 62-74, >60/100 -> >74
            threshold_low = 62
            threshold_high = 74
        elif fusion_method == "rgb_only":
            # RGB主导时（已经是50-90范围）
            threshold_low = 65
            threshold_high = 77.5
        elif fusion_method == "eeg_priority":
            # EEG优先时，使用稍低阈值（更敏感）
            threshold_low = 63
            threshold_high = 75
        else:
            # 标准阈值（融合后的50-90范围）
            threshold_low = 65
            threshold_high = 77.5
        
        if final_score < threshold_low:
            level = "正常"
            emoji = "😊"
            suggestion = "状态良好，可以继续工作"
        elif final_score < threshold_high:
            level = "轻度疲劳"
            emoji = "😐"
            suggestion = "建议适当休息，避免长时间工作"
        else:
            level = "重度疲劳"
            emoji = "😴"
            suggestion = "强烈建议立即休息，避免继续工作"
        
        # 输出融合结果
        if rgb_valid and eeg_valid:
            self.logger.info(f"  EEG: {eeg_score:.2f}/100 → {eeg_score_50_90:.2f}/90 (权重={used_eeg_weight:.2f}) ✓")
            self.logger.info(f"  RGB: {rgb_score:.2f}/90 (权重={used_rgb_weight:.2f}) ✓")
        else:
            self.logger.info(f"  EEG: {eeg_score:.2f}/100 (权重={used_eeg_weight:.2f}) {'✓' if eeg_valid else '✗'}")
            self.logger.info(f"  RGB: {rgb_score:.2f}/90 (权重={used_rgb_weight:.2f}) {'✓' if rgb_valid else '✗'}")
        self.logger.info(f"  融合方法: {fusion_method}")
        self.logger.info(f"  综合分数: {final_score:.2f}/90 (分数越高越疲劳)")
        self.logger.info(f"  疲劳等级: {level} {emoji}")
        self.logger.info(f"  置信度: {confidence:.1%}")
        
        return {
            "status": status,
            "message": message,
            "fatigue_score": round(final_score, 2),
            "fatigue_level": level,
            "emoji": emoji,
            "suggestion": suggestion,
            "confidence": round(confidence, 2),
            "fusion_method": fusion_method,
            "components": {
                "eeg": {
                    "score": round(eeg_score, 2),
                    "weight": used_eeg_weight,
                    "valid": eeg_valid,
                    "windows": eeg_result.get('num_windows', 0) if eeg_valid else 0
                },
                "rgb": {
                    "score": round(rgb_score, 2),
                    "weight": used_rgb_weight,
                    "valid": rgb_valid,
                    "frames": rgb_result.get('num_rgb_frames', 0) if rgb_valid else 0
                }
            }
        }
    
    def _publish_result(self, result: Dict[str, Any]) -> None:
        """发布评估结果
        
        同时发布两种格式的事件：
        1. FATIGUE_ASSESSMENT_RESULT - 完整评估结果（用于后端日志/存储）
        2. DETECTION_RESULT - 前端兼容格式（用于UI实时显示）
        """
        # 发布完整评估结果事件
        self.bus.publish(Event(
            topic=EventTopic.FATIGUE_ASSESSMENT_RESULT,
            payload=result
        ))
        
        # 同时发布前端兼容格式的检测结果事件
        # 前端期望的格式: detector="model_fatigue", predictions={"fatigue_score": ..., "prediction_class": ...}
        if result.get("status") == "success":
            fatigue_score = result.get("fatigue_score", 0.0)
            fatigue_level = result.get("fatigue_level", "未知")
            
            # 转换为前端期望的格式
            self.bus.publish(Event(
                topic=EventTopic.DETECTION_RESULT,
                payload={
                    "detector": "model_fatigue",
                    "status": "detected",
                    "label": "fatigue",
                    "predictions": {
                        "fatigue_score": fatigue_score,
                        "prediction_class": fatigue_level,
                        "confidence": result.get("confidence", 0.0),
                        "fusion_method": result.get("fusion_method", "unknown"),
                        "components": result.get("components", {}),
                        "inference_mode": "session_assessment"  # 标识这是会话评估结果
                    },
                    "request_id": result.get("request_id"),
                    "timestamp": time.time()
                }
            ))
            
            self.logger.info(f"✅ 同时发布前端兼容格式: fatigue_score={fatigue_score:.2f}, level={fatigue_level}")
        
        self.logger.info(f"\n✅ 疲劳度评估完成")
        self.logger.info(f"{'='*60}\n")
    
    def _publish_error(self, request_id: str, error_message: str) -> None:
        """发布错误结果"""
        result = {
            "status": "error",
            "request_id": request_id,
            "error": error_message,
            "fatigue_score": 0.0
        }
        
        self.bus.publish(Event(
            topic=EventTopic.FATIGUE_ASSESSMENT_RESULT,
            payload=result
        ))


__all__ = ["FatigueAssessmentService"]
