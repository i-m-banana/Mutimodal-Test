"""RGB疲劳度模型 - MediaPipe版本

基于MediaPipe面部检测的疲劳度评估
使用 fatigue_rgb 模块中的评估算法
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict

from .base_inference_model import BaseInferenceModel

# 添加 fatigue_rgb 路径
_FATIGUE_RGB_PATH = Path(__file__).parent / "fatigue_rgb"
sys.path.insert(0, str(_FATIGUE_RGB_PATH))

try:
    from fershowv3 import evaluate_fatigue_from_video
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    _import_error = e


class RGBFatigueModel(BaseInferenceModel):
    """RGB疲劳度模型（MediaPipe版本）
    
    基于RGB视频的疲劳度评估
    使用MediaPipe进行面部特征检测
    
    功能:
    - 处理RGB视频帧
    - 检测眨眼、打哈欠、低头等疲劳特征
    - 输出疲劳度分数 (50-90分，分数越高越清醒)
    """
    
    def initialize(self) -> None:
        """初始化RGB疲劳度模型"""
        if not HAS_DEPS:
            raise RuntimeError(f"无法加载依赖: {_import_error}")
        
        self.logger.debug("✅ RGB疲劳度模型初始化完成 (MediaPipe版本)")
    
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行RGB疲劳度推理（仅支持文件模式）
        
        Args:
            data: 输入数据字典
                - file_mode: bool = True
                - rgb_video_path: str - RGB视频文件路径
                - start_sec: float = 0.0 - 开始时间（秒）
                - duration_sec: float = None - 分析时长（秒）
                - stats_output_path: str = None - 统计输出路径
        
        Returns:
            推理结果:
                - rgb_fatigue_score: 疲劳度分数 (50-90，分数越高越清醒)
                - state_description: 状态描述
                - totals: 累计统计（眨眼、打哈欠、低头、疲劳事件）
                - summary: 各状态占比
                - face_frames: 检测到人脸的帧数
                - total_frames: 总处理帧数
                - face_ratio: 人脸检测率
        """
        return self._infer_from_file(data)
    
    def _infer_from_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从视频文件推理"""
        start_time = time.time()
        
        rgb_video_path = data.get("rgb_video_path")
        start_sec = data.get("start_sec", 0.0)
        duration_sec = data.get("duration_sec", None)
        
        if not rgb_video_path:
            return {
                "status": "error",
                "error": "缺少必需的视频文件路径",
                "rgb_fatigue_score": 0.0
            }
        
        # 验证文件存在
        if not Path(rgb_video_path).exists():
            return {
                "status": "error",
                "error": f"RGB视频文件不存在: {rgb_video_path}",
                "rgb_fatigue_score": 0.0
            }
        
        try:
            # 调用 MediaPipe 疲劳评估函数
            result = evaluate_fatigue_from_video(
                video_path=str(rgb_video_path),
                start_sec=start_sec,
                duration_sec=duration_sec,
                stats_output_path=None  # 不保存统计文件
            )
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 检查是否有错误
            if result.get("fatigue_score") is None:
                error_msg = result.get("error", "评估失败")
                self.logger.warning(f"❌ RGB疲劳度评估失败: {error_msg}")
                return {
                    "status": "error",
                    "error": error_msg,
                    "rgb_fatigue_score": 0.0
                }
            
            # 提取结果
            fatigue_score = result["fatigue_score"]
            state_desc = result["state_description"]
            totals = result["totals"]
            
            # RGB原始分数: 50-90分，分数越高越清醒（90=最清醒，50=最疲劳）
            # 为了与EEG疲劳度统一（分数越高越疲劳），需要反转
            # 反转公式: rgb_fatigue = 140 - raw_score
            # 当 raw_score=90（最清醒）时，rgb_fatigue=50（不疲劳）
            # 当 raw_score=50（最疲劳）时，rgb_fatigue=90（疲劳）
            raw_score = fatigue_score
            rgb_fatigue_score = fatigue_score
            
            # 判断疲劳等级（基于50-90分，分数越高越好）
            if rgb_fatigue_score < 65:  # <65分 = 疲劳
                fatigue_level = "重度疲劳😴"
            elif rgb_fatigue_score < 77.5:  # 65-77.5 = 轻度疲劳
                fatigue_level = "轻度疲劳😐"
            else:  # >=77.5 = 正常
                fatigue_level = "正常😊"
            
            # 单行输出推理结果
            self.logger.info(
                f"📷😴 RGB疲劳度: {round(rgb_fatigue_score, 2)}/90 ({fatigue_level}, "
                f"原始={round(raw_score, 1)}, {state_desc}, 眨眼{totals['blinks']}次, 打哈欠{totals['yawns']}次, "
                f"{round(inference_time, 1)}ms)"
            )
            
            return {
                "status": "success",
                "rgb_fatigue_score": round(rgb_fatigue_score, 2),
                "raw_fatigue_score": round(fatigue_score, 2),  # 原始50-90分
                "state_description": state_desc,
                "totals": totals,
                "summary": result.get("summary", {}),
                "face_frames": result.get("face_frames", 0),
                "total_frames": result.get("total_frames", 0),
                "face_ratio": result.get("face_ratio", 0.0),
                "inference_mode": "file",
                "inference_time_ms": round(inference_time, 1)
            }
            
        except Exception as e:
            self.logger.error(f"从文件推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "rgb_fatigue_score": 0.0
            }
    
    def cleanup(self) -> None:
        """清理模型资源"""
        self.logger.info("RGB疲劳度模型资源已清理")


__all__ = ["RGBFatigueModel"]
