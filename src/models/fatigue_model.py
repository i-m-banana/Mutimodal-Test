"""疲劳度模型 - 直接集成版本

直接集成 emotion_fatigue_infer/fatigue 中的推理代码
"""

import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from .base_inference_model import BaseInferenceModel

# 添加 emotion_fatigue_infer 路径
_EMOTION_FATIGUE_PATH = Path(__file__).parent.parent.parent / "model_backends" / "emotion_fatigue_infer"
_FATIGUE_PATH = _EMOTION_FATIGUE_PATH / "fatigue"
sys.path.insert(0, str(_EMOTION_FATIGUE_PATH))
sys.path.insert(0, str(_FATIGUE_PATH))

try:
    import torch
    from fatigue.infer_multimodal import (
        FatigueFaceOnlyCNN,
        extract_eye_features_from_samples,
        extract_face_features_from_frames
    )
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    _import_error = e


class FatigueModel(BaseInferenceModel):
    """疲劳度模型（集成版本）
    
    直接在后端进程中运行，无需独立进程
    """
    
    def initialize(self) -> None:
        """初始化疲劳度模型"""
        if not HAS_TORCH:
            raise RuntimeError(f"无法加载PyTorch依赖: {_import_error}")
        
        # 确定模型路径 - 从根目录的models_data文件夹加载
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models_data" / "fatigue_models"
        model_path = models_dir / "fatigue_best_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = FatigueFaceOnlyCNN().to(self.device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model.eval()
        
        self.logger.info("疲劳度模型初始化完成")
    
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理
        
        支持三种输入模式：
        1. 内存模式（推荐，零I/O开销）：
           - memory_mode: bool = True
           - rgb_frames_memory: List[np.ndarray] - RGB图像numpy数组
           - depth_frames_memory: List[np.ndarray] - 深度图像numpy数组
           - eyetrack_memory: List[List[float]] - 眼动特征数据
           
        2. 文件路径模式（存档备份）：
           - file_mode: bool = True
           - rgb_video_path: str - RGB视频文件路径
           - depth_video_path: str - 深度视频文件路径
           - eyetrack_json_path: str - 眼动数据JSON文件路径
           - max_frames: int = 30 - 最大读取帧数
           
        3. base64 数据模式（兼容旧接口）：
           - rgb_frames: List[str] - base64编码的RGB图像
           - depth_frames: List[str] - base64编码的深度图像
           - eyetrack_samples: List[Dict] - 眼动数据
        
        Args:
            data: 输入数据字典
        
        Returns:
            推理结果:
                - fatigue_score: 疲劳度分数 (0-100)
                - prediction_class: 预测类别
        """
        # 优先使用内存模式(避免文件I/O)
        if data.get("memory_mode") == True:
            return self._infer_from_memory(data)
        elif data.get("file_mode") == True:
            return self._infer_from_files(data)
        else:
            return self._infer_from_base64(data)
    
    def _infer_from_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从内存中的numpy数组直接推理（零I/O开销）"""
        import time
        start_time = time.time()
        
        rgb_frames = data.get("rgb_frames_memory", [])
        depth_frames = data.get("depth_frames_memory", [])
        eyetrack_samples = data.get("eyetrack_memory", [])
        
        if not rgb_frames or not depth_frames:
            return {
                "status": "error",
                "error": "缺少必需的图像数据",
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
        
        try:
            # 直接使用numpy数组,无需解码或I/O
            frames = min(len(rgb_frames), len(depth_frames))
            
            # 提取特征
            face_feat = extract_face_features_from_frames(
                rgb_frames, depth_frames, frames=frames
            ).to(self.device)
            
            eye_feat = extract_eye_features_from_samples(eyetrack_samples).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.model(eye_feat, face_feat)
                probs = output.cpu().numpy()[0]
                num_classes = output.shape[1]
                
                scores = np.linspace(0, 100, num_classes)
                score = float(np.dot(probs, scores))
                pred = int(np.argmax(probs))
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 单行输出推理结果
            fatigue_level = "正常😊" if score < 30 else "轻度疲劳😐" if score < 60 else "重度疲劳😴"
            self.logger.info(
                f"😴 疲劳度: {round(score, 2)} ({fatigue_level}, "
                f"RGB{len(rgb_frames)}+深度{len(depth_frames)}+眼动{len(eyetrack_samples)}, {round(inference_time, 1)}ms)"
            )
            
            return {
                "status": "success",
                "fatigue_score": round(score, 2),
                "prediction_class": pred,
                "num_rgb_frames": len(rgb_frames),
                "num_depth_frames": len(depth_frames),
                "num_eyetrack_samples": len(eyetrack_samples),
                "inference_mode": "memory",
                "inference_time_ms": round(inference_time, 1)
            }
            
        except Exception as e:
            self.logger.error(f"内存推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
    
    def _infer_from_files(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从文件路径读取数据并推理"""
        import cv2
        import json
        import time
        from pathlib import Path
        
        start_time = time.time()
        
        rgb_video_path = data.get("rgb_video_path")
        depth_video_path = data.get("depth_video_path")
        eyetrack_json_path = data.get("eyetrack_json_path")
        max_frames = data.get("max_frames", 30)
        
        if not rgb_video_path or not depth_video_path:
            return {
                "status": "error",
                "error": "缺少必需的视频文件路径",
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
        
        # 验证文件存在
        if not Path(rgb_video_path).exists():
            return {
                "status": "error",
                "error": f"RGB视频文件不存在: {rgb_video_path}",
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
        
        if not Path(depth_video_path).exists():
            return {
                "status": "error",
                "error": f"深度视频文件不存在: {depth_video_path}",
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
        
        try:
            # 1. 读取RGB视频
            rgb_frames = []
            cap = cv2.VideoCapture(str(rgb_video_path))
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame_count += 1
            cap.release()
            
            # 2. 读取深度视频
            depth_frames = []
            cap = cv2.VideoCapture(str(depth_video_path))
            frame_count = 0
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # 转换为灰度图
                if len(frame.shape) == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                depth_frames.append(frame)
                frame_count += 1
            cap.release()
            
            # 3. 读取眼动数据（支持JSONL格式）
            eyetrack_samples = []
            if eyetrack_json_path and Path(eyetrack_json_path).exists():
                with open(eyetrack_json_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            # 提取8维眼动特征：gaze(2) + eye_position(6)
                            # gaze_point: [x, y]
                            # eye_position: [left_x, left_y, left_z, right_x, right_y, right_z]
                            gaze = data.get('gaze_point', [0.0, 0.0])
                            eye_pos = data.get('eye_position', [0.0] * 6)
                            
                            # 确保维度正确
                            if len(gaze) < 2:
                                gaze = [0.0, 0.0]
                            if len(eye_pos) < 6:
                                eye_pos = [0.0] * 6
                            
                            # 组合为8维特征
                            sample = list(gaze[:2]) + list(eye_pos[:6])
                            eyetrack_samples.append(sample)
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"� 疲劳度分析 - 文件模式")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"📊 数据统计:")
            self.logger.info(f"   RGB帧数: {len(rgb_frames)}")
            self.logger.info(f"   深度帧数: {len(depth_frames)}")
            self.logger.info(f"   眼动样本数: {len(eyetrack_samples)}")
            self.logger.info(f"📂 文件路径:")
            self.logger.info(f"   RGB视频: {Path(rgb_video_path).name}")
            self.logger.info(f"   深度视频: {Path(depth_video_path).name}")
            if eyetrack_json_path:
                self.logger.info(f"   眼动数据: {Path(eyetrack_json_path).name}")
            
            # 4. 提取特征
            self.logger.info(f"🔍 提取面部和眼动特征...")
            frames = min(len(rgb_frames), len(depth_frames))
            face_feat = extract_face_features_from_frames(
                rgb_frames, depth_frames, frames=frames
            ).to(self.device)
            
            eye_feat = extract_eye_features_from_samples(eyetrack_samples).to(self.device)
            
            # 5. 模型推理
            with torch.no_grad():
                output = self.model(eye_feat, face_feat)
                probs = output.cpu().numpy()[0]
                num_classes = output.shape[1]
                
                scores = np.linspace(0, 100, num_classes)
                score = float(np.dot(probs, scores))
                pred = int(np.argmax(probs))
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            # 输出推理结果
            fatigue_level = "正常😊" if score < 30 else "轻度疲劳😐" if score < 60 else "重度疲劳😴"
            self.logger.info(f"✅ 推理完成:")
            self.logger.info(f"   疲劳度分数: {round(score, 2)}")
            self.logger.info(f"   疲劳等级: {fatigue_level}")
            self.logger.info(f"   预测类别: {pred}")
            self.logger.info(f"   推理耗时: {round(inference_time, 1)}ms")
            self.logger.info(f"   推理模式: 文件模式")
            self.logger.info(f"{'='*60}\n")
            
            return {
                "status": "success",
                "fatigue_score": round(score, 2),
                "prediction_class": pred,
                "num_rgb_frames": len(rgb_frames),
                "num_depth_frames": len(depth_frames),
                "num_eyetrack_samples": len(eyetrack_samples),
                "inference_mode": "file",
                "inference_time_ms": round(inference_time, 1)
            }
            
        except Exception as e:
            self.logger.error(f"从文件推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
    
    def _infer_from_base64(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从base64数据推理（原有逻辑）"""
        import time
        start_time = time.time()
        
        rgb_b64_list = data.get("rgb_frames", [])
        depth_b64_list = data.get("depth_frames", [])
        eyetrack_samples = data.get("eyetrack_samples", [])
        elapsed = data.get("elapsed_time", 0.0)
        
        if not rgb_b64_list or not depth_b64_list:
            return {
                "status": "error",
                "error": "缺少必需的图像数据",
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
        
        try:
            # 1. 解码图像
            rgb_frames = []
            depth_frames = []
            
            for rgb_b64 in rgb_b64_list:
                rgb_bytes = base64.b64decode(rgb_b64)
                rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                rgb_array = np.array(rgb_image)
                rgb_frames.append(rgb_array)
            
            for depth_b64 in depth_b64_list:
                depth_bytes = base64.b64decode(depth_b64)
                depth_image = Image.open(io.BytesIO(depth_bytes)).convert("L")
                depth_array = np.array(depth_image)
                depth_frames.append(depth_array)
            
            # 2. 提取特征
            frames = min(len(rgb_frames), len(depth_frames))
            face_feat = extract_face_features_from_frames(
                rgb_frames, depth_frames, frames=frames
            ).to(self.device)
            
            eye_feat = extract_eye_features_from_samples(eyetrack_samples).to(self.device)
            
            # 3. 模型推理
            with torch.no_grad():
                output = self.model(eye_feat, face_feat)
                probs = output.cpu().numpy()[0]
                num_classes = output.shape[1]
                
                # 加权计算分数
                scores = np.linspace(0, 100, num_classes)
                score = float(np.dot(probs, scores))
                pred = int(np.argmax(probs))
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            self.logger.info(
                f"✅ Base64推理完成: 分数={round(score, 2)}, 类别={pred}, 耗时={round(inference_time, 1)}ms"
            )
            
            return {
                "status": "success",
                "fatigue_score": round(score, 2),
                "prediction_class": pred,
                "elapsed_time": round(elapsed, 2),
                "num_rgb_frames": len(rgb_frames),
                "num_depth_frames": len(depth_frames),
                "num_eyetrack_samples": len(eyetrack_samples),
                "inference_mode": "base64",
                "inference_time_ms": round(inference_time, 1)
            }
            
        except Exception as e:
            self.logger.error(f"疲劳度推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "fatigue_score": 0.0,
                "prediction_class": 0
            }
    
    def cleanup(self) -> None:
        """清理模型资源"""
        if hasattr(self, 'model'):
            del self.model
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()


__all__ = ["FatigueModel"]
