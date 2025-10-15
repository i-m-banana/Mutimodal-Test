"""情绪模型 - 直接集成版本

直接集成 emotion_fatigue_infer/emotion 中的推理代码
"""

import base64
import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base_inference_model import BaseInferenceModel

# 添加 emotion_fatigue_infer 路径
_EMOTION_FATIGUE_PATH = Path(__file__).parent.parent.parent / "model_backends" / "emotion_fatigue_infer"
_EMOTION_PATH = _EMOTION_FATIGUE_PATH / "emotion"
sys.path.insert(0, str(_EMOTION_FATIGUE_PATH))
sys.path.insert(0, str(_EMOTION_PATH))

try:
    import torch
    import cv2
    import soundfile as sf
    from transformers import VivitImageProcessor, Wav2Vec2Processor, AutoTokenizer, AutoModel, Wav2Vec2Model
    from emotion.inference_standalone_all import (
        SimpleMultimodalClassifier,
        extract_vision_feature,
        extract_audio_feature,
        extract_text_feature
    )
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    _import_error = e


class EmotionModel(BaseInferenceModel):
    """情绪模型（集成版本）
    
    直接在后端进程中运行，无需独立进程
    """
    
    def initialize(self) -> None:
        """初始化情绪模型"""
        if not HAS_DEPS:
            raise RuntimeError(f"无法加载依赖: {_import_error}")
        
        # 确定模型路径
        model_path = _EMOTION_PATH / "best_model.pt"
        model_dir = _EMOTION_PATH / "model"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载预处理器
        self.logger.info("加载预训练模型...")
        self.vision_processor = VivitImageProcessor.from_pretrained(str(model_dir / "TIMESFORMER"))
        self.audio_processor = Wav2Vec2Processor.from_pretrained(str(model_dir / "WAV2VEC2"))
        self.text_tokenizer = AutoTokenizer.from_pretrained(str(model_dir / "ROBBERTA"))
        
        # 加载特征提取模型
        self.vision_model = AutoModel.from_pretrained(
            str(model_dir / "TIMESFORMER"), 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.vision_model.eval()
        
        self.audio_model = Wav2Vec2Model.from_pretrained(
            str(model_dir / "WAV2VEC2"), 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.audio_model.eval()
        
        self.text_model = AutoModel.from_pretrained(
            str(model_dir / "ROBBERTA"), 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.text_model.eval()
        
        # 预热并获取特征维度
        self.logger.info("预热模型...")
        dummy_video = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        dummy_text = "test"
        
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp_video:
            tmp_video_path = tmp_video.name
            out = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (224, 224))
            for _ in range(8):
                out.write(dummy_video)
            out.release()
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
            sf.write(tmp_audio_path, dummy_audio, 16000)
        
        try:
            vision_feat = extract_vision_feature(tmp_video_path, self.vision_processor, self.vision_model, self.device)
            audio_feat = extract_audio_feature(tmp_audio_path, self.audio_processor, self.audio_model, self.device)
            text_feat = extract_text_feature(dummy_text, self.text_tokenizer, self.text_model, self.device)
            
            # 初始化分类器
            self.model = SimpleMultimodalClassifier(
                vision_feat_dim=vision_feat.shape[-1],
                audio_feat_dim=audio_feat.shape[-1],
                text_feat_dim=text_feat.shape[-1],
                hidden_dim=512,
                num_classes=2,
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
            self.model.eval()
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
        
        self.logger.info("情绪模型初始化完成")
    
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理
        
        支持两种输入模式：
        1. base64 数据模式（原有）：
           - samples: List[Dict] - 包含video_b64, audio_b64, text的样本列表
           
        2. 文件路径模式（新增）：
           - file_mode: bool = True
           - video_path: str - 视频文件路径
           - audio_path: str - 音频文件路径
           - text: str = "" - 文本内容（可选）
        
        Args:
            data: 输入数据字典
        
        Returns:
            推理结果:
                - emotion_score: 情绪分数 (0-100)
                - sample_results: 每个样本的结果（如果是多样本）
        """
        # 检查是否为文件路径模式
        if data.get("file_mode") == True:
            return self._infer_from_files(data)
        else:
            return self._infer_from_base64(data)
    
    def _infer_from_files(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从文件路径读取数据并推理"""
        from pathlib import Path
        
        video_path = data.get("video_path")
        audio_path = data.get("audio_path")
        text = data.get("text", "")
        
        if not video_path or not audio_path:
            return {
                "status": "error",
                "error": "缺少必需的视频或音频文件路径",
                "emotion_score": 0.0
            }
        
        # 验证文件存在
        if not Path(video_path).exists():
            return {
                "status": "error",
                "error": f"视频文件不存在: {video_path}",
                "emotion_score": 0.0
            }
        
        if not Path(audio_path).exists():
            return {
                "status": "error",
                "error": f"音频文件不存在: {audio_path}",
                "emotion_score": 0.0
            }
        
        try:
            self.logger.info(f"📂 从文件读取: video={video_path}, audio={audio_path}")
            
            # 提取特征
            v_feat = extract_vision_feature(str(video_path), self.vision_processor, self.vision_model, self.device)
            a_feat = extract_audio_feature(str(audio_path), self.audio_processor, self.audio_model, self.device)
            t_feat = extract_text_feature(text, self.text_tokenizer, self.text_model, self.device)
            
            # 推理
            with torch.no_grad():
                logits = self.model(v_feat.unsqueeze(0), a_feat.unsqueeze(0), t_feat.unsqueeze(0))
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(logits, dim=1).item()
                prob_values = probs.squeeze(0).cpu().numpy()
            
            # 计算分数（0-100）
            score = float(prob_values[1] * 100) if len(prob_values) > 1 else float(prob_values[0] * 100)
            
            # 记录推理结果
            self.logger.info(
                f"情绪推理完成: 分数={round(score, 2)}, 类别={pred}, "
                f"文本='{text[:50]}{'...' if len(text) > 50 else ''}'"
            )
            
            return {
                "status": "success",
                "emotion_score": round(score, 2),
                "prediction": pred,
                "probabilities": prob_values.tolist(),
                "text_input": text
            }
            
        except Exception as e:
            self.logger.error(f"从文件推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "emotion_score": 0.0
            }
    
    def _infer_from_base64(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从base64数据推理（原有逻辑）"""
        samples = data.get("samples", [])
        """执行情绪推理
        
        Args:
            data: 输入数据，包含:
                - samples: 样本列表，每个包含 video_b64, audio_b64, text
        
        Returns:
            推理结果:
                - emotion_score: 情绪分数 (0-100)
                - sample_results: 每个样本的结果
        """
        samples = data.get("samples", [])
        
        if not samples:
            return {
                "status": "error",
                "error": "缺少样本数据",
                "emotion_score": 0.0
            }
        
        temp_files = []
        try:
            logits_list = []
            sample_results = []
            
            for idx, sample in enumerate(samples):
                video_b64 = sample.get("video_b64", "")
                audio_b64 = sample.get("audio_b64", "")
                text = sample.get("text", "")
                question_index = sample.get("question_index", idx + 1)
                
                # 解码并保存视频
                video_bytes = base64.b64decode(video_b64)
                tmp_video = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
                tmp_video.write(video_bytes)
                tmp_video.close()
                temp_files.append(tmp_video.name)
                
                # 解码并保存音频
                audio_bytes = base64.b64decode(audio_b64)
                tmp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp_audio.write(audio_bytes)
                tmp_audio.close()
                temp_files.append(tmp_audio.name)
                
                # 提取特征
                v_feat = extract_vision_feature(tmp_video.name, self.vision_processor, self.vision_model, self.device)
                a_feat = extract_audio_feature(tmp_audio.name, self.audio_processor, self.audio_model, self.device)
                t_feat = extract_text_feature(text, self.text_tokenizer, self.text_model, self.device)
                
                # 推理
                with torch.no_grad():
                    logits = self.model(v_feat.unsqueeze(0), a_feat.unsqueeze(0), t_feat.unsqueeze(0))
                    probs = torch.softmax(logits, dim=1)
                    logits_list.append(logits.squeeze(0).cpu().numpy())
                    pred = torch.argmax(logits, dim=1).item()
                    prob_values = probs.squeeze(0).cpu().numpy()
                
                sample_results.append({
                    "question_index": question_index,
                    "prediction": pred,
                    "probabilities": prob_values.tolist()
                })
            
            # 计算最终分数
            logits_arr = np.array(logits_list)
            probs = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()
            pos_probs = probs[:, 1]
            
            min_prob, max_prob = pos_probs.min(), pos_probs.max()
            if max_prob - min_prob < 1e-6:
                scores = np.full_like(pos_probs, 50.0)
            else:
                scores = (pos_probs - min_prob) / (max_prob - min_prob) * 100
            
            final_score = float(np.mean(scores))
            
            return {
                "status": "success",
                "emotion_score": round(final_score, 2),
                "sample_scores": scores.tolist(),
                "sample_results": sample_results,
                "num_samples": len(samples)
            }
            
        except Exception as e:
            self.logger.error(f"情绪推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "emotion_score": 0.0
            }
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
    
    def cleanup(self) -> None:
        """清理模型资源"""
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'vision_model'):
            del self.vision_model
        if hasattr(self, 'audio_model'):
            del self.audio_model
        if hasattr(self, 'text_model'):
            del self.text_model
        
        if HAS_DEPS and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()


__all__ = ["EmotionModel"]
