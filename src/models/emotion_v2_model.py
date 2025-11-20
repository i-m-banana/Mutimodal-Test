"""Emotion V2 模型封装（二模态：视频+音频，无文本）

基于 `emotion_infer_v2` 的新架构，兼容现有 `BaseInferenceModel` 接口，
支持单样本(file_mode)与多样本(multi_sample_mode)文件路径输入。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base_inference_model import BaseInferenceModel


class EmotionV2Model(BaseInferenceModel):
    """使用 emotion_infer_v2 的二模态情绪模型。

    约定输入：
    - 多样本：
      {
        "multi_sample_mode": True,
        "video_paths": List[str],
        "audio_paths": List[str]
      }
    - 单样本：
      {
        "file_mode": True,
        "video_path": str,
        "audio_path": str
      }
    文本(text)将被忽略。
    """

    def initialize(self) -> None:
        # 确保项目根路径在 sys.path 中，便于导入 emotion_infer_v2
        project_root = Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # 延迟导入第三方与新架构模块（避免安装缺失时提前失败）
        try:
            import torch  # noqa: F401
            from .emotion_v2.infer_v2 import prepare_components  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"加载 emotion_infer_v2 失败: {exc}") from exc

        # 设备选择
        device_opt = str(self.options.get("device", "auto")).lower()
        try:
            import torch
            if device_opt == "auto":
                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            elif device_opt in {"cuda", "cpu"}:
                self.device_str = device_opt
            else:
                self.device_str = "cpu"
        except Exception:
            self.device_str = "cpu"

        # 模型与预训练路径
        # 默认从 models_data/emotion_pretrained_models 查找 TIMESFORMER / WAV2VEC2
        models_data = project_root / "models_data"
        default_vision = models_data / "emotion_pretrained_models" / "TIMESFORMER"
        default_audio = models_data / "emotion_pretrained_models" / "WAV2VEC2"

        # checkpoint 默认使用项目内 models_data/emotion_models 下的权重
        default_ckpt = models_data / "emotion_models" / "best_model_emotion.pt"

        self.vision_model_path = str(self.options.get("vision_model_path", default_vision))
        self.audio_model_path = str(self.options.get("audio_model_path", default_audio))
        self.checkpoint_path = str(self.options.get("checkpoint_path", default_ckpt))

        # 分类头维度/类别（需与训练时一致）
        self.hidden_dim = int(self.options.get("hidden_dim", 512))
        self.num_classes = int(self.options.get("num_classes", 3))

        # 采样/预处理参数
        self.vision_frames = int(self.options.get("vision_frames", 8))
        self.audio_sampling_rate = int(self.options.get("audio_sampling_rate", 16000))
        self.audio_max_length = int(self.options.get("audio_max_length", 160000))

        # 实例化组件（内部模块）
        from .emotion_v2.infer_v2 import prepare_components  # type: ignore

        (
            self.device,
            self.vision_processor,
            self.audio_processor,
            self.vision_model,
            self.audio_model,
            self.classifier,
        ) = prepare_components(
            checkpoint=Path(self.checkpoint_path),
            vision_model_path=self.vision_model_path,
            audio_model_path=self.audio_model_path,
            device=self.device_str,
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
        )

        self.logger.debug(
            f"EmotionV2 初始化完成 | device={self.device} | frames={self.vision_frames} | classes={self.num_classes}"
        )

    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行情绪推理
        
        多样本模式：调用 run_directory_inference 完成切分/批量推理/CSV输出/汇总
        单样本模式：直接调用 infer_sample 进行单文件推理
        """
        from .emotion_v2.infer_v2 import infer_sample, run_directory_inference  # type: ignore
        import tempfile
        import shutil

        start_ts = time.time()
        # 类别名称映射（按 class_0/1/2 → 开心/中性/消极）
        label_names = ["开心", "中性", "消极"]

        # 多样本模式：使用 run_directory_inference 完成切分+批量推理
        if bool(data.get("multi_sample_mode")):
            video_paths: List[str] = list(map(str, data.get("video_paths", [])))
            audio_paths: List[str] = list(map(str, data.get("audio_paths", [])))

            n = min(len(video_paths), len(audio_paths))
            if n == 0:
                return {"status": "error", "error": "没有可用的音视频样本", "emotion_score": 0.0}

            # 确定CSV保存路径：如果输入文件来自recordings目录，则保存到同一会话目录
            # 否则保存到项目根目录的 recordings/emotion_results/
            project_root = Path(__file__).resolve().parents[2]
            first_video = Path(video_paths[0])
            
            # 尝试从第一个视频路径推断会话目录
            csv_output = None
            if "recordings" in first_video.parts:
                # 找到 recordings 目录后的用户和会话路径
                try:
                    rec_idx = first_video.parts.index("recordings")
                    if len(first_video.parts) > rec_idx + 2:
                        # 路径格式: recordings/user/session/emotion/1.avi
                        user_name = first_video.parts[rec_idx + 1]
                        session_name = first_video.parts[rec_idx + 2]
                        session_dir = project_root / "recordings" / user_name / session_name
                        csv_output = session_dir / "emotion" / "emotion_predictions.csv"
                except (ValueError, IndexError):
                    pass
            
            # 回退：保存到专用的结果目录
            if csv_output is None:
                results_dir = project_root / "recordings" / "emotion_results"
                results_dir.mkdir(parents=True, exist_ok=True)
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_output = results_dir / f"predictions_{timestamp}.csv"
            
            # 创建临时目录存放所有输入文件
            temp_dir = None
            try:
                temp_dir = tempfile.mkdtemp(prefix="emotion_v2_")
                temp_path = Path(temp_dir)
                
                # 复制所有视频音频对到临时目录（统一扩展名以便 run_directory_inference 识别）
                for idx in range(n):
                    src_video = Path(video_paths[idx])
                    src_audio = Path(audio_paths[idx])
                    
                    # 使用索引作为文件名，保持配对关系
                    dst_video = temp_path / f"sample_{idx+1}.avi"
                    dst_audio = temp_path / f"sample_{idx+1}.wav"
                    
                    shutil.copy2(src_video, dst_video)
                    shutil.copy2(src_audio, dst_audio)
                
                self.logger.debug(f"📹 已准备 {n} 对音视频文件，开始切分+批量推理...")
                
                # 调用 run_directory_inference：自动切分+推理+汇总（临时CSV）
                temp_csv = temp_path / "predictions.csv"
                results, summary = run_directory_inference(
                    media_dir=temp_path,
                    checkpoint=self.checkpoint_path,
                    vision_model_path=self.vision_model_path,
                    audio_model_path=self.audio_model_path,
                    device=str(self.device),
                    vision_frames=self.vision_frames,
                    audio_sampling_rate=self.audio_sampling_rate,
                    audio_max_length=self.audio_max_length,
                    hidden_dim=self.hidden_dim,
                    num_classes=self.num_classes,
                    video_extension=".avi",  # 临时目录中的视频格式
                    audio_extension=".wav",
                    preprocess_with_crop=True,  # 启用人脸裁剪+切分为3-6秒片段
                    raw_video_extension=".avi",  # 原始视频扩展名
                    raw_audio_extension=".wav",  # 原始音频扩展名
                    crop_output_dir="crop",  # 裁剪临时目录（已废弃但保留兼容性）
                    output_csv=temp_csv,
                )
                
                # 将CSV从临时目录移动到项目目录
                if temp_csv.exists():
                    csv_output.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(temp_csv), str(csv_output))
                    self.logger.debug(f"📄 CSV结果已保存到: {csv_output}")
                
                if not results:
                    return {"status": "error", "error": "所有样本推理失败", "emotion_score": 0.0}
                
                # 提取汇总分数与标签分布
                emotion_score = summary.get("emotion_score", 0.0) if summary else 0.0
                mean_probs = summary.get("mean_probabilities", {}) if summary else {}
                
                # 构建样本结果（带标签）
                sample_results = []
                for idx, rec in enumerate(results):
                    pred = int(rec.get("predicted_label", 0))
                    probs = [rec.get(f"prob_class_{i}", 0.0) for i in range(self.num_classes)]
                    sample_results.append({
                        "sample_index": idx + 1,
                        "prediction": pred,
                        "label": label_names[pred] if pred < len(label_names) else str(pred),
                        "probabilities": probs,
                        "video_file": Path(rec.get("video_path", "")).name,
                        "audio_file": Path(rec.get("audio_path", "")).name,
                    })
                
                elapsed_ms = round((time.time() - start_ts) * 1000.0, 1)
                
                # 统计标签分布
                from collections import Counter
                labels_dist = Counter([s["label"] for s in sample_results])
                top_label, top_count = labels_dist.most_common(1)[0] if labels_dist else ("未知", 0)
                
                self.logger.info(
                    f"✅ 情绪(多样本): {emotion_score:.2f} [主:{top_label} {top_count}/{len(sample_results)}] "
                    f"{elapsed_ms:.0f}ms | CSV已保存: {csv_output}"
                )
                
                return {
                    "status": "success",
                    "emotion_score": round(emotion_score, 2),
                    "sample_results": sample_results,
                    "num_samples": len(sample_results),
                    "mean_probabilities": mean_probs,
                    "inference_time_ms": elapsed_ms,
                    "inference_mode": "multi_file",
                    "labels": label_names,
                    "csv_path": str(csv_output),
                }
                
            except Exception as exc:
                self.logger.error(f"多样本推理失败: {exc}", exc_info=True)
                return {"status": "error", "error": str(exc), "emotion_score": 0.0}
            finally:
                # 清理临时目录（可选：保留用于调试）
                if temp_dir and Path(temp_dir).exists():
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass

        # 单样本模式：直接推理单个文件
        if bool(data.get("file_mode")):
            video_path = Path(str(data.get("video_path", "")))
            audio_path = Path(str(data.get("audio_path", "")))
            if not video_path.exists() or not audio_path.exists():
                return {
                    "status": "error",
                    "error": f"文件不存在: video={video_path}, audio={audio_path}",
                    "emotion_score": 0.0,
                }

            probs = infer_sample(
                video_path,
                audio_path,
                device=self.device,
                vision_processor=self.vision_processor,
                audio_processor=self.audio_processor,
                vision_model=self.vision_model,
                audio_model=self.audio_model,
                classifier=self.classifier,
                vision_frames=self.vision_frames,
                audio_sampling_rate=self.audio_sampling_rate,
                audio_max_length=self.audio_max_length,
            )
            if probs is None:
                return {"status": "error", "error": "样本推理失败", "emotion_score": 0.0}

            probs_np = probs.detach().cpu().numpy()
            pred = int(np.argmax(probs_np))
            score = float(np.max(probs_np) * 100.0)  # 简单以最大概率映射 0-100

            elapsed_ms = round((time.time() - start_ts) * 1000.0, 1)
            self.logger.info(f"✅ 情绪(单样本): {score:.2f} [{label_names[pred]}] {elapsed_ms:.0f}ms")
            
            return {
                "status": "success",
                "emotion_score": round(score, 2),
                "prediction": pred,
                "prediction_label": label_names[pred] if pred < len(label_names) else str(pred),
                "probabilities": probs_np.tolist(),
                "inference_time_ms": elapsed_ms,
                "inference_mode": "file",
                "labels": label_names,
            }

        # 其他模式（例如 base64）不支持 —— 可按需扩展
        return {"status": "error", "error": "不支持的输入模式", "emotion_score": 0.0}

    def cleanup(self) -> None:
        # 释放显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        # 其余对象由 GC 清理


__all__ = ["EmotionV2Model"]
