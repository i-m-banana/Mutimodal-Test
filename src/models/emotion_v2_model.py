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

        self.logger.info(
            f"EmotionV2 初始化完成 | device={self.device} | frames={self.vision_frames} | classes={self.num_classes}"
        )

    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        from .emotion_v2.infer_v2 import infer_sample, summarize_emotion_predictions  # type: ignore

        start_ts = time.time()

        # 多样本模式
        if bool(data.get("multi_sample_mode")):
            video_paths: List[str] = list(map(str, data.get("video_paths", [])))
            audio_paths: List[str] = list(map(str, data.get("audio_paths", [])))

            n = min(len(video_paths), len(audio_paths))
            if n == 0:
                return {"status": "error", "error": "没有可用的音视频样本", "emotion_score": 0.0}

            probs_list: List[np.ndarray] = []
            sample_results: List[Dict[str, Any]] = []

            for idx in range(n):
                vp = Path(video_paths[idx])
                ap = Path(audio_paths[idx])
                probs = infer_sample(
                    vp,
                    ap,
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
                    self.logger.warning(f"样本失败(跳过): {vp.name}")
                    continue

                probs_np = probs.detach().cpu().numpy()
                pred = int(np.argmax(probs_np))
                probs_list.append(probs_np)
                sample_results.append({
                    "sample_index": idx + 1,
                    "prediction": pred,
                    "probabilities": probs_np.tolist(),
                    "video_file": vp.name,
                    "audio_file": ap.name,
                })

            if not probs_list:
                return {"status": "error", "error": "所有样本推理失败", "emotion_score": 0.0}

            predictions = []
            for arr in probs_list:
                rec = {f"prob_class_{i}": float(p) for i, p in enumerate(arr.tolist())}
                predictions.append(rec)

            # 汇总分数（使用 v2 提供的统计函数，返回 50-90 区间）
            summary = summarize_emotion_predictions(
                predictions=[{**r} for r in predictions], num_classes=self.num_classes
            )
            if summary is not None:
                emotion_score = float(summary.get("emotion_score", 0.0))
            else:
                # 回退: 用目标类(1)概率的均值映射到0-100
                target_probs = [rec.get("prob_class_1", 0.0) for rec in predictions]
                emotion_score = float(np.mean(target_probs) * 100.0)

            elapsed_ms = round((time.time() - start_ts) * 1000.0, 1)
            self.logger.info(f"✅ 情绪(多样本): {emotion_score:.2f} | {len(sample_results)} 样本 | {elapsed_ms:.0f}ms")
            return {
                "status": "success",
                "emotion_score": round(emotion_score, 2),
                "sample_results": sample_results,
                "num_samples": len(sample_results),
                "inference_time_ms": elapsed_ms,
                "inference_mode": "multi_file",
            }

        # 单样本模式
        if bool(data.get("file_mode")):
            from .emotion_v2.infer_v2 import infer_sample  # type: ignore

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
            return {
                "status": "success",
                "emotion_score": round(score, 2),
                "prediction": pred,
                "probabilities": probs_np.tolist(),
                "inference_time_ms": elapsed_ms,
                "inference_mode": "file",
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
