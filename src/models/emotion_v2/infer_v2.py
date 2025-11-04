# Vendored from emotion_infer_v2/infer.py with minimal path changes
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, VivitImageProcessor, Wav2Vec2Processor

from .dataprocess_v2 import crop_video_to_crop_dir


class EmotionClassifier(nn.Module):
    """
    二模态视觉/音频特征融合分类器，使用 Transformer 交叉注意力实现模态交互。

    - vision_feat: [B, Tv, Dv] 或 [B, Dv]
    - audio_feat:  [B, Ta, Da] 或 [B, Da]

    若输入为二维张量，则视为单帧特征，自动添加长度维。
    """

    def __init__(
        self,
        vision_feat_dim: int,
        audio_feat_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vision_proj = nn.Linear(vision_feat_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_dim)

        self.cross_attn_audio_to_vision = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_vision_to_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_v = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    @staticmethod
    def _ensure_seq(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        return x

    def forward(self, vision_feat: torch.Tensor, audio_feat: torch.Tensor) -> torch.Tensor:
        vision_feat = self._ensure_seq(vision_feat)
        audio_feat = self._ensure_seq(audio_feat)

        v_emb = self.vision_proj(vision_feat)
        a_emb = self.audio_proj(audio_feat)

        a_ctx, _ = self.cross_attn_audio_to_vision(query=a_emb, key=v_emb, value=v_emb)
        v_ctx, _ = self.cross_attn_vision_to_audio(query=v_emb, key=a_emb, value=a_emb)

        a_ctx = self.norm_a(a_ctx + a_emb)
        v_ctx = self.norm_v(v_ctx + v_emb)

        a_pooled = a_ctx.mean(dim=1)
        v_pooled = v_ctx.mean(dim=1)

        fused = torch.cat([v_pooled, a_pooled], dim=-1)
        logits = self.classifier(fused)
        return logits

def sample_video_frames(video_path: Path, frames: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise ValueError(f"Invalid frame count for video: {video_path}")

    indices = np.linspace(0, total_frames - 1, frames, dtype=int)
    sampled = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if not success:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled.append(frame)

    capture.release()
    return np.stack(sampled)


def extract_vision_feature(
    video_path: Path,
    processor: VivitImageProcessor,
    model: AutoModel,
    device: torch.device,
    frames: int,
) -> torch.Tensor:
    frames_array = sample_video_frames(video_path, frames)
    pixel_values = processor(list(frames_array), return_tensors="pt")
    vision_tensor = pixel_values["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(vision_tensor)
        feature = outputs.last_hidden_state[:, 0]
    return feature.squeeze(0)


def extract_audio_feature(
    audio_path: Path,
    processor: Wav2Vec2Processor,
    model: AutoModel,
    device: torch.device,
    target_sampling_rate: int,
    max_length: int,
) -> torch.Tensor:
    audio, _ = sf.read(str(audio_path))
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    inputs = processor(
        audio,
        sampling_rate=target_sampling_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
    )
    audio_tensor = inputs.input_values.to(device)
    with torch.no_grad():
        outputs = model(audio_tensor)
        feature = outputs.last_hidden_state[:, 0]
    return feature.squeeze(0)


def load_models(
    vision_model_path: str,
    audio_model_path: str,
    device: torch.device,
) -> Tuple[AutoModel, AutoModel]:
    vision_model = AutoModel.from_pretrained(vision_model_path, ignore_mismatched_sizes=True)
    audio_model = AutoModel.from_pretrained(audio_model_path)
    vision_model.to(device).eval()
    audio_model.to(device).eval()
    return vision_model, audio_model


def prepare_components(
    checkpoint: Path,
    vision_model_path: str,
    audio_model_path: str,
    *,
    device: str = "cuda",
    hidden_dim: int = 512,
    num_classes: int = 3,
) -> Tuple[
    torch.device,
    VivitImageProcessor,
    Wav2Vec2Processor,
    AutoModel,
    AutoModel,
    EmotionClassifier,
]:
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")

    vision_processor = VivitImageProcessor.from_pretrained(vision_model_path)
    audio_processor = Wav2Vec2Processor.from_pretrained(audio_model_path)
    vision_model, audio_model = load_models(vision_model_path, audio_model_path, resolved_device)

    classifier = EmotionClassifier(
        vision_feat_dim=vision_model.config.hidden_size,
        audio_feat_dim=audio_model.config.hidden_size,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(resolved_device)

    state_dict = torch.load(checkpoint, map_location=resolved_device)
    classifier.load_state_dict(state_dict)
    classifier.eval()

    return (
        resolved_device,
        vision_processor,
        audio_processor,
        vision_model,
        audio_model,
        classifier,
    )


def infer_sample(
    video_path: Path,
    audio_path: Path,
    *,
    device: torch.device,
    vision_processor: VivitImageProcessor,
    audio_processor: Wav2Vec2Processor,
    vision_model: AutoModel,
    audio_model: AutoModel,
    classifier: EmotionClassifier,
    vision_frames: int,
    audio_sampling_rate: int,
    audio_max_length: int,
) -> torch.Tensor | None:
    if not video_path.exists():
        print(f"[WARN] Video segment missing: {video_path}")
        return None
    if not audio_path.exists():
        print(f"[WARN] Audio segment missing: {audio_path}")
        return None

    try:
        vision_feat = extract_vision_feature(
            video_path,
            vision_processor,
            vision_model,
            device,
            frames=vision_frames,
        )
        audio_feat = extract_audio_feature(
            audio_path,
            audio_processor,
            audio_model,
            device,
            target_sampling_rate=audio_sampling_rate,
            max_length=audio_max_length,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Failed to extract features for {video_path.name}: {exc}")
        return None

    with torch.no_grad():
        logits = classifier(vision_feat.unsqueeze(0), audio_feat.unsqueeze(0))
        return F.softmax(logits, dim=1).squeeze(0)


def run_directory_inference(
    media_dir: str | Path,
    checkpoint: str | Path,
    vision_model_path: str,
    audio_model_path: str,
    *,
    device: str = "cuda",
    vision_frames: int = 8,
    audio_sampling_rate: int = 16000,
    audio_max_length: int = 160000,
    hidden_dim: int = 512,
    num_classes: int = 3,
    video_extension: str = ".mp4",
    audio_extension: str = ".wav",
    preprocess_with_crop: bool = False,
    raw_video_extension: str = ".avi",
    raw_audio_extension: str = ".wav",
    crop_output_dir: str = "crop",
    output_csv: str | Path | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object] | None]:
    media_root = Path(media_dir)
    if not media_root.exists() or not media_root.is_dir():
        raise FileNotFoundError(f"Media directory not found: {media_root}")

    checkpoint_path = Path(checkpoint)
    (
        resolved_device,
        vision_processor,
        audio_processor,
        vision_model,
        audio_model,
        classifier,
    ) = prepare_components(
        checkpoint_path,
        vision_model_path,
        audio_model_path,
        device=device,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    )

    results: List[Dict[str, object]] = []

    segment_pairs: List[Tuple[Path, Path]] = []
    if preprocess_with_crop:
        segment_pairs.extend(
            crop_video_to_crop_dir(
                media_root,
                video_extension=raw_video_extension,
                audio_extension=raw_audio_extension,
                output_dir_name=crop_output_dir,
            )
        )

    if not segment_pairs:
        for video_file in sorted(media_root.glob(f"*{video_extension}")):
            audio_file = video_file.with_suffix(audio_extension)
            segment_pairs.append((video_file, audio_file))

    # 类别标签映射
    label_names = {
        0: "开心",
        1: "中性", 
        2: "消极",
    }
    
    import time
    
    for idx, (video_file, audio_file) in enumerate(segment_pairs, 1):
        segment_start = time.time()
        
        probs = infer_sample(
            video_file,
            audio_file,
            device=resolved_device,
            vision_processor=vision_processor,
            audio_processor=audio_processor,
            vision_model=vision_model,
            audio_model=audio_model,
            classifier=classifier,
            vision_frames=vision_frames,
            audio_sampling_rate=audio_sampling_rate,
            audio_max_length=audio_max_length,
        )

        if probs is None:
            continue

        probs_np = probs.detach().cpu().numpy()
        predicted = int(np.argmax(probs_np))
        inference_time_ms = round((time.time() - segment_start) * 1000.0, 1)
        
        # 获取中文标签
        label_cn = label_names.get(predicted, f"未知({predicted})")
        max_prob = float(np.max(probs_np))

        record: Dict[str, object] = {
            "segment_index": idx,
            "video_file": video_file.name,
            "audio_file": audio_file.name,
            "video_path": str(video_file),
            "audio_path": str(audio_file),
            "predicted_label": predicted,  # 保持数字类型，用于程序读取
            "predicted_label_cn": label_cn,  # 中文标签，用于人类阅读
            "confidence": round(max_prob, 4),
        }

        # 添加各类别概率
        for class_idx in range(num_classes):
            prob_val = float(probs_np[class_idx]) if class_idx < len(probs_np) else 0.0
            record[f"prob_class_{class_idx}"] = round(prob_val, 4)
            record[f"prob_{label_names.get(class_idx, f'class_{class_idx}')}"] = round(prob_val, 4)
        
        record["inference_time_ms"] = inference_time_ms

        results.append(record)

        print(f"[{idx}/{len(segment_pairs)}] {video_file.name} -> {label_cn}({predicted}) | 置信度:{max_prob:.3f} | {inference_time_ms:.0f}ms")

    if output_csv and results:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入详细结果
        fieldnames = list(results[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
            
            # 添加空行分隔
            fh.write("\n")
            
            # 添加汇总统计信息
            fh.write("# 汇总统计\n")
            fh.write(f"总样本数,{len(results)}\n")
            
            # 统计各类别数量
            from collections import Counter
            label_counts = Counter([r.get("predicted_label_cn", "未知") for r in results])
            fh.write("\n# 标签分布\n")
            fh.write("标签,数量,百分比\n")
            for label, count in label_counts.most_common():
                percentage = (count / len(results)) * 100
                fh.write(f"{label},{count},{percentage:.1f}%\n")
            
            # 平均推理时间
            avg_time = sum(r.get("inference_time_ms", 0) for r in results) / len(results)
            total_time = sum(r.get("inference_time_ms", 0) for r in results)
            fh.write(f"\n# 性能统计\n")
            fh.write(f"平均推理时间(ms),{avg_time:.1f}\n")
            fh.write(f"总推理时间(ms),{total_time:.1f}\n")
            
        print(f"✅ 已保存 {len(results)} 条推理记录到: {output_path}")

    from .infer_v2 import summarize_emotion_predictions  # local import for reuse
    summary = summarize_emotion_predictions(results, num_classes=num_classes)
    if summary is not None:
        mean_probs = summary.get("mean_probabilities", {})
        label_alias = {
            "class_0": "happy (0)",
            "class_1": "neutral (1)",
            "class_2": "negative (2)",
        }
        print("\nAggregate emotion statistics:")
        for cls_name in sorted(mean_probs):
            label = label_alias.get(cls_name, cls_name)
            print(f"  {label}: {mean_probs[cls_name]:.4f}")
        print(f"Emotion score (50-90): {summary['emotion_score']:.2f}")

    return results, summary


def compute_overall_score(
    predictions: List[Dict[str, object]],
    *,
    target_class: int = 1,
) -> Dict[str, float] | None:
    key = f"prob_class_{target_class}"
    scores: List[float] = []
    for record in predictions:
        if key in record and record[key] is not None:
            scores.append(float(record[key]))

    if not scores:
        return None

    mean_prob = float(np.mean(scores))
    min_prob = float(np.min(scores))
    max_prob = float(np.max(scores))

    if max_prob - min_prob < 1e-6:
        normalized = mean_prob * 100.0
    else:
        normalized = (mean_prob - min_prob) / (max_prob - min_prob) * 100.0

    return {
        "mean_probability": mean_prob,
        "min_probability": min_prob,
        "max_probability": max_prob,
        "normalized_score": float(np.clip(normalized, 0.0, 100.0)),
    }


def summarize_emotion_predictions(
    predictions: List[Dict[str, object]],
    *,
    num_classes: int = 3,
) -> Dict[str, object] | None:
    """方案1: 加权投票法
    
    每个样本根据预测类别和置信度进行加权投票
    最终根据各类别得票比例计算情绪分数
    
    优点: 考虑了样本数量和置信度，更公平
    """
    if not predictions:
        return None
    
    # 统计各类别的加权票数
    class_votes = {i: 0.0 for i in range(num_classes)}
    total_confidence = 0.0
    
    for record in predictions:
        predicted_label = record.get("predicted_label")
        confidence = record.get("confidence", 0.0)
        
        if predicted_label is not None and confidence > 0:
            class_votes[predicted_label] += confidence
            total_confidence += confidence
    
    if total_confidence == 0:
        return None
    
    # 归一化为比例
    class_ratios = {
        f"class_{i}": votes / total_confidence 
        for i, votes in class_votes.items()
    }
    
    # 计算情绪分数
    # class_0 (开心) -> 高分, class_2 (消极) -> 低分
    positive_ratio = class_ratios.get("class_0", 0.0)
    neutral_ratio = class_ratios.get("class_1", 0.0)
    negative_ratio = class_ratios.get("class_2", 0.0)
    
    # 情绪分数 = 基准分70 + 积极占比×20 - 消极占比×20
    raw_score = 70.0 + positive_ratio * 20.0 - negative_ratio * 20.0
    emotion_score = float(np.clip(raw_score, 50.0, 90.0))
    
    return {
        "class_ratios": class_ratios,
        "class_votes": class_votes,
        "total_samples": len(predictions),
        "emotion_score": emotion_score,
        "method": "weighted_voting",
        "positive_ratio": round(positive_ratio, 4),
        "neutral_ratio": round(neutral_ratio, 4),
        "negative_ratio": round(negative_ratio, 4),
    }