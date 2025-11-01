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

from dataprocess import crop_video_to_crop_dir


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


def run_inference(
    video_path: str | Path,
    audio_path: str | Path,
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
) -> Dict[str, float]:
    """Run multimodal inference on a single cropped video/audio pair.

    Parameters
    ----------
    video_path, audio_path, checkpoint:
        Paths to the segment pair and trained classifier weights.
    vision_model_path, audio_model_path:
        Locations of the pretrained encoders used during training.
    device:
        Preferred device; falls back to CPU automatically when CUDA is absent.
    vision_frames / audio_sampling_rate / audio_max_length:
        Preprocessing hyperparameters mirroring the training pipeline. Audio is
        assumed to be already resampled to ``audio_sampling_rate`` as in
        training; a warning is emitted otherwise.
    hidden_dim / num_classes:
        Classifier configuration; must match the training setup.

    Returns
    -------
    Dict[str, float]
        Mapping of class indices to probabilities. The highest entry is the
        predicted label.
    """

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

    probs = infer_sample(
        Path(video_path),
        Path(audio_path),
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
        raise RuntimeError("Failed to compute probabilities for provided inputs.")

    probabilities = {f"class_{idx}": float(prob) for idx, prob in enumerate(probs.tolist())}
    predicted_label = max(probabilities, key=probabilities.get)

    print(f"Prediction: {predicted_label}")
    print("Probabilities:")
    for label, prob in probabilities.items():
        print(f"  {label}: {prob:.4f}")

    return probabilities


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

    for video_file, audio_file in segment_pairs:
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

        record: Dict[str, object] = {
            "video_path": str(video_file),
            "audio_path": str(audio_file),
            "predicted_label": predicted,
        }

        for idx, prob in enumerate(probs_np):
            record[f"prob_class_{idx}"] = float(prob)

        results.append(record)

        print(f"{video_file.name} -> class {predicted} | probs: {probs_np}")

    if output_csv and results:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(results[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved batch predictions to {output_path}")

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
    """Compute an aggregate score from per-sample prediction probabilities.

    Parameters
    ----------
    predictions:
        Output from ``run_directory_inference`` containing prob_class_* entries.
    target_class:
        Which class probability to aggregate; defaults to class 1.

    Returns
    -------
    dict | None
        Dictionary containing mean/min/max probabilities and a normalized
        score mapped to [0, 100]. Returns ``None`` if the target probability is
        unavailable.
    """

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
    """Summarize mean probabilities and derive a 50-90 emotion score."""

    if not predictions:
        return None

    mean_probabilities: Dict[str, float] = {}
    for idx in range(num_classes):
        key = f"prob_class_{idx}"
        values = [float(record[key]) for record in predictions if key in record and record[key] is not None]
        if values:
            mean_probabilities[f"class_{idx}"] = float(np.mean(values))

    if not mean_probabilities:
        return None

    positive = mean_probabilities.get("class_0", 0.0)
    negative = mean_probabilities.get("class_2", 0.0)
    diff = positive - negative
    raw_score = 70.0 + diff * 20.0
    emotion_score = float(np.clip(raw_score, 50.0, 90.0))

    return {
        "mean_probabilities": mean_probabilities,
        "emotion_score": emotion_score,
    }





if __name__ == "__main__":  # pragma: no cover - simple demo
    # Fill in the following paths before running directly.
    media_directory = Path(
        "/home/caixiaohui/workspace/multi_state/mulimodal_state/emotionv3/data"
    )
    checkpoint = Path("/home/caixiaohui/workspace/multi_state/mulimodal_state/emotionv3/best_model2-2-82.pt")
    vision_model_path = "/home/caixiaohui/workspace/multi_state/mulimodal_state/emotion/model/TIMESFORMER"
    audio_model_path = "/home/caixiaohui/workspace/multi_state/mulimodal_state/emotion/model/WAV2VEC2"

    if not Path(checkpoint).exists():
        raise SystemExit(f"Checkpoint missing: {checkpoint}")
    if not media_directory.exists():
        raise SystemExit(f"Media directory missing: {media_directory}")

    try:
        predictions, summary = run_directory_inference(
            media_dir=media_directory,
            checkpoint=checkpoint,
            vision_model_path=vision_model_path,
            audio_model_path=audio_model_path,
            device="cuda",
            vision_frames=8,
            preprocess_with_crop=True,
            raw_video_extension=".avi",
            raw_audio_extension=".wav",
            crop_output_dir="crop",
            output_csv=media_directory / "predictions.csv",
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Inference failed: {exc}") from exc

    print(f"\nProcessed {len(predictions)} samples.")

    if summary is None:
        print("No probabilities available to compute emotion score.")
