"""Heuristic fatigue estimator used when real model inference is unavailable."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

try:  # numpy is optional but preferred
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    np = None  # type: ignore


def _safe_len(sequence: Sequence[Any]) -> int:
    try:
        return len(sequence)
    except Exception:  # pragma: no cover - defensive
        count = 0
        for _ in sequence:
            count += 1
        return count


def _eye_valid_ratio(samples: Sequence[Any]) -> float:
    if not samples:
        return 0.0
    total = 0.0
    valid = 0.0
    for entry in samples:
        if isinstance(entry, dict):
            total += 1.0
            if entry.get("valid", True):
                valid += 1.0
            continue
        if np is not None and isinstance(entry, np.ndarray):  # type: ignore[arg-type]
            total += 1.0
            if entry.size > 0 and float(np.mean(entry)) > 0.05:  # type: ignore[call-arg]
                valid += 1.0
            continue
        try:
            magnitude = float(sum(float(v) for v in entry))
        except Exception:  # pragma: no cover - defensive
            magnitude = 0.0
        total += 1.0
        if magnitude > 0.1:
            valid += 1.0
    if total == 0.0:
        return 0.0
    return valid / total


def estimate_fatigue_score(
    rgb_frames: Sequence[Any],
    depth_frames: Sequence[Any],
    eyetrack_samples: Sequence[Any],
    elapsed_time: float = 0.0,
) -> float:
    """Return a pseudo fatigue score in the ``[0, 100]`` range.
    
    The score now incorporates temporal dynamics to simulate fatigue accumulation
    over time, rather than solely relying on queue lengths which plateau quickly.
    
    Args:
        rgb_frames: Sequence of RGB frame data
        depth_frames: Sequence of depth frame data
        eyetrack_samples: Sequence of eyetracking samples
        elapsed_time: Time elapsed since collection started (seconds)
    
    Returns:
        Float score in range [0, 100]
    """
    import math
    import random

    rgb_count = float(_safe_len(rgb_frames))
    depth_count = float(_safe_len(depth_frames))
    gaze_ratio = _eye_valid_ratio(eyetrack_samples)

    print('使用了模拟疲劳评分器,请勿在生产环境中使用!')

    # Base data quality score (reduced weight to make room for time factor)
    base = 35.0
    coverage = min(25.0, 0.12 * (rgb_count + depth_count))
    gaze_bonus = min(8.0, 8.0 * gaze_ratio)

    # Time accumulation factor: simulate fatigue growth over time
    time_factor = 0.0
    if elapsed_time > 0:
        if elapsed_time <= 30:
            # First 30 seconds: rapid growth 0 -> 18
            time_factor = elapsed_time * 0.6
        elif elapsed_time <= 300:
            # 30s - 5min: gradual growth 18 -> 48
            time_factor = 18.0 + (elapsed_time - 30) * 0.11
        else:
            # After 5min: slow growth 48 -> 60
            time_factor = 48.0 + min(12.0, (elapsed_time - 300) * 0.01)

    # Dynamic fluctuation to simulate realistic variance
    phase = elapsed_time * 0.08
    fluctuation = math.sin(phase) * 2.5 + math.cos(phase * 0.7) * 1.2
    noise = random.uniform(-1.0, 1.0)

    score = base + coverage + gaze_bonus + time_factor + fluctuation + noise

    # Smooth out extreme values when queues are still filling up.
    if rgb_count < 5 or depth_count < 2:
        score *= 0.80

    # Brightness adjustment from latest frame
    if np is not None and rgb_frames:
        try:
            brightness = float(np.mean(rgb_frames[-1]))  # type: ignore[index]
            brightness_factor = (brightness - 96.0) / 15.0
            score += max(-5.0, min(5.0, brightness_factor))
        except Exception:  # pragma: no cover - defensive
            pass

    score = max(0.0, min(100.0, score))
    return round(float(score), 2)


__all__ = ["estimate_fatigue_score"]
