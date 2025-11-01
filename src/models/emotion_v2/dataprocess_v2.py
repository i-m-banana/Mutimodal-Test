# Vendored from emotion_infer_v2/dataprocess.py with module path adjustments
from __future__ import annotations

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import mediapipe as mp
import soundfile as sf

OUTPUT_SIZE = (224, 224)
PADDING = 10
MIN_FRAMES = 1
SEGMENT_MIN_SECONDS = 3.0
SEGMENT_TARGET_SECONDS = 5.0
SEGMENT_MAX_SECONDS = 6.0


def collect_face_boxes(
    detector: mp.solutions.face_detection.FaceDetection,
    video_path: Path,
) -> Tuple[List[Tuple[int, int, int, int]], int, int, float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], 0, 0, 0.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    boxes: List[Tuple[int, int, int, int]] = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb_frame)
        detections = results.detections if results.detections else []
        if not detections:
            continue

        def detection_area(det) -> float:
            bbox = det.location_data.relative_bounding_box
            return max(bbox.width, 0.0) * max(bbox.height, 0.0)

        best_detection = max(detections, key=detection_area)
        bbox = best_detection.location_data.relative_bounding_box
        if bbox.width <= 0 or bbox.height <= 0:
            continue

        left = max(int(round(bbox.xmin * width)), 0)
        top = max(int(round(bbox.ymin * height)), 0)
        right = min(int(round((bbox.xmin + bbox.width) * width)), width)
        bottom = min(int(round((bbox.ymin + bbox.height) * height)), height)
        if right > left and bottom > top:
            boxes.append((left, top, right, bottom))

    capture.release()
    return boxes, width, height, fps


def aggregate_box(
    boxes: Sequence[Tuple[int, int, int, int]],
    width: int,
    height: int,
    padding: int,
) -> Tuple[int, int, int, int] | None:
    if not boxes:
        return None

    min_left = max(min(box[0] for box in boxes) - padding, 0)
    min_top = max(min(box[1] for box in boxes) - padding, 0)
    max_right = min(max(box[2] for box in boxes) + padding, width)
    max_bottom = min(max(box[3] for box in boxes) + padding, height)

    if max_right <= min_left or max_bottom <= min_top:
        return None
    return min_left, min_top, max_right, max_bottom


def next_output_path(video_path: Path, output_dir_name: str = "crop") -> Path:
    crop_dir = video_path.parent / output_dir_name
    crop_dir.mkdir(parents=True, exist_ok=True)

    index = 1
    while True:
        candidate = crop_dir / f"{index}.mp4"
        if not candidate.exists():
            return candidate
        index += 1


def crop_video_to_file(video_path: Path, output_path: Path) -> bool:
    with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
        boxes, width, height, fps = collect_face_boxes(detector, video_path)

    if not boxes:
        print(f"[WARN] No faces detected: {video_path}", file=sys.stderr)
        return False

    region = aggregate_box(boxes, width, height, PADDING)
    if region is None:
        print(f"[WARN] Invalid aggregate crop for: {video_path}", file=sys.stderr)
        return False

    left, top, right, bottom = region

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[WARN] Cannot reopen video: {video_path}", file=sys.stderr)
        return False

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, OUTPUT_SIZE)

    written_frames = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        cropped = frame[top:bottom, left:right]
        if cropped.size == 0:
            continue
        resized = cv2.resize(cropped, OUTPUT_SIZE, interpolation=cv2.INTER_LINEAR)
        writer.write(resized)
        written_frames += 1

    capture.release()
    writer.release()

    if written_frames < MIN_FRAMES:
        output_path.unlink(missing_ok=True)
        print(f"[WARN] No frames written for: {video_path}", file=sys.stderr)
        return False

    print(f"Saved cropped video -> {output_path}")
    return True


def get_video_duration(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    capture.release()
    if fps <= 1e-6 or frame_count <= 0:
        raise RuntimeError(f"Cannot determine duration for video: {path}")
    return frame_count / fps


def get_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    if info.samplerate <= 0:
        raise RuntimeError(f"Invalid audio samplerate for {path}")
    return info.frames / info.samplerate


def choose_segment_count(duration: float) -> int:
    if duration < SEGMENT_MIN_SECONDS:
        return 0
    candidate = max(1, round(duration / SEGMENT_TARGET_SECONDS))
    seg_len = duration / candidate
    while seg_len > SEGMENT_MAX_SECONDS:
        candidate += 1
        seg_len = duration / candidate
    while seg_len < SEGMENT_MIN_SECONDS and candidate > 1:
        candidate -= 1
        seg_len = duration / candidate
    if seg_len < SEGMENT_MIN_SECONDS:
        return 1
    return candidate


def build_segments(duration: float) -> List[Tuple[float, float]]:
    count = choose_segment_count(duration)
    if count == 0:
        return []
    segment_length = duration / count
    segments: List[Tuple[float, float]] = []
    start = 0.0
    for index in range(count):
        end = duration if index == count - 1 else start + segment_length
        segments.append((start, end))
        start = end

    if segments and (segments[-1][1] - segments[-1][0]) < SEGMENT_MIN_SECONDS and len(segments) >= 2:
        prev_start, _ = segments[-2]
        segments[-2] = (prev_start, duration)
        segments.pop()

    corrected: List[Tuple[float, float]] = []
    for seg_start, seg_end in segments:
        if corrected and (seg_end - seg_start) < SEGMENT_MIN_SECONDS:
            prev_start, _ = corrected.pop()
            corrected.append((prev_start, seg_end))
        else:
            corrected.append((seg_start, seg_end))

    final_segments: List[Tuple[float, float]] = []
    for seg_start, seg_end in corrected:
        length = seg_end - seg_start
        if length > SEGMENT_MAX_SECONDS + 1e-6:
            remaining = length
            current_start = seg_start
            chunk = min(SEGMENT_MAX_SECONDS, max(SEGMENT_MIN_SECONDS, SEGMENT_TARGET_SECONDS))
            while remaining > SEGMENT_MAX_SECONDS + 1e-6:
                final_segments.append((current_start, current_start + chunk))
                current_start += chunk
                remaining = seg_end - current_start
            if remaining >= SEGMENT_MIN_SECONDS:
                final_segments.append((current_start, seg_end))
            else:
                if final_segments:
                    prev_start, _ = final_segments.pop()
                    final_segments.append((prev_start, seg_end))
                else:
                    final_segments.append((seg_start, seg_end))
        else:
            final_segments.append((seg_start, seg_end))
    return final_segments


def run_ffmpeg_slice(src: Path, dst: Path, start: float, end: float, stream_copy: bool = True) -> None:
    duration = max(0.0, end - start)
    if duration < SEGMENT_MIN_SECONDS:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.3f}",
    ]
    if stream_copy:
        cmd += ["-c", "copy"]
    cmd.append(str(dst))
    subprocess.run(cmd, check=True)


def split_cropped_pair(
    video_path: Path,
    audio_path: Path,
    split_root: Path,
    audio_extension: str,
    stem: str,
) -> list[tuple[Path, Path]]:
    try:
        video_duration = get_video_duration(video_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {exc}; skipping {video_path}", file=sys.stderr)
        return []
    try:
        audio_duration = get_audio_duration(audio_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {exc}; skipping {audio_path}", file=sys.stderr)
        return []

    duration = min(video_duration, audio_duration)
    segments = build_segments(duration)
    if not segments:
        print(f"[WARN] No valid segments generated for {video_path}", file=sys.stderr)
        return []

    split_root.mkdir(parents=True, exist_ok=True)

    prefix = f"{stem}-"
    for stale in split_root.glob(f"{prefix}*"):
        stale.unlink(missing_ok=True)

    produced: list[tuple[Path, Path]] = []
    for idx, (start, end) in enumerate(segments, start=1):
        segment_video = split_root / f"{stem}-{idx}.mp4"
        segment_audio = split_root / f"{stem}-{idx}{audio_extension}"
        try:
            run_ffmpeg_slice(video_path, segment_video, start, end, stream_copy=True)
            run_ffmpeg_slice(audio_path, segment_audio, start, end, stream_copy=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] ffmpeg failed during split: {exc}", file=sys.stderr)
            segment_video.unlink(missing_ok=True)
            segment_audio.unlink(missing_ok=True)
            continue
        produced.append((segment_video, segment_audio))

    return produced


def crop_video_to_crop_dir(
    media_root: str | Path,
    video_extension: str = ".avi",
    audio_extension: str = ".wav",
    output_dir_name: str = "crop",
) -> list[tuple[Path, Path]]:
    media_root_path = Path(media_root)

    if not media_root_path.exists() or not media_root_path.is_dir():
        print(f"[ERROR] Directory not found: {media_root_path}", file=sys.stderr)
        return []

    _ = output_dir_name

    split_pairs: list[tuple[Path, Path]] = []

    for video_file in sorted(media_root_path.glob(f"*{video_extension}")):
        audio_file = video_file.with_suffix(audio_extension)
        if not audio_file.exists():
            print(f"[WARN] Audio counterpart missing for {video_file.name}", file=sys.stderr)
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_video = temp_dir_path / f"{video_file.stem}.mp4"

            try:
                success = crop_video_to_file(video_file, temp_video)
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] {exc}", file=sys.stderr)
                continue

            if not success:
                continue

            split_root = video_file.parent / "split"
            split_results = split_cropped_pair(temp_video, audio_file, split_root, audio_extension, video_file.stem)
            split_pairs.extend(split_results)

    return split_pairs
