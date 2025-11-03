from collections import Counter
import json
import os
import time

import cv2
import mediapipe as mp

from scipy.spatial import distance as dist


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(landmarks):
    left_corner = landmarks[MOUTH_LEFT_CORNER]
    right_corner = landmarks[MOUTH_RIGHT_CORNER]
    top_center = landmarks[MOUTH_TOP_CENTER]
    bottom_center = landmarks[MOUTH_BOTTOM_CENTER]

    horizontal = dist.euclidean(left_corner, right_corner)
    if horizontal == 0:
        return 0.0

    vertical = dist.euclidean(top_center, bottom_center)
    return vertical / horizontal


def compute_fatigue_score(summary: dict, totals: dict, face_frames: int) -> float:
    """
    计算疲劳分数 (50-90分)，分数越高表示状态越好
    
    基线数据：
    - 最好状态(90分): 疲劳事件1次/分钟, 打哈欠0次/分钟, 眨眼4次/分钟, 疲劳占比0.7%
    - 最差状态(50分): 疲劳事件10次/分钟, 打哈欠4次/分钟, 眨眼35次/分钟, 疲劳占比50%
    
    评分依据（基于频率/比率，消除时间影响）：
    1. 疲劳事件频率: 每分钟疲劳事件次数 (1-10次/分钟)
    2. 疲劳状态占比: 高疲劳概率状态占比 (0.7%-50%)
    3. 打哈欠频率: 每分钟打哈欠次数 (0-4次/分钟)
    4. 眨眼频率: 每分钟眨眼次数 (4-35次/分钟)
    """
    base_score = 90.0  # 基础满分
    
    # 估算视频时长（分钟）
    # 假设视频是30fps，face_frames可以估算时长
    estimated_minutes = face_frames / (30.0 * 60.0)
    if estimated_minutes < 0.1:  # 至少6秒
        estimated_minutes = 0.1
    
    # 1. 疲劳事件频率惩罚（每分钟）
    # 基线: 最好1次/分钟，最差10次/分钟，范围9次/分钟
    # 分配权重: 10分
    fatigue_events = totals.get("fatigue_events", 0)
    events_per_minute = fatigue_events / estimated_minutes
    
    if events_per_minute <= 1.0:
        event_penalty = 0.0  # 最好状态
    elif events_per_minute >= 10.0:
        event_penalty = 10.0  # 最差状态
    else:
        # 线性扣分: 1-10次/分钟之间
        event_penalty = ((events_per_minute - 1.0) / 9.0) * 10.0
    
    # 2. 疲劳状态占比惩罚
    # 基线: 最好0.7%，最差50%，范围49.3%
    # 分配权重: 20分
    fatigue_ratio = 0.0
    if "High probability of sleepiness" in summary:
        fatigue_ratio = summary["High probability of sleepiness"]["ratio"]
    
    BEST_RATIO = 0.007   # 0.7%
    WORST_RATIO = 0.50   # 50%
    
    if fatigue_ratio <= BEST_RATIO:
        ratio_penalty = 0.0  # 最好状态
    elif fatigue_ratio >= WORST_RATIO:
        ratio_penalty = 20.0  # 最差状态
    else:
        # 线性扣分: 0.7%-50%之间
        ratio_penalty = ((fatigue_ratio - BEST_RATIO) / (WORST_RATIO - BEST_RATIO)) * 20.0
    
    # 3. 打哈欠频率惩罚（每分钟）
    # 基线: 最好0次/分钟，最差4次/分钟，范围4次/分钟
    # 分配权重: 5分
    yawns = totals.get("yawns", 0)
    yawns_per_minute = yawns / estimated_minutes
    
    if yawns_per_minute <= 0.0:
        yawn_penalty = 0.0  # 最好状态
    elif yawns_per_minute >= 4.0:
        yawn_penalty = 5.0  # 最差状态
    else:
        # 线性扣分: 0-4次/分钟之间
        yawn_penalty = (yawns_per_minute / 4.0) * 5.0
    
    # 4. 眨眼频率惩罚（每分钟）
    # 基线: 最好4次/分钟，最差35次/分钟，范围31次/分钟
    # 分配权重: 5分
    blinks = totals.get("blinks", 0)
    blinks_per_minute = blinks / estimated_minutes
    
    if blinks_per_minute <= 4.0:
        blink_penalty = 0.0  # 最好状态
    elif blinks_per_minute >= 35.0:
        blink_penalty = 5.0  # 最差状态
    else:
        # 线性扣分: 4-35次/分钟之间
        blink_penalty = ((blinks_per_minute - 4.0) / 31.0) * 5.0
    
    # 计算最终分数
    total_penalty = event_penalty + ratio_penalty + yawn_penalty + blink_penalty
    final_score = base_score - total_penalty
    
    # 限制在50-90范围内
    return max(50.0, min(90.0, final_score))


def get_fatigue_state_description(score: float) -> str:
    """根据疲劳分数返回状态描述"""
    if score >= 85:
        return "Excellent - Very alert and focused"
    elif score >= 75:
        return "Good - Normal and stable state"
    elif score >= 65:
        return "Fair - Mild fatigue signs"
    elif score >= 55:
        return "Poor - Moderate fatigue detected"
    else:
        return "Critical - Severe fatigue, rest needed"


def evaluate_fatigue_from_video(
    video_path: str,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
    stats_output_path: str | None = None,
) -> dict:
    """
    从视频文件评估疲劳分数,不显示窗口
    
    Parameters
    ----------
    video_path : str
        视频文件路径
    start_sec : float
        起始时间(秒)
    duration_sec : float | None
        处理时长(秒),None表示处理整个视频
    stats_output_path : str | None
        可选的统计数据保存路径
    
    Returns
    -------
    dict
        包含以下键的字典:
        - fatigue_score: float, 疲劳分数(50-90)
        - state_description: str, 状态描述
        - totals: dict, 累计统计(眨眼、打哈欠、低头、疲劳事件)
        - summary: dict, 各状态占比
        - face_frames: int, 检测到人脸的帧数
        - total_frames: int, 总处理帧数
        - face_ratio: float, 人脸检测率
    """
    try:
        summary, ear_history, totals, face_frames = fatigue(
            COUNTER_BLINK=0,
            COUNTER_YAWN=0,
            COUNTER_HEAD_DROP=0,
            TOTAL_BLINKS=0,
            TOTAL_YAWNS=0,
            TOTAL_HEAD_DROPS=0,
            EYE_AR_THRESH=0.2,
            MOUTH_AR_THRESH=0.6,
            HEAD_DROP_ANGLE_THRESH=15,
            EYE_AR_CONSEC_FRAMES=3,
            MOUTH_AR_CONSEC_FRAMES=15,
            BLINK_THRESH=3,
            YAWN_THRESH=1,
            HEAD_DROP_THRESH=5,
            BLINK_RECOVERY_THRESH=10,
            YAWN_RECOVERY_THRESH=2,
            HEAD_RECOVERY_THRESH=5,
            RECOVERY_FRAMES=100,
            normal_counter=0,
            fatigue_counter=0,
            camera_index=0,
            video_path=video_path,
            display=False,  # 不显示窗口
            stats_path=stats_output_path,
            save_interval_sec=5.0,
            start_sec=start_sec,
            duration_sec=duration_sec,
        )
    except RuntimeError as e:
        raise RuntimeError(f"Failed to process video {video_path}: {e}") from e
    
    if face_frames == 0:
        return {
            "fatigue_score": None,
            "state_description": "No face detected",
            "totals": totals,
            "summary": summary,
            "face_frames": 0,
            "total_frames": 0,
            "face_ratio": 0.0,
            "error": "No face detected in video",
        }
    
    # 计算疲劳分数
    fatigue_score = compute_fatigue_score(summary, totals, face_frames)
    state_description = get_fatigue_state_description(fatigue_score)
    
    # 计算总帧数和人脸检测率
    total_frames = sum(item["frames"] for item in summary.values())
    face_ratio = face_frames / total_frames if total_frames > 0 else 0.0
    
    return {
        "fatigue_score": fatigue_score,
        "state_description": state_description,
        "totals": totals,
        "summary": summary,
        "face_frames": face_frames,
        "total_frames": total_frames,
        "face_ratio": face_ratio,
    }


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_LEFT_CORNER = 78
MOUTH_RIGHT_CORNER = 308
MOUTH_TOP_CENTER = 13
MOUTH_BOTTOM_CENTER = 14
NOSE_TIP_IDX = 1
CHIN_IDX = 152

def detect_drowsiness(total_blinks, total_yawns, total_head_drops, BLINK_THRESH, YAWN_THRESH, HEAD_DROP_THRESH,
                      BLINK_RECOVERY_THRESH, YAWN_RECOVERY_THRESH,
                      HEAD_RECOVERY_THRESH, RECOVERY_FRAMES, normal_counter, fatigue_counter, status):
    
    # 疲劳检测
    if total_blinks > BLINK_THRESH or total_yawns > YAWN_THRESH or total_head_drops > HEAD_DROP_THRESH:
        fatigue_counter += 1  # 记录疲劳状态的持续时间（累计帧数，不再被重置）
        status = "High probability of sleepiness"
        # 为避免重复触发，窗口内计数清零，但不清零 fatigue_counter（用于累计统计）
        total_blinks = 0
        total_yawns = 0
        total_head_drops = 0
    else:
        if (total_blinks < BLINK_RECOVERY_THRESH and total_yawns < YAWN_RECOVERY_THRESH and total_head_drops < HEAD_RECOVERY_THRESH):
            normal_counter =  normal_counter + 1
            if normal_counter >= RECOVERY_FRAMES:
                status = "Normal"
                normal_counter = 0

    return status, normal_counter, fatigue_counter, total_blinks, total_yawns, total_head_drops

def fatigue(
    COUNTER_BLINK,
    COUNTER_YAWN,
    COUNTER_HEAD_DROP,
    TOTAL_BLINKS,
    TOTAL_YAWNS,
    TOTAL_HEAD_DROPS,
    EYE_AR_THRESH,
    MOUTH_AR_THRESH,
    HEAD_DROP_ANGLE_THRESH,
    EYE_AR_CONSEC_FRAMES,
    MOUTH_AR_CONSEC_FRAMES,
    BLINK_THRESH,
    YAWN_THRESH,
    HEAD_DROP_THRESH,
    BLINK_RECOVERY_THRESH,
    YAWN_RECOVERY_THRESH,
    HEAD_RECOVERY_THRESH,
    RECOVERY_FRAMES,
    normal_counter,
    fatigue_counter,
    camera_index=0,
    video_path=None,
    display=True,
    stats_path=None,
    save_interval_sec=5.0,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
):
    if video_path is not None:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        source = video_path if video_path is not None else f"camera index {camera_index}"
        raise RuntimeError(f"Unable to open video source {source}")

    # 若为视频文件且指定了起始秒，跳转到对应时间
    if video_path is not None and start_sec and start_sec > 0:
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(start_sec) * 1000.0)
        except Exception:
            pass

    # 源视频帧率与处理帧率测量
    is_video_file = video_path is not None
    source_fps = cap.get(cv2.CAP_PROP_FPS) if is_video_file else 0.0
    proc_fps = None
    last_tick = time.perf_counter()
    last_sync_tick = last_tick

    last_status = "Monitoring"
    status_counts = Counter()
    ear_history = []
    # 累计计数（用于显示与保存，不随检测窗口复位）
    CUM_BLINKS = 0
    CUM_YAWNS = 0
    CUM_HEAD_DROPS = 0
    CUM_FATIGUE_EVENTS = 0

    last_save_time = time.time() if stats_path else None
    mp_face_mesh = mp.solutions.face_mesh
    total_processed_frames = 0
    
    # GPU加速配置
    # MediaPipe会自动检测GPU并使用，如果可用
    # 注意: MediaPipe 0.10.x 版本的FaceMesh支持的参数：
    #   - max_num_faces: 最大检测人脸数
    #   - refine_landmarks: 是否细化关键点（True会更准确但稍慢）
    #   - min_detection_confidence: 人脸检测置信度阈值
    #   - min_tracking_confidence: 人脸跟踪置信度阈值
    # 较新版本(0.10.14+)可能支持额外参数，但为了兼容性使用基础参数
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,  # 🚀 改为False加速30%（对疲劳检测影响很小）
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_processed_frames += 1
            
            # 🚀 性能优化：缩小帧尺寸加速推理（2-3倍提速）
            # 640x480对人脸检测足够，且MediaPipe CPU版本处理更快
            if frame.shape[1] > 640:  # 如果宽度大于640
                scale = 640 / frame.shape[1]
                new_width = 640
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))

            # 若为视频文件且设置了时长，超过则终止
            if video_path is not None and duration_sec is not None and duration_sec > 0:
                pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                # 当驱动不支持 POS_MSEC 时可能返回 0，这种情况下不做限制
                if pos_msec > 0 and pos_msec >= (float(start_sec) + float(duration_sec)) * 1000.0:
                    break

            # 计算处理 FPS（指数平滑）
            now_tick = time.perf_counter()
            dt = now_tick - last_tick
            if dt > 0:
                inst_fps = 1.0 / dt
                proc_fps = inst_fps if proc_fps is None else (0.9 * proc_fps + 0.1 * inst_fps)
            last_tick = now_tick

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            current_status = last_status
            current_ear = None
            face_detected = False

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                height, width = frame.shape[:2]
                landmarks = [(lm.x * width, lm.y * height) for lm in face_landmarks.landmark]

                left_eye = [landmarks[idx] for idx in LEFT_EYE_IDX]
                right_eye = [landmarks[idx] for idx in RIGHT_EYE_IDX]

                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0
                current_ear = ear
                ear_history.append(ear)

                mar = mouth_aspect_ratio(landmarks)

                if ear < EYE_AR_THRESH:
                    COUNTER_BLINK += 1
                else:
                    if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES:
                        TOTAL_BLINKS += 1
                        CUM_BLINKS += 1
                    COUNTER_BLINK = 0

                if mar > MOUTH_AR_THRESH:
                    COUNTER_YAWN += 1
                else:
                    if COUNTER_YAWN >= MOUTH_AR_CONSEC_FRAMES:
                        TOTAL_YAWNS += 1
                        CUM_YAWNS += 1
                    COUNTER_YAWN = 0

                nose_tip = landmarks[NOSE_TIP_IDX]
                chin = landmarks[CHIN_IDX]
                head_drop_angle = abs(nose_tip[1] - chin[1])

                if head_drop_angle > HEAD_DROP_ANGLE_THRESH:
                    COUNTER_HEAD_DROP += 1
                else:
                    if COUNTER_HEAD_DROP > 1:
                        TOTAL_HEAD_DROPS += 1
                        CUM_HEAD_DROPS += 1
                    COUNTER_HEAD_DROP = 0

                status, normal_counter, fatigue_counter, TOTAL_BLINKS, TOTAL_YAWNS, TOTAL_HEAD_DROPS = detect_drowsiness(
                    TOTAL_BLINKS,
                    TOTAL_YAWNS,
                    TOTAL_HEAD_DROPS,
                    BLINK_THRESH,
                    YAWN_THRESH,
                    HEAD_DROP_THRESH,
                    BLINK_RECOVERY_THRESH,
                    YAWN_RECOVERY_THRESH,
                    HEAD_RECOVERY_THRESH,
                    RECOVERY_FRAMES,
                    normal_counter,
                    fatigue_counter,
                    last_status,
                )

                # 检测疲劳事件从非疲劳到疲劳的跃迁
                if status == "High probability of sleepiness" and last_status != "High probability of sleepiness":
                    CUM_FATIGUE_EVENTS += 1

                current_status = status
                face_detected = True

                highlight_indices = set(
                    LEFT_EYE_IDX
                    + RIGHT_EYE_IDX
                    + [
                        MOUTH_LEFT_CORNER,
                        MOUTH_RIGHT_CORNER,
                        MOUTH_TOP_CENTER,
                        MOUTH_BOTTOM_CENTER,
                        NOSE_TIP_IDX,
                        CHIN_IDX,
                    ]
                )
                for idx in highlight_indices:
                    x, y = landmarks[idx]
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

            if face_detected:
                status_counts[current_status] += 1
                last_status = current_status
            else:
                current_status = "No face detected"

            if display:
                cv2.putText(frame, current_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # 界面显示累计统计
                cv2.putText(frame, f"Blinks: {CUM_BLINKS}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Yawns: {CUM_YAWNS}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Head Drops: {CUM_HEAD_DROPS}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Fatigue Events: {CUM_FATIGUE_EVENTS}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                if current_ear is not None:
                    cv2.putText(
                        frame,
                        f"EAR: {current_ear:.3f}",
                        (10, 155),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                # 显示 FPS：处理 FPS 与源 FPS（若为视频文件可用）
                if proc_fps is not None:
                    cv2.putText(frame, f"Proc FPS: {proc_fps:.1f}", (10, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if is_video_file and source_fps and source_fps > 0:
                    cv2.putText(frame, f"Src FPS:  {source_fps:.1f}", (10, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                # 显示人脸检测帧比例（检测到人脸的帧数 / 总处理帧数）
                face_frames_count = sum(status_counts.values())
                if total_processed_frames > 0:
                    face_ratio = 100.0 * face_frames_count / float(total_processed_frames)
                    cv2.putText(frame, f"Face frames: {face_frames_count}/{total_processed_frames} ({face_ratio:.1f}%)", (10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                total_frames = sum(status_counts.values())
                if total_frames:
                    sorted_status = sorted(status_counts.items(), key=lambda item: item[1], reverse=True)
                    for idx, (status_name, count) in enumerate(sorted_status[:3]):
                        ratio = (count / total_frames) * 100
                        y_pos = 180 + idx * 20
                        cv2.putText(
                            frame,
                            f"{status_name}: {ratio:.1f}%",
                            (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2,
                        )
                # 同步到源视频 FPS（可选）
                if is_video_file and SYNC_TO_VIDEO and source_fps and source_fps > 0:
                    period = 1.0 / float(source_fps)
                    elapsed = time.perf_counter() - last_sync_tick
                    if elapsed < period:
                        time.sleep(period - elapsed)
                    last_sync_tick = time.perf_counter()

                cv2.imshow("Fatigue Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # 周期性保存统计
            if stats_path:
                now = time.time()
                if last_save_time is None or (now - last_save_time) >= float(save_interval_sec):
                    try:
                        os.makedirs(os.path.dirname(stats_path), exist_ok=True) if os.path.dirname(stats_path) else None
                        tmp_path = stats_path + ".tmp"
                        stats_payload = {
                            "blinks": CUM_BLINKS,
                            "yawns": CUM_YAWNS,
                            "head_drops": CUM_HEAD_DROPS,
                            "fatigue_events": CUM_FATIGUE_EVENTS,
                            "frames_with_face": sum(status_counts.values()),
                            "frames_total": total_processed_frames,
                            "face_ratio": (sum(status_counts.values()) / float(total_processed_frames)) if total_processed_frames > 0 else None,
                            "timestamp": int(now),
                        }
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            json.dump(stats_payload, f, ensure_ascii=False, indent=2)
                        os.replace(tmp_path, stats_path)
                        last_save_time = now
                    except Exception as e:
                        # 仅打印，不中断主循环
                        print(f"Failed to save stats to {stats_path}: {e}")

    cap.release()
    if display:
        cv2.destroyAllWindows()

    total_face_frames = sum(status_counts.values())
    totals = {
        "blinks": CUM_BLINKS,
        "yawns": CUM_YAWNS,
        "head_drops": CUM_HEAD_DROPS,
        "fatigue_events": CUM_FATIGUE_EVENTS,
    }
    # 退出前进行一次最终保存
    if stats_path:
        try:
            os.makedirs(os.path.dirname(stats_path), exist_ok=True) if os.path.dirname(stats_path) else None
            with open(stats_path, "w", encoding="utf-8") as f:
                final_face_frames = sum(status_counts.values())
                final_total_frames = total_processed_frames
                final_payload = {
                    **totals,
                    "frames_with_face": final_face_frames,
                    "frames_total": final_total_frames,
                    "face_ratio": (final_face_frames / float(final_total_frames)) if final_total_frames > 0 else None,
                    "timestamp": int(time.time()),
                }
                json.dump(final_payload, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save final stats to {stats_path}: {e}")
    if total_face_frames == 0:
        return {}, ear_history, totals, total_face_frames

    summary = {
        status: {
            "frames": count,
            "ratio": count / total_face_frames,
        }
        for status, count in status_counts.items()
    }

    return summary, ear_history, totals, total_face_frames


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fatigue monitor: use camera or video file")
    parser.add_argument("--video", "-v", default="/home/caixiaohui/workspace/multi_state/mulimodal_state/fatiguev2/fff/wsh0/20251031_112615/fatigue/rgb0.avi", help="Path to video file. If omitted, camera will be used.")
    parser.add_argument("--camera-index", "-c", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--no-display", action="store_true", help="Disable display window (useful for headless).")
    parser.add_argument("--save-stats", type=str, default=None, help="Path to save cumulative stats as JSON (e.g., stats.json)")
    parser.add_argument("--save-interval", type=float, default=5.0, help="Seconds between periodic stats save (default 5s)")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Start time in seconds when reading from a video file")
    parser.add_argument("--duration-sec", type=float, default=None, help="Duration in seconds to read from the video file")
    parser.add_argument("--sync-to-video", action="store_true", help="Sync playback to source video FPS (video file only)")
    args = parser.parse_args()

    video_path = args.video
    # 若未指定 --video，尝试在脚本所在目录下自动寻找本地视频，优先 .mp4/.avi/.mov/.mkv 中最近修改的文件
    if not video_path:
        root_dir = os.path.abspath(os.path.dirname(__file__))
        candidates = []
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in exts:
                    full = os.path.join(dirpath, fn)
                    try:
                        mtime = os.path.getmtime(full)
                    except OSError:
                        mtime = 0
                    candidates.append((mtime, full))
        if candidates:
            candidates.sort(reverse=True)
            video_path = candidates[0][1]
            print(f"Auto-selected local video: {video_path}")
        else:
            print("No local video found. Falling back to camera.")
    show_choice = not args.no_display
    # 全局/外层同步标志（供 fatigue 内部使用）
    global SYNC_TO_VIDEO
    SYNC_TO_VIDEO = bool(args.sync_to_video)

    try:
        summary, ear_history, totals, face_frames = fatigue(
            COUNTER_BLINK=0,
            COUNTER_YAWN=0,
            COUNTER_HEAD_DROP=0,
            TOTAL_BLINKS=0,
            TOTAL_YAWNS=0,
            TOTAL_HEAD_DROPS=0,
            EYE_AR_THRESH=0.2,
            MOUTH_AR_THRESH=0.6,
            HEAD_DROP_ANGLE_THRESH=15,
            EYE_AR_CONSEC_FRAMES=3,
            MOUTH_AR_CONSEC_FRAMES=15,
            BLINK_THRESH=3,
            YAWN_THRESH=1,
            HEAD_DROP_THRESH=5,
            BLINK_RECOVERY_THRESH=10,
            YAWN_RECOVERY_THRESH=2,
            HEAD_RECOVERY_THRESH=5,
            RECOVERY_FRAMES=100,
            normal_counter=0,
            fatigue_counter=0,
            camera_index=args.camera_index,
            video_path=video_path,
            display=show_choice,
            stats_path=args.save_stats,
            save_interval_sec=args.save_interval,
            start_sec=args.start_sec,
            duration_sec=args.duration_sec,
        )
    except RuntimeError as e:
        print(f"Error opening source: {e}")
        raise SystemExit(1)
    if summary:
        total = sum(item["frames"] for item in summary.values())
        for status, stats in summary.items():
            ratio_percent = stats["ratio"] * 100
            print(f"{status}: {stats['frames']} frames ({ratio_percent:.2f}% of {total} frames)")
    else:
        print("No frames processed. Check that the video path and predictor model are correct.")

    if ear_history:
        avg_ear = sum(ear_history) / len(ear_history)
        min_ear = min(ear_history)
        max_ear = max(ear_history)
        print(f"EAR tracked on {len(ear_history)} frames | avg: {avg_ear:.4f}, min: {min_ear:.4f}, max: {max_ear:.4f}")
    else:
        print("No EAR values collected.")

    print(
        "Total events | blinks: {blinks}, yawns: {yawns}, head drops: {head_drops}, fatigue events: {fatigue_events}".format(
            **totals,
        )
    )
    print(f"Frames with face detected: {face_frames}")
    
    # 计算疲劳分数 (50-90分，分数越高状态越好)
    if face_frames > 0:
        fatigue_score = compute_fatigue_score(summary, totals, face_frames)
        print(f"\n{'='*50}")
        print(f"Fatigue Score: {fatigue_score:.2f} / 90")
        print(f"State: {get_fatigue_state_description(fatigue_score)}")
        print(f"{'='*50}")
