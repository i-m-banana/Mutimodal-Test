from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import mediapipe as mp

if TYPE_CHECKING:
    from typing import Any

    from cv2.typing import MatLike

    dlib: Any

def get_distance(part1, part2):
    """
    计算两个特征点之间的欧氏距离。
    输入：两个特征点对象
    返回：距离（float）
    """
    return pow(pow(part1.x - part2.x, 2) + pow(part1.y - part2.y, 2), 0.5)
class FaceDetector:
    """
    人脸检测与特征提取器（仅使用mediapipe）。
    get_data方法：给定彩色RGB图像和深度帧，输出面部特征（嘴巴开合度，眼部和嘴部关键点归一化坐标，平均深度）。
    特征维度说明：
    - 嘴巴开合度：1维
    - 眼部关键点：12个点 × 2 = 24维（x, y归一化）
    - 嘴部关键点：8个点 × 2 = 16维（x, y归一化）
    - 平均深度：1维
    - 总计：1 + 24 + 16 + 1 = 42维
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

    def get_data(self, color_image: np.ndarray, depth_frame) -> list[float]:
        """
        输入RGB图像和深度图，提取并组合面部特征信息。
        步骤：
          - 检测人脸
          - 计算嘴巴开合比
          - 提取并归一化眼睛（左眼33~42，右眼263~272）、嘴巴（61~68, 291~298）区域关键点
          - 获取嘴巴中心点的深度
        返回:
          [嘴巴开度, 眼部关键点（x1, y1...）, 嘴部关键点, 平均深度]
          若未检测到人脸，返回空列表
        """
        color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(color_image_rgb)
        if not results.multi_face_landmarks:
            return []

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = color_image.shape[:2]

        # 嘴巴开合度：上下嘴唇中点距离/嘴巴宽度
        mouth_open = np.sqrt(
            (landmarks[13].x - landmarks[14].x) ** 2 +
            (landmarks[13].y - landmarks[14].y) ** 2
        )
        mouth_width = np.sqrt(
            (landmarks[78].x - landmarks[308].x) ** 2 +
            (landmarks[78].y - landmarks[308].y) ** 2
        )
        mouth_ratio = mouth_open / mouth_width if mouth_width > 0 else 0

        # 眼部关键点（左眼33~42，右眼263~272）
        eye_features = []
        for idx in range(33, 43):  # 左眼
            eye_features.append(landmarks[idx].x)
            eye_features.append(landmarks[idx].y)
        for idx in range(263, 273):  # 右眼
            eye_features.append(landmarks[idx].x)
            eye_features.append(landmarks[idx].y)

        # 嘴部关键点（61~68, 291~298）
        mouth_features = []
        for idx in range(61, 69):  # 上嘴唇
            mouth_features.append(landmarks[idx].x)
            mouth_features.append(landmarks[idx].y)
        for idx in range(291, 299):  # 下嘴唇
            mouth_features.append(landmarks[idx].x)
            mouth_features.append(landmarks[idx].y)

        # ==== 新的深度均值采集逻辑开始 ====
        depth_indices = (
            list(range(33, 43)) +
            list(range(263, 273)) +
            list(range(61, 69)) +
            list(range(291, 299))
        )
        depth_values = []

        if depth_frame is not None:
            for idx in depth_indices:
                px = int(landmarks[idx].x * w)
                py = int(landmarks[idx].y * h)
                if hasattr(depth_frame, 'get_distance'):
                    d = float(depth_frame.get_distance(px, py))
                else:
                    d = float(depth_frame[py, px]) if depth_frame.ndim == 2 else 0
                if d > 0:
                    depth_values.append(d)
            depth_value = np.mean(depth_values) if depth_values else 0
        else:
            depth_value = 0
        # ==== 新的深度均值采集逻辑结束 ====

        return [mouth_ratio, *eye_features, *mouth_features, depth_value]


