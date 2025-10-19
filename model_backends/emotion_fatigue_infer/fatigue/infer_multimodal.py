import torch
import torch.nn as nn
import numpy as np
from facewap import FaceDetector
import time

class FatigueFaceOnlyCNN(nn.Module):
    def __init__(self):
        super(FatigueFaceOnlyCNN, self).__init__()
        # 头部特征提取子网络 (HeadNet)
        self.head_net = nn.Sequential(
            nn.Conv1d(in_channels=42, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 分类子网络
        self.gap = nn.AdaptiveAvgPool1d(1) # 全局平均池化层
        self.fc = nn.Linear(in_features=256, out_features=2) # 全连接层
        self.dropout = nn.Dropout(p=0.5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self,eye_input, face_input):
        # 头部特征提取
        head_features = self.head_net(face_input)  # [B, 256, T_head]
        # 分类
        pooled_features = self.gap(head_features).squeeze(-1)
        pooled_features = self.dropout(pooled_features)
        output = self.fc(pooled_features)
        output = self.softmax(output)
        return output

def extract_eye_features_from_samples(eyetrack_samples):
    '''
    输入eyetrack_samples: List[list[float]]，每个元素为8维，返回[1, 8, T]
    '''
    # eyetrack_samples: List[List[float]]，每个元素为8维list
    arr = np.array(eyetrack_samples, dtype=np.float32)  # [T, 8]
    arr = arr.T  # [8, T]
    arr = arr[np.newaxis, ...]  # [1, 8, T]
    return torch.tensor(arr, dtype=torch.float32)

def extract_face_features_from_frames(rgb_frames, depth_frames, frames=75):
    face_detector = FaceDetector()
    features = []
    for img, depth in zip(rgb_frames, depth_frames):
        feat = face_detector.get_data(img, depth)
        if not feat or len(feat) < 42:
            feat = [0.0] * 42
        features.append(feat[:42])
    while len(features) < frames:
        features.append([0.0] * 42)
    features = np.array(features, dtype=np.float32)  # [T, 42]
    features = features[:frames]
    features = torch.tensor(features, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
    return features

def infer_fatigue_score(rgb_frames, depth_frames, eyetrack_samples, model_path, device='cuda'):
    '''
    输入:
        rgb_frames: List[np.ndarray]
        depth_frames: List[np.ndarray]
        eyetrack_samples: List[List[float]]
        model_path: str
        device: str
    输出:
        pred: int
    '''
    start_time = time.time()
    # 可选：frames参数可由外部指定
    frames = min(len(rgb_frames), len(depth_frames))
    print(f"--------------------行推理")
    face_feat = extract_face_features_from_frames(rgb_frames, depth_frames, frames=frames).to(device)  # [1, 42, frames]
    eye_feat = extract_eye_features_from_samples(eyetrack_samples).to(device)  # [1, 8, T]
    print(f"--------------------行推理2222")
    model = FatigueFaceOnlyCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        output = model(eye_feat, face_feat)  # [1, num_classes]
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]  # [num_classes]
        num_classes = output.shape[1]
        # 分数加权拟合：sum_i(prob_i * (i * 100 / (num_classes-1)))
        scores = np.linspace(0, 100, num_classes)
        score = float(np.dot(probs, scores))
        pred = int(np.argmax(probs))
        elapsed = time.time() - start_time
        print(f"单样本推理耗时: {elapsed:.4f} 秒")
        print("预测类别:", pred)
        print("加权拟合分数(0-100):", score)
    return score

if __name__ == "__main__":
    # 模拟生成测试数据
    num_frames = 10
    rgb_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(num_frames)]
    depth_frames = [np.random.randint(0, 255, (224, 224), dtype=np.uint8) for _ in range(num_frames)]
    eyetrack_samples = [list(np.random.rand(8)) for _ in range(num_frames)]

    result = infer_fatigue_score(
        rgb_frames,
        depth_frames,
        eyetrack_samples,
        'fatigue_best_model.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('最终预测类别:', result)
