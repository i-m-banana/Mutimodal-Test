import torch
import cv2
import numpy as np
import soundfile as sf
from transformers import VivitImageProcessor, Wav2Vec2Processor, AutoTokenizer, AutoModel, Wav2Vec2Model
import time

class SimpleMultimodalClassifier(torch.nn.Module):
    def __init__(self, vision_feat_dim, audio_feat_dim, text_feat_dim, hidden_dim=512, num_classes=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(vision_feat_dim + audio_feat_dim + text_feat_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
    def forward(self, vision_feat, audio_feat, text_feat):
        x = torch.cat([vision_feat, audio_feat, text_feat], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

def preprocess_text(text):
    new_text = []
    for t in str(text).split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def extract_vision_feature(video_path, processor, model, device, frames=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, frames, dtype=int)
    sampled_frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled_frames.append(frame)
    cap.release()
    frames_array = np.array(sampled_frames)
    vision_tensor = processor(list(frames_array), return_tensors='pt')["pixel_values"].to(device)
    with torch.no_grad():
        feat = model(vision_tensor).last_hidden_state[:, 0]
    return feat.squeeze(0)

def extract_audio_feature(audio_path, processor, model, device):
    audio_input, _ = sf.read(audio_path)
    audio_tensor = processor(audio_input, return_tensors="pt", sampling_rate=16000, padding="max_length", max_length=1600000).input_values.to(device)
    with torch.no_grad():
        feat = model(audio_tensor).last_hidden_state[:, 0]
    return feat.squeeze(0)

def extract_text_feature(text, tokenizer, model, device):
    text = preprocess_text(text)
    text_tensor_dict = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=150)
    text_input_ids = text_tensor_dict['input_ids'].to(device)
    with torch.no_grad():
        feat = model(text_input_ids).last_hidden_state[:, -1]
    return feat.squeeze(0)

def infer_emotion(vision_path, audio_path, text, model_path, device='cuda'):
    
    # 加载预处理器和预训练模型
    vision_processor = VivitImageProcessor.from_pretrained('./model/TIMESFORMER')
    audio_processor = Wav2Vec2Processor.from_pretrained('./model/WAV2VEC2')
    text_tokenizer = AutoTokenizer.from_pretrained('./model/ROBBERTA')
    vision_model = AutoModel.from_pretrained('./model/TIMESFORMER', ignore_mismatched_sizes=True).to(device)
    audio_model = Wav2Vec2Model.from_pretrained('./model/WAV2VEC2', ignore_mismatched_sizes=True).to(device)
    text_model = AutoModel.from_pretrained('./model/ROBBERTA', ignore_mismatched_sizes=True).to(device)
    vision_model.eval(); audio_model.eval(); text_model.eval()
    # 特征抽取
    start_time = time.time()
    vision_feat = extract_vision_feature(vision_path, vision_processor, vision_model, device)
    audio_feat = extract_audio_feature(audio_path, audio_processor, audio_model, device)
    text_feat = extract_text_feature(text, text_tokenizer, text_model, device)
    print("特征维度:", vision_feat.shape, audio_feat.shape, text_feat.shape)
    # 分类器
    model = SimpleMultimodalClassifier(
        vision_feat_dim=vision_feat.shape[-1],
        audio_feat_dim=audio_feat.shape[-1],
        text_feat_dim=text_feat.shape[-1],
        hidden_dim=512,
        num_classes=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        
        logits = model(vision_feat.unsqueeze(0), audio_feat.unsqueeze(0), text_feat.unsqueeze(0))
        pred = torch.argmax(logits, dim=1).item()
        elapsed = time.time() - start_time
        print(f"单样本推理耗时: {elapsed:.4f} 秒")
        print("预测类别:", pred)
        return pred

if __name__ == "__main__":
    result = infer_emotion(
        "E:/code/mulimodal_state/recordings/cfy1/20250920_004013/emotion/1.avi",
        "E:/code/mulimodal_state/recordings/cfy1/20250920_004013/emotion/1.wav",
        '十一很好',
        'best_model.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('最终预测类别:', result)
