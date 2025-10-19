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
    print("-------------------原始文本:",text)
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
    # print('-----------vision_tensor',feat.shape)
    return feat.squeeze(0)

def extract_audio_feature(audio_path, processor, model, device):
    audio_input, _ = sf.read(audio_path)
    audio_tensor = processor(audio_input, return_tensors="pt", sampling_rate=16000, padding="max_length", max_length=1600000).input_values.to(device)
    with torch.no_grad():
        feat = model(audio_tensor).last_hidden_state[:, 0]
    # print('-----------audio_tensor',feat.shape)
    return feat.squeeze(0)

def extract_text_feature(text, tokenizer, model, device):
    text = preprocess_text(text)
    print("-------------------预处理后文本:", text)
    text_tensor_dict = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=150)
    text_input_ids = text_tensor_dict['input_ids'].to(device)
    with torch.no_grad():
        feat = model(text_input_ids).last_hidden_state[:, -1]
    # print('-----------text_tensor',feat.shape)
    return feat.squeeze(0)

def infer_emotion(vision_path, audio_path, text, model_path='D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/best_model.pt', device='cuda'):
    
    # 批量处理，输入均为列表（长度一致），text为字典列表。
    vision_processor = VivitImageProcessor.from_pretrained('D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/model/TIMESFORMER')
    print('-----------------vision_processor')
    audio_processor = Wav2Vec2Processor.from_pretrained('D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/model/WAV2VEC2')
    print('-----------------audio_processor')
    text_tokenizer = AutoTokenizer.from_pretrained('D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/model/ROBBERTA')
    print('-----------------text_tokenizer')
    vision_model = AutoModel.from_pretrained('D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/model/TIMESFORMER', ignore_mismatched_sizes=True).to(device)
    print('-----------------vision_model')
    audio_model = Wav2Vec2Model.from_pretrained('D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/model/WAV2VEC2', ignore_mismatched_sizes=True).to(device)
    print('-----------------audio_model')
    text_model = AutoModel.from_pretrained('D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/model/ROBBERTA', ignore_mismatched_sizes=True).to(device)
    print('-----------------text_model')
    vision_model.eval(); audio_model.eval(); text_model.eval()
    # 分类器初始化（用第一个样本的特征维度）
    vision_feat = extract_vision_feature(vision_path[0], vision_processor, vision_model, device)
    audio_feat = extract_audio_feature(audio_path[0], audio_processor, audio_model, device)
    print(audio_feat.shape)
    text_feat = extract_text_feature(text[0]['recognized_text'], text_tokenizer, text_model, device)
    print('-----------------提取第一个样本特征完成')
    model = SimpleMultimodalClassifier(
        vision_feat_dim=vision_feat.shape[-1],
        audio_feat_dim=audio_feat.shape[-1],
        text_feat_dim=text_feat.shape[-1],
        hidden_dim=512,
        num_classes=2,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logits_list = []
    for v_path, a_path, t_dict in zip(vision_path, audio_path, text):
        start_time = time.time()
        v_feat = extract_vision_feature(v_path, vision_processor, vision_model, device)
        a_feat = extract_audio_feature(a_path, audio_processor, audio_model, device)
        t_feat = extract_text_feature(t_dict['recognized_text'], text_tokenizer, text_model, device)
        print("-------------------处理样本:", t_dict.get('question_index', '?'))
        print("特征维度:", v_feat.shape, a_feat.shape, t_feat.shape)
        with torch.no_grad():
            logits = model(v_feat.unsqueeze(0), a_feat.unsqueeze(0), t_feat.unsqueeze(0))  # [1, 2]
            import torch.nn.functional as F
            probs = F.softmax(logits, dim=1)
            print("logits:", logits)
            print("softmax概率:", probs)
            logits_list.append(logits.squeeze(0).cpu().numpy())
            pred = torch.argmax(logits, dim=1).item()
            elapsed = time.time() - start_time
            print(f"样本 {t_dict.get('question_index', '?')} 推理耗时: {elapsed:.4f} 秒，预测类别: {pred}")
    logits_arr = np.array(logits_list)  # [5, 2]
    import torch.nn.functional as F
    probs = F.softmax(torch.tensor(logits_arr), dim=1).numpy()  # [5, 2]
    pos_probs = probs[:, 1]  # [5]
    min_prob, max_prob = pos_probs.min(), pos_probs.max()
    if max_prob - min_prob < 1e-6:
        scores = np.full_like(pos_probs, 50.0)
    else:
        scores = (pos_probs - min_prob) / (max_prob - min_prob) * 100
    final_score = float(np.mean(scores))
    print(f"五个样本情绪分数: {scores}")
    print(f"最终情绪分数: {final_score:.2f}")
    return final_score

if __name__ == "__main__":
    vision_paths = [
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/1.avi",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/2.avi",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/3.avi",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/4.avi",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/5.avi",
    ]
    audio_paths = [
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/1.wav",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/2.wav",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/3.wav",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/4.wav",
        "D:/非接触式安装程序/backend/recordings/cfy0/20250920_180902/emotion/5.wav",
    ]
    # text_dicts = [
    #     {'question_index': 1, 'question_text': '您最近两周有在担忧什么事情吗？', 'recognized_text': '没有担忧', 'audio_path': 'recordings/cfy1/20250920_004013/emotion/1.wav', 'timestamp': '2025-09-20 18:19:59'},
    #     {'question_index': 2, 'question_text': '您家庭成员会分享彼此的兴趣和爱好吗？', 'recognized_text': '会分享彼此的兴趣和爱好', 'audio_path': 'recordings/cfy1/20250920_004013/emotion/2.wav', 'timestamp': '2025-09-20 18:20:05'},
    #     {'question_index': 3, 'question_text': '您最近两周会有消极的想法吗？比如死掉或用某种方式伤害自己的念头。', 'recognized_text': '没有消息的消防', 'audio_path': 'recordings/cfy1/20250920_004013/emotion/3.wav', 'timestamp': '2025-09-20 18:20:12'},
    #     {'question_index': 4, 'question_text': '您最近两周有感到焦虑吗？', 'recognized_text': '没有焦虑', 'audio_path': 'recordings/cfy1/20250920_004013/emotion/4.wav', 'timestamp': '2025-09-20 18:20:18'},
    #     {'question_index': 5, 'question_text': '您最近两周有感到开心吗？', 'recognized_text': '有开心', 'audio_path': 'recordings/cfy1/20250920_004013/emotion/5.wav', 'timestamp': '2025-09-20 18:20:25'},
    # ]
    text_dicts = [
        {'question_index': 1, 'question_text': '您最近两周有在担忧什么事情吗？', 'recognized_text': '没有担忧', 'audio_path': 'recordings/cfy1/20250920_004013/emotion/1.wav', 'timestamp': '2025-09-20 18:19:59'},
        ]
    score = infer_emotion(
        vision_paths,
        audio_paths,
        text_dicts,
        'D:/非接触式安装程序/backend_modaltest/emotion_fatigue_infer/emotion/best_model.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print('最终情绪分数:', score)

    #   elif self.current_step == 2:  # 舒特格测试
    #         self.answer_stack.setCurrentIndex(2)
    #         self.btn_next.setVisible(False)  # 隐藏下一步按钮，由舒特格测试控件自己管理
    #         self.btn_finish.setVisible(False)
    #         if self.mic_anim.state() == QPropertyAnimation.Running:
    #             self.mic_anim.stop()

    #         #cxh
    #         text = get_recognition_results()
    #         self.infer_emotion(self._audio_paths, self._video_paths,text)
