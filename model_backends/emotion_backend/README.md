# 情绪模型后端

## 概述

基于音视频和文本的多模态情绪识别后端服务。

## 功能

- 音频情感特征提取
- 视频面部表情识别
- 文本语义情感分析
- 多模态融合评分

## 环境要求

- Python 3.9+
- PyTorch 1.13+
- transformers 4.30+
- librosa 0.10+
- opencv-python 4.7+

## 安装

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python main.py --host 127.0.0.1 --port 8767
```

## 接口规范

### 输入格式

```json
{
    "audio_paths": [
        "/path/to/audio1.wav",
        "/path/to/audio2.wav"
    ],
    "video_paths": [
        "/path/to/video1.avi",
        "/path/to/video2.avi"
    ],
    "text_data": [
        {
            "question_index": 1,
            "question_text": "您最近两周有在担忧什么事情吗？",
            "recognized_text": "没有担忧",
            "audio_path": "recordings/.../1.wav",
            "timestamp": "2025-09-20 18:19:59"
        }
    ]
}
```

### 输出格式

```json
{
    "emotion_score": 0.72,
    "emotion_label": "neutral",
    "audio_score": 0.68,
    "video_score": 0.75,
    "text_score": 0.73,
    "confidence": 0.88,
    "inference_time_ms": 156
}
```

## TODO

- [ ] 实现音频特征提取模型
- [ ] 实现视频表情识别模型
- [ ] 实现文本情感分析模型
- [ ] 实现多模态融合模型
- [ ] 添加模型文件管理
- [ ] 优化推理性能
