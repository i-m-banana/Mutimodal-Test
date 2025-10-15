# 疲劳度模型后端

基于 `emotion_fatigue_infer` 中的疲劳度推理模型的WebSocket后端服务。

## 功能

- 接收RGB图像、深度图像和眼动数据
- 使用训练好的疲劳度模型进行推理
- 返回0-100的疲劳度分数

## 依赖

```bash
pip install torch numpy pillow websockets
```

## 启动服务

```bash
python main.py
```

默认监听地址: `ws://127.0.0.1:8767`

## API

### 推理请求

```json
{
    "type": "inference_request",
    "request_id": "uuid-string",
    "data": {
        "rgb_frames": ["base64_encoded_image_1", "base64_encoded_image_2", ...],
        "depth_frames": ["base64_encoded_depth_1", "base64_encoded_depth_2", ...],
        "eyetrack_samples": [[8维特征], [8维特征], ...],
        "elapsed_time": 30.5
    }
}
```

### 推理响应

```json
{
    "type": "inference_response",
    "request_id": "uuid-string",
    "model_type": "fatigue",
    "timestamp": 1234567890.123,
    "result": {
        "status": "success",
        "predictions": {
            "fatigue_score": 65.23,
            "prediction_class": 1,
            "elapsed_time": 30.5,
            "inference_time_ms": 150.5,
            "num_rgb_frames": 75,
            "num_depth_frames": 75,
            "num_eyetrack_samples": 375
        },
        "latency_ms": 152.3
    }
}
```

## 模型

- 模型文件: `model_backends/emotion_fatigue_infer/fatigue/fatigue_best_model.pt`
- 模型架构: `FatigueFaceOnlyCNN` (1D CNN)
- 输入: 
  - 面部特征: [1, 42, T] (从RGB和深度图像提取)
  - 眼动特征: [1, 8, T]
- 输出: [1, 2] (二分类logits) → 加权拟合为0-100分数
