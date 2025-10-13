# 多模态模型后端

这是一个独立的Python服务,用于运行多模态疲劳检测模型。

## 功能

- 接收RGB图像、深度图像和眼动数据
- 执行多模态模型推理
- 返回疲劳度分数、注意力水平、姿态状态等指标

## 环境要求

- Python 3.9+
- PyTorch 1.13+

## 安装

1. 创建虚拟环境(推荐):
```bash
python -m venv venv_multimodal
venv_multimodal\Scripts\activate  # Windows
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 配置

模型后端配置在主后端的 `config/models.yaml` 中:

```yaml
model_backends:
  - name: multimodal_backend
    type: multimodal
    enabled: true
    connection:
      host: 127.0.0.1
      port: 8766
```

## 启动服务

单独启动:
```bash
python main.py
```

或通过主后端启动脚本:
```bash
python scripts/start_all_backends.py
```

## 接口协议

### 输入消息格式

```json
{
    "type": "inference_request",
    "request_id": "uuid-xxx",
    "model_type": "multimodal",
    "timestamp": 1234567890.123,
    "data": {
        "rgb_frame": "base64_encoded_jpeg_image",
        "depth_frame": "base64_encoded_depth_image",
        "metadata": {
            "frame_count": 123,
            "timestamp": "2025-10-13T00:00:00"
        }
    }
}
```

### 输出消息格式

```json
{
    "type": "inference_response",
    "request_id": "uuid-xxx",
    "model_type": "multimodal",
    "timestamp": 1234567890.456,
    "result": {
        "status": "success",
        "predictions": {
            "fatigue_score": 0.75,
            "attention_level": 0.82,
            "pose_status": "good",
            "blink_frequency": 15.5
        },
        "latency_ms": 123.45
    }
}
```

## 模型集成

### 替换模拟模型

在 `main.py` 中找到 `initialize_model` 方法:

```python
async def initialize_model(self) -> None:
    # 删除模拟代码
    # self.model = "模拟模型"
    
    # 加载实际模型
    self.model = torch.jit.load("models/multimodal_fatigue_v1.pt")
    self.model.eval()
    
    # 如果使用GPU
    if torch.cuda.is_available():
        self.model = self.model.cuda()
```

### 实现推理逻辑

在 `process_inference` 方法中:

```python
async def process_inference(self, data):
    # ... 图像预处理 ...
    
    # 模型推理
    with torch.no_grad():
        output = self.model(rgb_tensor, depth_tensor)
        
    # 后处理
    fatigue_score = output['fatigue'].sigmoid().item()
    attention_level = output['attention'].sigmoid().item()
    
    return {
        "fatigue_score": fatigue_score,
        "attention_level": attention_level,
        # ...
    }
```

## 测试

测试WebSocket连接:
```python
import asyncio
import websockets
import json

async def test():
    uri = "ws://127.0.0.1:8766"
    async with websockets.connect(uri) as ws:
        # 发送健康检查
        await ws.send(json.dumps({"type": "health_check"}))
        response = await ws.recv()
        print(response)

asyncio.run(test())
```

## 故障排查

### 端口被占用
```bash
netstat -ano | findstr 8766
taskkill /PID <pid> /F
```

### 内存不足
- 使用更小的batch size
- 使用模型量化
- 使用GPU推理

### 连接超时
- 检查防火墙设置
- 确认主后端配置正确
- 查看日志文件

## 性能优化

1. **使用GPU加速**:
```python
if torch.cuda.is_available():
    self.model = self.model.cuda()
    rgb_tensor = rgb_tensor.cuda()
```

2. **批处理**:
累积多个请求后批量推理

3. **模型量化**:
```python
self.model = torch.quantization.quantize_dynamic(
    self.model, {torch.nn.Linear}, dtype=torch.qint8
)
```

4. **ONNX导出** (更快推理):
```python
import onnxruntime as ort
self.session = ort.InferenceSession("model.onnx")
```

## 许可证

与主项目保持一致
