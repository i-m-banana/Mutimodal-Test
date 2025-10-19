# Model Backends - 独立模型后端服务

本目录包含所有独立运行的AI模型后端服务。每个后端运行在独立的Python环境中,通过WebSocket与主后端通信。

---

## 📁 目录结构

```
model_backends/
├── base/                      # 抽象基类
│   └── base_backend.py        # BaseModelBackend - 所有后端的父类
├── fatigue_backend/           # 疲劳度模型后端 ✅
│   ├── main.py               # 完整实现
│   ├── requirements.txt      # 独立依赖
│   └── README.md             # 使用说明
├── emotion_backend/           # 情绪模型后端 ✅
│   ├── main.py               # 完整实现
│   ├── requirements.txt      # 独立依赖
│   └── README.md             # 使用说明
└── eeg_backend/               # 脑电模型后端 (待实现) ⚠️
```

---

## 🚀 快速开始

### 1. 创建新的模型后端

使用 `fatigue_backend` 或 `emotion_backend` 作为模板:

```bash
# 复制模板
cp -r fatigue_backend your_backend

# 修改实现
cd your_backend
# 编辑 main.py, requirements.txt, README.md
```

### 2. 实现必需的方法

继承 `BaseModelBackend` 并实现三个抽象方法:

```python
from model_backends.base.base_backend import BaseModelBackend

class YourBackend(BaseModelBackend):
    def initialize_model(self):
        """初始化模型 - 在启动时调用一次"""
        self.model = load_your_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理推理请求 - 每次请求都会调用"""
        # 1. 解码输入数据
        image_base64 = data.get("image")
        image = decode_base64_image(image_base64)
        
        # 2. 预处理
        tensor = self.preprocess(image)
        
        # 3. 推理
        with torch.no_grad():
            output = self.model(tensor)
        
        # 4. 后处理
        result = self.postprocess(output)
        
        return result
    
    def cleanup(self):
        """清理资源 - 在停止时调用"""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
```

### 3. 配置端口

在 `config/models.yaml` 中添加配置:

```yaml
model_backends:
  your_backend:
    host: "localhost"
    port: 8769  # 选择未使用的端口
    enabled: false  # 开发时设为false,部署时设为true
    reconnect_interval: 5
    max_reconnect_attempts: 10
    request_timeout: 30
```

### 4. 启动后端

```bash
cd your_backend

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境 (Windows)
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 启动后端
python main.py --host localhost --port 8769
```

---

## 🔌 WebSocket协议

所有模型后端必须遵循统一的WebSocket协议。

### 消息格式

#### 推理请求 (主后端 → 模型后端)

```json
{
  "type": "inference_request",
  "request_id": "unique-uuid",
  "data": {
    "image": "base64-encoded-image",
    "timestamp": 1234567890.123,
    // ... 其他模型特定的数据
  }
}
```

#### 推理响应 (模型后端 → 主后端)

```json
{
  "type": "inference_response",
  "request_id": "same-uuid-from-request",
  "data": {
    "prediction": 0.85,
    "confidence": 0.92,
    // ... 模型输出结果
  }
}
```

#### 错误响应

```json
{
  "type": "error",
  "request_id": "same-uuid-from-request",
  "error": "Detailed error message"
}
```

#### 健康检查

请求:
```json
{
  "type": "health_check"
}
```

响应:
```json
{
  "type": "pong",
  "status": "healthy"
}
```

---

## 📦 依赖管理

每个后端都有独立的 `requirements.txt`:

### 必需依赖

```txt
# WebSocket通信
websockets>=10.0

# 如果需要访问基类
# 注意: 基类文件应该在Python路径中
```

### 框架特定依赖

**PyTorch后端**:
```txt
torch>=1.9.0
torchvision>=0.10.0
```

**TensorFlow后端**:
```txt
tensorflow>=2.6.0
# 或者
tensorflow-gpu>=2.6.0
```

**脑电处理后端**:
```txt
mne>=0.24.0
numpy>=1.19.0
scipy>=1.5.0
```

### 通用工具依赖

```txt
pillow>=8.0.0  # 图像处理
numpy>=1.19.0  # 数组操作
opencv-python>=4.5.0  # 计算机视觉
```

---

## 🗂️ 项目模板

每个模型后端应该包含以下文件:

```
your_backend/
├── main.py                 # 主入口文件
├── requirements.txt        # Python依赖
├── README.md               # 使用说明
├── model/                  # 模型文件目录
│   ├── weights.pth        # 模型权重
│   └── config.yaml        # 模型配置
└── utils/                  # 工具函数
    ├── preprocess.py      # 预处理
    └── postprocess.py     # 后处理
```

### main.py 模板

```python
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from model_backends.base.base_backend import BaseModelBackend

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YourBackend(BaseModelBackend):
    """您的模型后端实现"""
    
    def initialize_model(self):
        """初始化模型"""
        logger.info("Initializing model...")
        # TODO: 加载模型
        logger.info("Model initialized successfully")
    
    def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理推理请求"""
        logger.info("Processing inference request")
        # TODO: 实现推理逻辑
        result = {"prediction": 0.0}
        return result
    
    def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up resources")
        # TODO: 清理模型

def main():
    parser = argparse.ArgumentParser(description='Your Backend Server')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind')
    parser.add_argument('--port', type=int, default=8769, help='Port to bind')
    args = parser.parse_args()
    
    backend = YourBackend(host=args.host, port=args.port)
    
    try:
        logger.info(f"Starting YourBackend on ws://{args.host}:{args.port}")
        backend.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        backend.stop()

if __name__ == "__main__":
    main()
```

---

## 🧪 测试

### 单独测试后端

使用Python脚本测试WebSocket连接:

```python
import asyncio
import websockets
import json

async def test_backend():
    uri = "ws://localhost:8769"
    
    async with websockets.connect(uri) as websocket:
        # 1. 健康检查
        await websocket.send(json.dumps({
            "type": "health_check"
        }))
        response = await websocket.recv()
        print(f"Health check: {response}")
        
        # 2. 推理请求
        await websocket.send(json.dumps({
            "type": "inference_request",
            "request_id": "test-123",
            "data": {
                "image": "base64-encoded-image"
            }
        }))
        response = await websocket.recv()
        print(f"Inference result: {response}")

asyncio.run(test_backend())
```

### 集成测试

运行架构测试:

```bash
cd project-root
python tests/test_model_backend_architecture.py
```

---

## 📊 性能优化

### 1. 模型优化

- **量化**: INT8量化可减少模型大小和推理时间
- **剪枝**: 移除不重要的权重
- **蒸馏**: 训练更小的学生模型

### 2. 批处理

如果支持,可以批量处理多个请求:

```python
def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
    images = data.get("images", [data.get("image")])  # 支持单个或批量
    batch = self.preprocess_batch(images)
    results = self.model(batch)
    return self.postprocess_batch(results)
```

### 3. GPU优化

```python
def initialize_model(self):
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
        # 启用cudnn自动优化
        torch.backends.cudnn.benchmark = True
    else:
        self.device = torch.device("cpu")
    
    self.model.to(self.device)
    self.model.eval()
    
    # 预热GPU
    dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
    with torch.no_grad():
        _ = self.model(dummy_input)
```

### 4. 缓存

对于重复的预处理结果可以缓存:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def preprocess(self, image_hash):
    # 预处理逻辑
    pass
```

---

## 🐛 调试技巧

### 1. 详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
    logger.debug(f"Received data keys: {data.keys()}")
    logger.debug(f"Image size: {len(data.get('image', ''))}")
    # ... 推理逻辑
    logger.debug(f"Result: {result}")
    return result
```

### 2. 性能分析

```python
import time

def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()
    
    # 预处理
    t1 = time.time()
    image = self.preprocess(data)
    logger.info(f"Preprocess: {(time.time() - t1)*1000:.2f}ms")
    
    # 推理
    t2 = time.time()
    output = self.model(image)
    logger.info(f"Inference: {(time.time() - t2)*1000:.2f}ms")
    
    # 后处理
    t3 = time.time()
    result = self.postprocess(output)
    logger.info(f"Postprocess: {(time.time() - t3)*1000:.2f}ms")
    
    logger.info(f"Total: {(time.time() - start)*1000:.2f}ms")
    return result
```

### 3. 内存监控

```python
import torch

def cleanup(self):
    if hasattr(self, 'model'):
        del self.model
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory before cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        torch.cuda.empty_cache()
        logger.info(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
```

---

## 📚 相关文档

- **架构文档**: `../MULTI_MODEL_BACKEND_ARCHITECTURE.md`
- **快速接入**: `../MULTI_MODEL_BACKEND_QUICKSTART.md`
- **接口说明**: `../MULTI_MODEL_BACKEND_INTERFACES.md`
- **完整性检查**: `../MULTI_MODEL_BACKEND_CHECKLIST.md`
- **快速参考**: `../MULTI_MODEL_BACKEND_QUICK_REF.md`

---

## 🔗 端口分配

| 后端 | 端口 | 状态 | 框架 |
|-----|------|------|------|
| 主后端 | 8765 | ✅ 运行中 | - |
| fatigue | 8767 | ✅ 已实现 | PyTorch |
| emotion | 8768 | ✅ 已实现 | PyTorch + Transformers |
| eeg | 8769 | ⚠️ 待实现 | MNE-Python |
| (预留) | 8770+ | - | - |

---

## 💡 最佳实践

1. **独立虚拟环境**: 每个后端使用独立的venv
2. **明确的日志**: 使用logger而不是print
3. **优雅关闭**: 处理KeyboardInterrupt,清理资源
4. **错误处理**: 捕获异常,返回error消息
5. **配置优于硬编码**: 使用配置文件或命令行参数
6. **版本控制**: 在requirements.txt中固定依赖版本
7. **文档完善**: 保持README.md更新

---

## ❓ 常见问题

### Q: 如何选择端口?
A: 使用8766-8999范围,避免常用端口冲突。

### Q: 如何处理大图像?
A: 在发送前压缩,或使用更高效的编码 (如JPEG而非PNG)。

### Q: 如何支持多GPU?
A: 使用DataParallel或DistributedDataParallel:
```python
self.model = torch.nn.DataParallel(self.model)
```

### Q: 如何监控后端状态?
A: 主后端会定期发送health_check请求。

### Q: 如何处理模型更新?
A: 停止后端,更新模型文件,重启后端。考虑使用热重载机制。

---

**最后更新**: 2025-01-XX
**维护者**: GitHub Copilot
