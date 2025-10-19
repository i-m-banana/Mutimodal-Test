# 多模态人员状态评估平台

一个基于事件驱动架构的多模态数据采集与智能分析平台，专注于非接触式人员状态评估。

## 🎯 核心功能

- **疲劳度检测**: 基于视频、音频、深度图的多模态分析
- **情绪识别**: 面部表情、语音、文本的7类情绪分类
- **脑负荷评估**: EEG脑电信号的认知负荷等级评估
- **实时推理**: 毫秒级延迟的模型推理和状态评估
- **硬件自适应**: 自动检测硬件并支持模拟模式

## 🏗️ 系统架构

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   UI前端 (Qt)   │ ←─────────────→ │   后端服务      │
│                 │                 │                 │
│ • 登录/校准页面  │                 │ • 事件总线      │
│ • 测试界面      │                 │ • 模型推理      │
│ • 数据展示      │                 │ • 设备管理      │
└─────────────────┘                 └─────────────────┘
```

## 📁 项目结构

```
project-root-cut/
├── config/                    # 配置文件
│   ├── models.yaml           # 模型配置
│   └── interfaces.yaml       # 接口配置
├── src/                      # 后端服务
│   ├── core/                 # 核心组件（事件总线、调度器）
│   ├── models/               # 推理模型
│   ├── services/             # 业务服务
│   ├── devices/              # 硬件设备抽象
│   └── interfaces/           # 通信接口
├── ui/                       # 前端界面
│   ├── app/                  # 应用核心
│   ├── services/             # UI服务层
│   ├── widgets/              # UI组件
│   └── data/                 # 用户数据
├── models_data/              # 模型权重文件
├── requirements.txt          # 依赖列表
└── recordings/               # 录制数据（运行时生成）
```

## 🚀 快速开始

### 1. 环境要求
- Python 3.11+
- 4GB+ RAM
- （可选）CUDA GPU

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动系统

**启动后端服务**:
```bash
python -m src.main --root .
```

**启动UI应用**:
```bash
python -m ui.main
```

### 4. 模拟模式（无硬件）
```bash
# 设置环境变量启用模拟
set UI_FORCE_SIMULATION=1
set BACKEND_EEG_SIMULATION=1

# 启动系统
python -m src.main --root .
python -m ui.main
```

## ⚙️ 配置说明

### 环境变量
| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `UI_FORCE_SIMULATION` | UI全局模拟模式 | 0 |
| `BACKEND_EEG_SIMULATION` | 后端EEG模拟 | 0 |
| `UI_DEBUG_MODE` | UI调试模式 | 0 |
| `UI_SKIP_DATABASE` | 跳过数据库连接 | 0 |

### 模型配置 (config/models.yaml)
```yaml
inference_models:
  - name: fatigue
    type: fatigue
    mode: integrated
    enabled: true
  - name: emotion
    type: emotion
    mode: integrated
    enabled: true
  - name: eeg
    type: eeg
    mode: integrated
    enabled: true
```

## 🔧 开发指南

### 添加新模型
1. 继承 `BaseInferenceModel` 类
2. 实现 `initialize()` 和 `infer()` 方法
3. 在 `config/models.yaml` 中注册

### 添加新设备
1. 继承 `BaseDevice` 类
2. 实现设备特定的连接和数据采集逻辑
3. 在相应的服务中集成

## 📊 技术栈

- **后端**: Python 3.11+, PyTorch, ONNX Runtime
- **前端**: PyQt5, QtAwesome
- **通信**: WebSocket
- **数据处理**: NumPy, OpenCV, PyAudio
- **AI模型**: RoBERTa, TimesFormer, Wav2Vec2


---

**版本**: 0.1.0  
**最后更新**: 2025年10月