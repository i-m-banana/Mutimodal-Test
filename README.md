# 多模态人员状态评估平台

基于事件驱动架构的多模态数据采集与分析平台，集成 EEG 脑电、RGB 视频等传感器，用于非接触式人员疲劳度、情绪状态与认知负荷评估。

## 🎯 核心功能

### 疲劳度评估
- **RGB 疲劳检测**: 基于视频的面部疲劳特征识别
- **EEG 疲劳分析**: 基于脑电信号的疲劳度离线分析（SART 实验结束后）
- **融合策略**: EEG (70%) + RGB (30%) 加权融合，支持自适应权重调整

### 情绪识别
- **多模态输入**: 支持语音、文本情感分析
- **情绪分类**: 7 类情绪识别（快乐、悲伤、愤怒、恐惧、厌恶、惊讶、中性）

### 脑负荷评估
- **EEG 分析**: 认知负荷三级分类（低、中、高）
- **特征提取**: 频域功率、熵值、非线性特征

### 实验任务
- **SART 实验**: 持续注意反应测试（5分钟/25分钟可选）
- **朗读录音**: 语音采集与文本显示
- **舒尔特方格**: 注意力测试
- **血压测量**: 生理指标记录

### 系统特性
- **事件驱动架构**: 发布-订阅模式，松耦合设计
- **模拟模式**: 无硬件设备时可使用模拟数据
- **数据持久化**: 自动保存实验数据（CSV、视频、音频）
- **WebSocket 通信**: 前后端实时数据交互

## 🏗️ 系统架构

```
┌─────────────────────────────────────────┐
│         UI 前端 (PyQt5)                 │
│  • 登录  • SART  • 测试任务  • 结果    │
└──────────────┬──────────────────────────┘
               │ WebSocket (ws://localhost:8765)
┌──────────────┴──────────────────────────┐
│         后端服务 (Python)                │
│  ┌────────────────────────────────┐    │
│  │    事件总线 (EventBus)          │    │
│  │  发布-订阅 | 异步解耦           │    │
│  └─────┬──────────┬────────────┬──┘    │
│        ↓          ↓            ↓        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
│  │ 推理层  │ │ 服务层  │ │ 接口层  │  │
│  │ • 模型  │ │ • 融合  │ │WebSocket│  │
│  └─────────┘ └─────────┘ └─────────┘  │
└─────────────────────────────────────────┘
```

### 核心特性
- **事件驱动**: 发布-订阅模式，组件解耦
- **模块化设计**: 模型、服务、设备分层管理
- **异步处理**: 推理任务异步执行，不阻塞主流程

## 📁 项目结构

```
Mutimodal-Test/
├── config/                    # 配置文件
│   ├── models.yaml           # 模型配置
│   └── interfaces.yaml       # 接口配置
│
├── src/                      # 后端服务
│   ├── core/                 # 核心组件
│   │   ├── event_bus.py      # 事件总线
│   │   └── orchestrator.py   # 服务编排器
│   ├── models/               # AI 模型
│   │   ├── rgb_fatigue_model.py   # RGB疲劳检测
│   │   ├── eeg_fatigue_model.py   # EEG疲劳分析
│   │   ├── emotion_v2_model.py    # 情绪识别
│   │   └── eeg_model.py           # EEG脑负荷
│   ├── services/             # 业务服务
│   │   └── fatigue_assessment_service.py  # 疲劳度融合
│   ├── interfaces/           # 通信接口
│   │   └── websocket_server.py
│   └── main.py              # 后端入口
│
├── ui/                       # 前端界面
│   ├── app/                  # 应用核心
│   │   ├── pages/
│   │   │   ├── login.py      # 登录页
│   │   │   ├── sart.py       # SART实验
│   │   │   └── test.py       # 测试任务
│   │   └── application.py    # 主应用
│   ├── widgets/              # UI组件
│   │   ├── score_page.py     # 结果展示
│   │   └── schulte_grid.py   # 舒尔特方格
│   ├── services/             # UI服务
│   │   └── backend_client.py # 后端通信
│   └── main.py              # 前端入口
│
├── models_data/              # 模型文件
│   ├── eeg_models/           # EEG模型
│   ├── eeg_fatigue_models/   # EEG疲劳模型
│   └── emotion_models/       # 情绪模型
│
├── recordings/               # 录制数据（自动生成）
│   └── {用户名}/{时间戳}/
│       ├── eeg_data_*.csv
│       ├── rgb.avi
│       └── metadata.json
│
├── docs/                     # 文档
├── start.bat                 # 启动脚本
├── requirements.txt          # 依赖列表
└── README.md
```

## 🚀 快速开始

### 1. 系统要求

- **操作系统**: Windows 10/11 或 Linux
- **Python**: 3.8+
- **内存**: 4GB+ RAM
- **存储**: 5GB+ 可用空间

### 2. 安装依赖

```bash
cd d:\Mutimodal-Test
pip install -r requirements.txt
```

### 3. 启动系统

**方式一：一键启动（推荐）**

双击 `start.bat` 或运行：
```bash
start.bat
```

**方式二：手动启动**

```bash
# 终端1 - 启动后端
python -m src.main

# 终端2 - 启动前端
python -m ui.main
```

> ⚠️ 必须使用 `-m` 参数以模块方式运行

**模拟模式（无硬件）**

```bash
set UI_FORCE_SIMULATION=1
python -m src.main
python -m ui.main --debug
```

### 4. 使用流程

1. **登录**: 输入用户名
2. **SART实验**: 5分钟持续注意力测试
3. **测试任务**: 朗读录音 → 血压测量 → 舒尔特方格
4. **查看结果**: 综合评估报告

### 5. 常见问题

**Q: ModuleNotFoundError**  
A: 使用 `python -m src.main`，不要直接运行 `python src/main.py`

**Q: WebSocket 连接失败**  
A: 确保后端已启动，端口 8765 未被占用

**Q: 无法检测硬件**  
A: 使用模拟模式测试

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认 |
|------|------|------|
| `UI_FORCE_SIMULATION` | 前端模拟模式 | 0 |
| `BACKEND_EEG_SIMULATION` | 后端EEG模拟 | 0 |
| `UI_DEBUG_MODE` | 调试模式 | 0 |

### 模型配置 (config/models.yaml)

```yaml
inference_models:
  - name: rgb_fatigue      # RGB疲劳检测
    type: rgb_fatigue
    enabled: true
  
  - name: eeg_fatigue      # EEG疲劳分析（离线）
    type: eeg_fatigue
    enabled: false         # 不实时推理
  
  - name: emotion          # 情绪识别
    type: emotion
    enabled: true
  
  - name: eeg              # EEG脑负荷
    type: eeg
    enabled: true
```

### 疲劳评估权重

在 `fatigue_assessment_service.py` 中配置:
```python
EEG_WEIGHT = 0.7    # EEG权重70%
RGB_WEIGHT = 0.3    # RGB权重30%
```

## 🔧 开发指南

### 添加新模型

1. 继承 `BaseInferenceModel` 创建模型类
2. 实现 `initialize()` 和 `infer()` 方法
3. 在 `config/models.yaml` 注册

### 事件系统

**发布事件**:
```python
from src.constants import EventTopic
bus.publish(Event(topic=EventTopic.CUSTOM, payload=data))
```

**订阅事件**:
```python
def handler(event):
    print(event.payload)

bus.subscribe(EventTopic.CUSTOM, handler)
```

### 调试

```bash
# 显示所有事件
python -m src.main --full-events

# 模拟模式
set UI_FORCE_SIMULATION=1
python -m ui.main --debug
```

## 📊 技术栈

**后端**
- Python 3.8+, PyYAML, WebSockets
- NumPy, OpenCV, PyAudio
- Scikit-learn (可选: PyTorch, ONNX)

**前端**
- PyQt5, QtAwesome

**硬件接口**
- PySerial (血压计)
- Faster-Whisper (语音识别)

## 📚 文档

- [快速启动指南](docs/QUICK_START.md)
- [系统架构](docs/ARCHITECTURE.md)

## 📄 许可证

MIT License

---

**更新时间**: 2025年11月