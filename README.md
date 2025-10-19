# 多模态人员状态评估平台

## 一、项目概述

这是一个**模块化、配置驱动、可扩展**的多模态数据采集与智能分析平台，专注于非接触式人员状态评估。系统通过整合**音视频、脑电（EEG）、眼动、深度相机**等多种传感器数据，实时推理用户的**疲劳度、情绪、脑负荷**等认知状态，为人机交互、驾驶安全、工作效能评估等场景提供技术支持。

### 核心特性

- 🎯 **配置驱动架构**: 通过 YAML 文件灵活配置采集器、检测器、模型参数
- 🔌 **模块化设计**: 基于事件总线的松耦合架构，组件可独立开发和测试
- 🚀 **集成部署**: 模型直接集成在后端进程中，高性能低延迟
- 🧠 **多模态融合**: 统一管理视觉、音频、生理信号等多源数据
- 📊 **实时推理**: 毫秒级延迟的模型推理和状态评估
- 🔧 **硬件自适应**: 自动检测硬件并无缝切换到模拟模式
- 🎨 **响应式UI**: Qt界面适配多种分辨率（1280x720 到 4K）


## 二、技术亮点

### 2.1 智能推理系统

| 模型类型 | 输入模态 | 输出指标 | 技术栈 |
|---------|---------|---------|--------|
| **疲劳度检测** | 视频、音频、深度图 | 疲劳度评分 (0-100) | PyTorch、多模态特征融合 |
| **情绪识别** | 面部表情、语音、文本 | 7类情绪分类 + 效价/唤醒度 | RoBERTa、TimesFormer、Wav2Vec2 |
| **脑负荷评估** | EEG脑电信号 | 认知负荷等级 (低/中/高) | 机器学习特征工程 |

### 2.2 系统架构优势

| 特性 | 说明 | 价值 |
|------|------|------|
| **事件驱动** | 基于发布-订阅模式的事件总线 | 组件解耦、易扩展 |
| **异步处理** | 采集器、检测器、推理服务并行运行 | 高吞吐、低延迟 |
| **配置热更新** | YAML配置修改后自动生效 | 快速调试、灵活调优 |
| **高性能推理** | 模型直接集成，无网络开销 | 实时响应，低延迟 |
| **标准化通信** | WebSocket统一协议 | 跨语言、跨平台 |
| **完整日志** | 分模块日志记录 + 性能统计 | 可观测性强 |


## 三、系统架构


本系统采用事件驱动架构，所有数据采集、推理、展示与存储均基于实际代码实现，主要流程如下：

1. **音视频数据**：由 `AVService` 负责，直接通过摄像头和麦克风采集，支持预览和录制，录制后保存为本地文件（.avi/.wav），并通过事件总线实时推送帧和音量信息。
2. **脑电（EEG）数据**：由 UI 层通过 `eeg_start_collection` 等接口发起采集，采集结果通过事件总线推送，最终用于推理。
3. **血压、舒特格等其他数据**：同样由 UI 层通过相关接口采集，采集结果直接在 UI 展示或存储。
4. **统一推理服务**：`UnifiedInferenceService` 负责所有模型推理，包括疲劳度、情绪、脑负荷等。推理服务支持集成和远程两种模式，自动根据配置加载模型。多模态数据通过事件总线推送到推理服务，推理结果通过 `DETECTION_RESULT` 事件推送回 UI。
5. **UI 展示**：`TestPage` 负责所有数据的实时展示，包括疲劳度、脑负荷分数的实时显示、音视频预览、舒特格测试界面、血压结果等。
6. **数据存储**：`DatabaseService` 负责将测试结果、分数、音视频路径、识别文本等写入数据库（MySQL），音视频、语音识别文本等均保存为本地文件。
7. **事件总线**：`EventBus` 实现了线程安全的发布/订阅机制，所有数据采集、推理、结果分发均通过事件总线完成，确保 UI、服务、模型之间解耦。

各类数据均遵循“采集 → 推理 → 展示/存储”三步流程，具体实现详见上述相关代码文件。


## 四、目录结构

```
project-root-cut/
├── 📋 配置文件
│   ├── config/                      # 系统配置（YAML格式）
│   │   ├── system.yaml             # 全局配置（心跳、超时、UI自动启动）
│   │   ├── collectors.yaml         # 采集器配置（采样率、设备参数）
│   │   ├── detectors.yaml          # 检测器配置（阈值、订阅主题）
│   │   ├── models.yaml             # 模型配置（部署模式、路径、启用状态）
│   │   └── interfaces.yaml         # 接口配置（WebSocket地址端口）
│   ├── pyproject.toml              # 项目元数据和依赖（Poetry）
│   ├── requirements.txt            # Python依赖列表（pip）
│   └── AGENTS.MD                   # 开发规范和编码准则
│
├── 🔧 后端服务 (src/)
│   ├── core/                       # 核心组件
│   │   ├── orchestrator.py         # 系统调度中心
│   │   ├── event_bus.py            # 事件总线（发布-订阅）
│   │   ├── system_monitor.py       # 系统监控和心跳
│   │   └── thread_pool.py          # 线程池管理
│   ├── collectors/                 # 数据采集器
│   │   ├── base_collector.py       # 采集器基类
│   │   ├── camera_collector.py     # 摄像头采集
│   │   ├── sensor_collector.py     # 多模态传感器
│   │   └── file_collector.py       # 文件数据源
│   ├── detectors/                  # 智能检测器
│   │   ├── base_detector.py        # 检测器基类
│   │   ├── object_detector.py      # 目标检测
│   │   ├── anomaly_detector.py     # 异常检测
│   │   └── ocr_detector.py         # 文字识别
│   ├── models/                     # 集成模型（integrated模式）
│   │   ├── base_inference_model.py # 推理模型基类
│   │   ├── fatigue_model.py        # 疲劳度检测模型
│   │   ├── emotion_model.py        # 情绪识别模型
│   │   ├── eeg_model.py            # 脑电分析模型
│   │   └── model_manager.py        # 模型管理器
│   ├── services/                   # 后端服务
│   │   ├── unified_inference_service.py  # 统一推理调度
│   │   ├── eeg_service.py          # EEG设备管理
│   │   ├── av_service.py           # 音视频录制
│   │   ├── multimodal_service.py   # 多模态数据融合
│   │   ├── ui_command_router.py    # UI命令路由
│   │   ├── tts_service.py          # 文字转语音
│   │   └── database.py             # 数据库访问
│   ├── interfaces/                 # 通信接口
│   │   ├── base.py                 # 接口基类
│   │   ├── websocket_server.py     # WebSocket服务器
│   │   └── model_ws_client.py      # 模型后端客户端
│   ├── devices/                    # 硬件设备抽象
│   │   ├── base.py                 # 设备基类
│   │   ├── eeg.py                  # EEG脑电设备（BLE）
│   │   ├── tobii.py                # Tobii眼动仪
│   │   └── maibobo.py              # 麦博麦克风
│   ├── utils/                      # 工具类
│   │   └── logger.py               # 日志配置
│   ├── constants.py                # 常量定义
│   └── main.py                     # 后端入口
│
├── 🖥️ 前端界面 (ui/)
│   ├── app/                        # 应用核心
│   │   ├── application.py          # 主窗口和应用启动
│   │   ├── config.py               # UI配置管理
│   │   ├── qt.py                   # Qt组件导入封装
│   │   ├── pages/                  # 页面组件
│   │   │   ├── login.py            # 登录页面
│   │   │   ├── calibration.py      # 校准页面
│   │   │   └── test.py             # 测试页面
│   │   └── utils/                  # UI工具
│   │       ├── responsive.py       # 响应式缩放
│   │       └── widgets.py          # 自定义控件
│   ├── services/                   # UI服务层
│   │   ├── backend_client.py       # 后端WebSocket客户端
│   │   ├── backend_proxy.py        # 后端服务代理
│   │   └── backend_launcher.py     # 后端启动器
│   ├── widgets/                    # UI组件库
│   │   ├── brain_load_bar.py       # 脑负荷条
│   │   ├── multimodal_preview.py   # 多模态预览
│   │   └── schulte_grid.py         # 舒尔特方格
│   ├── data/                       # UI资源数据
│   │   ├── users/                  # 用户数据（CSV）
│   │   └── questionnaires/         # 问卷配置（YAML）
│   ├── utils_common/               # 通用工具
│   │   └── thread_process_manager.py  # 线程/进程管理
│   ├── runtime/                    # 运行时组件
│   │   └── ui_thread_pool.py       # UI线程池
│   ├── style.qss                   # Qt样式表
│   └── main.py                     # UI入口
│
├── 🤖 模型后端 (model_backends/)  # 已移除，现使用集成模式
│   ├── base/                       # 模型后端基类
│   │   └── base_backend.py         # WebSocket服务器框架
│   ├── fatigue_backend/            # 疲劳度模型后端
│   │   ├── main.py                 # 疲劳度服务入口
│   │   ├── requirements.txt        # 依赖列表
│   │   └── README.md               # 使用说明
│   ├── emotion_backend/            # 情绪模型后端
│   │   ├── main.py                 # 情绪服务入口
│   │   ├── requirements.txt
│   │   └── README.md
│   └── eeg_backend/                # 脑电模型后端
│       ├── main.py                 # EEG服务入口
│       ├── requirements.txt
│       └── brain_load/             # 脑负荷计算模块
│
├── 📊 模型资源 (models_data/)
│   ├── fatigue_models/             # 疲劳度模型权重
│   ├── emotion_models/             # 情绪模型权重
│   ├── eeg_models/                 # EEG模型权重
│   └── emotion_pretrained_models/  # 预训练模型
│       ├── ROBBERTA/               # RoBERTa文本模型
│       ├── TIMESFORMER/            # TimesFormer视频模型
│       └── WAV2VEC2/               # Wav2Vec2语音模型
│
├── 📝 文档
│   ├── README.md                   # 项目总览（本文件）
│   ├── QUICK_START.md              # 快速启动指南
│   └── AGENTS.MD                   # 开发规范
│
└── 🗂️ 运行时目录（自动生成，不提交Git）
    ├── logs/                       # 日志文件
    │   ├── orchestrator.log        # 后端主日志
    │   ├── collector/*.log         # 采集器日志
    │   ├── detector/*.log          # 检测器日志
    │   ├── model/*.log             # 模型推理日志
    │   └── interface/*.log         # 接口日志
    └── recordings/                 # 测试录制数据
        └── [用户名]/[时间戳]/     # 按用户和时间组织
```


## 五、快速开始

### 5.1 环境准备

**系统要求**:
- Python 3.11+
- Windows 10/11 / Linux / macOS
- 4GB+ RAM
- （可选）CUDA GPU（用于加速深度学习推理）

**安装步骤**:

```bash
# 1. 克隆仓库
git clone <repository-url>
cd project-root-cut

# 2. 创建虚拟环境（推荐）
python -m venv .venv

# 3. 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. 安装核心依赖
pip install -r requirements.txt

# 5. 安装UI依赖（如果需要使用图形界面）
pip install pyqt5 qtawesome opencv-python pyaudio numpy matplotlib

# 6. （可选）安装深度学习依赖
pip install torch torchvision torchaudio faster-whisper opencc
```

### 5.2 启动系统

#### ⚡ 标准启动流程（推荐）

**步骤1: 启动后端服务**
```bash
cd project-root-cut
python -m src.main --root .
```

**期望输出**:
```
INFO  orchestrator | Starting orchestrator
INFO  WebsocketPushInterface | Starting WebSocket interface on 127.0.0.1:8765
INFO  inference | ✅ 统一推理服务已启动 (共 3 个模型)
```

**步骤2: 启动UI应用**
```bash
# 新开一个终端窗口
cd project-root-cut
python -m ui.main
```

**期望输出**:
```
应用程序主窗口初始化完成
已成功连接到后端服务器 ws://127.0.0.1:8765
```

#### 🔧 开发/调试模式

**使用模拟数据（无需硬件）**:
```bash
# 后端启用EEG模拟
set BACKEND_EEG_SIMULATION=1
python -m src.main --root .

# UI启用全局模拟
set UI_FORCE_SIMULATION=1
python -m ui.main --debug
```

**减少日志输出**:
```bash
# 后端：仅显示关键信息
python -m src.main --root . --no-listeners

# 后端：显示全部事件日志
python -m src.main --root . --full-events
```

### 5.3 启动UI应用# 启动服务（监听8767端口）
python main.py
```

**注意**: 集成模式（`mode: integrated`）不需要此步骤，模型会自动在后端主进程中加载。

### 5.4 验证安装

**检查后端状态**:
```bash
# Windows
netstat -ano | findstr :8765

# Linux/macOS
lsof -i :8765
```

**测试WebSocket连接**:
```bash
# 使用wscat工具（需先安装：npm install -g wscat）
wscat -c ws://127.0.0.1:8765

# 发送测试消息
{"type": "ping"}
```

**查看日志**:
```bash
# 后端日志
type logs\orchestrator.log  # Windows
cat logs/orchestrator.log   # Linux/macOS

# 模型推理日志
type logs\model\model.log
```


## 六、配置指南

### 6.1 环境变量

| 变量名 | 说明 | 默认值 | 使用场景 |
|--------|------|--------|----------|
| `UI_FORCE_SIMULATION` | UI全局模拟模式 | 0 | 无硬件环境开发 |
| `BACKEND_EEG_SIMULATION` | 后端EEG模拟 | 0 | 无EEG设备测试 |
| `UI_DEBUG` | UI调试模式 | 0 | 前端问题排查 |
| `BACKEND_KEY_INFO_MODE` | 后端简化日志 | 1 | 减少控制台输出 |
| `UI_SKIP_DATABASE` | 跳过数据库连接 | 0 | 演示模式 |
| `UI_CAMERA_INDEX` | 摄像头索引 | 0 | 多摄像头切换 |
| `UI_DB_HOST` | 数据库地址 | localhost | 自定义数据库 |

**设置方法**:
```bash
# Windows (临时)
set UI_FORCE_SIMULATION=1

# Windows (永久)
setx UI_FORCE_SIMULATION 1

# Linux/macOS (临时)
export UI_FORCE_SIMULATION=1

# Linux/macOS (永久，添加到 ~/.bashrc)
echo 'export UI_FORCE_SIMULATION=1' >> ~/.bashrc
```

### 6.2 配置文件详解

#### system.yaml - 系统全局配置
```yaml
heartbeat_interval: 2.0          # 心跳间隔（秒）
enable_monitor: true             # 是否启用系统监控
collector_stop_timeout: 3.0      # 采集器停止超时（秒）

ui:
  auto_start_backend: false      # UI是否自动启动后端
```

#### collectors.yaml - 采集器配置
```yaml
collectors:
  - name: camera
    class: collectors.camera_collector.CameraCollector
    enabled: true
    options:
      sample_rate: 30            # 采样率（Hz）
      camera_index: 0            # 摄像头索引
```

#### detectors.yaml - 检测器配置
```yaml
detectors:
  - name: object_detector
    class: detectors.object_detector.ObjectDetector
    enabled: true
    options:
      confidence_threshold: 0.5  # 置信度阈值
      subscribed_topics:         # 订阅的事件主题
        - camera.frame
```

#### models.yaml - 模型配置
```yaml
inference_models:
  - name: fatigue
    type: fatigue
    mode: integrated             # integrated 或 remote
    enabled: true
    integrated:
      class: models.fatigue_model.FatigueModel
      options:
        device: auto             # auto, cuda, cpu
    remote:
      host: 127.0.0.1
      port: 8767
      timeout: 10.0
```

**集成模式 vs 远程模式选择**:
- 开发测试 → `mode: integrated` （一键启动，性能最优）
- 生产部署 → `mode: remote` （环境隔离，故障隔离）
- 多服务器 → `mode: remote` （分布式推理）

#### interfaces.yaml - 通信接口配置
```yaml
interfaces:
  - name: websocket
    class: interfaces.websocket_server.WebsocketPushInterface
    enabled: true
    options:
      host: 127.0.0.1
      port: 8765
      topics:                    # 推送的事件主题
        - detector.result
        - system.heartbeat
        - camera.frame
```

### 6.3 常见配置场景

**场景1: 完全离线模式（无任何硬件）**
```bash
# 环境变量
set UI_FORCE_SIMULATION=1
set BACKEND_EEG_SIMULATION=1
set UI_SKIP_DATABASE=1

# 启动
python -m src.main --root .
python -m ui.main --debug
```

**场景2: 仅模拟EEG设备**
```bash
set BACKEND_EEG_SIMULATION=1
python -m src.main --root .
```

**场景3: 禁用某个模型**
```yaml
inference_models:
  - name: emotion
    enabled: false         # 设置为false即可
```


## 七、开发指南

### 7.1 扩展新的数据采集器

**步骤1**: 创建采集器类

```python
# src/collectors/my_collector.py
from .base_collector import BaseCollector
from ..constants import EventTopic
from ..core.event_bus import Event

class MyCollector(BaseCollector):
    def run_once(self) -> None:
        """采集数据并发布事件"""
        # 1. 采集数据
        data = self._collect_data()
        
        # 2. 发布事件
        self.bus.publish(Event(
            topic=EventTopic.CUSTOM_DATA,
            payload={"data": data}
        ))
    
    def _collect_data(self):
        # 实现具体的采集逻辑
        return {"value": 123}
```

**步骤2**: 注册到配置文件

```yaml
# config/collectors.yaml
collectors:
  - name: my_collector
    class: collectors.my_collector.MyCollector
    enabled: true
    options:
      sample_rate: 10  # 自定义参数
      device_id: 0
```

**步骤3**: 重启后端生效

```bash
python -m src.main --root .
```

### 7.2 扩展新的推理模型

#### 方式1: 集成模式（推荐，简单快速）

**步骤1**: 创建模型类

```python
# src/models/my_model.py
from .base_inference_model import BaseInferenceModel
from typing import Dict, Any

class MyModel(BaseInferenceModel):
    def initialize(self) -> None:
        """初始化模型（加载权重等）"""
        import torch
        self.model = torch.load("models_data/my_model/model.pt")
        self.model.eval()
    
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行推理"""
        # 1. 预处理
        input_tensor = self._preprocess(data)
        
        # 2. 推理
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 3. 后处理
        result = self._postprocess(output)
        return {"prediction": result, "confidence": 0.95}
```

**步骤2**: 配置模型

```yaml
# config/models.yaml
inference_models:
  - name: my_model
    type: my_model
    mode: integrated
    enabled: true
    integrated:
      class: models.my_model.MyModel
      options:
        device: auto
```

### 7.3 扩展新的检测器

**步骤1**: 创建检测器类

```python
# src/detectors/my_detector.py
from .base_detector import BaseDetector
from ..constants import EventTopic
from ..core.event_bus import Event

class MyDetector(BaseDetector):
    def start(self) -> None:
        """订阅需要的事件"""
        self.bus.subscribe(EventTopic.CAMERA_FRAME, self.handle_event)
    
    def handle_event(self, event: Event) -> None:
        """处理事件"""
        frame = event.payload.get("frame")
        
        # 执行检测逻辑
        result = self._detect(frame)
        
        # 发布检测结果
        self.bus.publish(Event(
            topic=EventTopic.DETECTION_RESULT,
            payload={
                "detector": self.name,
                "result": result
            }
        ))
    
    def _detect(self, frame):
        # 实现检测算法
        return {"found": True, "confidence": 0.9}
```

**步骤2**: 注册检测器

```yaml
# config/detectors.yaml
detectors:
  - name: my_detector
    class: detectors.my_detector.MyDetector
    enabled: true
    options:
      threshold: 0.7
```

### 7.4 调试技巧

**启用详细日志**:
```bash
# 后端全量事件日志
python -m src.main --root . --full-events

# UI调试模式
python -m ui.main --debug
```

**实时查看日志**:
```bash
# Windows
powershell -Command "Get-Content logs\model\model.log -Wait"

# Linux/macOS
tail -f logs/model/model.log
```

**使用Python调试器**:
```python
# 在代码中插入断点
import pdb; pdb.set_trace()

# 或使用 IPython
from IPython import embed; embed()
```

**测试单个组件**:
```python
# test_my_model.py
from src.models.my_model import MyModel

model = MyModel(name="test", logger=None)
model.initialize()

result = model.infer({"input": "test_data"})
print(result)
```

### 7.5 代码规范

参考 `AGENTS.MD` 中的开发准则：

- ✅ 不要猜测API，先阅读文档或源码
- ✅ 遇到模糊需求，先确认再实现
- ✅ 优先复用现有代码，避免重复造轮子
- ✅ 所有功能必须添加验证和测试
- ✅ 遵循项目架构和约定
- ✅ 最小化修改，谨慎重构
- ✅ 避免生成不必要的文档或测试文件


## 八、硬件支持与模拟模式

### 8.1 支持的硬件设备

| 设备类型 | 型号 | 用途 | 是否必需 |
|---------|------|------|---------|
| **摄像头** | 任意USB摄像头 | 视频采集、表情识别 | 否（可模拟） |
| **深度相机** | Intel RealSense D435 | 深度图采集 | 否（可模拟） |
| **EEG设备** | 蓝牙脑电头环 | 脑电信号采集 | 否（可模拟） |
| **眼动仪** | Tobii Eye Tracker | 眼动数据采集 | 否（可模拟） |
| **麦克风** | 麦博麦克风阵列 | 音频采集、语音识别 | 否（可模拟） |

### 8.2 模拟模式说明

系统支持**完全无硬件运行**，通过模拟数据进行功能演示和开发测试：

**启用方式**:
```bash
# 全局模拟（推荐）
set UI_FORCE_SIMULATION=1
set BACKEND_EEG_SIMULATION=1
python -m ui.main --debug

# 仅模拟特定硬件
set BACKEND_EEG_SIMULATION=1      # 仅EEG模拟
set UI_MULTIMODAL_SIMULATION=1    # 仅多模态传感器模拟
```

**模拟数据特性**:
- 📷 **视频**: 渐变RGB图像（640x480）
- 🎤 **音频**: 静音音频流（16kHz采样）
- 🧠 **EEG**: 正弦波叠加高斯噪声（500Hz采样，8通道）
- 👁️ **眼动**: 随机注视点和瞳孔直径
- 📊 **传感器**: 随机波动的生理指标数据

### 8.3 硬件检测流程

```
启动采集器 → 尝试连接硬件
    │
    ├─ 成功 → 使用真实硬件数据
    │
    └─ 失败 → 自动切换到模拟模式
              ↓
         记录警告日志
              ↓
         继续正常运行
```

## 九、常见问题（FAQ）

### Q1: 模型推理结果不准确？

1. **检查模型文件**: 确认 `models_data/` 下有完整的模型权重
2. **查看推理日志**: `logs/model/model.log` 中的详细信息
3. **确认输入数据**: 检查数据格式和质量
4. **调整配置参数**: 修改 `config/models.yaml` 中的阈值等参数

### Q2: 后端服务启动失败怎么办?

**问题: 端口被占用**
```bash
# 查找占用8765端口的进程
netstat -ano | findstr :8765  # Windows
lsof -i :8765                 # Linux/macOS

# 结束进程
taskkill /F /PID <PID>        # Windows
kill -9 <PID>                 # Linux/macOS
```

**问题: 模块导入错误**
```bash
# 确保在项目根目录执行
cd d:\duomotai\project-root-cut
python -m src.main --root .

# 检查Python路径
python -c "import sys; print('\n'.join(sys.path))"
```

**问题: 配置文件找不到**
```bash
# 确认config目录存在
dir config  # Windows
ls config   # Linux/macOS

# 使用--root参数指定项目根目录
python -m src.main --root d:\duomotai\project-root-cut
```

### Q3: UI无法连接后端？

1. **确认后端已启动**: 查看控制台输出 `WebSocket interface on 127.0.0.1:8765`
2. **检查防火墙**: 允许8765端口通信
3. **查看UI日志**: `ui/logs/app_log_*.txt`
4. **测试连接**:
   ```bash
   # 使用wscat测试
   wscat -c ws://127.0.0.1:8765
   ```
5. **尝试重启**: 先停止UI和后端，再按顺序重新启动

### Q4: 模型推理结果不准确？

1. **检查模型文件**: 确认 `models_data/` 下有完整的模型权重
2. **查看推理日志**: `logs/model/model.log` 中的详细信息
3. **确认输入数据**: 检查数据格式和质量
4. **调整配置参数**: 修改 `config/models.yaml` 中的阈值等参数

### Q5: EEG设备连接失败？

**问题诊断**:
```bash
# 检查蓝牙适配器
# Windows: 设置 → 蓝牙和其他设备
# Linux: bluetoothctl

# 检查bleak库
python -c "from bleak import BleakClient; print('OK')"
```

**解决方案**:
```bash
# 1. 使用模拟模式（推荐用于开发）
set BACKEND_EEG_SIMULATION=1

# 2. 检查设备地址配置
# 编辑 src/devices/eeg.py 中的 DEFAULT_DEVICE_ADDRESS

# 3. 重新配对设备
# Windows: 移除设备后重新配对
```

### Q6: 如何查看实时推理性能？

```bash
# 查看模型推理日志（包含耗时统计）
type logs\model\model.log | findstr "推理"  # Windows
grep "推理" logs/model/model.log          # Linux/macOS

# 示例输出：
# INFO  inference.fatigue | 疲劳度推理完成: score=0.65, time=120ms
```

### Q7: 如何添加自定义用户？

编辑 `ui/data/users/users.csv`:
```csv
username,password,role,department
admin,admin123,admin,技术部
user1,pass123,user,测试部
```

### Q8: 录制数据保存在哪里？

```
recordings/
└── [用户名]/
    └── [时间戳_YYYYMMDD_HHMMSS]/
        ├── video.mp4          # 视频录制
        ├── audio.wav          # 音频录制
        ├── eeg/               # EEG数据
        │   └── eeg_data.csv
        └── multimodal/        # 多模态数据
            └── frame_*.json
```

## 十、技术栈

| 类别 | 技术 |
|------|------|
| **编程语言** | Python 3.11+ |
| **UI框架** | PyQt5 |
| **深度学习** | PyTorch, ONNX Runtime |
| **音视频处理** | OpenCV, PyAudio |
| **语音识别** | Faster-Whisper, Wav2Vec2 |
| **自然语言处理** | RoBERTa (Transformers) |
| **视频理解** | TimesFormer |
| **数据处理** | NumPy, Pandas, Matplotlib |
| **通信协议** | WebSocket (websockets) |
| **硬件接口** | Bleak (BLE), PySerial |
| **配置管理** | PyYAML |
| **日志记录** | Python logging |
| **项目管理** | Poetry, pip |

## 十一、性能指标

| 指标 | 数值 |
|------|------|
| **模型推理延迟** | < 150ms (集成模式) |
| **视频采集帧率** | 30 FPS |
| **EEG采样率** | 500 Hz |
| **WebSocket延迟** | < 10ms (本地) |
| **内存占用** | ~ 2GB (集成3个模型) |
| **CPU占用** | 20-40% (Intel i7) |

## 十二、许可证与贡献

### 许可证
本项目采用 **MIT 许可证**，允许自由使用、修改和分发，需保留版权声明。

### 贡献指南
欢迎贡献代码、报告问题或提出建议！

1. **Fork 本仓库**
2. **创建功能分支**: `git checkout -b feature/amazing-feature`
3. **提交更改**: `git commit -m 'Add amazing feature'`
4. **推送到分支**: `git push origin feature/amazing-feature`
5. **提交 Pull Request**

### 开发规范
- 遵循 `AGENTS.MD` 中的编码准则
- 添加必要的注释和文档
- 确保代码通过测试
- 保持提交信息清晰

### 联系方式
- 🐛 问题反馈: GitHub Issues
- 📧 项目维护: 联系仓库管理员
- 📚 文档更新: 提交 PR

---

**项目状态**: 🚀 积极开发中  
**最后更新**: 2025年10月16日  
**版本**: 0.1.0  
**Python要求**: 3.11+