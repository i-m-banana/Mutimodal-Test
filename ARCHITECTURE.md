# 系统架构

## 1. 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                         UI Layer (Qt)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Login Page │  │ Calib Page │  │ Test Page  │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│         │                │                │                  │
│         └────────────────┴────────────────┘                  │
│                          │                                    │
│                WebSocket Client (8765)                        │
└───────────────────────────┬───────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │  Event Bus    │
                    │ (Pub/Sub)     │
                    └───┬───────┬───┘
                        │       │
        ┌───────────────┼───────┼──────────────────┐
        │               │       │                  │
   ┌────▼────┐    ┌────▼────┐ ┌▼───────┐   ┌─────▼────┐
   │Collectors│    │Detectors│ │ Models │   │Interfaces│
   │(Audio/   │    │(Object/ │ │(Fatigue│   │(WebSocket│
   │ Video/   │    │ Anomaly)│ │Emotion)│   │ Server)  │
   │ EEG/     │    │         │ │        │   │          │
   │ Sensor)  │    │         │ └────────┘   │          │
   └──────────┘    └─────────┘               └──────────┘
                                                   │
        ┌──────────────────────────────────────────┘
        │
   ┌────▼───────────────────────────────────┐
   │  Unified Inference Service              │
   │  ┌────────────┐      ┌────────────┐   │
   │  │ Integrated │      │   Remote   │   │
   │  │   Models   │      │   Clients  │   │
   │  │  (in-proc) │      │(WebSocket) │   │
   │  └────────────┘      └──────┬─────┘   │
   └─────────────────────────────┼─────────┘
                                  │
                 ┌────────────────┴────────────────┐
                 │                                  │
        ┌────────▼────────┐              ┌─────────▼────────┐
        │ Fatigue Backend │              │ Emotion Backend  │
        │   (Port 8767)   │              │   (Port 8768)    │
        └─────────────────┘              └──────────────────┘
```

## 2. 核心组件说明

| 组件层 | 模块 | 职责 |
|--------|------|------|
| **后端核心**<br>(src/) | Orchestrator | 系统调度中心，管理所有组件生命周期 |
| | EventBus | 事件总线，实现组件间解耦通信 |
| | SystemMonitor | 系统心跳监控，定期发送健康状态 |
| **数据采集**<br>(src/collectors/) | CameraCollector | 摄像头数据采集（RGB/Depth） |
| | SensorCollector | 多模态传感器数据采集（Tobii眼动等） |
| | FileCollector | 文件数据源读取 |
| **智能检测**<br>(src/detectors/) | ObjectDetector | 目标检测（人脸、手势等） |
| | AnomalyDetector | 异常行为检测 |
| | OCRDetector | 文字识别 |
| **模型推理**<br>(src/models/) | FatigueModel | 疲劳度检测模型（集成模式） |
| | EmotionModel | 情绪识别模型（集成模式） |
| | EEGModel | 脑电分析模型（集成模式） |
| **后端服务**<br>(src/services/) | UnifiedInferenceService | 统一推理调度服务 |
| | EEGService | EEG设备管理和数据采集 |
| | AVService | 音视频录制服务 |
| | UICommandRouter | UI命令路由和处理 |
| **通信接口**<br>(src/interfaces/) | WebsocketPushInterface | WebSocket服务器（8765端口） |
| | ModelBackendClient | 模型后端WebSocket客户端 |
| **前端UI**<br>(ui/) | MainWindow | 主窗口和页面流控制 |
| | LoginPage | 用户登录页面 |
| | CalibrationPage | 系统校准页面 |
| | TestPage | 测试执行页面 |
| | BackendClient | 后端WebSocket客户端 |
| **模型后端**<br>(model_backends/) | fatigue_backend | 疲劳度模型独立进程（远程模式） |
| | emotion_backend | 情绪模型独立进程（远程模式） |
| | eeg_backend | 脑电模型独立进程（远程模式） |

## 3. 多模态数据流转详解（基于文件结构）

### 3.1 情绪推理数据流转

**节点文件与服务**
- **音视频采集与录制**：`src/services/av_service.py`（摄像头/麦克风采集、录制、帧推送）
- **语音识别文本**：`ui/app/pages/test.py`（UI端集成语音识别与文本收集）
- **推理服务**：`src/services/unified_inference_service.py`（统一推理，情绪模型集成或远程）
- **推理模型**：集成模型 `src/models/emotion_model.py`
- **推理结果分发**：`src/core/event_bus.py`（事件总线）、`src/interfaces/websocket_server.py`（WebSocket接口）
- **UI展示**：`ui/app/pages/test.py`

**真实流转路径**：  
AVService(音视频采集/录制) → TestPage(语音识别/文本) → UnifiedInferenceService(情绪推理) → EventBus → WebSocket → TestPage(UI展示)

---

### 3.2 疲劳度推理数据流转

**节点文件与服务**
- **视频采集与录制**：`src/services/av_service.py`（RGB/深度视频采集、录制、帧推送）
- **眼动/多模态数据**：`ui/app/pages/test.py`（UI端采集与管理，部分硬件直连）
- **推理服务**：`src/services/unified_inference_service.py`（统一推理，疲劳模型集成或远程）
- **推理模型**：集成模型 `src/models/fatigue_model.py`
- **推理结果分发**：`src/core/event_bus.py`、`src/interfaces/websocket_server.py`
- **UI展示**：`ui/app/pages/test.py`

**真实流转路径**：  
AVService(视频采集/录制) → TestPage(眼动/多模态数据) → UnifiedInferenceService(疲劳推理) → EventBus → WebSocket → TestPage(UI展示)

---

### 3.3 脑负荷推理数据流转

**节点文件与服务**
- **脑电采集**：`src/devices/eeg.py`（硬件采集/驱动）、`ui/app/pages/test.py`（UI端采集控制）
- **推理服务**：`src/services/unified_inference_service.py`（统一推理，脑负荷模型集成或远程）
- **推理模型**：集成模型 `src/models/eeg_model.py`
- **推理结果分发**：`src/core/event_bus.py`、`src/interfaces/websocket_server.py`
- **UI展示**：`ui/app/pages/test.py`

**真实流转路径**：  
eeg.py(脑电采集) → TestPage(采集控制) → UnifiedInferenceService(脑负荷推理) → EventBus → WebSocket → TestPage(UI展示)

---

### 3.4 生理数据流转

**节点文件与服务**
- **血压/脉搏采集**：`ui/app/pages/test.py`（UI端采集与硬件交互）
- **分发与展示**：`src/core/event_bus.py`、`src/interfaces/websocket_server.py`、`ui/app/pages/test.py`

**真实流转路径**：  
TestPage(血压/脉搏采集) → EventBus → WebSocket → TestPage(UI展示)

---

### 3.5 舒尔特测试结果流转

**节点文件与服务**
- **UI交互与结果生成**：`ui/app/pages/test.py`（测试页面、分数/准确率/用时计算）
- **分发与存储**：`src/services/db_service.py`（数据库持久化）、`src/interfaces/model_ws_client.py`（WebSocket接口，选用）、`ui/app/pages/test.py`（UI展示）

**真实流转路径**：  
TestPage(交互/结果计算) → db_service.py(存储) → ModelWSClient(WebSocket，可选) → TestPage(UI展示)

---

### 3.6 总览流程图（按文件结构分层）

```
[UI操作]
   │
   ▼
[ui/app/pages/test.py]
   │
   ▼
[WebSocket: src/interfaces/websocket_server.py]
   │
   ▼
[EventBus: src/core/event_bus.py]
   │
   ├─[CameraCollector: src/collectors/camera_collector.py]
   │      │
   │      ├─[EmotionModel: src/models/emotion_model.py]
   │      ├─[FatigueModel: src/models/fatigue_model.py]
   │      └─[EEGModel: src/models/eeg_model.py]
   │
   ├─[SensorCollector: src/collectors/sensor_collector.py]
   │      │
   │      ├─[AnomalyDetector: src/detectors/anomaly_detector.py]
   │      └─[EEGModel: src/models/eeg_model.py]
   │
   └─[Test结果: ui/app/pages/test.py]
         │
         └─[WebSocket/后端存储: src/services/database.py]
```

---

## 4. 两种模型部署模式

| 特性 | 集成模式 (Integrated) | 远程模式 (Remote) |
|------|----------------------|-------------------|
| **启动方式** | 后端自动加载 | 需手动启动独立进程 |
| **性能** | ⭐⭐⭐⭐⭐ (内存共享) | ⭐⭐⭐⭐ (网络开销) |
| **隔离性** | ⭐⭐ (共享环境) | ⭐⭐⭐⭐⭐ (完全隔离) |
| **可靠性** | ⭐⭐⭐ (崩溃影响主进程) | ⭐⭐⭐⭐⭐ (故障隔离) |
| **配置复杂度** | ⭐⭐⭐⭐⭐ (极简) | ⭐⭐⭐ (需管理多进程) |
| **适用场景** | 开发测试、单机部署 | 生产环境、分布式部署 |
| **依赖管理** | 统一环境 | 各模型独立环境 |

**配置示例** (`config/models.yaml`):
```yaml
inference_models:
  - name: fatigue
    type: fatigue
    mode: integrated  # 集成模式（推荐）
    enabled: true
    
  - name: emotion
    type: emotion
    mode: remote      # 远程模式
    enabled: true
    remote:
      host: 127.0.0.1
      port: 8768
```

## 5. 端口分配

| 端口 | 服务 | 协议 | 说明 |
|------|------|------|------|
| 8765 | 主后端 WebSocket | WebSocket | UI与后端通信主接口 |
| 8767 | 疲劳度模型后端 | WebSocket | 远程模式下的疲劳度推理 |
| 8768 | 情绪模型后端 | WebSocket | 远程模式下的情绪推理 |
| 8769 | 脑电模型后端 | WebSocket | 远程模式下的EEG分析 |

## 6. 启动流程详解

### 6.1 后端启动流程 (`python -m src.main --root .`)

```
main.py
│
├─ parse_args()
│  └─ 解析命令行参数: --root, --no-listeners, --full-events
│
├─ Orchestrator.from_config_directory(root)
│  │
│  ├─ 读取配置文件
│  │  ├─ config/system.yaml          # 系统全局配置
│  │  ├─ config/collectors.yaml      # 采集器配置
│  │  ├─ config/detectors.yaml       # 检测器配置
│  │  ├─ config/models.yaml          # 模型配置（关键：mode字段）
│  │  └─ config/interfaces.yaml      # 接口配置
│  │
│  ├─ __init__()
│  │  ├─ 创建 EventBus                # 事件总线（核心通信机制）
│  │  ├─ 创建 ModelManager            # 模型管理器
│  │  ├─ 创建 SystemMonitor           # 系统监控器
│  │  └─ 创建 UICommandRouter         # UI命令路由器
│  │
│  └─ 返回 orchestrator 实例
│
└─ run(orchestrator)
   │
   ├─ orchestrator.start()
   │  │
   │  ├─ 实例化并启动 Collectors (from collectors.yaml)
   │  │  ├─ CameraCollector.start()
   │  │  ├─ SensorCollector.start()
   │  │  └─ FileCollector.start()
   │  │
   │  ├─ 实例化并启动 Detectors (from detectors.yaml)
   │  │  ├─ ObjectDetector.start()
   │  │  │  └─ 订阅 camera.frame 事件
   │  │  ├─ AnomalyDetector.start()
   │  │  └─ OCRDetector.start()
   │  │
   │  ├─ 实例化并启动 Interfaces (from interfaces.yaml)
   │  │  └─ WebsocketPushInterface.start()
   │  │     └─ 启动 WebSocket 服务器 (127.0.0.1:8765)
   │  │
   │  ├─ 创建并启动 UnifiedInferenceService
   │  │  │
   │  │  ├─ 解析 config/models.yaml
   │  │  │  ├─ name: fatigue, mode: integrated
   │  │  │  ├─ name: emotion, mode: integrated
   │  │  │  └─ name: eeg, mode: integrated
   │  │  │
   │  │  ├─ 加载 Integrated Models (mode=integrated)
   │  │  │  ├─ FatigueModel.__init__()
   │  │  │  │  └─ 加载模型权重: models_data/fatigue_models/
   │  │  │  ├─ EmotionModel.__init__()
   │  │  │  │  └─ 加载预训练模型: ROBBERTA, TIMESFORMER, WAV2VEC2
   │  │  │  └─ EEGModel.__init__()
   │  │  │     └─ 加载模型: mymodel_clf.joblib, mymodel_scaler.joblib
   │  │  │
   │  │  ├─ 连接 Remote Backends (mode=remote, 如果有)
   │  │  │  └─ ModelBackendClient 连接到 ws://host:port
   │  │  │
   │  │  └─ 订阅相关事件并启动推理循环
   │  │
   │  ├─ 创建并启动 EEGService
   │  │  └─ 初始化 EEG 设备连接（或启用模拟模式）
   │  │
   │  ├─ 创建并启动 AVService
   │  │  └─ 初始化音视频录制组件
   │  │
   │  └─ SystemMonitor.start()
   │     └─ 定时发布系统心跳事件 (每2秒)
   │
   ├─ 安装信号处理器 (Ctrl+C)
   │
   └─ 进入主循环 (asyncio.sleep)
      └─ 保持进程运行，响应事件
```

### 6.2 前端启动流程 (`python -m ui.main`)

```
ui/main.py
│
├─ create_application()
│  │
│  ├─ 创建 QApplication
│  │
│  ├─ 读取 config/system.yaml
│  │  └─ 获取 ui.auto_start_backend 配置
│  │
│  ├─ 如果 auto_start_backend=true
│  │  └─ BackendLauncher.start_backend_process()
│  │     └─ 启动后端子进程: python -m src.main --root .
│  │
│  └─ 返回 app 实例
│
├─ 创建 MainWindow
│  │
│  ├─ __init__()
│  │  │
│  │  ├─ 初始化 Qt 主窗口
│  │  │  └─ 设置窗口大小、标题、样式
│  │  │
│  │  ├─ 创建 3 个页面对象
│  │  │  ├─ LoginPage()
│  │  │  │  └─ 加载用户数据: ui/data/users/users.csv
│  │  │  ├─ CalibrationPage()
│  │  │  │  └─ 初始化校准界面组件
│  │  │  └─ TestPage()
│  │  │     └─ 初始化测试界面组件
│  │  │
│  │  ├─ 创建 QStackedWidget
│  │  │  └─ 将 3 个页面加入堆栈
│  │  │
│  │  ├─ 连接页面切换信号
│  │  │  ├─ login_page.login_success → show_calibration_page()
│  │  │  └─ calibration_page.calibration_complete → show_test_page()
│  │  │
│  │  └─ 设置调试快捷键
│  │     ├─ Ctrl+1 → 切换到登录页
│  │     ├─ Ctrl+2 → 切换到校准页
│  │     └─ Ctrl+3 → 切换到测试页
│  │
│  └─ show_login_page()
│     └─ 显示登录页面
│
├─ 初始化后端服务连接
│  │
│  ├─ BackendClient.connect("ws://127.0.0.1:8765")
│  │  │
│  │  ├─ 建立 WebSocket 连接
│  │  │
│  │  ├─ 注册消息处理器
│  │  │  ├─ "system.heartbeat" → 显示后端状态
│  │  │  ├─ "inference.result" → 更新推理结果显示
│  │  │  ├─ "camera.frame" → 更新视频预览
│  │  │  └─ "detector.result" → 处理检测结果
│  │  │
│  │  └─ 启动消息接收循环
│  │
│  └─ BackendProxy.initialize()
│     └─ 初始化后端服务代理
│        ├─ EEGService 代理
│        ├─ AVService 代理
│        └─ MultimodalService 代理
│
└─ app.exec_()
   └─ 进入 Qt 事件循环
      └─ 等待用户交互
```

### 6.3 运行时事件流

```
┌─────────────────────────────────────────────────────────────┐
│                        运行时数据流                           │
└─────────────────────────────────────────────────────────────┘

1️⃣ 数据采集流
   CameraCollector (30 FPS)
      │
      ├─ 发布 Event("camera.frame", {frame: ndarray, timestamp: ...})
      │
      └─ EventBus 广播 → ObjectDetector, UnifiedInferenceService

2️⃣ 目标检测流
   ObjectDetector.handle_event(camera.frame)
      │
      ├─ 运行 YOLO/SSD 目标检测
      │
      └─ 发布 Event("detector.face", {bbox: [x,y,w,h], confidence: 0.95})

3️⃣ 模型推理流（集成模式）
   UnifiedInferenceService.handle_event(camera.frame)
      │
      ├─ 调用 FatigueModel.infer(frame)
      │  └─ 返回 {fatigue_score: 0.65, features: {...}}
      │
      ├─ 调用 EmotionModel.infer(frame)
      │  └─ 返回 {emotion: "happy", valence: 0.8, arousal: 0.6}
      │
      └─ 发布 Event("inference.result", {fatigue: 0.65, emotion: "happy"})

4️⃣ 模型推理流（远程模式）
   UnifiedInferenceService.handle_event(camera.frame)
      │
      ├─ ModelBackendClient.send_inference_request(frame)
      │  │
      │  └─ WebSocket → ws://127.0.0.1:8767 (Fatigue Backend)
      │     │
      │     └─ 返回 {fatigue_score: 0.65}
      │
      └─ 发布 Event("inference.result", {fatigue: 0.65})

5️⃣ EEG数据流
   EEGService.poll_device()
      │
      ├─ 从 BLE 设备读取 8 通道数据 (500 Hz)
      │
      ├─ 发布 Event("eeg.data", {channels: [ch1, ch2, ...], timestamp: ...})
      │
      └─ EEGModel.infer(eeg_data)
         └─ 返回 {brain_load: "medium", confidence: 0.88}

6️⃣ WebSocket推送流
   WebsocketPushInterface.handle_event(inference.result)
      │
      └─ 推送到所有连接的客户端
         │
         └─ UI BackendClient 接收 → 更新界面显示

7️⃣ UI命令流
   TestPage.start_test_button_clicked()
      │
      ├─ BackendClient.send_command({"cmd": "start_test", "user": "admin"})
      │
      └─ 后端 UICommandRouter 接收
         │
         ├─ 启动 AVService.start_recording()
         ├─ 启动 EEGService.start_collection()
         └─ 返回 {"status": "success"}

8️⃣ 系统监控流
   SystemMonitor (每2秒)
      │
      └─ 发布 Event("system.heartbeat", {cpu: 35%, mem: 2.1GB, uptime: 3600s})
         │
         └─ WebSocket → UI 状态栏显示
```

### 6.4 典型用户会话的完整调用序列

```
┌──────┐      ┌────┐      ┌──────────┐      ┌───────┐      ┌────────────┐      ┌───────┐
│ User │      │ UI │      │WebSocket │      │Backend│      │Inference   │      │ Model │
│      │      │    │      │  Client  │      │       │      │  Service   │      │       │
└──┬───┘      └─┬──┘      └────┬─────┘      └───┬───┘      └─────┬──────┘      └───┬───┘
   │             │              │                │                 │                 │
   │             │              │                │                 │                 │
   │ [阶段1: 系统启动]           │                │                 │                 │
   │             │              │                │                 │                 │
   │             │<───────────────────── Start Backend Process ────────────────────────
   │             │              │                │                 │                 │
   │             │              │<───────────────┤ WebSocket       │                 │
   │             │              │                │ Server Start    │                 │
   │             │              │                │ (0.0.0.0:8765)  │                 │
   │             │              │                │                 │                 │
   │             │              │                ├─────────────────>                 │
   │             │              │                │ Initialize      │                 │
   │             │              │                │ InferenceService│                 │
   │             │              │                │                 │                 │
   │             │              │                │                 ├────────────────>│
   │             │              │                │                

---

*本文档描述了多模态人员状态评估平台的完整系统架构，包括组件设计、数据流转、部署模式和启动流程。*  
*最后更新: 2025年10月16日*
