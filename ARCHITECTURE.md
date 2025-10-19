# 系统架构设计

## 整体架构

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│   UI前端 (Qt)   │ ←─────────────→ │   后端服务      │
│                 │                 │                 │
│ • 登录/校准页面  │                 │ • 事件总线      │
│ • 测试界面      │                 │ • 模型推理      │
│ • 数据展示      │                 │ • 设备管理      │
└─────────────────┘                 └─────────────────┘
```

## 核心组件

### 后端核心 (src/)
- **Orchestrator**: 系统调度中心，管理所有组件生命周期
- **EventBus**: 事件总线，实现组件间解耦通信
- **SystemMonitor**: 系统心跳监控

### 模型推理 (src/models/)
- **FatigueModel**: 疲劳度检测模型
- **EmotionModel**: 情绪识别模型  
- **EEGModel**: 脑电分析模型

### 服务层 (src/services/)
- **UnifiedInferenceService**: 统一推理调度
- **EEGService**: EEG设备管理
- **AVService**: 音视频录制
- **UICommandRouter**: UI命令路由

### 前端UI (ui/)
- **MainWindow**: 主窗口和页面流控制
- **LoginPage**: 用户登录
- **CalibrationPage**: 系统校准
- **TestPage**: 测试执行

## 数据流转

### 1. 情绪推理流
```
UI测试页面 (ui/app/pages/test.py)
    ↓ 用户答题完成
UI命令路由 (src/services/ui_command_router.py)
    ↓ 调用情绪服务
情绪服务 (src/services/emotion_service.py)
    ↓ 发布情绪请求事件
统一推理服务 (src/services/unified_inference_service.py)
    ↓ 调用情绪模型
情绪模型 (src/models/emotion_model.py)
    ↓ 返回推理结果
事件总线 (src/core/event_bus.py)
    ↓ 发布检测结果事件
WebSocket接口 (src/interfaces/websocket_server.py)
    ↓ 推送到UI
UI测试页面 (ui/app/pages/test.py) - 结果展示
```

### 2. 疲劳度推理流
```
UI测试页面 (ui/app/pages/test.py)
    ↓ 启动多模态采集
多模态服务 (src/services/multimodal_service.py)
    ↓ 采集RGB/深度视频 + 眼动数据
多模态数据收集器 (src/services/multimodal_service.py::MultiModalDataCollector)
    ↓ 发布多模态数据事件
统一推理服务 (src/services/unified_inference_service.py)
    ↓ 调用疲劳度模型
疲劳度模型 (src/models/fatigue_model.py)
    ↓ 调用推理算法 (src/models/emotion_fatigue_infer/fatigue/infer_multimodal.py)
事件总线 (src/core/event_bus.py)
    ↓ 发布检测结果事件
WebSocket接口 (src/interfaces/websocket_server.py)
    ↓ 推送到UI
UI测试页面 (ui/app/pages/test.py) - 结果展示
```

### 3. 脑负荷推理流
```
UI测试页面 (ui/app/pages/test.py)
    ↓ 启动EEG采集
UI命令路由 (src/services/ui_command_router.py)
    ↓ 调用EEG服务
EEG服务 (src/services/eeg_service.py)
    ↓ 启动EEG采集器
EEG设备 (src/devices/eeg.py)
    ↓ 采集脑电数据 (500Hz采样)
多模态服务 (src/services/multimodal_service.py)
    ↓ EEG轮询循环，获取2秒窗口数据
统一推理服务 (src/services/unified_inference_service.py)
    ↓ 调用EEG模型
EEG模型 (src/models/eeg_model.py)
    ↓ 调用脑电算法 (src/models/eeg_algorithms/online_inference.py)
事件总线 (src/core/event_bus.py)
    ↓ 发布检测结果事件
WebSocket接口 (src/interfaces/websocket_server.py)
    ↓ 推送到UI
UI测试页面 (ui/app/pages/test.py) - 结果展示
```

### 4. 音视频采集流
```
UI测试页面 (ui/app/pages/test.py)
    ↓ 启动音视频采集
UI命令路由 (src/services/ui_command_router.py)
    ↓ 调用AV服务
AV服务 (src/services/av_service.py)
    ↓ 启动摄像头/麦克风
音视频录制组件 (src/services/av_service.py)
    ↓ 实时录制并保存文件
事件总线 (src/core/event_bus.py)
    ↓ 发布帧数据事件
WebSocket接口 (src/interfaces/websocket_server.py)
    ↓ 推送到UI
UI测试页面 (ui/app/pages/test.py) - 实时预览
```

## 端口分配

| 端口 | 服务 | 说明 |
|------|------|------|
| 8765 | 主后端 WebSocket | UI与后端通信主接口 |
| 8767 | 疲劳度模型后端 | 远程模式下的疲劳度推理 |
| 8768 | 情绪模型后端 | 远程模式下的情绪推理 |
| 8769 | 脑电模型后端 | 远程模式下的EEG分析 |

## 事件驱动架构

系统采用发布-订阅模式的事件总线架构：

```
数据采集器 → 发布事件 → 事件总线 → 订阅者处理 → 发布结果事件
```

### 主要事件类型
- `camera.frame`: 摄像头帧数据
- `eeg.data`: 脑电数据
- `inference.result`: 推理结果
- `system.heartbeat`: 系统心跳

## 配置驱动

系统通过YAML配置文件控制组件行为：

- `config/models.yaml`: 模型配置和部署模式
- `config/interfaces.yaml`: 通信接口配置

---

**最后更新**: 2025年10月