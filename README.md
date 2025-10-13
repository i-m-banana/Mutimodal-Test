# Multimodal Platform (Refactored)



一个模块化、可配置、可扩展的多模态数据采集与检测平台，支持音视频、脑电（EEG）、多模态传感器等数据源的统一管理和智能分析。



## ✨ 功能特性一个模块化、可配置、可扩展的多模态数据采集与检测平台，支持音视频、脑电（EEG）、多模态传感器等数据源的统一管理和智能分析。该项目是一个模块化的多模态采集与检测平台，支持配置驱动、可扩展且便于测试。



- **🎯 配置驱动**: 通过 YAML 文件灵活配置采集器、检测器和模型

- **🔌 事件解耦**: 基于事件总线的松耦合架构，易于扩展新功能

- **📹 多模态采集**: 支持摄像头、传感器、文件等多种数据源，硬件/模拟模式自动切换## ✨ 功能特性## 功能亮点

- **🤖 统一模型管理**: PyTorch/ONNX 模型统一注册与推理调用

- **📊 可观测性**: 标准化日志输出和系统心跳监控

- **🚀 微服务架构**: 多模型后端隔离部署，通过 WebSocket 通信

- **🎯 配置驱动**: 通过 YAML 文件灵活配置采集器、检测器和模型- **配置驱动**: 通过 YAML 文件启停 Collector / Detector / Model

## 🏗️ 架构设计

- **🔌 事件解耦**: 基于事件总线的松耦合架构，易于扩展新功能- **事件解耦**: 内部组件经事件总线通信，易于扩展新功能

### 核心组件

- **📹 多模态采集**: 支持摄像头、传感器、文件等多种数据源，硬件/模拟模式自动切换- **多模态采集**: 内置摄像头、传感器、文件目录三类采集器，支持硬件与模拟双模式

**后端服务** (`src/`):

- 调度中心、事件总线、系统监控- **🤖 统一模型管理**: PyTorch/ONNX 模型统一注册与推理调用- **统一模型管理**: PyTorch / ONNX 模型统一注册与推理调用

- 数据采集器（音视频、EEG、多模态传感器）

- 检测器和模型推理引擎- **📊 可观测性**: 标准化日志输出和系统心跳监控- **可观测性**: 标准化日志输出 + 系统心跳事件，便于运维监控

- WebSocket 接口服务

- **🚀 微服务架构**: 多模型后端隔离部署，通过 WebSocket 通信- **🚀 多模型后端架构**: 支持将不同环境的AI模型部署为独立后端服务，通过WebSocket与主后端通信，实现完全的环境隔离

**前端界面** (`ui/`):

- Qt 应用程序框架

- 响应式 UI（自适应 1080p-4K 分辨率）

- 实时监控和数据可视化组件## 🏗️ 架构设计## 架构特性

- 用户管理和测评系统



**模型后端** (`model_backends/`):

- 独立进程运行，环境完全隔离### 核心组件### 🚀 多模型后端架构

- WebSocket 协议标准化接口

- 支持 PyTorch、TensorFlow 等多种框架



### 数据流- **后端服务** (`src/`): 为解决不同AI模型（多模态、情绪、脑电）需要不同Python环境的问题，采用了基于WebSocket的微服务架构：



```  - 调度中心、事件总线、系统监控

UI ─→ 主后端 (EventBus) ─→ ModelProxyService ─→ 模型后端 (隔离环境)

       ↓                        ↓  - 数据采集器（音视频、EEG、多模态传感器）#### 核心组件

   数据采集               检测结果聚合

```  - 检测器和模型推理引擎



### 端口分配  - WebSocket 接口服务- **BaseModelBackend** (`model_backends/base/`): 抽象基类，提供WebSocket服务器框架



- **8765**: 主后端 WebSocket 接口- **ModelBackendClient** (`src/interfaces/model_ws_client.py`): 异步WebSocket客户端，支持自动重连

- **8766**: 多模态模型后端

- **8767**: 情绪模型后端- **前端界面** (`ui/`):- **ModelProxyService** (`src/services/model_proxy_service.py`): 代理服务，管理多个模型后端连接

- **8768**: 脑电模型后端

  - Qt 应用程序框架- **示例实现** (`model_backends/multimodal_backend/`): 完整的PyTorch多模态模型后端模板

## 📁 目录结构

  - 响应式 UI（自适应 1080p-4K 分辨率）

```

project-root/  - 实时监控和数据可视化组件#### 架构优势

├── aidebug/         # 调试文档和测试脚本

├── config/          # 配置文件 (YAML)  - 用户管理和测评系统

├── data/            # 数据目录

├── docs/            # 架构与API文档- ✅ **完全环境隔离**: 每个模型运行在独立Python环境/进程中，避免依赖冲突

├── logs/            # 日志输出

├── model_backends/  # 独立模型后端服务- **模型后端** (`model_backends/`):- ✅ **独立进程管理**: 模型崩溃不影响主系统，自动重连机制保证可靠性

│   ├── base/                 # 抽象基类

│   ├── multimodal_backend/   # 多模态模型示例  - 独立进程运行，环境完全隔离- ✅ **标准化接口**: 统一的WebSocket协议，快速添加新模型

│   ├── emotion_backend/      # 情绪模型

│   └── eeg_backend/          # 脑电模型  - WebSocket 协议标准化接口- ✅ **生产级错误处理**: Future-based API + 超时控制 + 健康检查

├── src/             # 后端源代码

│   ├── core/        # 核心调度和事件总线  - 支持 PyTorch、TensorFlow 等多种框架

│   ├── collectors/  # 数据采集器

│   ├── detectors/   # 检测模块#### 数据流

│   ├── services/    # 后端服务

│   ├── interfaces/  # WebSocket 接口### 数据流```

│   └── utils/       # 工具类

├── ui/              # Qt 前端UI → 主后端(EventBus) → ModelProxyService → 多个模型后端(隔离环境)

│   ├── app/         # 应用核心

│   ├── data/        # 资源文件```     ↓                      ↓

│   ├── services/    # UI 服务层

│   ├── widgets/     # UI 组件UI ─→ 主后端 (EventBus) ─→ ModelProxyService ─→ 模型后端 (隔离环境)MULTIMODAL_SNAPSHOT → 分发到各模型 → DETECTION_RESULT → 结果聚合

│   └── models/      # 模型推理

├── tests/           # 单元与集成测试       ↓                        ↓```

├── pyproject.toml   # 项目配置

├── requirements.txt # 依赖列表   数据采集               检测结果聚合

└── README.md

``````#### 端口分配



## 🚀 快速开始- 8765: 主后端WebSocket接口



### 1. 环境准备### 端口分配- 8766: 多模态模型后端



```bash- 8767: 情绪模型后端

# 克隆仓库

git clone <repository-url>- **8765**: 主后端 WebSocket 接口- 8768: 脑电模型后端

cd project-root

- **8766**: 多模态模型后端

# 创建虚拟环境

python -m venv .venv- **8767**: 情绪模型后端#### 快速开始



# 激活虚拟环境- **8768**: 脑电模型后端1. **📋 文档索引**: `MULTI_MODEL_BACKEND_INDEX.md` (完整导航,从这里开始!)

# Windows:

.venv\Scripts\activate2. **✅ 完整性检查**: `MULTI_MODEL_BACKEND_CHECKLIST.md` (验证所有组件)

# Linux/Mac:

source .venv/bin/activate## 📁 目录结构3. **🚀 快速接入**: `MULTI_MODEL_BACKEND_QUICKSTART.md` (5步接入指南)



# 安装依赖4. **📐 架构设计**: `MULTI_MODEL_BACKEND_ARCHITECTURE.md` (完整设计文档)

pip install -r requirements.txt

``````5. **🔌 接口说明**: `MULTI_MODEL_BACKEND_INTERFACES.md` (API文档)



### 2. 启动后端服务project-root/6. **📝 实现总结**: `MULTI_MODEL_BACKEND_SUMMARY.md` (实现清单)



```bash├── aidebug/         # 调试文档和测试脚本

python -m src.main --root .

```├── config/          # 配置文件 (YAML)#### 测试验证



后端启动后会监听 `8765` 端口，提供 WebSocket 接口。├── data/            # 数据目录```bash



### 3. 启动前端界面├── docs/            # 架构与API文档# 测试架构组件



```bash├── logs/            # 日志输出python tests/test_model_backend_architecture.py

# 监控看板（轻量级）

python -m ui.dashboard├── model_backends/  # 独立模型后端服务



# 完整界面│   ├── base/                 # 抽象基类# 启动示例模型后端

python -m ui.main           # 正常模式

python -m ui.main --debug   # 调试模式（模拟数据源）│   ├── multimodal_backend/   # 多模态模型示例cd model_backends/multimodal_backend

```

│   ├── emotion_backend/      # 情绪模型python -m venv .venv

### 4. 启动模型后端（可选）

│   └── eeg_backend/          # 脑电模型.venv\Scripts\activate  # Windows

```bash

cd model_backends/multimodal_backend├── src/             # 后端源代码pip install -r requirements.txt

python -m venv .venv

.venv\Scripts\activate  # Windows│   ├── core/        # 核心调度和事件总线python main.py

pip install -r requirements.txt

python main.py│   ├── collectors/  # 数据采集器```

```

│   ├── detectors/   # 检测模块

## ⚙️ 配置说明

│   ├── services/    # 后端服务---

### 环境变量

│   ├── interfaces/  # WebSocket 接口

| 变量名 | 说明 | 默认值 |

|--------|------|--------|│   └── utils/       # 工具类### Phase 3 UI增强(2025-10-07)🎨

| `UI_FORCE_SIMULATION` | 全局模拟模式 | 0 |

| `UI_DEBUG_MODE` | 调试模式 | 0 |├── ui/              # Qt 前端- **响应式分辨率适配**：解决2K开发环境到1080p部署环境的显示问题：

| `BACKEND_EEG_SIMULATION` | 后端 EEG 模拟 | 0 |

| `UI_SKIP_DATABASE` | 跳过数据库 | 0 |│   ├── app/         # 应用核心  - 新建 `ui/app/utils/responsive.py` - 响应式缩放系统，基于2560x1440基准自动缩放

| `UI_DB_HOST` | 数据库主机 | localhost |

| `UI_TTS_BACKEND` | 语音后端 | powershell (Windows) |│   ├── data/        # 资源文件  - 支持分辨率范围：1280x720 ~ 4K，自动检测并计算缩放因子



### 配置文件│   ├── services/    # UI 服务层  - 更新所有主要页面使用响应式API：`scale()`, `scale_size()`, `scale_font()`



- `config/system.yaml`: 系统参数（心跳间隔、超时设置等）│   ├── widgets/     # UI 组件  - 1080p环境缩放因子0.75，所有UI组件按比例缩放，完美适配小屏幕

- `config/collectors.yaml`: 采集器配置（采样率、工作模式）

- `config/detectors.yaml`: 检测器配置（阈值、启用状态）│   └── models/      # 模型推理- **实时监控弹窗**：测试页面智能布局优化：

- `config/models.yaml`: 模型配置（路径、参数）

├── tests/           # 单元与集成测试  - 新建 `ui/app/utils/monitor_window.py` - 独立监控窗口，显示摄像头+检测结果

## 🔧 开发指南

├── pyproject.toml   # 项目配置  - 小屏幕（<0.85缩放因子）：隐藏内嵌画面，显示"显示实时监控"按钮

### 扩展新采集器

├── requirements.txt # 依赖列表  - 大屏幕（>=0.85缩放因子）：保留原有内嵌画面布局

1. 在 `src/collectors/` 创建类，继承 `BaseCollector`

2. 实现 `run_once()` 方法└── README.md  - 监控窗口自适应尺寸（2K: 1000x750, 1080p: 800x600），30fps实时刷新

3. 在 `config/collectors.yaml` 注册

```- **测试验证**：`test_responsive.py` 和 `test_responsive_simulated.py` 验证多分辨率缩放 ✅

### 扩展新模型

- **文档完善**：`RESOLUTION_ADAPTATION.md` 详细说明，`TESTING_GUIDE.md` 测试指南

1. 实现 `BaseModel` 子类

2. 配置到 `config/models.yaml`## 🚀 快速开始

3. 由 `ModelManager` 自动加载

### Phase 2 重构（2025-01-07）✨

### 扩展新检测器

### 1. 环境准备- **资源文件组织化**：`ui/` 根目录的 CSV/YAML 文件统一迁移到 `ui/data/` 目录，实现资源集中管理：

1. 继承 `BaseDetector`

2. 声明订阅的事件主题  - `ui/data/users/` - 用户数据（users.csv, scores.csv, schulte_scores.csv）

3. 实现事件处理逻辑

```bash  - `ui/data/questionnaires/` - 问卷配置（questionnaire.yaml）

## 🧪 测试

# 克隆仓库  - 所有路径配置集中到 `ui/app/config.py` 的 `DATA_DIR` 常量

```bash

# 运行所有测试git clone <repository-url>- **EEG服务后端迁移**：将UI层的EEG硬件连接代码（602行）完全迁移到后端：

pytest

cd project-root  - 新建 `src/services/eeg_service.py` (368行) - 后端EEG服务，管理BLE连接和数据采集

# 测试特定模块

python tests/test_model_backend_architecture.py  - 精简 `ui/services/eeg_service.py` (602→122行, -79.7%) - 改为纯WebSocket客户端代理

python tests/test_responsive.py

```# 创建虚拟环境  - UI层不再依赖 `bleak`、`asyncio` 等硬件库，实现完全的前后端解耦



## 📚 文档python -m venv .venv  - WebSocket命令协议：`eeg.start`, `eeg.stop`, `eeg.snapshot`, `eeg.file_paths`



详细文档位于 `aidebug/` 目录：  - 集成测试验证：后端服务、命令路由、UI客户端全部测试通过 ✅



- **快速开始**: `QUICK_START.md`# 激活虚拟环境- **环境变量增强**：新增 `BACKEND_EEG_SIMULATION=1` 用于后端EEG模拟模式（独立于UI的模拟控制）

- **架构设计**: `MULTI_MODEL_BACKEND_ARCHITECTURE.md`

- **接口文档**: `MULTI_MODEL_BACKEND_INTERFACES.md`# Windows:

- **快速接入**: `MULTI_MODEL_BACKEND_QUICKSTART.md`

- **测试指南**: `TESTING_GUIDE.md`.venv\Scripts\activate### Phase 1 重构（2025-10-07）

- **分辨率适配**: `RESOLUTION_ADAPTATION.md`

# Linux/Mac:- **Qt 前端模块化**：`ui/main.py` 现仅作为薄封装，核心窗口与页面迁移至 `ui/app/application.py` 和 `ui/app/pages/*`。调试模式开关逻辑仍由 `ui/app/config.py` 解析 `--debug` / `--mode` 参数，兼容 `python -m ui.main --debug` 的原有体验。

## 🎨 UI 特性

source .venv/bin/activate- **TTS 后端自愈**：`src/services/tts_service.py` 针对 PowerShell 管道新增超时检测，若 18 秒内未完成朗读会自动禁用该后端并回退到 `pyttsx3`，同时沿用用户请求中的语速、音量等参数。前端无需改动即可获得稳定播报，也可通过 `UI_TTS_BACKEND=pyttsx3` 强制指定。

- **响应式设计**: 自动适配 1280x720 到 4K 分辨率

- **智能布局**: 小屏幕自动切换独立监控窗口- **摄像头缺失自动降级**：`src/services/av_service.py` 在硬件初始化失败（或 `UI_FORCE_SIMULATION=1`）时会自动启用合成画面生成器，并在日志中提示“Camera unavailable…启用模拟视频帧”。该模式生成的彩条 Frame 仍会广播到 `camera.frame` 主题，方便前端和调试脚本继续验证链路。

- **实时监控**: 30fps 视频流和检测结果显示

- **模块化架构**: 页面组件完全解耦，易于维护# 安装依赖- **数据库错误透传**：当 MySQL 服务不可达时，`DatabaseUnavailable` 异常会同步到 UI 并在日志中给出原始错误码（如 `WinError 10061`），避免静默失败。若仅需演示，可设置 `UI_SKIP_DATABASE=1` 暂时跳过持久化。



## 🔒 模拟模式pip install -r requirements.txt- **录制指令保护**：AV 录制接口在预览未成功启动时会直接返回 `Preview not started`，防止误触发空音视频文件。请先确认 `av.start_preview` 成功（或启用模拟模式）再调用 `av.start_recording`。



支持在无硬件环境下运行，便于开发和演示：```



```bash## 目录结构

# 方式1: 启动参数（推荐）

python -m ui.main --debug### 2. 启动后端服务



# 方式2: 环境变量（Windows）```

set UI_FORCE_SIMULATION=1

```bashproject-root/

# 方式3: 细分控制

set UI_MULTIMODAL_SIMULATION=1python -m src.main --root .├── config/          # 配置文件

set BACKEND_EEG_SIMULATION=1

``````│   └── models.yaml  # ✨ 新增model_backends配置节



### 模拟模式功能├── data/            # 原始/处理/结果数据目录



- ✅ 生成合成音视频数据后端启动后会监听 `8765` 端口，提供 WebSocket 接口。├── docs/            # 架构与部署文档

- ✅ 模拟传感器读数

- ✅ 模拟 EEG 信号├── logs/            # 日志输出目录（运行时自动生成）

- ✅ 跳过硬件设备连接

### 3. 启动前端界面├── model_backends/  # 🚀 独立模型后端服务(新增)

## 📝 许可证

│   ├── base/                      # 抽象基类

本项目采用 MIT 许可证。

```bash│   │   └── base_backend.py        # BaseModelBackend - WebSocket服务器框架

## 🤝 贡献

# 监控看板（轻量级）│   ├── multimodal_backend/        # 示例: PyTorch多模态模型

欢迎提交 Issue 和 Pull Request！

python -m ui.dashboard│   │   ├── main.py               # 完整实现(含模拟推理)

## 📧 联系方式

│   │   ├── requirements.txt      # 独立依赖(torch, torchvision, websockets)

如有问题，请通过 GitHub Issues 联系。

# 完整界面│   │   └── README.md             # 使用文档

---

python -m ui.main           # 正常模式│   ├── emotion_backend/           # 待实现: TensorFlow情绪模型

**注意事项**:

- 首次运行需要联网下载模型文件，后续可离线运行python -m ui.main --debug   # 调试模式（模拟数据源）│   └── eeg_backend/               # 待实现: MNE-Python脑电模型

- 部分功能需要特定硬件支持（摄像头、EEG 设备等），可使用模拟模式进行测试

- 详细的架构说明和重构历史请参考 `aidebug/` 目录中的相关文档```├── src/             # 后端源代码


│   ├── core/        # 调度中心、事件总线、系统监控

### 4. 启动模型后端（可选）│   ├── collectors/  # 数据采集器

│   ├── models/      # 模型封装与管理

```bash│   ├── detectors/   # 检测模块

cd model_backends/multimodal_backend│   ├── services/    # 后端服务

python -m venv .venv│   │   ├── av_service.py          # 音视频服务

.venv\Scripts\activate  # Windows│   │   ├── eeg_service.py         # EEG服务

# source .venv/bin/activate  # Linux/Mac│   │   ├── multimodal_service.py  # 多模态数据采集

pip install -r requirements.txt│   │   ├── tts_service.py         # 语音播报

python main.py│   │   ├── model_proxy_service.py # 🚀 新增: 模型后端代理服务

```│   │   └── database_service.py    # 数据库服务

│   ├── interfaces/  # WebSocket服务器

## ⚙️ 配置说明│   │   ├── ws_server.py           # 主WebSocket服务器(8765)

│   │   └── model_ws_client.py     # 🚀 新增: 模型后端WebSocket客户端

### 环境变量│   └── utils/       # 公共工具类

├── ui/              # Qt 前端

| 变量名 | 说明 | 默认值 |│   ├── app/         # 应用核心（应用入口、配置、页面）

|--------|------|--------|│   │   ├── utils/   # 🎨 UI工具（响应式缩放、监控弹窗 - Phase 3新增）

| `UI_FORCE_SIMULATION` | 全局模拟模式 | 0 |│   │   └── pages/   # 页面组件（登录、校准、测试、血压、成绩等）

| `UI_DEBUG_MODE` | 调试模式 | 0 |│   ├── data/        # 📁 资源文件（Phase 2新增）

| `BACKEND_EEG_SIMULATION` | 后端 EEG 模拟 | 0 |│   │   ├── users/            # 用户数据CSV

| `UI_SKIP_DATABASE` | 跳过数据库 | 0 |│   │   └── questionnaires/   # 问卷配置YAML

| `UI_DB_HOST` | 数据库主机 | localhost |│   ├── services/    # UI服务层（后端客户端、EEG代理、AV代理、多模态代理）

| `UI_TTS_BACKEND` | 语音后端 | powershell (Windows) / pyttsx3 (其他) |│   ├── widgets/     # UI组件（多模态预览、脑负荷条、舒尔特表、仪表盘、成绩页）

│   ├── models/      # 模型推理

### 配置文件│   └── utils_common/ # 通用工具（数据库、线程管理、工具函数）

├── tests/           # 单元与集成测试

- `config/system.yaml`: 系统参数（心跳间隔、超时设置等）│   ├── test_model_backend_architecture.py  # 🚀 新增: 多模型后端架构测试

- `config/collectors.yaml`: 采集器配置（采样率、工作模式）│   ├── test_integration_thread_optimization.py  # 线程池集成测试

- `config/detectors.yaml`: 检测器配置（阈值、启用状态）│   ├── test_eeg_integration.py      # Phase 2 EEG服务集成测试

- `config/models.yaml`: 模型配置（路径、参数）│   ├── test_responsive.py           # Phase 3 响应式缩放测试 🎨

│   └── test_responsive_simulated.py # Phase 3 多分辨率模拟测试 🎨

## 🔧 开发指南├── MULTI_MODEL_BACKEND_INDEX.md              # 🚀 多模型后端文档索引(从这里开始!)

├── MULTI_MODEL_BACKEND_CHECKLIST.md          # 🚀 多模型后端完整性检查清单

### 扩展新采集器├── MULTI_MODEL_BACKEND_ARCHITECTURE.md       # 🚀 多模型后端完整设计文档

├── MULTI_MODEL_BACKEND_QUICKSTART.md         # 🚀 快速接入指南

1. 在 `src/collectors/` 创建类，继承 `BaseCollector`├── MULTI_MODEL_BACKEND_INTERFACES.md         # 🚀 接口使用文档

2. 实现 `run_once()` 方法├── MULTI_MODEL_BACKEND_SUMMARY.md            # 🚀 实现总结与清单

3. 在 `config/collectors.yaml` 注册├── BACKEND_THREAD_OPTIMIZATION.md            # 🧵 后端线程池优化文档

├── UI_THREAD_OPTIMIZATION.md                 # 🧵 UI线程池优化文档

### 扩展新模型├── THREAD_POOL_QUICK_REF.md                  # 🧵 线程池快速参考

├── RESOLUTION_ADAPTATION.md        # Phase 3 分辨率适配详细说明 🎨

1. 实现 `BaseModel` 子类├── RESOLUTION_ADAPTATION_SUMMARY.md # Phase 3 完成总结 🎨

2. 配置到 `config/models.yaml`├── TESTING_GUIDE.md                # Phase 3 测试指南 🎨

3. 由 `ModelManager` 自动加载├── REFACTORING_PHASE2_SUMMARY.md   # Phase 2 重构总结 ✨

├── REFACTORING_REPORT.md           # Phase 1 完整报告

### 扩展新检测器├── REFACTORING_SUMMARY.md        # Phase 1详细总结

├── test_ui_refactor.py          # Phase 1 UI重构测试

1. 继承 `BaseDetector`├── pyproject.toml   # 依赖与构建配置

2. 声明订阅的事件主题├── requirements.txt # 运行依赖列表

3. 实现事件处理逻辑├── QUICK_START.md   # 快速启动指南

└── README.md

## 🧪 测试```



```bash## 快速开始

# 运行所有测试

pytest1. 安装依赖：



# 测试特定模块```bash

python tests/test_model_backend_architecture.pypython -m venv .venv

python tests/test_responsive.py. .venv/bin/activate  # Windows: .venv\Scripts\activate

```pip install -r requirements.txt

```

## 📚 文档

2. 运行 Orchestrator：

详细文档位于 `aidebug/` 目录：

```bash

- **快速开始**: `QUICK_START.md`python -m src.main --root .

- **架构设计**: `MULTI_MODEL_BACKEND_ARCHITECTURE.md````

- **接口文档**: `MULTI_MODEL_BACKEND_INTERFACES.md`

- **快速接入**: `MULTI_MODEL_BACKEND_QUICKSTART.md`默认会在控制台打印 Detector 的检测结果与系统心跳，如需关闭监听可添加 `--no-listeners`。

- **测试指南**: `TESTING_GUIDE.md`

- **分辨率适配**: `RESOLUTION_ADAPTATION.md`3. 启动 Qt 前端监控看板（可选）：



## 🎨 UI 特性```bash

python -m ui.dashboard

- **响应式设计**: 自动适配 1280x720 到 4K 分辨率```

- **智能布局**: 小屏幕自动切换独立监控窗口

- **实时监控**: 30fps 视频流和检测结果显示请先确保 Orchestrator 已运行并启用了 `interfaces.yaml` 中的 WebSocket 接口。监控看板仅负责展示后端推送的检测结果与系统心跳，不再直接驱动采集任务。

- **模块化架构**: 页面组件完全解耦，易于维护

> **提示**：若需要在 UI 中查看实时摄像头画面或音频电平，请确认 `config/interfaces.yaml` 的 `topics` 列表包含 `camera.frame` 与 `audio.level`（仓库默认已开启）。

## 🔒 模拟模式

4. 启动完整旧版 UI（依赖后端命令链路）：

支持在无硬件环境下运行，便于开发和演示：

```powershell

```bashcd F:\mutilUI\project-root

# 方式1: 启动参数（推荐）python -m ui.main           # 正常模式：真实信号，需要后端在线

python -m ui.main --debugpython -m ui.main --debug   # 调试模式：启用模拟信号，界面窗口化

```

# 方式2: 环境变量（Windows）

set UI_FORCE_SIMULATION=1确保第二步的 Orchestrator 已启动，否则正常模式会提示“Backend connection not established”。该界面保留了历史测评流程，但所有音视频操作都改由后端服务执行。



# 方式3: 细分控制> **离线演示**：在无后端或无硬件的环境下，请使用 `python -m ui.main --debug`，系统会自动切换到模拟信号并以窗口化模式运行，便于调试。正常模式始终请求真实信号，若缺少设备或后台将直接报错提醒。

set UI_MULTIMODAL_SIMULATION=1

set BACKEND_EEG_SIMULATION=15. 运行测试：

```

```bash

模拟模式功能：pytest

- ✅ 生成合成音视频数据```

- ✅ 模拟传感器读数

- ✅ 模拟 EEG 信号## 模拟模式切换

- ✅ 跳过硬件设备连接

前端与采集脚本现已支持统一的模拟模式，便于在无硬件/无后台的环境下演示：

## 📝 许可证

- **启动参数（推荐）**：

本项目采用 MIT 许可证。

   ```powershell

## 🤝 贡献   python -m ui.main --debug

   ```

欢迎提交 Issue 和 Pull Request！

   该命令会在当前进程内设置 `UI_DEBUG_MODE=1` 与 `UI_FORCE_SIMULATION=1`，并禁用所有真实设备调用。若需强制回到真实信号，可执行 `python -m ui.main --mode normal`。

## 📧 联系方式

- **全局开关**：

如有问题，请通过 GitHub Issues 联系。

  ```powershell

---  setx UI_FORCE_SIMULATION 1

  ```

**注意**: 

- 首次运行需要联网下载模型文件，后续可离线运行  重新打开终端后，音视频、多模态、脑电模块都会使用内置的合成数据源。

- 部分功能需要特定硬件支持（摄像头、EEG 设备等），可使用模拟模式进行测试

- **后端模拟开关**（Phase 2新增）：

  ```powershell
  # 后端EEG服务使用模拟数据（不需要真实蓝牙设备）
  setx BACKEND_EEG_SIMULATION 1
  ```

  后端服务支持独立的模拟模式控制，允许后端生成模拟数据而UI仍可连接真实后端，便于开发和调试。

- **细分开关**：需要单独控制时，可使用：

  | 模块       | 环境变量                | 说明 |
  |------------|-------------------------|------|
  | 多模态采集 | `UI_MULTIMODAL_SIMULATION=1` | 使用渐变 RGB、噪声深度与随机 gaze/head pose；同步写入录制文件。
  | 脑电采集（UI）| `UI_EEG_SIMULATION=1`   | UI侧使用模拟模式（后端不可用时的降级方案）。
  | 脑电采集（后端）| `BACKEND_EEG_SIMULATION=1` | 后端以 500Hz 生成正弦叠加噪声的双通道数据，保存 CSV/NPY/JSON 及元数据。✨
  | 血压测试   | `UI_BP_SIMULATION=1`         | UI 可直接生成合成血压/脉搏结果，省略 Maibobo 设备连接。

关闭模拟模式：

```powershell
setx UI_FORCE_SIMULATION ""
setx BACKEND_EEG_SIMULATION ""
setx UI_MULTIMODAL_SIMULATION ""
setx UI_EEG_SIMULATION ""
```

> **提示**：若缺少 RealSense、Tobii 或 `bleak` 等依赖，即使未设置环境变量，也会自动降级到模拟模式并在日志中提示。Phase 2 重构后，UI层不再依赖 `bleak`（EEG的BLE连接库），硬件依赖完全转移到后端。✨

## 数据库与语音识别配置

- `UI_SKIP_DATABASE=1`：前端与采集脚本将跳过所有 MySQL 读写操作，适合无数据库或调试环境。
- `UI_DB_HOST` / `UI_DB_USER` / `UI_DB_PASSWORD` / `UI_DB_NAME`：自定义数据库连接参数，默认指向 `localhost` 的 `tired` 库。
- `UI_WHISPER_LOCAL_ONLY=1`：强制 Faster-Whisper 仅使用本地缓存模型，避免在离线环境下触发 Hugging Face 下载错误。
- `UI_WHISPER_MODEL`：指定 Faster-Whisper 模型大小（默认 `base`）。首次联网运行会自动缓存，后续离线即可使用。
- `UI_WHISPER_MODEL_DIR`：指定本地 Faster-Whisper 模型目录（例如离线复制的 `Systran/faster-whisper-base` 文件夹）。设置该变量后将始终离线加载。
- `UI_TTS_BACKEND`：语音播报后端选择。Windows 默认使用 `powershell`（调用 `System.Speech`），其余平台默认 `pyttsx3`；可显式设置为 `pyttsx3` 或 `powershell` 并与 `UI_TTS_VOICE` 搭配指定目标语音名称片段。

当 `UI_SKIP_DATABASE=1` 时，所有成绩统计、血压/多模态数据持久化都会自动降级为内存模式；成绩页将回退到模拟数据展示，核心流程仍可继续。语音识别在联网失败时会自动尝试使用本地缓存，并提示如何下载或开启离线模式。

## 后端启动步骤

1. 安装依赖并激活虚拟环境（参考“快速开始”）。
2. 在 `project-root` 目录运行：

   ```powershell
   cd F:/mutilUI/project-root
   # 等效写法：python -m src.main
   python -m src.main --root .
   ```

3. 观察控制台，确认出现如下日志：

   ```text
   INFO  entrypoint | Loaded interfaces: websocket
   INFO  WebsocketPushInterface | Starting WebSocket interface on 127.0.0.1:8765
   ```

   若提示 `websockets package not installed; interface disabled`，请执行 `pip install websockets` 后重启。

4. 前端启动后仍出现 “目标计算机积极拒绝” 时，请检查：

   - WebSocket 接口是否占用同一端口（默认 8765），可使用 `netstat -ano | findstr 8765` 验证。
   - 网络安全软件/防火墙是否拦截本地回环端口连接。
   - 配置文件 `config/interfaces.yaml` 中的 `host` 是否与 UI 期望一致（默认为 `127.0.0.1`）。必要时可改为 `0.0.0.0` 并重启后端。

5. 后端运行过程中每 5 秒会输出一条 `system.heartbeat` 日志（示例中 severity=warn 仅表示内存占用门限提醒），代表主循环正常。当 UI 成功连上后，可以在前端日志看到 “Backend connection established”。

## 配置说明

- `config/system.yaml`：系统级参数（心跳间隔、停止超时等）。
- `config/collectors.yaml`：采集器列表与采样频率、工作模式。
- `config/detectors.yaml`：检测器启用状态及阈值设定。
- `config/models.yaml`：模型类型、路径与运行参数。

详见 `docs/api_spec.md` 与 `docs/architecture.md`。

## 扩展指南

1. 新增采集器：在 `src/collectors` 下创建类继承 `BaseCollector`，实现 `run_once` 并在配置文件注册。
2. 新增模型：实现 `BaseModel` 子类，配置到 `models.yaml` 后由 `ModelManager` 自动加载。
3. 新增检测器：继承 `BaseDetector`，声明订阅主题并处理事件。

## 迁移说明

相比旧版 `Multimodal-Project`：

- UI 逻辑被解耦为独立模块，可按需集成。
- 所有设备调用统一封装，降低多线程和资源泄漏风险。
- 数据与日志目录结构统一，便于部署与审计。

## 前端架构概述

### Phase 2 架构优化（2025-01-07）✨
- **资源文件集中管理**：`ui/data/` 统一存放所有配置和数据文件，通过 `ui/app/config.py` 的常量访问，避免硬编码路径。
- **EEG服务前后端分离**：
  - UI层 `ui/services/eeg_service.py` 精简为122行的WebSocket客户端代理，提供函数式接口（`start`, `stop`, `get_file_paths`, `get_realtime_sample`, `get_recent_5s`）
  - 后端 `src/services/eeg_service.py` (368行) 负责BLE连接（F4:3C:7C:A6:29:E0）、数据采集、解析和保存
  - 通过WebSocket命令协议通信：`eeg.start`, `eeg.stop`, `eeg.snapshot`, `eeg.file_paths`
  - UI层不再依赖硬件库（bleak），实现完全解耦

### Phase 1 架构（2025-10-07）
- `ui/dashboard.py` 提供轻量监控看板，仅订阅 WebSocket 推流的检测结果与系统心跳，用于实时可视化。
- `ui/services/backend_client.py` 封装了与后端 WebSocket 的连接、自动重连与信号分发，便于在 Qt 中使用。
- `ui/services/av_service.py` 通过 `BackendClient` 将 UI 操作转换为 `av.*` 命令并解析广播帧，实现与旧版接口兼容的同时将采集逻辑完全迁移到后端。
- `ui/app/application.py` 负责主窗口装配与资源清理，`ui/app/pages/*` 管理登录 / 校准 / 测试页面，`ui/app/config.py` 解析 `--debug` / `--mode` 参数并维护调试模式标志。
- `ui/main.py` 退化为薄封装，仅调用 `ui.app.application.main()` 保持 `python -m ui.main` CLI 兼容性。
- 离线场景下 `ui/services/av_service.py` 会切换到模拟模式，生成合成帧与静音音视频文件，以确保完整界面可预览。
- `ui/widgets/brain_load_bar.py` 改为纯 UI 组件，由后端分数驱动，也可选择模拟模式用于离线演示。
- 调试快捷键仍在主窗口内启用：`Ctrl+Alt+1/2/3` 分别跳转到登录/校准/测试页，便于在演示或调试时快速切换。
- 旧版采集/测试界面仍以 `legacy` 模块形式保留，后续如需完全废弃可直接移除相关文件。

新前端默认不再直接调用多模态采集脚本；所有采集、推理任务均在后端 Orchestrator 中运行，通过事件总线与接口服务对外暴露结果。需要二次开发时，可扩展新的接口（gRPC/REST/WebSocket）或在 `interfaces.yaml` 中启用多种输出通道。 
