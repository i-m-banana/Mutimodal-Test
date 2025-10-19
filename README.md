# Multimodal Platform (Refactored) 多模态平台（已重构）

## 一、项目概述
一个模块化、可配置、可扩展的多模态数据采集与检测平台，支持音视频、脑电（EEG）、多模态传感器等数据源的统一管理和智能分析，核心优势包括配置驱动、环境隔离、前后端解耦，适配团队协作与多场景测试需求。


## 二、核心功能特性
| 分类         | 具体能力                                                                 |
|--------------|--------------------------------------------------------------------------|
| 配置驱动     | 通过 YAML 文件灵活配置采集器、检测器、模型参数，支持启停 Collector/Detector/Model |
| 架构设计     | 基于事件总线的解耦架构，微服务模式部署多模型后端，WebSocket 标准化通信       |
| 多模态采集   | 支持摄像头、传感器、文件数据源，自动切换硬件/模拟模式                         |
| 模型管理     | PyTorch/ONNX 模型统一注册与推理调用，独立进程隔离运行（避免依赖冲突）         |
| 可观测性     | 标准化日志输出 + 系统心跳监控，便于运维排查问题                               |
| UI 特性      | Qt 响应式界面（适配 1280x720~4K 分辨率），实时监控与数据可视化                 |


## 三、架构设计
### 3.1 核心组件
| 组件类型     | 包含模块                                                                 |
|--------------|--------------------------------------------------------------------------|
| 后端服务（src/） | 调度中心、事件总线、系统监控、数据采集器（音视频/EEG/传感器）、检测器、统一推理服务、WebSocket 接口服务 |
| 前端界面（ui/）  | Qt 应用框架、响应式 UI 组件、实时监控面板、用户管理与测评系统、数据可视化模块       |
| 模型后端（model_backends/） | 抽象基类（base/）、疲劳度模型（fatigue_backend/）、情绪模型（emotion_backend/）、脑电模型（eeg_backend/） |
| 集成模型（src/models/） | 疲劳度模型（FatigueModel）、情绪模型（EmotionModel），支持集成模式直接在主进程中运行 |

### 3.2 数据流
```
UI → 主后端 (WebSocket:8765) → UnifiedInferenceService → 模型实例 (集成/远程)
       ↓                              ↓                        ↓
   数据采集                    模型推理调度              推理结果返回
       ↓                              ↓                        ↓
   检测结果聚合                  日志记录                UI显示结果
```

**两种模型部署模式**：
- **集成模式（Integrated）**：模型直接在主进程中运行，性能最优，内存共享
- **远程模式（Remote）**：模型运行在独立进程，通过WebSocket通信，环境隔离

**详细调用流程**：参见 `UI_TO_BACKEND_FLOW.md` 文档

### 3.3 端口分配
| 端口  | 用途                     |
|-------|--------------------------|
| 8765  | 主后端 WebSocket 接口    |
| 8767  | 疲劳度模型后端接口        |
| 8768  | 情绪模型后端接口          |
| 8769  | 脑电模型后端接口          |

### 3.4 架构优势
- **灵活部署**：支持集成模式（高性能）和远程模式（环境隔离），配置文件即可切换
- **环境隔离**：远程模式下每个模型运行在独立 Python 进程/环境，避免依赖冲突
- **可靠性**：模型崩溃不影响主系统，支持自动重连机制
- **扩展性**：统一 WebSocket 协议，快速新增模型后端
- **错误处理**：Future-based API + 超时控制 + 健康检查（生产级保障）
- **完整日志**：全链路推理日志记录，方便性能分析和问题排查（参见 `MODEL_INFERENCE_LOGGING.md`）


## 四、目录结构
```
project-root/
├── config/          # 配置文件（YAML 格式）
│   ├── system.yaml  # 系统参数（心跳间隔、超时设置）
│   ├── collectors.yaml  # 采集器配置（采样率、工作模式）
│   ├── detectors.yaml   # 检测器配置（阈值、启用状态）
│   └── app_config.yaml  # 模型部署配置（模式、后端地址、启用状态）
├── model_backends/  # 独立模型后端服务
│   ├── base/                 # 抽象基类（BaseModelBackend：WebSocket 服务器框架）
│   ├── fatigue_backend/      # 疲劳度模型（PyTorch 实现，多模态推理）
│   ├── emotion_backend/      # 情绪模型（PyTorch + Transformers，视觉+音频+文本）
│   └── eeg_backend/          # 脑电模型（待实现，MNE-Python 框架）
├── src/             # 后端源代码
│   ├── core/        # 核心模块（调度中心、事件总线、系统监控）
│   ├── collectors/  # 数据采集器（音视频、EEG、多模态传感器）
│   ├── detectors/   # 检测模块（事件订阅与处理逻辑）
│   ├── models/      # 集成模型（FatigueModel、EmotionModel）
│   ├── services/    # 后端服务（AV/EEG/TTS/数据库/统一推理服务）
│   ├── interfaces/  # 接口模块（WebSocket 服务器/客户端）
│   └── utils/       # 工具类（通用函数、格式处理）
├── ui/              # Qt 前端界面
│   ├── app/         # 应用核心（入口、配置、页面逻辑）
│   │   ├── utils/   # UI 工具（响应式缩放、监控弹窗）
│   │   └── pages/   # 页面组件（登录、校准、测试、成绩）
│   ├── data/        # 前端资源（用户数据、问卷配置）
│   ├── services/    # UI 服务（后端客户端、EEG/AV 代理）
│   ├── widgets/     # UI 组件（多模态预览、脑负荷条、仪表盘）
│   └── models/      # 前端模型推理（轻量场景）
├── requirements.txt # 运行依赖列表
├── QUICK_START.md   # 快速开始指南
├── UI_TO_BACKEND_FLOW.md        # UI 到后端的完整调用流程
├── MODEL_INFERENCE_LOGGING.md   # 模型推理日志配置说明
├── FRONTEND_BACKEND_BINDING.md  # 前后端绑定架构说明
└── README.md        # 项目总览文档（本文件）

注意：以下目录在运行时自动生成或用于本地开发，不会提交到 Git：
├── logs/            # 日志输出（运行时自动生成）
├── recordings/      # 录制文件（运行时生成）
├── data/            # 数据目录（本地数据）
├── debug/           # 调试脚本与测试工具（本地开发）
├── tests/           # 单元测试（本地开发）
├── scripts/         # 工具脚本（本地开发）
└── docs/            # 文档归档（本地开发）
```


## 五、快速开始
### 5.1 环境准备
1. **克隆仓库**
   ```bash
   git clone <repository-url>
   cd project-root
   ```

2. **创建并激活虚拟环境**
   ```bash
   # 创建虚拟环境
   python -m venv .venv

   # 激活环境（Windows）
   .venv\Scripts\activate
   # 激活环境（Linux/Mac）
   source .venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```


### 5.2 启动后端服务
```bash
# 根目录执行，默认监听 8765 端口（WebSocket 接口）
python -m src.main --root .

# 可选：关闭控制台监听输出
python -m src.main --root . --no-listeners
```
- 验证：控制台输出 `INFO  WebsocketPushInterface | Starting WebSocket interface on 127.0.0.1:8765` 表示启动成功


### 5.3 启动前端界面
#### 方式1：轻量级监控看板（仅展示结果）
```bash
python -m ui.dashboard
```

#### 方式2：完整功能界面
```bash
# 正常模式（依赖真实硬件/后端）
python -m ui.main

# 调试模式（启用模拟数据，无硬件也可运行）
python -m ui.main --debug
```


### 5.4 启动模型后端（可选 - 仅远程模式需要）

**注意**：如果配置文件中使用 `mode: integrated`（集成模式），模型会在主后端进程中运行，**无需启动独立的模型后端**。

只有在使用 `mode: remote`（远程模式）时，才需要手动启动独立的模型后端进程：

以「疲劳度模型后端」为例：
```bash
# 进入模型后端目录
cd model_backends/fatigue_backend

# 创建并激活独立虚拟环境（避免依赖冲突）
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装模型依赖
pip install -r requirements.txt

# 启动模型后端（默认端口8767）
python main.py
```

**情绪模型后端**：
```bash
cd model_backends/emotion_backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py  # 默认端口8768
```

**模式对比**：
| 特性 | 集成模式 (integrated) | 远程模式 (remote) |
|------|----------------------|-------------------|
| 启动方式 | 主后端自动加载 | 需手动启动独立进程 |
| 性能 | 最优（内存共享） | 略低（网络通信） |
| 隔离性 | 共享环境 | 完全隔离 |
| 稳定性 | 模型崩溃影响主进程 | 模型崩溃不影响主系统 |
| 适用场景 | 开发测试、单机部署 | 生产环境、分布式部署 |


## 六、配置说明
### 6.1 环境变量
| 变量名                  | 说明                                  | 默认值                |
|-------------------------|---------------------------------------|-----------------------|
| UI_FORCE_SIMULATION     | 全局启用模拟模式（所有模块）          | 0（关闭）             |
| UI_DEBUG_MODE           | 前端调试模式（显示日志/模拟数据）     | 0（关闭）             |
| BACKEND_EEG_SIMULATION  | 后端 EEG 模拟模式（无需蓝牙设备）     | 0（关闭）             |
| UI_SKIP_DATABASE        | 前端跳过数据库连接（仅演示用）        | 0（关闭）             |
| UI_DB_HOST              | 数据库主机地址                        | localhost             |
| UI_TTS_BACKEND          | 语音播报后端（Windows：powershell）   | powershell（Windows） |
| UI_WHISPER_MODEL        | Faster-Whisper 模型大小（语音识别）   | base                  |

#### 启用方式（Windows 示例）
```bash
# 临时启用（当前终端有效）
set UI_FORCE_SIMULATION=1

# 永久启用（需重启终端）
setx UI_FORCE_SIMULATION 1
```


### 6.2 配置文件
| 文件名                | 用途                                  | 核心配置项                |
|-----------------------|---------------------------------------|---------------------------|
| config/system.yaml    | 系统全局参数                          | 心跳间隔、超时时间        |
| config/collectors.yaml| 采集器配置                            | 采样率、设备类型、启用状态 |
| config/detectors.yaml | 检测器配置                            | 检测阈值、订阅事件主题    |
| config/app_config.yaml| 模型部署配置                          | 模型模式(集成/远程)、后端地址、启用状态 |

**模型配置示例** (`config/app_config.yaml`):
```yaml
model_backends:
  # 集成模式 - 模型在主进程中运行
  - name: "fatigue"
    type: "fatigue"
    enabled: true
    mode: "integrated"  # ✅ 推荐：性能最优
    
  # 远程模式 - 模型在独立进程中运行
  - name: "emotion"
    type: "emotion"
    enabled: true
    mode: "remote"
    remote_url: "ws://127.0.0.1:8768"  # 需手动启动emotion_backend
```


## 七、开发指南
### 7.1 扩展新采集器
1. 在 `src/collectors/` 新建类，继承 `BaseCollector`
2. 实现核心方法 `run_once()`（定义采集逻辑）
3. 在 `config/collectors.yaml` 注册采集器（示例）：
   ```yaml
   - name: new_collector
     type: src.collectors.NewCollector
     enabled: true
     sample_rate: 30  # 自定义参数
   ```


### 7.2 扩展新模型
#### 方式1：集成模式（推荐）
1. 在 `src/models/` 新建模型类，继承 `BaseInferenceModel`
2. 实现 `initialize()` 和 `infer()` 方法
3. 在 `config/app_config.yaml` 配置：
   ```yaml
   model_backends:
     - name: "new_model"
       type: "new_model"
       enabled: true
       mode: "integrated"
   ```

#### 方式2：远程模式（环境隔离）
1. 在 `model_backends/` 新建目录（如 `new_model_backend/`）
2. 继承 `base/base_backend.py` 中的 `BaseModelBackend`
3. 实现 `initialize_model()` 和 `infer()` 推理方法
4. 在 `config/app_config.yaml` 配置：
   ```yaml
   model_backends:
     - name: "new_model"
       type: "new_model"
       enabled: true
       mode: "remote"
       remote_url: "ws://127.0.0.1:8770"
   ```
5. 手动启动后端：`python model_backends/new_model_backend/main.py`


### 7.3 扩展新检测器
1. 在 `src/detectors/` 新建类，继承 `BaseDetector`
2. 声明订阅的事件主题（如 `camera.frame`）
3. 实现 `handle_event()` 方法（定义检测逻辑）


## 八、参考文档
项目根目录包含以下核心文档：

| 文档名称 | 说明 |
|---------|------|
| `README.md` | 项目总览文档（本文件） |
| `QUICK_START.md` | 快速启动指南（简化版） |
| `UI_TO_BACKEND_FLOW.md` | UI通过WebSocket调用模型的完整流程（含代码示例） |
| `MODEL_INFERENCE_LOGGING.md` | 模型推理日志配置与使用说明 |
| `FRONTEND_BACKEND_BINDING.md` | 前后端绑定架构说明 |

**注意**：更多开发文档（测试指南、架构设计、调试工具等）位于本地 `debug/`、`docs/`、`tests/` 目录，不会提交到 GitHub。如需这些文档，请联系项目维护者或查看本地开发环境。


## 九、模拟模式说明
### 9.1 启用方式
| 启用方式       | 操作命令（Windows 示例）                          | 适用场景                  |
|----------------|---------------------------------------------------|---------------------------|
| 启动参数（推荐）| `python -m ui.main --debug`                       | 快速调试、无硬件演示      |
| 全局环境变量   | `set UI_FORCE_SIMULATION=1`                       | 所有模块统一模拟          |
| 细分控制       | `set UI_MULTIMODAL_SIMULATION=1`（仅多模态模拟）  | 单独测试某模块            |


### 9.2 模拟功能支持
- ✅ 合成音视频数据（渐变 RGB 画面、静音音频）
- ✅ 模拟传感器读数（随机波动数值）
- ✅ 模拟 EEG 信号（正弦波叠加噪声，500Hz 采样）
- ✅ 跳过硬件连接（自动忽略蓝牙/摄像头设备检查）


## 十、迁移与兼容性说明
### 10.1 与旧版（Multimodal-Project）差异
- **UI 解耦**：UI 层不再依赖硬件库（如 EEG 的 `bleak`），通过 WebSocket 调用后端服务
- **资源管理**：前端资源（CSV/YAML）统一迁移到 `ui/data/`，路径配置集中在 `ui/app/config.py`
- **模型隔离**：多模型后端独立部署，避免 Python 环境依赖冲突


### 10.2 后端启动问题排查
- 端口占用：执行 `netstat -ano | findstr 8765`（Windows）查看占用进程，关闭后重启
- WebSocket 依赖缺失：执行 `pip install websockets` 补充安装
- 防火墙拦截：关闭本地防火墙或添加端口例外（8765~8768）


## 十一、常见问题（FAQ）

### Q1: 集成模式和远程模式如何选择？
**A**: 
- **集成模式**（推荐）：适合单机部署、开发测试，性能最优，配置简单，无需启动独立进程
- **远程模式**：适合生产环境、分布式部署，环境完全隔离，模型崩溃不影响主系统

### Q2: 如何查看模型推理日志？
**A**: 
- 集成模式日志：`logs/model/model.log`
- 远程模式日志：
  - 疲劳度：`logs/model/fatigue_backend.log`
  - 情绪：`logs/model/emotion_backend.log`
- 日志内容包括：推理分数、耗时、数据量统计等
- 详细说明参见 `MODEL_INFERENCE_LOGGING.md`

### Q3: UI如何调用后端模型？
**A**: 
两种调用方式：
1. **命令模式**：通过 `backend_proxy.emotion_analyze()` 发送命令
2. **推理模式**：直接发送 `model_inference` WebSocket消息

完整调用流程和代码示例参见 `UI_TO_BACKEND_FLOW.md`

### Q4: 模型后端启动失败怎么办？
**A**: 
1. 检查端口占用：`netstat -ano | findstr 8767`
2. 检查依赖安装：`pip install -r requirements.txt`
3. 检查配置文件：确认 `config/app_config.yaml` 中模型启用且端口正确
4. 查看日志：检查 `logs/model/` 目录下对应的日志文件

### Q5: 如何切换模型部署模式？
**A**: 
修改 `config/app_config.yaml`：
```yaml
model_backends:
  - name: "fatigue"
    mode: "integrated"  # 改为 "remote" 即可切换
```
切换到远程模式后需要手动启动对应的模型后端进程。


## 十二、许可证与贡献
- **许可证**：本项目采用 MIT 许可证（可自由使用、修改，需保留版权声明）
- **贡献方式**：欢迎提交 Issue（问题反馈）或 Pull Request（代码贡献）
- **联系方式**：通过 GitHub Issues 沟通，或联系项目维护者