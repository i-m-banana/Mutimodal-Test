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
| 后端服务（src/） | 调度中心、事件总线、系统监控、数据采集器（音视频/EEG/传感器）、检测器、推理引擎、WebSocket 接口服务 |
| 前端界面（ui/）  | Qt 应用框架、响应式 UI 组件、实时监控面板、用户管理与测评系统、数据可视化模块       |
| 模型后端（model_backends/） | 抽象基类（base/）、疲劳度模型（fatigue_backend/）、情绪模型（emotion_backend/）、脑电模型（eeg_backend/） |

### 3.2 数据流
```
UI → 主后端 (EventBus) → ModelProxyService → 模型后端 (隔离环境)
       ↓                        ↓
   数据采集               检测结果聚合
```

### 3.3 端口分配
| 端口  | 用途                     |
|-------|--------------------------|
| 8765  | 主后端 WebSocket 接口    |
| 8767  | 疲劳度模型后端接口        |
| 8768  | 情绪模型后端接口          |
| 8769  | 脑电模型后端接口          |

### 3.4 架构优势
- **环境隔离**：每个模型运行在独立 Python 进程/环境，避免依赖冲突
- **可靠性**：模型崩溃不影响主系统，支持自动重连机制
- **扩展性**：统一 WebSocket 协议，快速新增模型后端
- **错误处理**：Future-based API + 超时控制 + 健康检查（生产级保障）


## 四、目录结构
```
project-root/
├── aidebug/         # 调试文档与测试脚本（含快速开始、架构设计、接口说明等）
├── config/          # 配置文件（YAML 格式）
│   ├── system.yaml  # 系统参数（心跳间隔、超时设置）
│   ├── collectors.yaml  # 采集器配置（采样率、工作模式）
│   ├── detectors.yaml   # 检测器配置（阈值、启用状态）
│   └── models.yaml      # 模型配置（路径、参数、后端关联）
├── data/            # 数据目录（原始数据、处理结果、日志输出）
├── docs/            # 架构与部署文档
├── logs/            # 日志输出（运行时自动生成）
├── model_backends/  # 独立模型后端服务
│   ├── base/                 # 抽象基类（BaseModelBackend：WebSocket 服务器框架）
│   ├── fatigue_backend/      # 疲劳度模型（PyTorch 实现，多模态推理）
│   ├── emotion_backend/      # 情绪模型（PyTorch + Transformers，视觉+音频+文本）
│   └── eeg_backend/          # 脑电模型（待实现，MNE-Python 框架）
├── src/             # 后端源代码
│   ├── core/        # 核心模块（调度中心、事件总线、系统监控）
│   ├── collectors/  # 数据采集器（音视频、EEG、多模态传感器）
│   ├── detectors/   # 检测模块（事件订阅与处理逻辑）
│   ├── services/    # 后端服务（AV/EEG/TTS/数据库/模型代理）
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
├── tests/           # 单元与集成测试
│   ├── test_model_backend_architecture.py  # 模型后端架构测试
│   ├── test_responsive.py                  # 响应式 UI 测试
│   └── test_eeg_integration.py             # EEG 服务集成测试
├── pyproject.toml   # 项目配置（依赖与构建）
├── requirements.txt # 运行依赖列表
└── README.md        # 项目首页说明
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


### 5.4 启动模型后端（可选）
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
| config/models.yaml    | 模型配置                              | 模型路径、后端关联、参数  |


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
1. 在 `model_backends/` 新建目录（如 `new_model_backend/`）
2. 继承 `base/base_backend.py` 中的 `BaseModelBackend`
3. 实现 `infer()` 推理方法与 WebSocket 通信逻辑
4. 在 `config/models.yaml` 关联模型与后端：
   ```yaml
   - name: new_model
     path: ./models/new_model.pth
     backend_port: 8769  # 新模型后端端口
   ```


### 7.3 扩展新检测器
1. 在 `src/detectors/` 新建类，继承 `BaseDetector`
2. 声明订阅的事件主题（如 `camera.frame`）
3. 实现 `handle_event()` 方法（定义检测逻辑）


## 八、测试与文档
### 8.1 运行测试
```bash
# 运行所有测试
pytest

# 运行特定模块测试
python tests/test_model_backend_architecture.py  # 模型后端架构
python tests/test_responsive.py                  # 响应式 UI
```


### 8.2 参考文档
所有详细文档位于 `aidebug/` 目录：
- `QUICK_START.md`：快速启动指南（简化版）
- `MULTI_MODEL_BACKEND_ARCHITECTURE.md`：多模型后端架构设计
- `MULTI_MODEL_BACKEND_INTERFACES.md`：WebSocket 接口协议说明
- `TESTING_GUIDE.md`：测试用例编写指南
- `RESOLUTION_ADAPTATION.md`：UI 分辨率适配说明


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


## 十一、许可证与贡献
- **许可证**：本项目采用 MIT 许可证（可自由使用、修改，需保留版权声明）
- **贡献方式**：欢迎提交 Issue（问题反馈）或 Pull Request（代码贡献）
- **联系方式**：通过 GitHub Issues 沟通，或联系项目维护者