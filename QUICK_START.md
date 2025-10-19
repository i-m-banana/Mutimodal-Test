# 🚀 快速启动指南

## 系统要求

- Python 3.11+
- Windows 10/11 / Linux / macOS
- 所需依赖包已安装

## 📋 启动步骤

### 1. 环境准备

首次使用前需要安装依赖：

```bash
# 进入项目根目录
cd d:\duomotai\project-root-cut

# 安装核心依赖
pip install -r requirements.txt

# 如果需要使用UI，安装额外依赖
pip install pyqt5 qtawesome opencv-python pyaudio numpy matplotlib
```

### 2. 启动后端服务

打开命令行，运行:
```bash
cd d:\duomotai\project-root-cut
python -m src.main --root .

# 可选：关闭控制台监听输出（减少日志）
python -m src.main --root . --no-listeners

# 可选：显示全部事件日志
python -m src.main --root . --full-events
```

**期望输出**:
```
INFO  WebsocketPushInterface | Starting WebSocket interface on 127.0.0.1:8765
INFO  orchestrator | Orchestrator started
INFO  inference | ✅ 统一推理服务已启动 (共 3 个模型)
```

### 3. 启动UI应用

另开一个命令行窗口，运行:
```bash
cd d:\duomotai\project-root-cut
python -m ui.main

# 可选：启用调试模式（使用模拟数据，无需硬件）
python -m ui.main --debug
```

**期望输出**:
```
应用程序主窗口初始化完成
已成功连接到后端服务器 ws://127.0.0.1:8765
```

## 🔌 模型部署模式

系统使用**集成模式（Integrated）**部署所有推理模型，模型直接在后端主进程中运行，**无需额外启动独立进程**：

```yaml
inference_models:
  - name: fatigue
    type: fatigue
    mode: integrated  # 集成模式
    enabled: true
```

**优点**：
- ✅ 性能最优（无网络开销）
- ✅ 配置简单（自动启动）
- ✅ 内存共享（效率高）
- ✅ 便于调试和维护

配置文件位置：`config/models.yaml`

## 🎯 基本使用流程

### 使用UI系统

1. **登录**
   - 默认用户名: `admin`
   - 默认密码: `admin123`
   - 用户数据存储在 `ui/data/users/users.csv`

2. **校准**
   - 等待摄像头初始化（自动检测或使用模拟模式）
   - 调整摄像头位置确保人脸清晰可见
   - 点击"校准完成"进入测试阶段

3. **测试**
   - 系统自动开始采集多模态数据
   - 实时推理疲劳度、情绪、脑负荷等指标
   - 数据自动保存到 `recordings/用户名/时间戳/` 目录

4. **查看结果**
   - 测试完成后查看评估结果和统计图表
   - 历史记录可在系统中查询

## 🔧 常见问题

### 问题1: 后端服务无法启动

**现象**: `端口 8765 已被占用`

**解决**:
```bash
# Windows 查找占用端口的进程
netstat -ano | findstr :8765
# 结束该进程
taskkill /F /PID <进程ID>

# Linux/macOS
lsof -i :8765
kill -9 <PID>
```

### 问题2: UI无法连接后端

**现象**: `Backend connection not established` 或 `连接超时`

**解决**:
1. 确认后端服务已启动（查看控制台输出）
2. 检查防火墙是否阻止了 8765 端口
3. 确认 `config/interfaces.yaml` 中的地址配置正确
4. 尝试使用 `python -m ui.main --debug` 启用调试模式

### 问题3: 摄像头初始化失败

**现象**: `无法打开摄像头` 或 `Camera not found`

**解决方案1 - 使用模拟模式**:
```bash
# Windows
set UI_FORCE_SIMULATION=1
python -m ui.main

# Linux/macOS
export UI_FORCE_SIMULATION=1
python -m ui.main
```

**解决方案2 - 指定摄像头索引**:
```bash
# 尝试不同的摄像头索引（0, 1, 2...）
set UI_CAMERA_INDEX=1
python -m ui.main
```

### 问题4: 模型推理失败

**现象**: `Model inference failed` 或 `模型未响应`

**解决**:
1. 检查 `config/models.yaml` 中模型是否启用
2. 查看 `logs/model/` 目录下的日志文件
3. 确认模型文件是否存在于 `models_data/` 目录
4. 检查 GPU/CUDA 环境（如果使用 GPU）

### 问题5: EEG设备连接失败

**现象**: `EEG device not found` 或 `BLE connection failed`

**解决**:
```bash
# 使用EEG模拟模式（不需要真实硬件）
set BACKEND_EEG_SIMULATION=1
python -m src.main --root .
```

### 问题6: 依赖包缺失

**现象**: `ModuleNotFoundError: No module named 'xxx'`

**解决**:
```bash
# 重新安装依赖
pip install -r requirements.txt

# 如果是UI相关的依赖
pip install pyqt5 qtawesome opencv-python pyaudio numpy matplotlib

# 如果是模型相关的依赖
pip install torch torchvision torchaudio faster-whisper opencc
```

## 🐛 调试与日志

### 启用调试模式

**UI调试**:
```bash
python -m ui.main --debug
```

**后端全量日志**:
```bash
python -m src.main --root . --full-events
```

### 日志文件位置

| 日志类型 | 路径 | 说明 |
|---------|------|------|
| UI日志 | `ui/logs/app_log_*.txt` | UI应用运行日志 |
| 后端核心 | `logs/orchestrator.log` | 后端主进程日志 |
| 采集器 | `logs/collector/*.log` | 数据采集器日志 |
| 检测器 | `logs/detector/*.log` | 检测器日志 |
| 模型推理 | `logs/model/*.log` | 模型推理日志（含性能统计）|
| WebSocket | `logs/interface/*.log` | WebSocket通信日志 |

### 性能监控

**查看模型推理统计**:
```bash
# 查看最近的推理日志
type logs\model\model.log | findstr "推理"

# Linux/macOS
grep "推理" logs/model/model.log
```

**测试后端连接**:
```bash
# 使用 wscat 工具测试 WebSocket
npm install -g wscat
wscat -c ws://127.0.0.1:8765

# 发送心跳测试
{"type": "ping"}
```

## 📝 环境变量配置

### 常用环境变量

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `UI_FORCE_SIMULATION` | UI全局模拟模式（所有硬件） | 0 | `set UI_FORCE_SIMULATION=1` |
| `BACKEND_EEG_SIMULATION` | 后端EEG模拟模式 | 0 | `set BACKEND_EEG_SIMULATION=1` |
| `UI_CAMERA_INDEX` | 指定摄像头索引 | 0 | `set UI_CAMERA_INDEX=1` |
| `UI_DEBUG` | UI调试模式 | 0 | `set UI_DEBUG=1` |
| `UI_SKIP_DATABASE` | 跳过数据库连接 | 0 | `set UI_SKIP_DATABASE=1` |
| `BACKEND_KEY_INFO_MODE` | 后端关键信息模式（减少日志） | 1 | `set BACKEND_KEY_INFO_MODE=0` |

### 设置方式

**Windows (临时)**:
```bash
set UI_FORCE_SIMULATION=1
python -m ui.main
```

**Windows (永久)**:
```bash
setx UI_FORCE_SIMULATION 1
# 需要重启终端生效
```

**Linux/macOS (临时)**:
```bash
export UI_FORCE_SIMULATION=1
python -m ui.main
```

**Linux/macOS (永久)**:
```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
echo 'export UI_FORCE_SIMULATION=1' >> ~/.bashrc
source ~/.bashrc
```

### 模拟模式说明

系统支持多层次的模拟模式，适用于无硬件环境下的开发和测试：

- **`UI_FORCE_SIMULATION=1`**: UI层全局模拟，包括摄像头、音频、多模态传感器
- **`BACKEND_EEG_SIMULATION=1`**: 后端EEG服务使用模拟数据（无需蓝牙设备）
- **`UI_EEG_SIMULATION=1`**: UI层EEG模拟（后端不可用时的降级方案）

**推荐组合**：
```bash
# 完全无硬件环境（开发/演示）
set UI_FORCE_SIMULATION=1
set BACKEND_EEG_SIMULATION=1

# 仅模拟EEG（其他硬件正常）
set BACKEND_EEG_SIMULATION=1

# 仅模拟摄像头
set UI_FORCE_SIMULATION=1
```

## 🎓 开发者指南

### 代码修改后重启流程

1. **停止服务**
   ```bash
   # 在对应的终端窗口按 Ctrl+C
   # 或者强制结束进程
   taskkill /F /IM python.exe  # Windows (慎用，会结束所有python进程)
   ```

2. **重新启动**
   ```bash
   # 先启动后端
   python -m src.main --root .
   
   # 再启动UI
   python -m ui.main
   ```

### 配置文件说明

主要配置文件位于 `config/` 目录：

| 文件 | 用途 |
|------|------|
| `system.yaml` | 系统全局配置（心跳间隔、超时设置、UI自动启动等） |
| `collectors.yaml` | 数据采集器配置（采样率、设备参数） |
| `detectors.yaml` | 检测器配置（阈值、订阅主题） |
| `models.yaml` | 模型配置（部署模式、模型路径、启用状态） |
| `interfaces.yaml` | 接口配置（WebSocket地址、端口） |

### 项目目录结构

```
project-root-cut/
├── src/                        # 后端源代码
│   ├── core/                   # 核心模块（事件总线、调度器、监控）
│   ├── collectors/             # 数据采集器（摄像头、传感器、文件）
│   ├── detectors/              # 检测器（异常检测、对象检测、OCR等）
│   ├── models/                 # 集成模型（疲劳、情绪、EEG）
│   ├── services/               # 后端服务（统一推理、EEG、AV、命令路由）
│   ├── interfaces/             # 通信接口（WebSocket服务器/客户端）
│   ├── devices/                # 设备抽象（EEG、Tobii、麦博等）
│   └── utils/                  # 工具类
├── ui/                         # 前端UI应用
│   ├── app/                    # 应用核心（页面、配置、工具）
│   │   ├── pages/              # 页面组件（登录、校准、测试）
│   │   └── utils/              # UI工具（响应式、监控弹窗）
│   ├── data/                   # UI资源数据（用户、问卷）
│   ├── services/               # UI服务（后端客户端、启动器、代理）
│   ├── widgets/                # UI组件（脑负荷条、仪表盘、多模态预览）
│   └── models/                 # 前端轻量推理
├── model_backends/             # 独立模型后端（已移除）
├── config/                     # 配置文件（YAML格式）
├── models_data/                # 模型权重文件和数据
└── logs/                       # 运行时日志（自动生成）
```

### 添加新功能

**添加新的数据采集器**:
1. 在 `src/collectors/` 创建新文件继承 `BaseCollector`
2. 实现 `run_once()` 方法
3. 在 `config/collectors.yaml` 注册

**添加新的检测器**:
1. 在 `src/detectors/` 创建新文件继承 `BaseDetector`
2. 实现 `handle_event()` 方法
3. 在 `config/detectors.yaml` 注册

**添加新的推理模型**:
1. 在 `src/models/` 创建新文件继承 `BaseInferenceModel`
2. 实现 `initialize()` 和 `infer()` 方法
3. 在 `config/models.yaml` 配置

## 📞 获取帮助

### 参考文档
- **AGENTS.MD**: 编码规范和开发准则
- **pyproject.toml**: 项目依赖和元数据配置

### 问题反馈
- 查看 `logs/` 目录下的日志文件
- 在项目仓库提交 Issue
- 联系项目维护团队

---

**最后更新**: 2025年10月16日  
**项目版本**: 0.1.0  
**Python版本**: 3.11+
