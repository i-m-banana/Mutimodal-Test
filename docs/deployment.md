# 部署指南

## 1. 环境准备

1. 建议使用 Python 3.11+。
2. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
. .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
pip install -r requirements.txt
```

> 若仅部署基础管线且无需真实模型，可忽略 `torch` / `onnxruntime` 等重量依赖，系统将以桩实现运行。

## 2. 目录结构

```
project-root/
├── config/        # 组件配置
├── data/          # 数据输入输出
├── logs/          # 自动创建的日志目录
├── src/           # 后端源代码
├── ui/            # Qt 前端（监控看板 + 兼容层）
└── tests/         # 测试脚本
```

部署时需确保 `data/raw`、`data/processed`、`data/results` 可读写，用于存放采集与检测结果。

## 3. 配置调整

- `config/system.yaml`：系统级参数（心跳间隔、停止超时时间等）。
- `config/collectors.yaml`：采集器列表及运行参数。
- `config/detectors.yaml`：检测器启用状态及阈值设置。
- `config/models.yaml`：模型路径、类型与启用状态。
- `config/interfaces.yaml`：对外接口（如 WebSocket 推流）的启用与端口配置。

修改配置后无需修改代码即可扩展组件。

## 4. 运行服务

```bash
python -m src.main --root ./project-root
```

- 默认会订阅 `detector.result` 与 `system.heartbeat` 事件并在控制台打印。
- 使用 `--no-listeners` 可关闭控制台监听，将输出交由上层系统处理。
- 若与 UI 协同运行，请保持该进程常驻，以便提供真实音视频采集服务。

## 5. 启动前端（可选）

启动前端看板前，请确保 Orchestrator 已运行并启用了 WebSocket 接口：

```yaml
interfaces:
	- name: websocket
		class: interfaces.websocket_server.WebsocketPushInterface
		enabled: true
		options:
			host: 127.0.0.1
			port: 8765
```

```bash
python -m ui.dashboard
```

看板仅展示后端推送的检测结果与系统状态，不会直接驱动采集器或检测器。若在无 GUI 环境运行，可跳过此步骤，仅依赖后端事件总线。

如需启动完整问答流程 UI，可选择以下模式：

```powershell
cd F:\mutilUI\project-root
python -m ui.main           # 正常模式：使用真实设备/后台
python -m ui.main --debug   # 调试模式：启用模拟信号并窗口化
```

- 调试模式会在当前进程内设置 `UI_DEBUG_MODE=1`，并同步 `UI_FORCE_SIMULATION=1`，避免未连接后端时出现错误。
- 正常模式需确保后端 `python -m src.main --root .` 已成功启动，否则 UI 会提示连接失败。
- 也可通过 `python -m ui.main --mode normal` / `--mode debug` 显式切换，或设定环境变量 `UI_FORCE_SIMULATION=1` 强制启用模拟信号。

## 6. 集成外部系统

可直接订阅 `EventTopic.DETECTION_RESULT` 与 `EventTopic.SYSTEM_HEARTBEAT`，或在 `src/main.py` 中注入自定义监听逻辑。

## 7. 停止服务

- 使用 `Ctrl+C` 触发优雅退出，Orchestrator 会依次停止采集器、检测器并卸载模型。
- 异常退出时可检查 `logs/collector/*.log`、`logs/model/*.log`、`logs/detector/*.log` 获取详细信息。

## 8. 测试与验证

```bash
pytest
```

单元测试覆盖关键逻辑，集成测试会拉起 Orchestrator 并验证事件链路。

## 9. 常见问题

- **OpenCV 或音频库缺失**：`CameraCollector` 会自动回退至模拟模式，不影响流水线。
- **模型文件未准备**：`ModelManager` 会返回桩推理结果，确保流程不中断，可在部署完成后替换真实模型文件并重新加载。
- **PyAudio/设备驱动安装失败**：Windows 可使用 `pip install pipwin && pipwin install pyaudio` 安装预编译轮子；Linux/macOS 需提前安装 PortAudio。
- **WebSocket 接口连接不上**：检查 `config/interfaces.yaml` 的端口是否被占用，必要时修改 `host/port`，并确认后端日志中 `WebSocket interface on ...` 已成功启动。
