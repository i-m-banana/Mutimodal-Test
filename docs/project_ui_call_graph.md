# 项目与 UI 调用关系梳理

## 总体结构
- `src/main.py` 是后台程序（Orchestrator）的 CLI 入口，负责读取配置、启动采集器、检测器、模型与接口，并共享一个全局的 `EventBus`。
- UI 侧使用 `python -m ui.main` 启动，以 PyQt5 构建交互界面，通过内部服务模块与后台通信。

## 事件总线与核心流程
- `Orchestrator` 通过 YAML 配置实例化 collectors、detectors、interfaces，并将它们注册到 `EventBus`。
- 采集器（如 `src/collectors/camera_collector.py`）定时采集数据并发布到事件总线。
- 检测器（`src/detectors/*`）订阅相关主题，执行模型推理，将结果写回 `detector.result` 等主题。
- `SystemMonitor` 按配置发布 `system.heartbeat`，用于健康监控。

## 后台接口与 UI 通道
- `src/interfaces/websocket_server.py` 监听配置的主题，将事件推送给 WebSocket 客户端，同时接收客户端命令并包装成 `EventTopic.UI_COMMAND` 事件。
- `src/services/ui_command_router.py` 订阅 `UI_COMMAND`，根据 action 分发给：
  - `AVService`（音视频采集与录制）
  - `DatabaseService`（MySQL 记录读写）
  - `TTSService`（文本转语音）
- 处理结果包装成 `UI_RESPONSE`，再次经 WebSocket 回传 UI。

## UI 侧调用链
- `ui/services/backend_client.py` 建立 WebSocket 连接，维护命令 Future 与事件信号，是 UI 与后台的唯一通信层。
  - 监听后台 `event` 消息并发射 Qt 信号（如 `camera_frame`、`audio_level`、`detection_result`）。
  - 向后台发送命令（`send_command_sync`/`future`），等待 `response`。
- `ui/get_avdata.py` 封装音视频相关操作：
  - 正常模式下调用 `BackendClient` 触发 `av.*` 命令，接收帧/音量信号更新界面。
  - 调试模式下提供本地模拟数据。
- `ui/main.py`（PyQt 主程序）集成多页面逻辑：
  - 登录后通过 `TestPage` 引导语音问答、血压、舒特格测试等流程。
  - 通过 `get_backend_client()` 请求 TTS、数据库写入（`db.*`）、音视频录制。
  - 借助 `thread_process_manager` 提供的线程/进程池执行耗时任务并与 UI 解耦。
- `ui/services/backend_client.py` 将后台 `UI_RESPONSE` 转化为 Future 结果/错误提示，使 UI 同步更新状态、提示异常。

## 数据与配置交互
- 后台 `AVService` 在 `session_dir` 下生成音视频段；UI 侧通过 `av.list_paths` 获取并缓存路径。
- 数据库交互由 `DatabaseService` 调用 `ui.database.TestTableStore` 完成；UI 通过 `_queue_db_update` 等方法将流程结果写入。
- `TTSService` 支持 PowerShell 与 `pyttsx3` 后端，UI 侧通过 `tts.speak` 调用，并在 `TestPage` 中排队朗读问答题目。

## 典型调用链举例
1. UI 点击“开始录音” → `TestPage` 调用 `get_avdata.start_recording()` → `BackendClient` 发送 `av.start_recording` → `UICommandRouter` 调用 `AVService.start_recording()`，成功后 UI 继续更新录制状态。
2. 检测器产出结果 → 在事件总线上发布 `detector.result` → `WebsocketPushInterface` 推送给 UI → `BackendClient.detection_result` 信号触发界面更新。
3. UI 请求保存测试结果 → `_queue_db_update` 构造 payload → 发送 `db.update_test_record` → `DatabaseService` 写入 MySQL → 返回 `UI_RESPONSE`，UI 收到后记录成功或错误。

---
如需补充具体模块源文件或绘制流程图，可在此文档基础上继续扩展。