# 模块接口说明

## 1. Collectors

所有采集器继承自 `collectors.base_collector.BaseCollector`，需要实现：

- `run_once(self) -> None`：单次采集逻辑；采集完成后自行将数据封装为事件并通过 `self.bus.publish(...)` 对外广播。
- 可选重载：`on_start(self)` / `on_stop(self)` 进行资源初始化与释放。
- 支持配置项示例：

```yaml
collectors:
  - name: camera
    class: collectors.camera_collector.CameraCollector
    options:
      mode: simulator  # simulator / hardware
      interval: 0.5    # 采集频率（秒）
      width: 320
      height: 240
```

事件主题：

| Collector | Topic | Payload 关键字段 |
| --------- | ----- | ---------------- |
| `CameraCollector` | `camera.frame` | `sequence`, `timestamp`, `encoding`, `data` |
| `SensorCollector` | `sensor.packet` | `blood_pressure`, `pulse`, `eeg_alpha`, `stress_index` |
| `FileCollector` | `file.batch` | `files`（新增文件路径列表） |

## 2. Model Manager

`models.model_manager.ModelManager` 暴露：

- `load_enabled()`：读取配置并初始化所有启用的模型。
- `infer(model_name: str, payload: dict) -> dict`：执行一次推理并返回标准化结果。
- `unload_all()`：释放所有模型资源。

模型配置格式：

```yaml
models:
  - name: vision
    class: models.torch_model.TorchModel
    path: models/versions/vision.pt
    enabled: true
    options:
      jit: true
```

## 3. Detectors

检测器继承自 `detectors.base_detector.BaseDetector`，需实现：

- `topics(self) -> Iterable[EventTopic]`：订阅的事件主题列表。
- `handle_event(self, event)`：对事件进行处理，必要时调用 `self.model_manager.infer(...)`。

标准事件输出：

| Detector | Input Topics | Output Topic | Payload |
| -------- | ------------ | ------------ | ------- |
| `ObjectDetector` | `camera.frame` | `detector.result` | `confidence`, `model`, `source_sequence` |
| `AnomalyDetector` | `sensor.packet` | `detector.result` | `severity`, `metrics` |
| `OCRDetector` | `file.batch` | `detector.result` | `files`, `summaries` |

## 4. Orchestrator

`core.orchestrator.Orchestrator` 对外能力：

- `start()` / `stop()`：启动或停止所有组件。
- `run_for(seconds: float)`：在指定时长内运行（测试用）。
- `bus`：事件总线实例，可用于外部监听。
- `model_manager`：模型管理器实例，可查询当前加载的模型列表。

支持从目录加载配置：

```python
from src.core.orchestrator import Orchestrator

orch = Orchestrator.from_config_directory("project-root")
orch.start()
```

## 5. 事件主题枚举

`src/constants.py` 定义了项目内部统一事件名称：

- `EventTopic.CAMERA_FRAME`
- `EventTopic.AUDIO_LEVEL`
- `EventTopic.SENSOR_PACKET`
- `EventTopic.FILE_BATCH`
- `EventTopic.MODEL_REQUEST`
- `EventTopic.MODEL_RESPONSE`
- `EventTopic.DETECTION_RESULT`
- `EventTopic.SYSTEM_HEARTBEAT`

确保新增模块使用枚举值，避免 Topic 拼写不一致。
