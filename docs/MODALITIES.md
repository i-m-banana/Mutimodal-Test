# 项目模态数据说明

下面表格按你提供的字段列出含义、来源与是否可能为模拟数据（synthetic）：

| 字段 | 含义 | 来源 | 模拟（或有占位实现） |
|---|---|---|---|
| video | 情绪模块的视频流 | 默认相机（ `av_service.py`） | 有 |
| audio | 麦克风音频流 | 麦克风（ `av_service`） | 有 |
| recoed | 语音识别得到的文本 | 语音识别服务（`faster whisper`） | 有 |
| rgb | RealSense 的可见光帧 | RealSense（`multimodal_service.py`） | 有 |
| depth | RealSense 的深度帧 | RealSense（`multimodal_service.py`） | 有 |
| tobii | 眼动追踪样本（注视点等） | Tobii 设备（`multimodal_service.py`） |有 |
| eeg1 | EEG 通道 1 信号 | 脑电额贴（`eeg_service.py`） | 有 |
| eeg2 | EEG 通道 2 信号 | 脑电额贴（`eeg_service.py`） | 有 |
| blood | 高压/低压/心率（systolic/diastolic/pulse） | Maibobo 血压设备（`bp_service.py`） | 有 |
| sart | SART 测试分数 | UI（`sart.py`） | 无 |
| accuracy | 舒尔特任务正确率 | UI（`schulte_grid.py`）| 有 |
| elapsed | 舒尔特任务用时 | UI（`schulte_grid.py`）| 有 |
| name | 被试姓名 | UI（`login.py`） | 有 |
| datetime | 测试时间戳 | 测试开始自动生成（`test.py`） | 无 |

---

## 数据字段详细说明（按表格顺序）

下面为每个字段逐项给出：

- 来源（存放采集或读取逻辑的脚本）
- 用途（调用/消费该数据的脚本或服务）
- 模拟（哪些脚本实现模拟/占位逻辑）
- 切换模拟数据的方法（环境变量、命令行或接口 payload）

---

### video
- 来源：`ui/services/av_service.py`（UI 代理）与 `src/services/av_service.py`（后端采集/发布帧）
- 用途：`src/services/emotion_service.py`、`src/services/unified_inference_service.py`（情绪/疲劳推理订阅 `MULTIMODAL_FRAME` / `MULTIMODAL_SNAPSHOT`）
- 模拟：UI 的 `_generate_simulated_frame` / `_write_dummy_video`；后端的 `_enable_synthetic_camera` / `_generate_synthetic_frame`
- 切换方法：设置环境变量 `UI_FORCE_SIMULATION=1` 或 `UI_MULTIMODAL_SIMULATION=1`；后端也可通过 `BACKEND_ALLOW_SYNTHETIC_CAMERA` 控制是否允许合成。`--debug` 在后端/后端不可用时会作为回退条件启用模拟。

---

### audio
- 来源：`ui/services/av_service.py`（UI 端音量 & 录音代理）与 `src/services/av_service.py`（后端实时录音与回调 `_audio_callback`）
- 用途：`src/services/emotion_service.py`（情绪分析）、`speech_recognition_service`（转写）、`unified_inference_service.py`（作为多模态输入）
- 模拟：UI 的 `_simulation_loop` 会周期性生成 `_audio_level`；UI `_write_silent_wav` 用于模拟录音片段；后端在无法打开音频设备时会记录警告并可回退到软件生成的占位。
- 切换方法：设置 `UI_FORCE_SIMULATION=1` 强制 UI 端模拟。若后端音频设备故障，配置或日志会显示回退；`--debug` 在后端不可用时可触发 UI 端模拟回退。

---

### recoed（语音识别文本）
- 来源：`speech_recognition_service`（UI 或后端可调用该服务，项目中引用位置在 `ui/app/config.py` 的导入处）
- 用途：情绪分析、文本问答模块、记录用户回答（`src/services/emotion_service.py` 等）
- 模拟：当 audio 为模拟时，识别模块可能返回空或测试/占位文本；项目中没有单独的识别模拟模块，但可通过提供静音 WAV 或测试数据触发假结果。
- 切换方法：间接通过 `UI_FORCE_SIMULATION`（模拟音频）或用测试脚本直接向 `speech_recognition_service` 注入测试音频/文本。

---

### rgb
- 来源：`src/services/multimodal_service.py`（RealSense color stream 或回退到普通相机）
- 用途：`src/services/unified_inference_service.py`（疲劳/姿态/表情模型），`src/services/emotion_service.py`（情绪推理）
- 模拟：`MultiModalDataCollector` 的 `simulate_rgb_depth` 分支会生成模拟帧；如果 `pyrealsense2` 不可用则自动切换为模拟。
- 切换方法：设置 `UI_MULTIMODAL_SIMULATION=1` 或 `UI_FORCE_SIMULATION=1`，或在无 RealSense 时自动回退。

---

### depth
- 来源：`src/services/multimodal_service.py`（RealSense 深度流）
- 用途：疲劳/姿态分析模型（`unified_inference_service`）
- 模拟：与 `rgb` 共用 `simulate_rgb_depth` 分支生成深度占位数据
- 切换方法：同 `rgb`（`UI_MULTIMODAL_SIMULATION` / `UI_FORCE_SIMULATION`，或当 `pyrealsense2` 不可用时自动模拟）

---

### tobii
- 来源：`src/services/multimodal_service.py`（Tobii 设备初始化在 `_init_tobii`）
- 用途：眼动样本用于疲劳/注意力模型（`unified_inference_service`、multimodal 推理）
- 模拟：`MultiModalDataCollector` 设置 `simulate_eyetrack=True` 时生成占位 eyetrack 数据
- 切换方法：设置 `UI_FORCE_SIMULATION=1` 或 `UI_MULTIMODAL_SIMULATION=1`；若 `HAS_TOBII` 为 False（未安装/未检测到设备）会自动使用模拟

---

### eeg1 / eeg2
- 来源：`src/services/eeg_service.py`（EEGRecorder 与后端 BLE 驱动）
- 用途：脑负荷/认知负荷推理（`src/services/unified_inference_service.py` 订阅 `EEG_REQUEST`）
- 模拟：`EEGRecorder` / `eeg_service` 中有 `FORCE_SIMULATION` 分支（依据 `BACKEND_EEG_SIMULATION` 或 `UI_EEG_SIMULATION`）产生模拟波形/样本
- 切换方法：设置 `BACKEND_EEG_SIMULATION=1` 或 `UI_EEG_SIMULATION=1` 以强制后端使用模拟数据

---

### blood
- 来源：`src/services/bp_service.py`（Maibobo Device 适配）
- 用途：血压/心率记录、保存在会话数据中，并显示在 UI 报告
- 模拟：`bp_service` 在 `start(payload)` 中支持 `simulation` 参数；若驱动/端口不可用且 `allow_simulation` 为真，会进入 `_simulate_until_complete()` 分支返回随机合理读数
- 切换方法：通过接口 payload 传入 `{"simulation": true}` 或设置环境/驱动不可用时允许回退（`allow_simulation`）

---

### sart
- 来源：UI SART 页面（`ui/app/pages/sart.py` 或 `ui/app/pages/test.py` 中的流程）
- 用途：注意力测试分数，用于行为/注意力评估并保存到结果
- 模拟：不自动走硬件模拟；但开发快捷键或测试脚本可触发自动运行以生成结果
- 切换方法：使用 UI 的调试快捷键（`Ctrl+Alt+3` 在 `MainWindow._setup_debug_shortcuts`）或在 `TestPage` 中调用相应方法运行测试流程

---

### accuracy / elapsed
- 来源：`ui/widgets/schulte_grid.py` 或 `ui/app/pages/test.py` 中舒尔特任务实现
- 用途：测试准确率与耗时用于结果评估
- 模拟：可以通过调试快捷键或测试脚本触发自动生成数据（并非自动硬件模拟）
- 切换方法：使用调试快捷键或通过测试页面 API 自动运行任务

---

### name
- 来源：`ui/app/pages/login.py`（用户登录/用户管理），用户存储在 `ui/data/users/users.csv`
- 用途：作为会话归属的元数据（保存路径、数据库记录）
- 模拟：不是硬件数据，但可在调试时使用默认用户（如 `admin` 或 `debug`）
- 切换方法：通过登录页面输入或在 `users.csv` 中添加/修改账号

---

### datetime
- 来源：`ui/app/pages/test.py` 或 `MainWindow` 在创建 session 时自动生成时间戳并作为 session 目录名
- 用途：记录测试开始时间，作为数据归档与索引字段
- 模拟：通常由系统时间生成，不属于硬件数据；在测试脚本中可伪造时间字段
- 切换方法：修改创建 session 的代码或在保存 metadata 时覆盖时间字段（不建议常规修改）
