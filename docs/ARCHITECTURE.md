# 系统架构设计文档

本文档详细描述多模态人员状态评估平台的系统架构、数据流转、核心组件设计与交互模式。

## 📋 目录

1. [整体架构](#整体架构)
2. [核心设计模式](#核心设计模式)
3. [核心组件](#核心组件)
4. [数据流转](#数据流转)
5. [疲劳度评估融合策略](#疲劳度评估融合策略)
6. [基线管理机制](#基线管理机制)
7. [事件系统](#事件系统)
8. [通信协议](#通信协议)
9. [配置管理](#配置管理)
10. [部署架构](#部署架构)

---

## 整体架构

### 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                         表示层 (UI Layer)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │ 登录页面 │  │ SART任务 │  │ 实验任务 │  │ 结果展示 │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
│         │                                          ↑             │
│         │    PyQt5 Signals/Slots + WebSocket      │             │
│         ↓                                          │             │
└─────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────┐
│                      通信接口层 (Interface Layer)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  WebSocket Server (127.0.0.1:8765)                       │  │
│  │  • 双向实时通信  • JSON 消息格式  • 事件推送机制        │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────┐
│                      核心调度层 (Orchestration Layer)            │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Event Bus (事件总线)                                     │  │
│  │  • 发布-订阅模式  • 异步事件路由  • 解耦组件通信        │  │
│  └──────────────────────────────────────────────────────────┘  │
│         ↑                    ↑                    ↑              │
│         │                    │                    │              │
│  ┌──────┴──────┐      ┌──────┴──────┐                          │
│  │ Orchestrator │      │ UI Router   │                          │
│  │ 系统编排器   │      │ UI命令路由  │                          │
│  └─────────────┘      └─────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────┐
│                      业务服务层 (Service Layer)                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Fatigue       │  │ Emotion       │  │ EEG Service   │      │
│  │ Assessment    │  │ Service       │  │               │      │
│  │ Service       │  │               │  │ • 设备管理    │      │
│  │               │  │ • 多模态情绪  │  │ • 数据采集    │      │
│  │ • EEG+RGB融合 │  │   识别        │  │ • 基线校准    │      │
│  │ • 加权策略    │  │ • 7类情绪     │  │               │      │
│  │ • 阈值判断    │  │               │  │               │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Unified       │  │ Baseline      │  │ Recording     │      │
│  │ Inference     │  │ Manager       │  │ Service       │      │
│  │ Service       │  │               │  │               │      │
│  │               │  │ • 个体基线    │  │ • 音视频录制  │      │
│  │ • 模型调度    │  │ • 门控更新    │  │ • EEG存储     │      │
│  │ • 结果聚合    │  │ • EMA平滑     │  │ • 元数据      │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────┐
│                      模型推理层 (Model Layer)                    │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Fatigue Model │  │ Emotion Model │  │ EEG Fatigue   │      │
│  │ (RGB)         │  │               │  │ Model         │      │
│  │               │  │ • RoBERTa     │  │               │      │
│  │ • 面部特征    │  │ • Wav2Vec2    │  │ • 特征提取    │      │
│  │ • 动作识别    │  │ • TimesFormer │  │ • XGBoost分类 │      │
│  │ • 深度学习    │  │               │  │ • 在线推理    │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ EEG Workload  │  │ Feature       │  │ Model Cache   │      │
│  │ Model         │  │ Extractor     │  │               │      │
│  │               │  │               │  │ • 模型预加载  │      │
│  │ • 认知负荷    │  │ • 频域分析    │  │ • 内存优化    │      │
│  │ • 三级分类    │  │ • 时域特征    │  │               │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
                                ↕
┌─────────────────────────────────────────────────────────────────┐
│                      设备抽象层 (Device Layer)                   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ EEG Device    │  │ Camera Device │  │ Depth Device  │      │
│  │               │  │               │  │               │      │
│  │ • Neuracle    │  │ • RealSense   │  │ • RealSense   │      │
│  │ • LSL协议     │  │ • USB Camera  │  │ • D435/D455   │      │
│  │ • 500Hz采样   │  │               │  │               │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
│                                                                  │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐      │
│  │ Audio Device  │  │ BP Device     │  │ Simulator     │      │
│  │               │  │               │  │               │      │
│  │ • 麦克风阵列  │  │ • 血压计      │  │ • 模拟数据    │      │
│  │ • 多声道采集  │  │ • 串口通信    │  │ • 硬件模拟    │      │
│  └───────────────┘  └───────────────┘  └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

## 核心设计模式

### 1. 事件驱动架构 (Event-Driven Architecture)
- **解耦合**: 各组件通过事件总线通信，不直接依赖
- **异步处理**: 模型推理异步执行，不阻塞主流程
- **可扩展**: 新增组件只需订阅/发布相应事件

### 2. 分层架构 (Layered Architecture)
- **表示层**: PyQt5 UI 组件
- **接口层**: WebSocket 通信
- **服务层**: 业务逻辑处理
- **模型层**: AI 推理引擎
- **设备层**: 硬件抽象

### 3. 服务定位模式 (Service Locator)
- **统一注册**: Orchestrator 管理所有服务生命周期
- **动态查找**: 通过服务名获取服务实例
- **依赖注入**: 服务间通过事件总线交互

### 4. 策略模式 (Strategy Pattern)
- **疲劳融合策略**: 多种权重分配策略可切换
- **情绪评分策略**: 支持多种算法（投票、加权、时序等）
- **基线更新策略**: 门控更新、EMA 平滑等

---

## 核心组件

### 后端核心 (src/core/)

#### Orchestrator (系统编排器)
```python
class Orchestrator:
    """
    职责:
    - 初始化和管理所有服务生命周期
    - 协调组件启动顺序
    - 监控系统健康状态
    """
    def __init__(self):
        self.event_bus = EventBus()
        self.services = {}
    
    def register_service(self, name, service):
        """注册服务到编排器"""
        
    def start_all(self):
        """按依赖顺序启动所有服务"""
```

**关键功能**:
- 服务注册与发现
- 依赖关系管理
- 优雅启动/关闭

#### EventBus (事件总线)
```python
class EventBus:
    """
    职责:
    - 事件发布-订阅机制
    - 事件路由与分发
    - 异步事件处理
    """
    def publish(self, event: Event):
        """发布事件到所有订阅者"""
    
    def subscribe(self, event_type: str, handler: Callable):
        """订阅特定类型事件"""
    
    def unsubscribe(self, event_type: str, handler: Callable):
        """取消订阅"""
```

**事件流程**:
1. 组件调用 `event_bus.publish(Event(...))`
2. 事件总线查找所有订阅者
3. 异步调用所有处理函数
4. 订阅者处理事件并可能发布新事件

---

### 服务层 (src/services/)

#### FatigueAssessmentService (疲劳度评估服务)
```python
class FatigueAssessmentService:
    """
    职责:
    - 融合 EEG 和 RGB 疲劳度结果
    - 实施自适应加权策略
    - 应用阈值判断疲劳等级
    """
    def _compute_weighted_score(self, eeg_result, rgb_result):
        """
        核心算法:
        1. 检查 EEG 和 RGB 结果有效性
        2. 统一分数范围 (EEG: 0-100 → 50-90, RGB: 50-90)
        3. 计算一致性 (consistency = abs(eeg - rgb))
        4. 根据一致性调整权重:
           - 一致 (diff < 5): EEG 70%, RGB 30%
           - 冲突 (diff >= 10): EEG 85%, RGB 15%
        5. 返回融合分数 + 疲劳等级
        """
```

**融合策略详见**: [疲劳度评估融合策略](#疲劳度评估融合策略)

#### BaselineManager (基线管理器)
```python
class BaselineManager:
    """
    职责:
    - 管理每个被试的个体 EEG 基线
    - 门控更新机制 (防止异常基线)
    - EMA 平滑更新
    """
    def get(self, user_id: str):
        """获取用户基线"""
    
    def gated_update(self, user_id: str, new_baseline: dict):
        """
        门控更新逻辑:
        1. 检查新基线与旧基线的偏差
        2. 若偏差 < 30%, 则通过门控
        3. 使用 EMA 更新: baseline = α*new + (1-α)*old
        """
    
    def save(self, user_id: str, baseline: dict):
        """保存基线到磁盘"""
```

**基线更新详见**: [基线管理机制](#基线管理机制)

#### UnifiedInferenceService (统一推理服务)
```python
class UnifiedInferenceService:
    """
    职责:
    - 管理所有模型实例
    - 调度推理请求
    - 聚合推理结果
    """
    def __init__(self, event_bus):
        self.models = self._load_models()
        self.event_bus = event_bus
        
        # 订阅推理请求事件
        event_bus.subscribe('INFERENCE_REQUEST', self._handle_inference)
    
    def _handle_inference(self, event):
        """
        推理流程:
        1. 根据请求类型选择模型
        2. 调用模型 infer() 方法
        3. 发布 INFERENCE_RESULT 事件
        """
```

#### EEGService (EEG 服务)
```python
class EEGService:
    """
    职责:
    - 管理 EEG 设备连接
    - 实时数据采集 (500Hz)
    - 数据缓冲与分发
    - CSV 文件保存
    """
    def start_recording(self, user_id: str):
        """启动 EEG 录制"""
    
    def stop_recording(self):
        """停止录制并保存 CSV"""
```

---

### 模型层 (src/models/)

#### BaseInferenceModel (基础推理模型)
```python
class BaseInferenceModel:
    """
    所有模型的基类
    """
    def initialize(self):
        """加载模型权重和配置"""
        raise NotImplementedError
    
    def infer(self, data: dict):
        """执行推理"""
        raise NotImplementedError
```

#### EEGFatigueModel (EEG 疲劳模型)
```python
class EEGFatigueModel(BaseInferenceModel):
    """
    职责:
    - EEG 特征提取 (频域、熵值等)
    - 疲劳分类 (XGBoost)
    - 会话模式推理 (批量处理)
    - 基线候选计算
    """
    def _infer_from_session(self, session_dir: str):
        """
        会话推理流程:
        1. 查找 EEG CSV 文件 (eeg_data_*.csv)
        2. 读取 part_timestamps.json 分段信息
        3. 批量提取特征
        4. 预测疲劳度
        5. 计算基线候选
        6. 调用 baseline_manager.gated_update()
        """
```

**支持的文件命名格式**:
- `eeg_data_YYYYMMDD_HHMMSS.csv` (推荐)
- `part*.csv` (旧格式兼容)

#### FatigueModel (RGB 疲劳模型)
```python
class FatigueModel(BaseInferenceModel):
    """
    职责:
    - RGB 视频特征提取
    - 面部疲劳特征识别
    - 输出 50-90 分数段
    """
```

#### EmotionModel (情绪识别模型)
```python
class EmotionModel(BaseInferenceModel):
    """
    职责:
    - 多模态情绪识别 (视频/音频/文本)
    - 7 类情绪分类
    - 情绪评分计算
    """
```

---

### 前端层 (ui/)

#### MainWindow (主窗口)
```python
class MainWindow(QMainWindow):
    """
    职责:
    - 页面导航与状态管理
    - 与后端 WebSocket 通信
    - 全局事件处理
    """
    def __init__(self):
        self.backend_client = BackendClient()
        self.stacked_widget = QStackedWidget()
        
        # 连接信号
        self.backend_client.detection_result.connect(
            self._on_detection_result
        )
```

#### ScorePage (结果页面)
```python
class ScorePage(QWidget):
    """
    职责:
    - 展示综合评估结果
    - 接收延迟到达的推理结果
    - 计算综合得分
    """
    def __init__(self):
        # 监听后端事件
        self.backend_client.detection_result.connect(
            self._on_detection_result
        )
    
    def _on_detection_result(self, result: dict):
        """
        处理检测结果:
        1. 判断 detector 类型
        2. 更新 _test_results 字典
        3. 触发界面刷新
        """
    
    def _calculate_comprehensive_score(self):
        """
        综合评分算法:
        1. 从 _test_results 获取各项分数
        2. 应用权重计算总分
        3. 生成评级和建议
        """
```

**数据流路径**:
```
Backend Event (DETECTION_RESULT)
    ↓
WebSocket Push
    ↓
backend_client.detection_result (PyQt Signal)
    ↓
ScorePage._on_detection_result()
    ↓
_test_results["疲劳检测"] = fatigue_score
    ↓
_fetch_data() → _current_data
    ↓
_calculate_comprehensive_score()
    ↓
界面显示
```

## 数据流转

### 1. 疲劳度评估完整流程

```
┌──────────────────────────────────────────────────────────────────┐
│                        Step 1: 数据采集                          │
└──────────────────────────────────────────────────────────────────┘
                                ↓
UI 实验页面发起采集请求
    ↓ (WebSocket Command)
后端 UI 命令路由器
    ↓ (调用服务)
┌─────────────────────┐          ┌─────────────────────┐
│ EEG Service         │          │ RGB Camera Service  │
│ • 启动 EEG 设备     │          │ • 启动摄像头        │
│ • 500Hz 采样        │          │ • 30fps 视频流      │
│ • 实时缓冲数据      │          │ • 深度图像          │
└─────────────────────┘          └─────────────────────┘
         ↓                                  ↓
    EEG 数据流                         RGB 视频流
         ↓                                  ↓

┌──────────────────────────────────────────────────────────────────┐
│                        Step 2: 模型推理                          │
└──────────────────────────────────────────────────────────────────┘
         ↓                                  ↓
┌─────────────────────┐          ┌─────────────────────┐
│ EEG Fatigue Model   │          │ RGB Fatigue Model   │
│                     │          │                     │
│ 1. 特征提取         │          │ 1. 面部检测         │
│    - 频域功率       │          │ 2. 特征提取         │
│    - 熵值计算       │          │    - 眨眼频率       │
│    - 非线性特征     │          │    - 眼睛闭合度     │
│                     │          │    - 打哈欠         │
│ 2. XGBoost 分类     │          │ 3. 深度学习推理     │
│                     │          │                     │
│ 3. 输出: 0-100 分数 │          │ 4. 输出: 50-90 分数 │
└─────────────────────┘          └─────────────────────┘
         ↓                                  ↓
    EEG 推理结果                        RGB 推理结果
    (detector="eeg")                   (detector="rgb")
         ↓                                  ↓
         └──────────────┬───────────────────┘
                        ↓

┌──────────────────────────────────────────────────────────────────┐
│                        Step 3: 结果融合                          │
└──────────────────────────────────────────────────────────────────┘
                        ↓
    Fatigue Assessment Service (_compute_weighted_score)
                        │
    ┌───────────────────┴───────────────────┐
    ↓                                       ↓
分数范围统一                            一致性检测
(EEG: 0-100 → 50-90)                (consistency = |eeg - rgb|)
    ↓                                       ↓
    └───────────────────┬───────────────────┘
                        ↓
            ┌───────────┴───────────┐
            │   自适应权重调整       │
            │                       │
            │ • 高一致 (<5 差异):   │
            │   EEG 70% + RGB 30%   │
            │                       │
            │ • 中等 (5-10 差异):   │
            │   EEG 70% + RGB 30%   │
            │                       │
            │ • 冲突 (>=10 差异):   │
            │   EEG 85% + RGB 15%   │
            └───────────────────────┘
                        ↓
            融合分数 (50-90 范围)
                        ↓
            ┌───────────┴───────────┐
            │   阈值判断疲劳等级     │
            │                       │
            │ • < 65: 正常          │
            │ • 65-77.5: 轻度疲劳   │
            │ • >= 77.5: 重度疲劳   │
            └───────────────────────┘
                        ↓

┌──────────────────────────────────────────────────────────────────┐
│                        Step 4: 结果推送                          │
└──────────────────────────────────────────────────────────────────┘
                        ↓
    Event Bus 发布事件 (DETECTION_RESULT)
    {
        "detector": "model_fatigue",
        "fatigue_score": 72.5,
        "fatigue_level": "轻度疲劳",
        "eeg_score": 75,
        "rgb_score": 65,
        "consistency": "medium"
    }
                        ↓
    WebSocket Interface 推送到前端
                        ↓
    UI ScorePage 接收并显示
                        ↓
    _on_detection_result() 更新 _test_results
                        ↓
    _calculate_comprehensive_score() 计算总分
                        ↓
    界面更新显示
```

### 2. EEG 基线更新流程

```
实验结束
    ↓
EEG Fatigue Model (_infer_from_session)
    ↓
1. 定位 EEG CSV 文件
   - 优先: eeg_data_YYYYMMDD_HHMMSS.csv
   - 兼容: part*.csv
    ↓
2. 读取 part_timestamps.json
   (获取各实验阶段时间戳)
    ↓
3. 提取基线候选 (baseline phase)
   - 计算频段功率均值
   - 计算特征统计量
    ↓
4. 调用 BaselineManager.gated_update()
    ↓
    ┌─────────────────────────────┐
    │  门控更新检查                │
    │                             │
    │  old_baseline = get(user_id)│
    │  diff = |new - old| / old   │
    │                             │
    │  if diff < 0.3:  # 允许±30% │
    │      通过门控               │
    │  else:                      │
    │      拒绝更新               │
    └─────────────────────────────┘
    ↓ (通过)
    ┌─────────────────────────────┐
    │  EMA 平滑更新                │
    │                             │
    │  α = 0.3  # 新数据权重30%   │
    │  baseline_new =             │
    │      α * new +              │
    │      (1-α) * old            │
    └─────────────────────────────┘
    ↓
保存到磁盘 (models_data/eeg_online_textdata/{user}/baseline.json)
    ↓
下次推理使用新基线
```

### 3. 情绪识别流程

```
用户完成情绪任务 (观看视频/听音频/阅读文本)
    ↓
UI 发送情绪推理请求
    ↓
Emotion Service 收集多模态数据
    ↓
    ┌──────────┬──────────┬──────────┐
    ↓          ↓          ↓          ↓
  Video      Audio      Text    (可选)
    │          │          │
    ↓          ↓          ↓
TimesFormer  Wav2Vec2  RoBERTa
    │          │          │
    ↓          ↓          ↓
7类情绪     7类情绪     7类情绪
概率分布    概率分布    概率分布
    └──────────┴──────────┘
              ↓
    Emotion Model 融合算法
    (投票/概率加权/时序等)
              ↓
    最终情绪类别 + 置信度
              ↓
    Event Bus 发布 DETECTION_RESULT
              ↓
    UI 显示情绪结果
```

### 4. SART 实验流程

```
用户登录 → 选择被试 ID
    ↓
进入 SART 页面
    ↓
开始 SART 实验
    ↓
    ┌─────────────────────────┐
    │  持续注意反应测试        │
    │                         │
    │  • 数字刺激呈现         │
    │  • 响应时间记录         │
    │  • 正确率统计           │
    │                         │
    │  同时采集:              │
    │  - EEG 数据 (500Hz)     │
    │  - RGB 视频             │
    │  - 键盘响应             │
    └─────────────────────────┘
    ↓
实验结束
    ↓
    ┌──────────────────────┬─────────────────────┐
    ↓                      ↓                     ↓
保存 EEG CSV          保存 RGB 视频        保存 metadata.json
    ↓                      ↓                     ↓
触发 EEG 基线更新    (可选) 疲劳度分析   记录实验参数
    ↓
进入下一任务
```

## 疲劳度评估融合策略

### 设计目标
- **准确性**: 结合 EEG 脑电信号和 RGB 面部特征的优势
- **鲁棒性**: 当单模态失败时系统仍可工作
- **自适应性**: 根据两模态一致性动态调整权重

### 融合算法详解

#### 1. 分数范围统一

**问题**: EEG 模型输出 0-100,RGB 模型输出 50-90,需统一到同一范围

**解决方案**:
```python
# EEG 分数转换 (0-100 → 50-90)
eeg_score_50_90 = 50 + (1 - eeg_score / 100) * 40

# RGB 分数保持不变 (已在 50-90 范围)
rgb_score_50_90 = rgb_score
```

> ⚠️ **注意**: EEG 转换公式可能存在语义反转问题,待验证:
> - 若 EEG=0 表示清醒,转换后=90 (高疲劳) ❌
> - 若 EEG=100 表示清醒,转换后=50 (低疲劳) ✅

#### 2. 一致性检测

计算两模态分数差异,判断一致性程度:
```python
consistency = abs(eeg_score_50_90 - rgb_score_50_90)

if consistency < 5:
    level = "high"      # 高一致性
elif consistency < 10:
    level = "medium"    # 中等一致性
else:
    level = "low"       # 低一致性 (冲突)
```

#### 3. 自适应权重调整

根据一致性动态调整权重:

| 一致性等级 | 差异范围 | EEG 权重 | RGB 权重 | 原因 |
|-----------|---------|---------|---------|------|
| High | <5 | 70% | 30% | EEG 更可靠,但给 RGB 一定权重 |
| Medium | 5-10 | 70% | 30% | 保持默认权重 |
| Low | >=10 | 85% | 15% | 冲突时更信任 EEG |

**代码实现**:
```python
def _compute_weighted_score(self, eeg_result, rgb_result):
    # 场景 1: 两者都失败
    if not eeg_valid and not rgb_valid:
        return {"score": 50, "level": "未知", "status": "both_failed"}
    
    # 场景 2: 仅 EEG 有效
    if eeg_valid and not rgb_valid:
        return {
            "score": eeg_score_50_90,
            "level": self._classify_fatigue(eeg_score_50_90, "eeg_only"),
            "status": "eeg_only"
        }
    
    # 场景 3: 仅 RGB 有效
    if rgb_valid and not eeg_valid:
        return {
            "score": rgb_score_50_90,
            "level": self._classify_fatigue(rgb_score_50_90, "rgb_only"),
            "status": "rgb_only"
        }
    
    # 场景 4: 两者都有效
    consistency = abs(eeg_score_50_90 - rgb_score_50_90)
    
    if consistency >= 10:  # 冲突
        weight_eeg = 0.85
        weight_rgb = 0.15
    else:  # 一致
        weight_eeg = 0.70
        weight_rgb = 0.30
    
    final_score = (eeg_score_50_90 * weight_eeg + 
                   rgb_score_50_90 * weight_rgb)
    
    return {
        "score": final_score,
        "level": self._classify_fatigue(final_score, "standard"),
        "consistency": "high" if consistency < 5 else 
                       "medium" if consistency < 10 else "low",
        "eeg_score": eeg_score_50_90,
        "rgb_score": rgb_score_50_90,
        "status": "both_valid"
    }
```

#### 4. 疲劳等级判断

不同场景使用不同阈值:

| 场景 | 轻度疲劳阈值 | 重度疲劳阈值 | 说明 |
|------|-------------|-------------|------|
| 标准 (EEG+RGB) | 65 | 77.5 | 默认阈值 |
| 仅 EEG | 62 | 74 | 更保守 (避免漏检) |
| EEG 优先 | 63 | 75 | 介于两者之间 |
| 仅 RGB | 65 | 77.5 | 与标准相同 |

### 融合策略优势

1. **容错性**: 单模态失败不影响系统运行
2. **自适应**: 根据一致性动态调整权重
3. **透明性**: 返回详细的融合过程信息
4. **可调优**: 权重和阈值可根据实际数据调整

---

## 基线管理机制

### 设计理念

EEG 信号存在显著的**个体差异**和**状态依赖性**:
- 不同被试的基线脑电模式不同
- 同一被试在不同时间/状态下基线也会变化

因此需要**动态更新**每个被试的个体基线。

### 基线更新流程

#### 1. 基线采集 (SART 实验)

在 SART 实验的基线阶段采集:
```
SART 实验开始
    ↓
基线阶段 (前 N 秒)
    ↓
采集 EEG 数据
    ↓
计算频段功率均值
    - Delta (0.5-4 Hz)
    - Theta (4-8 Hz)
    - Alpha (8-13 Hz)
    - Beta (13-30 Hz)
    - Gamma (30-50 Hz)
    ↓
存储为基线候选
```

#### 2. 门控更新 (Gated Update)

防止异常数据污染基线:
```python
def gated_update(self, user_id: str, new_baseline: dict):
    old_baseline = self.get(user_id)
    
    # 计算相对变化率
    for feature in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        old_value = old_baseline[feature]
        new_value = new_baseline[feature]
        
        diff_ratio = abs(new_value - old_value) / old_value
        
        if diff_ratio > 0.3:  # 超过 ±30%
            logger.warning(f"{feature} 变化过大 ({diff_ratio:.1%}),拒绝更新")
            return False  # 拒绝通过门控
    
    # 通过门控,执行 EMA 更新
    self._ema_update(user_id, new_baseline)
    return True
```

**门控参数**:
- `GATE_THRESHOLD = 0.3`: 允许 ±30% 变化
- 超过阈值则拒绝更新,保护基线稳定性

#### 3. EMA 平滑更新

使用指数移动平均避免基线突变:
```python
def _ema_update(self, user_id: str, new_baseline: dict):
    old_baseline = self.get(user_id)
    alpha = 0.3  # 新数据权重 30%
    
    updated_baseline = {}
    for feature in new_baseline:
        updated_baseline[feature] = (
            alpha * new_baseline[feature] +
            (1 - alpha) * old_baseline[feature]
        )
    
    self.save(user_id, updated_baseline)
```

**EMA 参数**:
- `α = 0.3`: 新数据权重 30%,旧数据权重 70%
- 平滑更新,避免基线剧烈波动

### 基线使用

在 EEG 疲劳推理时:
```python
def infer(self, eeg_data: np.ndarray, user_id: str):
    # 1. 获取用户基线
    baseline = baseline_manager.get(user_id)
    
    # 2. 提取当前特征
    features = self._extract_features(eeg_data)
    
    # 3. 计算相对于基线的偏移
    relative_features = {}
    for feat in features:
        relative_features[feat] = features[feat] / baseline[feat]
    
    # 4. 使用相对特征进行分类
    fatigue_score = self.model.predict(relative_features)
    
    return fatigue_score
```

### 基线存储

**存储位置**: `models_data/eeg_online_textdata/{user_id}/baseline.json`

**文件格式**:
```json
{
    "user_id": "shh0",
    "delta_power": 12.34,
    "theta_power": 8.76,
    "alpha_power": 15.23,
    "beta_power": 6.45,
    "gamma_power": 3.21,
    "last_updated": "2025-01-15T10:30:00",
    "update_count": 15
}
```

---

## 事件系统

### 事件类型定义

系统中流转的主要事件类型:

| 事件类型 | 触发条件 | 数据内容 | 订阅者 |
|---------|---------|---------|-------|
| `CAMERA_FRAME` | 摄像头捕获新帧 | frame, timestamp | 疲劳模型、录制服务 |
| `EEG_DATA` | EEG 设备采集数据 | samples, channels, timestamp | EEG 模型、录制服务 |
| `INFERENCE_REQUEST` | UI 请求推理 | model_type, data | 统一推理服务 |
| `INFERENCE_RESULT` | 模型推理完成 | model, result, confidence | WebSocket 接口 |
| `DETECTION_RESULT` | 检测任务完成 | detector, score, level | WebSocket 接口、UI |
| `RECORDING_START` | 开始录制 | user_id, session_id | 录制服务 |
| `RECORDING_STOP` | 停止录制 | session_id, file_path | 录制服务 |
| `SYSTEM_HEARTBEAT` | 定时心跳 | timestamp, status | 系统监控 |

### 事件流转示例

**疲劳检测事件链**:
```
1. UI 发送 START_FATIGUE_DETECTION 命令
     ↓
2. UI Router 发布 INFERENCE_REQUEST 事件
     ↓
3. Unified Inference Service 订阅并处理
     ↓
4. 调用 Fatigue Model.infer()
     ↓
5. 模型发布 INFERENCE_RESULT 事件
     ↓
6. Fatigue Assessment Service 订阅并融合
     ↓
7. 发布 DETECTION_RESULT 事件
     ↓
8. WebSocket Interface 订阅并推送到 UI
     ↓
9. UI 接收并显示结果
```

### 事件数据格式

**DETECTION_RESULT 事件**:
```python
{
    "type": "DETECTION_RESULT",
    "data": {
        "detector": "model_fatigue",  # 检测器类型
        "timestamp": "2025-01-15T10:30:00",
        "fatigue_score": 72.5,  # 融合分数 (50-90)
        "fatigue_level": "轻度疲劳",
        "eeg_score": 75,  # EEG 原始分数
        "rgb_score": 65,  # RGB 原始分数
        "consistency": "medium",  # 一致性等级
        "confidence": 0.85,
        "metadata": {
            "eeg_valid": True,
            "rgb_valid": True,
            "fusion_status": "both_valid"
        }
    },
    "source": "fatigue_assessment_service"
}
```

---

## 通信协议

### WebSocket 协议

**连接信息**:
- URL: `ws://127.0.0.1:8765`
- 协议: WebSocket (RFC 6455)
- 消息格式: JSON

**消息类型**:

#### 1. UI → Backend (命令)
```json
{
    "type": "command",
    "command": "start_recording",
    "data": {
        "user_id": "shh0",
        "session_id": "20250115_103000"
    }
}
```

#### 2. Backend → UI (事件推送)
```json
{
    "type": "event",
    "event_name": "DETECTION_RESULT",
    "data": {
        "detector": "model_fatigue",
        "fatigue_score": 72.5,
        "timestamp": "2025-01-15T10:30:00"
    }
}
```

#### 3. UI → Backend (请求-响应)
```json
// 请求
{
    "type": "request",
    "request_id": "req_123",
    "method": "get_baseline",
    "params": {
        "user_id": "shh0"
    }
}

// 响应
{
    "type": "response",
    "request_id": "req_123",
    "status": "success",
    "data": {
        "baseline": {...}
    }
}
```

### 前端信号系统 (PyQt)

**信号定义** (`ui/app/backend_client.py`):
```python
class BackendClient(QObject):
    # 检测结果信号
    detection_result = pyqtSignal(dict)
    
    # 连接状态信号
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    
    # 错误信号
    error = pyqtSignal(str)
```

**信号使用**:
```python
# 订阅信号
self.backend_client.detection_result.connect(
    self._on_detection_result
)

# 处理函数
def _on_detection_result(self, result: dict):
    detector = result.get("detector")
    if detector == "model_fatigue":
        score = result.get("fatigue_score", 50)
        self._update_fatigue_display(score)
```

---

## 配置管理

### 模型配置 (config/models.yaml)

```yaml
inference_models:
  # 疲劳检测模型
  - name: fatigue
    type: fatigue
    mode: integrated        # integrated: 集成模式 | standalone: 独立服务
    enabled: true           # 是否启用
    model_path: models_data/emotion_models/best_model.pt
    device: cuda            # cuda | cpu
    batch_size: 1
    
  # 情绪识别模型
  - name: emotion
    type: emotion
    mode: integrated
    enabled: true
    modalities:
      - video               # 启用的模态
      - audio
      - text
    pretrained:
      robberta: models_data/emotion_pretrained_models/ROBBERTA
      wav2vec2: models_data/emotion_pretrained_models/WAV2VEC2
      timesformer: models_data/emotion_pretrained_models/TIMESFORMER
    
  # EEG 疲劳模型
  - name: eeg_fatigue
    type: eeg
    mode: integrated
    enabled: true
    model_path: models_data/eeg_fatigue_models/fatigue_model.joblib
    feature_config: models_data/eeg_fatigue_models/fatigue_feature_def.json
    baseline_manager:
      gate_threshold: 0.3   # 门控阈值 ±30%
      ema_alpha: 0.3        # EMA 权重 30%
    
  # EEG 脑负荷模型
  - name: eeg_workload
    type: eeg
    mode: integrated
    enabled: true
    model_path: models_data/eeg_models/mymodel_clf.joblib
    scaler_path: models_data/eeg_models/mymodel_scaler.joblib
```

### 接口配置 (config/interfaces.yaml)

```yaml
websocket:
  host: 127.0.0.1
  port: 8765
  max_connections: 10
  max_message_size: 10485760  # 10MB
  ping_interval: 30           # 心跳间隔 (秒)
  ping_timeout: 10            # 心跳超时 (秒)

rest_api:
  host: 127.0.0.1
  port: 8080
  cors_enabled: true
  cors_origins:
    - http://localhost:3000
    - http://127.0.0.1:5000

logging:
  level: INFO                 # DEBUG | INFO | WARNING | ERROR
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
  file: logs/backend.log
  max_bytes: 10485760         # 10MB
  backup_count: 5             # 保留 5 个备份
```

### 环境变量配置

| 变量名 | 说明 | 示例值 |
|--------|------|--------|
| `UI_FORCE_SIMULATION` | 前端全局模拟 | 0/1 |
| `BACKEND_EEG_SIMULATION` | 后端 EEG 模拟 | 0/1 |
| `UI_DEBUG_MODE` | 前端调试模式 | 0/1 |
| `UI_SKIP_DATABASE` | 跳过数据库 | 0/1 |
| `CORS_ALLOWED_ORIGINS` | CORS 白名单 | * 或 http://... |
| `LOG_LEVEL` | 日志级别 | DEBUG/INFO/WARNING/ERROR |

---

## 部署架构

### 单机部署 (默认)

```
┌────────────────────────────────────┐
│        同一台机器                   │
│                                    │
│  ┌──────────────┐                 │
│  │   UI 进程    │                 │
│  │ (PyQt5)      │                 │
│  └──────┬───────┘                 │
│         │ WebSocket               │
│         ↓                          │
│  ┌──────────────┐                 │
│  │  后端进程    │                 │
│  │  (Python)    │                 │
│  │              │                 │
│  │ • 事件总线   │                 │
│  │ • 模型推理   │                 │
│  │ • 设备管理   │                 │
│  └──────────────┘                 │
│                                    │
│  录制数据存储: recordings/         │
│  模型权重: models_data/            │
└────────────────────────────────────┘
```

**启动方式**:
```bash
# 终端 1: 启动后端
python -m src.main

# 终端 2: 启动前端
python -m ui.main
```

### 分布式部署 (可选)

```
┌────────────────┐           ┌────────────────┐
│   UI 机器       │           │   后端机器      │
│                │           │                │
│  ┌──────────┐ │  WebSocket │  ┌──────────┐ │
│  │  UI 进程 │ │◄──────────►│  │ 后端进程 │ │
│  └──────────┘ │           │  └──────────┘ │
│                │           │                │
└────────────────┘           │  ┌──────────┐ │
                             │  │ GPU 加速 │ │
                             │  └──────────┘ │
                             │                │
                             │  录制数据存储  │
                             └────────────────┘
```

**配置调整**:
```yaml
# config/interfaces.yaml
websocket:
  host: 0.0.0.0  # 监听所有网卡
  port: 8765
```

**前端连接**:
```python
# ui/app/backend_client.py
BACKEND_URL = "ws://192.168.1.100:8765"  # 后端机器 IP
```

---

## 性能优化

### 模型推理优化

1. **模型预加载**: 启动时加载所有模型到内存
2. **批处理**: 支持批量推理 (batch_size 可配置)
3. **GPU 加速**: 自动检测 CUDA 并使用 GPU
4. **ONNX 转换**: 部分模型转 ONNX 加速推理

### 数据传输优化

1. **WebSocket**: 低延迟双向通信
2. **压缩**: 大数据包 gzip 压缩
3. **分片传输**: 大文件分片发送
4. **缓冲区**: 数据缓冲减少 I/O 开销

### 录制优化

1. **异步写入**: 录制数据异步写磁盘
2. **缓冲队列**: 数据先入队列,批量写入
3. **压缩存储**: H.264 视频编码压缩

---

**文档版本**: 2.0  
**最后更新**: 2025年1月15日  
**维护者**: 系统架构组