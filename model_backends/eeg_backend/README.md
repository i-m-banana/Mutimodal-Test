# EEG脑负荷模型后端

## 功能概述

EEG脑负荷模型后端用于处理双通道脑电信号（Fp1, Fp2），输出脑负荷分数（0-100）。

### 主要特性

- **双通道EEG信号处理**: 支持Fp1和Fp2通道
- **实时推理**: 滑动窗口处理，输出连续的脑负荷评估
- **特征提取**: 时频域特征、Hjorth参数、相干性等
- **伪迹检测**: 自动过滤异常窗口
- **状态判定**: 基于迟滞阈值的高/低负荷状态

## 模型架构

### 信号处理流程

```
原始信号 (Fp1, Fp2)
    ↓
带通滤波 (0.5-45Hz)
    ↓
陷波滤波 (50/60Hz)
    ↓
滑动窗口分割 (2秒窗口, 1秒步长)
    ↓
伪迹检测与剔除
    ↓
特征提取 (42维)
    ↓
特征标准化
    ↓
分类器预测
    ↓
指数移动平均 (EMA)
    ↓
状态判定 (高/低负荷)
```

### 特征工程

**频域特征** (32维):
- Theta (4-7Hz)
- Alpha (8-12Hz)
- Beta (13-30Hz)
- Gamma (30-45Hz)
- 各通道功率 + Fp1-Fp2差值功率

**频率比率** (9维):
- Theta/Beta比率
- Alpha/Beta比率
- Beta/(Alpha+Theta)
- FAA (Frontal Alpha Asymmetry)

**相干性** (2维):
- Alpha频段相干性
- Beta频段相干性

**Hjorth参数** (9维):
- Activity, Mobility, Complexity
- Fp1, Fp2, Diff各通道

**熵** (1维):
- 频谱熵

## 使用方法

### 1. 启动后端

```bash
cd model_backends/eeg_backend
python main.py
```

后端将在 `ws://127.0.0.1:8769` 启动。

### 2. API接口

#### 推理请求

```json
{
  "type": "inference_request",
  "request_id": "req_001",
  "data": {
    "eeg_signal": [[1.2, 3.4], [2.1, 4.3], ...],  // [N, 2] 数组或base64编码文件
    "sampling_rate": 250,                          // 采样率 (默认250Hz)
    "subject_id": "test001"                        // 被试ID (可选)
  }
}
```

**eeg_signal格式**:
- 2D数组: `[[Fp1_1, Fp2_1], [Fp1_2, Fp2_2], ...]`
- Base64编码的TXT文件: 每行两列，空格分隔

#### 推理响应

```json
{
  "type": "inference_response",
  "request_id": "req_001",
  "model_type": "eeg",
  "timestamp": 1697123456.789,
  "result": {
    "status": "success",
    "predictions": {
      "brain_load_score": 65.3,           // 脑负荷分数 (0-100)
      "state": "high",                    // 状态: "low"/"high"
      "window_results": [                 // 各窗口详细结果
        {
          "window_index": 0,
          "t_start_s": 0.0,
          "t_end_s": 2.0,
          "score_raw": 62.5,
          "score_ema": 62.5,
          "state": "high",
          "state_changed": true
        },
        ...
      ],
      "num_windows": 10,
      "subject_id": "test001"
    },
    "latency_ms": 123.45
  }
}
```

#### 健康检查

```json
{
  "type": "health_check"
}
```

响应:
```json
{
  "type": "health_response",
  "status": "healthy",
  "model_type": "eeg",
  "model_loaded": true
}
```

## 模型参数

### 可调参数

```python
WIN_SEC = 2.0      # 窗口长度(秒)
STEP_SEC = 1.0     # 滑动步长(秒)
EMA_ALPHA = 0.7    # 指数移动平均系数 (越大越平滑)
TH_UP = 60.0       # 高负荷阈值
TH_DN = 50.0       # 低负荷阈值
```

### 状态转换逻辑

使用迟滞阈值避免状态抖动:
- `low → high`: 当 EMA ≥ 60
- `high → low`: 当 EMA ≤ 50

## 依赖安装

```bash
pip install -r requirements.txt
```

主要依赖:
- `joblib`: 模型加载
- `numpy`: 数值计算
- `scipy`: 信号处理
- `pandas`: 数据处理
- `websockets`: WebSocket通信

## 模型文件

位于 `brain_load/models_subj1/`:
- `mymodel_scaler.joblib`: 特征标准化器
- `mymodel_calibrator.joblib`: 校准分类器
- `feature_names.json`: 特征名称列表

## 测试

### 单独测试推理模块

```bash
cd brain_load
python online_inference.py
```

### 测试WebSocket连接

```python
import asyncio
import websockets
import json

async def test():
    async with websockets.connect("ws://127.0.0.1:8769") as ws:
        # 健康检查
        await ws.send(json.dumps({"type": "health_check"}))
        response = await ws.recv()
        print(response)

asyncio.run(test())
```

## 日志

日志文件位置: `logs/model/eeg_backend.log`

日志级别:
- INFO: 启动、关闭、推理概要
- DEBUG: 详细推理过程
- ERROR: 异常信息

## 性能指标

- **推理延迟**: ~50-200ms (取决于窗口数)
- **内存占用**: ~100MB
- **支持的信号长度**: 任意长度 (自动分窗处理)

## 故障排查

### 问题1: 模型文件未找到

**症状**: `FileNotFoundError: 模型文件不存在`

**解决**:
```bash
# 确保模型文件存在
ls model_backends/eeg_backend/brain_load/models_subj1/
# 应包含: mymodel_scaler.joblib, mymodel_calibrator.joblib
```

### 问题2: 所有窗口被判定为伪迹

**症状**: 返回 `"message": "所有窗口都被判定为伪迹"`

**原因**: 信号质量差或幅值异常

**解决**:
- 检查EEG信号采集质量
- 调整伪迹检测参数 (在 `eeg_utils.py` 中)

### 问题3: 端口占用

**症状**: `OSError: [Errno 48] Address already in use`

**解决**:
```bash
# 查找占用端口的进程
netstat -ano | findstr 8769
# 终止进程
taskkill /PID <PID> /F
# 或修改配置使用其他端口
```

## 开发者指南

### 添加新特征

在 `eeg_utils.py` 中:

```python
def extract_features_one_window(win2, fs=FS):
    # ... 现有特征 ...
    
    # 添加新特征
    new_feature = compute_my_feature(win2)
    feats.append(new_feature)
    names.append('my_feature_name')
    
    return np.array(feats), names
```

### 自定义预处理

修改 `preprocess_eeg()` 函数:

```python
def preprocess_eeg(arr_2ch, fs=FS):
    x = butter_bandpass_filter(arr_2ch, fs=fs, band=(1.0, 40.0))  # 自定义频段
    x = notch_filter(x, fs=fs, f0=50)  # 50Hz工频
    # 添加其他预处理步骤
    return x
```

## 参考文献

1. 脑负荷特征工程基于认知神经科学相关研究
2. Hjorth参数: Hjorth, B. (1970). EEG analysis based on time domain properties.
3. 频谱熵: Inouye et al. (1991). Quantification of EEG irregularity.

## 许可证

内部项目，未公开许可。
