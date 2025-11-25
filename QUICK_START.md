# 🚀 快速启动指南

## 系统要求

- Python 3.11+
- Windows 10/11 / Linux / macOS
- 8GB+ RAM（推荐16GB）
- NVIDIA GPU with CUDA支持（用于加速推理）
- 所需依赖包已安装

## 📋 启动步骤

### 1. 环境准备

首次使用前需要安装依赖：

```bash
# 进入项目根目录
cd d:\Mutimodal-Test

# 安装核心依赖
pip install -r requirements.txt
```

### 2. 初始化数据库

项目默认使用 MySQL 风格的表结构, 下面是一个示例表 `test` 的建表语句：

```sql
CREATE TABLE `test` (
   `id` int NOT NULL AUTO_INCREMENT COMMENT '记录id',
   `name` varchar(255) NOT NULL COMMENT '被试姓名',
   `datetime` datetime NOT NULL COMMENT '测试时间',
   `audio` json DEFAULT NULL COMMENT '音频路径列表（JSON数组）',
   `video` json DEFAULT NULL COMMENT '视频路径列表（JSON数组）',
   `record` json DEFAULT NULL COMMENT '录音文本列表（JSON数组）',
   `rgb` varchar(512) DEFAULT NULL COMMENT '可见光',
   `depth` varchar(512) DEFAULT NULL COMMENT '深度',
   `tobii` varchar(512) DEFAULT NULL COMMENT '眼动',
   `blood` varchar(512) DEFAULT NULL COMMENT '高压/低压/心率',
   `ptime` varchar(512) DEFAULT NULL COMMENT '调用疲劳队列时间戳',
   `eeg1` varchar(512) DEFAULT NULL COMMENT '脑电1.txt',
   `eeg2` varchar(512) DEFAULT NULL COMMENT '脑电2.txt',
   `score` float DEFAULT NULL COMMENT '综合舒尔特分数',
   `accuracy` float DEFAULT NULL COMMENT '舒尔特准确率',
   `elapsed` float DEFAULT NULL COMMENT '舒尔特用时',
   `fatigue_score` float DEFAULT NULL COMMENT '疲劳检测分数(0-100)',
   `brain_load_score` float DEFAULT NULL COMMENT '脑负荷分数(0-100)',
   `emotion_score` float DEFAULT NULL COMMENT '情绪分数(0-100)',
   PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

在 MySQL 中运行：

```powershell
# 登录 MySQL 并选择数据库 (示例)
mysql -u root -p 123456
CREATE DATABASE multimodal_test;
USE multimodal_test;
# 复制并执行上面的 CREATE TABLE 语句
```

>注意：UI 中默认数据库配置位于 `ui/app/config.py`，默认值为 host=localhost, user=root, password=123456, db=test。请根据你的实际数据库修改。

### 3. 启动程序

本程序有自动和手动两种启动方式，任选一种即可。

#### 3.0 启动后端服务（自动启动）
双击根目录下的start.bat即可开始测试


#### 3.1 启动后端服务（手动启动）

打开命令行，运行:
```bash
cd d:\Mutimodal-Test

python -m src.main
```

**期望输出**:
```
| INFO     | orchestrator |  协调器启动
| INFO     | inference | 启动统一推理服务...
```

#### 3.2. 启动UI应用（手动启动）

另开一个命令行窗口，运行:
```bash
cd d:\Mutimodal-Test

python -m ui.main
```

**期望输出**:
```
INFO - 应用程序主窗口初始化完成。
INFO - 应用程序启动（模式：正常）。
```


## 🎯 测试流程

### 完整流程（约3-5分钟）

```
登录 → 设备校准 → 基线校准(30s) → SART实验(1分钟) → 文本朗读 → 血压测量 → 舒尔特方格 → 结果展示
```

### 各阶段详细说明

1. **登录**
   - 默认用户名: `admin`
   - 默认密码: `123456`
   - 用户数据存储在 `ui/data/users/users.csv`

2. **设备校准**
   - 等待摄像头初始化
   - 调整座椅位置确保人脸清晰可见
   - 调整脑电设备直到显示"已连接"
   - 点击"校准完成"进入基线校准

3. **基线校准（30秒）**
   - 注视屏幕中央的"+"号
   - 保持放松，自然呼吸
   - 系统采集静息态脑电基线数据

4. **SART注意力测试**
   - 1分钟，约60个试次
   - 规则：看到除"3"外的数字按【空格】，看到"3"不要按
   - 保持安静、注视中央，不说话
   - 系统采集脑电和多模态数据

5. **文本朗读**
   - 点击麦克风图标开始录制
   - 朗读系统提供的文本
   - 再次点击图标以停止录制
   - 系统采集情绪数据和生理数据

6. **血压测量**
   - 按提示进行血压测量
   - 系统记录生理健康指标

7. **舒尔特方格**
   - 按顺序点击数字
   - 测试注意力和高脑负荷状态

8. **结果展示**
   - 查看评估结果和统计图表
   - 数据自动保存到 `recordings/用户名/时间戳/` 目录

## 🔧 常见问题

### 问题1: 后端服务无法启动

**现象**: `OSError: [Errno 10048] error while attempting to bind on address ('127.0.0.1', 8765): 通常每个套接字地址(协议/网络地址/端口)只允许使用一次。`

**解决**:
1. 有一个终端已经在运行后端程序了，关闭那个后端，确保只有一个终端在运行后端程序。


### 问题2: 摄像头初始化失败

**现象**: `无法打开摄像头` 或 `Camera not found`

**解决**:
```bash
# 尝试不同的摄像头索引（0, 1, 2...）
set UI_CAMERA_INDEX=1
python -m ui.main
```

### 问题3: 模型推理失败

**现象**: `Model inference failed` 或 `模型未响应`

**解决**:
1. 检查 `config/models.yaml` 中模型是否启用
2. 确认模型文件是否存在于 `models_data/` 目录
3. 检查 GPU/CUDA 环境（如果使用 GPU）

### 问题4: 分数页面雷达图为空

**现象**: 测试完成后分数页面显示"暂无测试数据"或雷达图为空

**解决**:
1. **疲劳度评估未完成**: 
   - 日志显示 `没有收集到疲劳度分数数据`
   - 疲劳评估是异步的,需要等待1-3秒
   
2. **某些阶段未完成**:
   - 血压测量被跳过 (`血压脉搏检测: False`)
   - 对应指标不会显示在雷达图上

**预期日志输出**:
```
✅ 疲劳度分数已添加到列表, 当前列表长度=1
🎯 收到会话疲劳评估结果: 94.28
🔄 分数页面已显示,立即更新疲劳度分数
✅ 疲劳度数据已更新到分数展示页面
```

**临时解决**: 如果只是部分指标缺失,雷达图会显示已完成的指标。疲劳度为0时不会显示,但其他有效指标(情绪、脑负荷、专注度、血压)仍会正常显示。

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

日志以会话（session）为单位保存，UI 与后端共享同一会话 ID（保存在 `logs/session_timestamp.txt` 的最后一行）。每次运行会追加一个新的时间戳行，从而生成新的日志文件并保留历史记录。

| 日志类型 | 路径 | 说明 |
|---------|------|------|
| UI 日志 | `logs/ui/app_log_<timestamp>.txt` | UI 应用运行日志，`<timestamp>` 由 `logs/session_timestamp.txt` 的最后一行决定 |
| 后端日志 | `logs/src/src_log_<timestamp>.txt` | 后端运行日志，使用与 UI 相同的 `<timestamp>` |


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
| `models.yaml` | 模型配置（部署模式、模型路径、启用状态） |
| `interfaces.yaml` | 接口配置（WebSocket地址、端口） |



## 📞 获取帮助

### 问题反馈
- 在项目仓库提交 Issue
- 联系项目维护团队

---

**最后更新**: 2025年10月19日  
**项目版本**: 0.1.0  
**Python版本**: 3.11+
