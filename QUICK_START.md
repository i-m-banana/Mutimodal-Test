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


## 🎯 测试流程

### 使用UI系统

1. **登录**
   - 默认用户名: `admin`
   - 默认密码: `123456`
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

**解决方案 - 指定摄像头索引**:
```bash
# 尝试不同的摄像头索引（0, 1, 2...）
set UI_CAMERA_INDEX=1
python -m ui.main
```

### 问题4: 模型推理失败

**现象**: `Model inference failed` 或 `模型未响应`

**解决**:
1. 检查 `config/models.yaml` 中模型是否启用
2. 确认模型文件是否存在于 `models_data/` 目录
3. 检查 GPU/CUDA 环境（如果使用 GPU）

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

### 性能监控


**测试后端连接**:
```bash
# 使用 wscat 工具测试 WebSocket
npm install -g wscat
wscat -c ws://127.0.0.1:8765

# 发送心跳测试
{"type": "ping"}
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
| `models.yaml` | 模型配置（部署模式、模型路径、启用状态） |
| `interfaces.yaml` | 接口配置（WebSocket地址、端口） |



## 📞 获取帮助

### 参考文档
- **AGENTS.MD**: 编码规范和开发准则

### 问题反馈
- 在项目仓库提交 Issue
- 联系项目维护团队

---

**最后更新**: 2025年10月19日  
**项目版本**: 0.1.0  
**Python版本**: 3.11+
