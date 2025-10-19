# 🚀 快速启动指南

## 系统要求

- Python 3.8+
- Windows 10/11
- 所需依赖包已安装

## 📋 启动步骤

### 1. 启动后端服务 (必须先启动)

打开命令行，运行:
```bash
cd d:\非接触式安装程序\project-root
python -m src.main --root .

# 可选：关闭控制台监听输出
python -m src.main --root . --no-listeners
```

**期望输出**:
```
WebSocket服务器启动在 ws://0.0.0.0:8765
事件总线已初始化
系统准备就绪
```

### 2. 启动UI应用

另开一个命令行窗口，运行:
```bash
cd d:\duomotai\src\project-root
python -m ui.main
```

**期望输出**:
```
应用程序主窗口初始化完成
应用程序启动（模式：正常）
```

## 🔌 设备连接测试（可选）

### 测试EEG和多模态设备

在启动后端服务后，可以运行设备连接测试：

```bash
cd d:\duomotai\src\project-root
python aidebug\test_device_connections.py
```

**测试内容**:
1. EEG脑电设备连接和数据采集
2. 多模态设备（RealSense + Tobii）连接
3. 数据保存验证

**期望输出**:
```
============================================================
测试 EEG 脑电设备服务
============================================================
1. 检查EEG硬件状态...
   硬件驱动可用: True/False
   强制模拟模式: False
   当前运行状态: False
2. 启动EEG采集...
   启动结果: {'status': 'started', 'save_dir': '...'}
...
EEG测试完成！

============================================================
测试多模态服务（RealSense + Tobii）
============================================================
1. 检查硬件能力...
   RealSense驱动: True/False
   Tobii驱动: True/False
...
多模态测试完成！
```

**注意**: 
- 如果硬件不可用，系统会自动切换到模拟模式
- 模拟模式下仍然可以测试完整的数据采集流程
- 测试数据保存在 `recordings/test_eeg/` 和 `recordings/test_multimodal/`

## 🎯 测试流程

### 快速验证

1. **登录**
   - 用户名: `admin`
   - 密码: `admin123`

2. **校准**
   - 等待摄像头初始化
   - 点击"校准完成"

3. **测试**
   - 系统自动开始测试流程
   - 观察数据采集是否正常

## 🔧 常见问题

### 问题1: 后端服务无法启动
**现象**: 端口8765已被占用
**解决**:
```bash
# 查找占用端口的进程
netstat -ano | findstr :8765
# 结束该进程
taskkill /F /PID <进程ID>
```

### 问题2: UI无法连接后端
**现象**: "Backend connection not established"
**解决**:
1. 确认后端服务已启动
2. 检查防火墙设置
3. 确认端口8765可访问

### 问题3: 摄像头初始化失败
**现象**: "无法打开摄像头"
**解决**:
```bash
# 使用模拟模式
set UI_FORCE_SIMULATION=1
python -m ui.main
```

## 🐛 调试模式

### 启用详细日志
```bash
python -m ui.main --debug
```

### 查看日志文件
- UI日志: `ui/logs/app_log_*.txt`
- 后端日志: `logs/*.log`

## 📊 性能监控

### 检查系统状态
```python
python -c "from ui.app import config; print('配置加载成功')"
```

### 测试后端连接
```python
python -c "from ui.services.backend_client import get_backend_client; client = get_backend_client(); print('后端连接:', client.url)"
```

## 📝 环境变量

### 常用配置
```bash
# 强制模拟模式（无硬件测试）
set UI_FORCE_SIMULATION=1

# EEG设备模拟模式
set BACKEND_EEG_SIMULATION=1

# 指定摄像头索引
set UI_CAMERA_INDEX=0

# 调试模式
set UI_DEBUG=1

# 跳过数据库
set UI_SKIP_DATABASE=1
```

### 模拟模式说明
**Phase 2新增**: EEG服务支持独立的模拟模式
- `BACKEND_EEG_SIMULATION=1` - 后端使用模拟EEG数据（不需要真实蓝牙设备）
- `UI_EEG_SIMULATION=1` - UI使用模拟模式（后端不可用时的降级方案）
- `UI_FORCE_SIMULATION=1` - 强制UI所有硬件使用模拟模式

## 🎓 开发模式

### 代码修改后重启
1. 停止UI程序 (Ctrl+C)
2. 停止后端服务 (Ctrl+C)
3. 重新启动后端
4. 重新启动UI

### 运行测试
```bash
# Phase 1: UI重构测试
python aidebug\test_ui_refactor.py

# Phase 2: EEG服务集成测试
python aidebug\test_eeg_integration.py
```

### 项目结构 (Phase 2)
```
ui/
├── data/                    # 资源文件 (Phase 2新增)
│   ├── users/              # 用户数据CSV
│   └── questionnaires/     # 问卷配置YAML
├── services/
│   ├── eeg_service.py      # EEG客户端代理 (Phase 2精简到122行)
│   ├── av_service.py       # 音视频采集
│   └── multimodal_service.py
├── widgets/                # UI组件
├── models/                 # 模型推理
└── utils_common/           # 通用工具

src/
├── services/
│   ├── eeg_service.py      # EEG后端服务 (Phase 2新增, 368行)
│   ├── av_service.py       # 音视频后端
│   └── ui_command_router.py # WebSocket命令路由
├── interfaces/
│   └── websocket_server.py # WebSocket服务器
└── core/
    └── event_bus.py        # 事件总线
```

## 📞 获取帮助

遇到问题请查看:
- `REFACTORING_PHASE2_SUMMARY.md` - Phase 2重构总结（资源组织+EEG后端迁移）
- `REFACTORING_REPORT.md` - Phase 1完整报告（UI结构重组）
- `docs/ui_architecture.md` - 架构文档
- `REFACTORING_SUMMARY.md` - Phase 1详细总结

---

**最后更新**: 2025年1月7日  
**版本**: 2.0 (Phase 2重构完成)
