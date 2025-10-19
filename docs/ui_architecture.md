# UI文件夹架构说明

## 📁 目录结构

```
ui/
├── app/                          # 主应用程序
│   ├── pages/                   # UI页面
│   │   ├── calibration.py      # 校准页面
│   │   ├── login.py            # 登录页面
│   │   ├── test.py             # 测试页面
│   │   └── ...
│   ├── utils/                   # UI专用工具
│   │   ├── helpers.py          # 辅助函数
│   │   └── widgets.py          # 自定义控件
│   ├── application.py           # 主应用类
│   ├── config.py                # 全局配置
│   └── qt.py                    # Qt导入管理
│
├── services/                     # 后端服务客户端（通过WebSocket通信）
│   ├── backend_client.py        # WebSocket客户端基础类
│   ├── av_service.py            # 音视频服务代理
│   ├── eeg_service.py           # EEG服务代理
│   └── multimodal_service.py    # 多模态服务代理
│
├── widgets/                      # 可重用UI组件
│   ├── multimodal_preview.py    # 多模态预览组件
│   ├── brain_load_bar.py        # 脑负荷进度条
│   ├── schulte_grid.py          # 舒尔特方格
│   ├── dashboard.py             # 仪表盘
│   └── score_page.py            # 评分页面
│
├── models/                       # 模型推理
│   └── model_inference.py       # 模型推理封装
│
├── utils_common/                 # 通用工具类
│   ├── database.py              # 数据库操作
│   ├── tools.py                 # 语音识别等工具
│   └── thread_process_manager.py # 线程进程管理
│
├── runtime/                      # 运行时管理
│   ├── lifecycle_manager.py     # 生命周期管理
│   ├── thread_manager.py        # 线程管理器
│   └── process_manager.py       # 进程管理器
│
├── assets/                       # 静态资源
├── logs/                         # 日志文件
├── recordings/                   # 录制数据
├── main.py                       # 程序入口
└── style.qss                     # 样式表
```

## 🔗 前后端通信架构

### 后端实现 (src/)
- 包含真实的硬件操作和数据采集逻辑
- 通过WebSocket服务器对外提供API
- 端口: 8765

### 前端实现 (ui/)
- 纯UI展示和用户交互
- 通过 `services/backend_client.py` 与后端WebSocket通信
- 所有硬件操作都委托给后端

### 通信流程

```
用户操作 → UI页面 → services/xxx_service.py → WebSocket → src/services/xxx_service.py → 硬件
                                                            ↓
用户界面 ← UI页面 ← services/xxx_service.py ← WebSocket ← 数据/事件
```

## 🎯 服务代理说明

### 1. av_service.py (UI端)
**职责**: 音视频数据采集的客户端代理
- 发送命令给后端启动/停止采集
- 接收摄像头帧数据
- 接收音频电平数据
- 提供模拟模式（当后端不可用时）

**后端对应**: `src/services/av_service.py`

### 2. multimodal_service.py (UI端)
**职责**: 多模态数据采集的客户端代理
- 启动/停止多模态采集
- 获取疲劳度评分
- 获取RGB、深度、眼动数据路径

**后端对应**: `src/services/multimodal_service.py`

### 3. eeg_service.py (UI端)
**职责**: EEG脑电数据采集的客户端代理
- 蓝牙设备连接管理
- EEG数据采集控制
- 提供模拟模式

**后端对应**: EEG设备直接通过蓝牙采集（独立模块）

## 🚀 启动流程

### 1. 启动后端服务
```bash
python -m src.main --root .
```
- 启动WebSocket服务器 (端口8765)
- 初始化事件总线
- 加载配置和模型

### 2. 启动UI应用
```bash
python -m ui.main
```
- 连接到后端WebSocket服务器
- 初始化UI界面
- 准备好接收后端事件

## ⚙️ 配置说明

### 环境变量

```bash
# 后端配置
BACKEND_ALLOW_SYNTHETIC_CAMERA=1  # 后端允许模拟摄像头
BACKEND_WS_PORT=8765              # WebSocket端口

# UI配置  
UI_FORCE_SIMULATION=1             # UI强制模拟模式
UI_CAMERA_INDEX=0                 # 摄像头索引
UI_AUDIO_DEVICE_INDEX=0           # 音频设备索引
UI_WHISPER_MODEL=base             # 语音识别模型
UI_EEG_SIMULATION=1               # EEG模拟模式
UI_MULTIMODAL_SIMULATION=1        # 多模态模拟模式
```

### 模拟模式

当后端服务不可用时，UI的服务代理会自动切换到模拟模式：
- 生成模拟的摄像头帧
- 生成模拟的音频电平
- 生成模拟的疲劳度评分
- 不保存真实数据

## 📝 开发指南

### 添加新的硬件设备

1. **后端实现** (src/)
   - 在 `src/services/` 创建服务类
   - 在 `src/interfaces/websocket_server.py` 注册命令处理器
   - 实现硬件操作逻辑

2. **前端代理** (ui/)
   - 在 `ui/services/` 创建客户端代理类
   - 通过 `backend_client.py` 发送命令
   - 在 `ui/app/config.py` 导出便捷接口
   - UI页面中使用代理函数

### 调试技巧

1. 启用调试模式
```bash
python -m ui.main --debug
```

2. 查看日志
- 后端日志: `logs/` (项目根目录)
- UI日志: `ui/logs/`

3. 使用模拟模式测试
```bash
UI_FORCE_SIMULATION=1 python -m ui.main
```

## 🔧 已移除的冗余代码

### ❌ 已删除: ui/backend/
**原因**: 
- 硬件驱动应该在后端实现
- UI不应该直接操作硬件
- 导致架构混乱和代码重复

### ✅ 现在的做法:
- 硬件操作全部在 `src/` 后端实现
- UI通过WebSocket与后端通信
- 前后端职责清晰分离

## 🎨 样式和主题

- 全局样式: `ui/style.qss`
- 使用Qt样式表进行主题定制
- 支持深色/浅色主题切换

## 📊 数据流

```
数据采集流程:
用户开始测试 
  → UI发送start命令 
  → 后端启动硬件采集 
  → 后端处理数据并发布事件
  → UI接收事件并更新界面
  → 后端保存数据到文件
  → 用户停止测试
  → UI发送stop命令
  → 后端停止采集并返回文件路径
  → UI将路径写入数据库
```

## 🔐 安全注意事项

1. **WebSocket通信**: 当前使用未加密的ws://协议，生产环境应使用wss://
2. **数据库密码**: 不要在代码中硬编码，使用环境变量
3. **文件权限**: 确保录制目录有适当的访问权限

## 📚 参考文档

- Qt5文档: https://doc.qt.io/qt-5/
- WebSocket协议: https://tools.ietf.org/html/rfc6455
- OpenCV文档: https://docs.opencv.org/
