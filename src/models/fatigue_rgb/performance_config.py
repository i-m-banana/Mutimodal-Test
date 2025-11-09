"""
RGB疲劳检测性能优化配置

这个文件提供了可选的性能优化选项，可以根据具体需求调整
"""

# ====================
# GPU加速配置
# ====================

# MediaPipe自动检测并使用GPU（无需配置）
# 如果系统有NVIDIA GPU + CUDA + cuDNN，会自动启用GPU加速

# ====================
# 模型配置
# ====================

# 模型复杂度 (0=最快, 1=平衡, 2=最精确)
# 当前使用: 0 (最快) - 已在 fershowv3.py 中配置
MODEL_COMPLEXITY = 0

# 静态图像模式 (False=视频流模式，更快)
STATIC_IMAGE_MODE = False

# 人脸检测置信度阈值 (0.0-1.0)
# 降低阈值可以检测更多人脸，但可能增加误检
MIN_DETECTION_CONFIDENCE = 0.5

# 人脸跟踪置信度阈值 (0.0-1.0)
MIN_TRACKING_CONFIDENCE = 0.5

# ====================
# 视频处理优化
# ====================

# 是否启用分辨率缩放
ENABLE_RESIZE = True

# 目标分辨率 (宽, 高)
# 推荐: (640, 480) - 平衡速度和质量
# 更快: (480, 360) 或 (320, 240)
# 更好: (1280, 720) - 需要更强的GPU
TARGET_RESOLUTION = (640, 480)

# 是否保持宽高比
KEEP_ASPECT_RATIO = True

# ====================
# 跳帧处理
# ====================

# 是否启用跳帧
ENABLE_FRAME_SKIP = False

# 跳帧间隔 (每N帧处理1帧)
# 1 = 不跳帧（最准确）
# 2 = 每2帧处理1帧（2倍速度）
# 3 = 每3帧处理1帧（3倍速度）
FRAME_SKIP_INTERVAL = 2

# ====================
# 并行处理
# ====================

# 是否启用多进程批量处理（仅用于离线分析多个视频）
ENABLE_MULTIPROCESSING = False

# 并行工作进程数 (None=自动检测CPU核心数)
MAX_WORKERS = None

# ====================
# 性能监控
# ====================

# 是否显示实时FPS
SHOW_FPS = True

# 是否显示GPU使用率（需要安装 gpustat）
SHOW_GPU_STATS = False

# 性能日志间隔（秒）
PERFORMANCE_LOG_INTERVAL = 5.0

# ====================
# 预设配置
# ====================

PRESETS = {
    "fastest": {
        "MODEL_COMPLEXITY": 0,
        "ENABLE_RESIZE": True,
        "TARGET_RESOLUTION": (480, 360),
        "ENABLE_FRAME_SKIP": True,
        "FRAME_SKIP_INTERVAL": 2,
        "description": "最快速度（牺牲一些精度）"
    },
    "balanced": {
        "MODEL_COMPLEXITY": 0,
        "ENABLE_RESIZE": True,
        "TARGET_RESOLUTION": (640, 480),
        "ENABLE_FRAME_SKIP": False,
        "FRAME_SKIP_INTERVAL": 1,
        "description": "平衡速度和精度 ✅ 推荐"
    },
    "quality": {
        "MODEL_COMPLEXITY": 1,
        "ENABLE_RESIZE": False,
        "TARGET_RESOLUTION": (1280, 720),
        "ENABLE_FRAME_SKIP": False,
        "FRAME_SKIP_INTERVAL": 1,
        "description": "最高质量（需要强劲GPU）"
    },
    "realtime": {
        "MODEL_COMPLEXITY": 0,
        "ENABLE_RESIZE": True,
        "TARGET_RESOLUTION": (640, 480),
        "ENABLE_FRAME_SKIP": False,
        "FRAME_SKIP_INTERVAL": 1,
        "description": "实时监控（前端UI使用）"
    }
}

# 当前使用的预设
CURRENT_PRESET = "balanced"


def get_config(preset_name: str = None):
    """获取配置
    
    Args:
        preset_name: 预设名称，可选值: fastest, balanced, quality, realtime
                     如果为None，使用CURRENT_PRESET
    
    Returns:
        dict: 配置字典
    """
    preset_name = preset_name or CURRENT_PRESET
    
    if preset_name not in PRESETS:
        raise ValueError(f"未知的预设: {preset_name}，可选值: {list(PRESETS.keys())}")
    
    return PRESETS[preset_name]


def print_config(preset_name: str = None):
    """打印配置信息"""
    preset_name = preset_name or CURRENT_PRESET
    config = get_config(preset_name)
    
    print(f"\n{'='*60}")
    print(f"RGB疲劳检测配置: {preset_name}")
    print(f"{'='*60}")
    print(f"说明: {config['description']}")
    print(f"\n配置详情:")
    for key, value in config.items():
        if key != "description":
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 显示所有预设
    print("可用的预设配置:\n")
    for name, preset in PRESETS.items():
        print(f"📋 {name}")
        print(f"   {preset['description']}")
        print(f"   模型复杂度: {preset['MODEL_COMPLEXITY']}")
        print(f"   分辨率: {preset['TARGET_RESOLUTION']}")
        print(f"   跳帧: {'是' if preset['ENABLE_FRAME_SKIP'] else '否'}")
        print()
    
    # 显示当前配置
    print_config()
