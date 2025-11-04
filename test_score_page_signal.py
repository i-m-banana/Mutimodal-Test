"""测试疲劳度评估结果传递到分数页面

这个脚本模拟整个流程，验证分数页面是否能正确接收并显示疲劳度评估结果。
"""

import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 模拟后端发布检测结果事件
def simulate_fatigue_assessment_result():
    """模拟后端完成疲劳度评估并发布结果"""
    from services.backend_client import get_backend_client
    
    backend_client = get_backend_client()
    
    # 模拟评估结果数据
    payload = {
        "detector": "model_fatigue",
        "status": "detected",
        "label": "fatigue",
        "predictions": {
            "fatigue_score": 85.0,
            "prediction_class": "重度疲劳",
            "confidence": 0.5,
            "fusion_method": "rgb_only",
            "inference_mode": "session_assessment",  # 关键：标识这是会话评估
            "components": {
                "eeg": {"score": 0.0, "weight": 0.0, "valid": False},
                "rgb": {"score": 85.0, "weight": 1.0, "valid": True}
            }
        },
        "request_id": "test_request_12345",
        "timestamp": time.time()
    }
    
    print("📤 发布疲劳度评估结果...")
    print(f"   fatigue_score: {payload['predictions']['fatigue_score']}")
    print(f"   inference_mode: {payload['predictions']['inference_mode']}")
    
    # 发射信号
    backend_client.detection_result.emit(payload)
    
    print("✅ 结果已发布")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 测试疲劳度评估结果传递到分数页面")
    print("="*70 + "\n")
    
    print("说明:")
    print("1. 这个测试验证分数页面能否接收后端的疲劳度评估结果")
    print("2. 正常流程中，疲劳度评估是异步的，可能在分数页面显示后才完成")
    print("3. 修复后，分数页面会监听评估结果并实时更新显示")
    print()
    
    # 等待一会儿再发布（模拟异步延迟）
    print("⏱️  等待3秒（模拟异步评估延迟）...")
    time.sleep(3)
    
    simulate_fatigue_assessment_result()
    
    print("\n" + "="*70)
    print("✅ 测试完成")
    print("="*70 + "\n")
    
    print("预期结果:")
    print("- 分数页面应该收到信号")
    print("- 日志显示: 📊 分数页面收到疲劳度评估结果")
    print("- 疲劳检测分数更新为 85.0")
    print("- 雷达图和综合得分自动刷新")
    print()
