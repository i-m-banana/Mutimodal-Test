"""测试EEG疲劳度模型的基线自动更新功能

使用方法:
    python test_baseline_update.py
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.eeg_fatigue_model import EEGFatigueModel


def test_baseline_update():
    """测试基线更新功能"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建模型实例
    model = EEGFatigueModel("eeg_fatigue")
    model.load()
    
    # 选择一个测试会话（以shh0为例）
    test_session = PROJECT_ROOT / "recordings" / "admin" / "20251101_202605"
    
    if not test_session.exists():
        print(f"❌ 测试会话不存在: {test_session}")
        print("\n可用的会话目录:")
        admin_dir = PROJECT_ROOT / "recordings" / "admin"
        if admin_dir.exists():
            sessions = sorted(admin_dir.iterdir(), reverse=True)[:5]
            for s in sessions:
                print(f"  - {s.name}")
        return
    
    print(f"\n{'='*70}")
    print(f"🧪 测试EEG疲劳度模型基线自动更新")
    print(f"{'='*70}")
    print(f"📂 测试会话: {test_session.name}")
    print(f"👤 被试标识: admin")
    print()
    
    # 执行推理（含基线更新）
    result = model.infer({
        "session_dir": str(test_session),
        "subject_base": "admin",
        "qc_is_lowload": True,
        "update_baseline": True
    })
    
    print(f"\n{'='*70}")
    print(f"📊 推理结果")
    print(f"{'='*70}")
    print(f"状态: {result.get('status')}")
    print(f"疲劳度分数: {result.get('eeg_fatigue_score', 0):.2f}")
    print(f"窗口数量: {result.get('num_windows', 0)}")
    
    if result.get('baseline_updated') is not None:
        print(f"\n🔄 基线更新信息:")
        print(f"  更新状态: {'✅已更新' if result['baseline_updated'] else '❌未更新'}")
        print(f"  原因: {result.get('baseline_reason', '未知')}")
    
    if result.get('status') != 'success':
        print(f"\n❌ 错误: {result.get('error', '未知错误')}")
    
    # 清理
    model.cleanup()
    
    print(f"\n{'='*70}")
    print(f"✅ 测试完成")
    print(f"{'='*70}\n")
    
    # 显示基线文件位置
    baseline_dir = PROJECT_ROOT / "models_data" / "eeg_fatigue_models" / "baselines"
    baseline_file = baseline_dir / "admin.json"
    
    print(f"💾 基线文件位置: {baseline_file}")
    if baseline_file.exists():
        import json
        with open(baseline_file, 'r', encoding='utf-8') as f:
            baseline = json.load(f)
        print(f"   med_tb: {baseline['med_tb']:.4f}")
        print(f"   mad_tb: {baseline['mad_tb']:.4f}")
        print(f"   med_ta: {baseline['med_ta']:.4f}")
        print(f"   mad_ta: {baseline['mad_ta']:.4f}")
    else:
        print(f"   (基线文件尚未创建)")
    
    print()


if __name__ == "__main__":
    test_baseline_update()
