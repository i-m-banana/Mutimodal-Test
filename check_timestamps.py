"""检查脑电时间戳记录是否符合要求"""

import json
import os
from datetime import datetime

def analyze_timestamps(json_path):
    """分析时间戳文件"""
    print("=" * 80)
    print("脑电时间戳分析")
    print("=" * 80)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        timestamps = json.load(f)
    
    print(f"\n总共记录了 {len(timestamps)} 个时间戳\n")
    
    # 定义预期的时间戳含义
    expected_stages = {
        0: "系统启动",
        1: "设备校准开始 OR SART开始(short)",
        2: "设备校准结束",
        3: "基线开始(未使用标准索引10)",
        4: "基线结束(未使用标准索引11) OR SART结束(short,标准索引4)",
        5: "SART开始(重复?) OR SART结束(重复?)",
        6: "文本问答开始",
        7: "文本问答结束",
        8: "血压测量开始",
        9: "血压测量结束",
        10: "舒尔特开始",
        11: "舒尔特结束"
    }
    
    print("时间戳详情:")
    print("-" * 80)
    print(f"{'索引':<6} {'时间':<20} {'相对时间(秒)':<15} {'预期含义':<30}")
    print("-" * 80)
    
    start_time = timestamps[0]['timestamp']
    
    for i, ts in enumerate(timestamps):
        relative_time = ts['timestamp'] - start_time
        dt = datetime.fromisoformat(ts['datetime'])
        time_str = dt.strftime('%H:%M:%S.%f')[:-3]
        expected = expected_stages.get(i, "未知阶段")
        
        print(f"{i:<6} {time_str:<20} {relative_time:<15.3f} {expected:<30}")
    
    print("\n" + "=" * 80)
    print("标准要求对比:")
    print("=" * 80)
    
    print("\n📋 Short模式应有的时间戳:")
    print("   0: 系统启动")
    print("  10: 基线开始 ⚠️ (实际缺失，使用了索引3)")
    print("  11: 基线结束 ⚠️ (实际缺失，使用了索引4)")
    print("   1: SART开始 ⚠️ (实际可能被设备校准占用)")
    print("   4: SART结束 ⚠️ (实际可能和基线结束混淆)")
    print("   -: 文本问答开始")
    print("   -: 文本问答结束")
    print("   -: 血压测量开始")
    print("   -: 血压测量结束")
    print("   -: 舒尔特测试开始")
    print("   -: 舒尔特测试结束")
    
    print("\n⚠️  发现的问题:")
    print("   1. 基线阶段没有使用标准索引 10→11")
    print("   2. SART阶段可能没有使用标准索引 1→4 (short) 或 20→21 (long)")
    print("   3. call_index 是自动生成的(enumerate)，不符合原始范式要求")
    print("   4. 用户跳过时(按Q键)只记录了连续的时间戳，没有保留原始索引含义")
    
    print("\n✅ 建议修改:")
    print("   方案1: 修改 part_timestamps 为字典列表，保存 (call_index, timestamp)")
    print("   方案2: 在跳过时仍然使用正确的 call_index 标记")
    print("   方案3: 添加阶段名称字段，便于后续分析")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 查找最新的时间戳文件
    recordings_dir = "recordings/admin"
    
    if os.path.exists(recordings_dir):
        sessions = [d for d in os.listdir(recordings_dir) if os.path.isdir(os.path.join(recordings_dir, d))]
        if sessions:
            latest_session = sorted(sessions)[-1]
            json_path = os.path.join(recordings_dir, latest_session, "eeg", "part_timestamps.json")
            
            if os.path.exists(json_path):
                print(f"\n分析文件: {json_path}\n")
                analyze_timestamps(json_path)
            else:
                print(f"❌ 找不到文件: {json_path}")
        else:
            print(f"❌ 没有找到任何会话记录")
    else:
        print(f"❌ 录制目录不存在: {recordings_dir}")
