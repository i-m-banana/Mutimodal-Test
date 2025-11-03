"""
疲劳评估函数使用示例
"""
from fershowv3 import evaluate_fatigue_from_video


def main():
    # 示例1: 评估单个视频
    video_path = "/home/caixiaohui/workspace/multi_state/mulimodal_state/fatiguev2/fff/wsh0/20251031_113230/fatigue/rgb0.avi"
    
    print("正在评估视频疲劳分数...")
    result = evaluate_fatigue_from_video(
        video_path=video_path,
        start_sec=0.0,
        duration_sec=None,  # 处理整个视频
        stats_output_path="fatigue_stats.json",  # 可选: 保存统计数据
    )
    
    print("\n" + "=" * 60)
    print("疲劳评估结果:")
    print("=" * 60)
    
    if result["fatigue_score"] is None:
        print(f"错误: {result.get('error', 'Unknown error')}")
    else:
        print(f"疲劳分数: {result['fatigue_score']:.2f} / 90")
        print(f"状态评级: {result['state_description']}")
        print(f"\n累计统计:")
        print(f"  - 眨眼次数: {result['totals']['blinks']}")
        print(f"  - 打哈欠次数: {result['totals']['yawns']}")
        print(f"  - 低头次数: {result['totals']['head_drops']}")
        print(f"  - 疲劳事件: {result['totals']['fatigue_events']}")
        print(f"\n人脸检测:")
        print(f"  - 检测到人脸的帧数: {result['face_frames']}")
        print(f"  - 总处理帧数: {result['total_frames']}")
        print(f"  - 人脸检测率: {result['face_ratio']*100:.1f}%")
        
        if result['summary']:
            print(f"\n状态分布:")
            for status, stats in result['summary'].items():
                print(f"  - {status}: {stats['ratio']*100:.1f}%")
    
    print("=" * 60)


def batch_evaluate_videos(video_paths: list[str]):
    """批量评估多个视频"""
    results = []
    for video_path in video_paths:
        print(f"\n处理视频: {video_path}")
        try:
            result = evaluate_fatigue_from_video(video_path)
            results.append({
                "video_path": video_path,
                "score": result["fatigue_score"],
                "state": result["state_description"],
            })
            print(f"  分数: {result['fatigue_score']:.2f} - {result['state_description']}")
        except Exception as e:
            print(f"  处理失败: {e}")
            results.append({
                "video_path": video_path,
                "score": None,
                "error": str(e),
            })
    
    return results


if __name__ == "__main__":
    main()
    
    # 批量评估示例 (取消注释使用)
    # video_list = [
    #     "/path/to/video1.avi",
    #     "/path/to/video2.avi",
    #     "/path/to/video3.avi",
    # ]
    # batch_results = batch_evaluate_videos(video_list)
    # print("\n批量评估结果:")
    # for res in batch_results:
    #     print(f"{res['video_path']}: {res.get('score', 'Failed')}")
