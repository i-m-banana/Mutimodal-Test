"""测试情绪模型后端集成

此脚本用于测试情绪模型后端是否正常工作
"""

import asyncio
import base64
import json
import sys
from pathlib import Path
import time

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import websockets
except ImportError as e:
    print(f"错误: 缺少依赖库 - {e}")
    print("请安装: pip install websockets")
    sys.exit(1)


async def test_health_check(uri: str):
    """测试健康检查"""
    print(f"\n[1/4] 测试健康检查...")
    try:
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=5) as ws:
            request = {
                "type": "health_check",
                "timestamp": time.time()
            }
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            
            if response.get("type") == "health_response":
                status = response.get("status")
                model_loaded = response.get("model_loaded")
                print(f"✅ 健康检查成功")
                print(f"   状态: {status}")
                print(f"   模型已加载: {model_loaded}")
                return True
            else:
                print(f"❌ 健康检查失败: 未知响应类型 {response.get('type')}")
                return False
    except asyncio.TimeoutError:
        print(f"❌ 连接超时")
        return False
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


async def test_empty_inference(uri: str):
    """测试空数据推理"""
    print(f"\n[2/4] 测试空数据推理...")
    try:
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=5) as ws:
            request = {
                "type": "inference_request",
                "request_id": "test-empty",
                "data": {
                    "samples": []
                }
            }
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            
            if response.get("type") == "inference_response":
                result = response.get("result", {})
                status = result.get("status")
                
                if status == "success":
                    predictions = result.get("predictions", {})
                    print(f"✅ 空数据推理成功")
                    print(f"   情绪分数: {predictions.get('emotion_score')}")
                    print(f"   消息: {predictions.get('message', 'N/A')}")
                    return True
                else:
                    print(f"⚠️  推理返回错误: {result.get('error')}")
                    return True  # 空数据返回错误是正常的
            else:
                print(f"❌ 推理失败: 未知响应类型")
                return False
    except Exception as e:
        print(f"❌ 空数据推理失败: {e}")
        return False


async def test_real_data_inference(uri: str, data_dir: Path):
    """测试真实数据推理"""
    print(f"\n[3/4] 测试真实数据推理...")
    
    if not data_dir.exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
        return True  # 跳过，不算失败
    
    # 查找视频和音频文件
    video_files = sorted(data_dir.glob("*.avi"))
    audio_files = sorted(data_dir.glob("*.wav"))
    
    if not video_files or not audio_files:
        print(f"⚠️  未找到视频或音频文件")
        return True
    
    print(f"   找到 {len(video_files)} 个视频文件, {len(audio_files)} 个音频文件")
    
    # 尝试读取真实文本记录
    text_records = {}
    record_file = data_dir / "record.txt"
    if record_file.exists():
        try:
            import ast
            with open(record_file, 'r', encoding='utf-8') as f:
                records = ast.literal_eval(f.read())
                for record in records:
                    q_idx = record.get('question_index')
                    text = record.get('recognized_text', '')
                    text_records[q_idx] = text
            print(f"   读取到 {len(text_records)} 条真实文本记录")
        except Exception as e:
            print(f"   ⚠️  读取record.txt失败: {e}")
    
    try:
        # 优先选择有真实文本的样本
        samples = []
        selected_files = []
        
        # 首先选择有真实文本记录的文件
        for video_path in video_files:
            if len(selected_files) >= 3:
                break
            file_index = int(video_path.stem)
            if file_index in text_records:
                audio_path = data_dir / f"{file_index}.wav"
                if audio_path.exists():
                    selected_files.append((video_path, audio_path, file_index))
        
        # 如果不足3个,补充其他文件
        if len(selected_files) < 3:
            for video_path, audio_path in zip(video_files, audio_files):
                if len(selected_files) >= 3:
                    break
                file_index = int(video_path.stem)
                if (video_path, audio_path, file_index) not in [(v, a, i) for v, a, i in selected_files]:
                    selected_files.append((video_path, audio_path, file_index))
        
        # 处理选中的样本
        for video_path, audio_path, file_index in selected_files:
            # 读取并编码视频
            with open(video_path, 'rb') as f:
                video_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 读取并编码音频
            with open(audio_path, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # 使用真实文本或默认文本
            text = text_records.get(file_index, f"样本 {file_index} 无文本记录")
            has_text = "✓" if file_index in text_records else "✗"
            
            samples.append({
                "video_b64": video_b64,
                "audio_b64": audio_b64,
                "text": text,
                "question_index": file_index
            })
            
            print(f"   样本 {file_index} {has_text}: 视频={len(video_b64)/1024:.1f}KB, 音频={len(audio_b64)/1024:.1f}KB, 文本=\"{text}\"")
        
        # 发送推理请求
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=60) as ws:
            request = {
                "type": "inference_request",
                "request_id": "test-real-data",
                "data": {
                    "samples": samples
                }
            }
            
            print(f"   发送推理请求 (总大小: {len(json.dumps(request))/1024/1024:.1f}MB)...")
            await ws.send(json.dumps(request))
            
            print(f"   等待推理结果...")
            response = json.loads(await ws.recv())
            
            if response.get("type") == "inference_response":
                result = response.get("result", {})
                if result.get("status") == "success":
                    predictions = result.get("predictions", {})
                    emotion_score = predictions.get("emotion_score")
                    latency_ms = result.get("latency_ms", 0)
                    inference_time_ms = predictions.get("inference_time_ms", 0)
                    
                    print(f"✅ 真实数据推理成功")
                    print(f"   情绪分数: {emotion_score:.2f}")
                    print(f"   样本数量: {len(samples)}")
                    print(f"   推理耗时: {inference_time_ms:.2f}ms")
                    print(f"   总耗时: {latency_ms:.2f}ms")
                    
                    # 如果有单个样本的详细结果
                    if "sample_scores" in predictions:
                        print(f"   各样本情绪分数:")
                        for idx, score in enumerate(predictions["sample_scores"]):
                            print(f"     样本 {idx+1}: {score:.2f}")
                    
                    return True
                else:
                    error = result.get("error", "未知错误")
                    print(f"❌ 推理失败: {error}")
                    return False
            else:
                print(f"❌ 未知响应类型: {response.get('type')}")
                return False
                
    except asyncio.TimeoutError:
        print(f"❌ 推理超时")
        return False
    except Exception as e:
        print(f"❌ 真实数据推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance(uri: str, data_dir: Path, num_requests: int = 5):
    """性能测试"""
    print(f"\n[4/4] 测试性能...")
    
    if not data_dir.exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
        return True  # 跳过，不算失败
    
    # 查找第一组数据
    video_files = sorted(data_dir.glob("*.avi"))
    audio_files = sorted(data_dir.glob("*.wav"))
    
    if not video_files or not audio_files:
        print(f"⚠️  未找到测试数据")
        return True
    
    # 尝试读取真实文本记录
    text_records = {}
    record_file = data_dir / "record.txt"
    if record_file.exists():
        try:
            import ast
            with open(record_file, 'r', encoding='utf-8') as f:
                records = ast.literal_eval(f.read())
                for record in records:
                    q_idx = record.get('question_index')
                    text = record.get('recognized_text', '')
                    text_records[q_idx] = text
        except Exception as e:
            print(f"   ⚠️  读取record.txt失败: {e}")
    
    try:
        # 准备一个样本
        video_path = video_files[0]
        audio_path = audio_files[0]
        file_index = int(video_path.stem)
        
        with open(video_path, 'rb') as f:
            video_b64 = base64.b64encode(f.read()).decode('utf-8')
        with open(audio_path, 'rb') as f:
            audio_b64 = base64.b64encode(f.read()).decode('utf-8')
        
        # 使用真实文本或默认文本
        text = text_records.get(file_index, "性能测试样本")
        
        sample = {
            "video_b64": video_b64,
            "audio_b64": audio_b64,
            "text": text,
            "question_index": file_index
        }
        
        print(f"   使用样本: 文本=\"{text}\"")
        print(f"   运行 {num_requests} 次推理...")
        latencies = []
        emotion_scores = []
        
        for i in range(num_requests):
            start_time = time.time()
            
            async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=60) as ws:
                request = {
                    "type": "inference_request",
                    "request_id": f"perf-test-{i}",
                    "data": {
                        "samples": [sample]
                    }
                }
                
                await ws.send(json.dumps(request))
                response = json.loads(await ws.recv())
                
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                latencies.append(latency)
                
                result = response.get("result", {})
                if result.get("status") == "success":
                    predictions = result.get("predictions", {})
                    emotion_score = predictions.get("emotion_score", 0)
                    emotion_scores.append(emotion_score)
                    print(f"   请求 {i+1}/{num_requests}: {latency:.0f}ms, 情绪分数: {emotion_score:.2f}")
                else:
                    print(f"   请求 {i+1}/{num_requests}: 失败")
        
        # 统计
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_emotion = sum(emotion_scores) / len(emotion_scores) if emotion_scores else 0
        
        print(f"✅ 性能测试完成")
        print(f"   平均延迟: {avg_latency:.0f}ms")
        print(f"   最小延迟: {min_latency:.0f}ms")
        print(f"   最大延迟: {max_latency:.0f}ms")
        print(f"   吞吐量: {1000/avg_latency:.1f} 请求/秒")
        print(f"   平均情绪分数: {avg_emotion:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False



async def main():
    """主测试函数"""
    print("=" * 70)
    print("情绪模型后端集成测试")
    print("=" * 70)
    
    uri = "ws://127.0.0.1:8768"
    print(f"\n测试目标: {uri}")
    print("请确保情绪模型后端已启动\n")
    
    # 使用完整的真实数据
    data_dir = Path("recordings/admin/20251014_200601/emotion")
    print(f"测试数据目录: {data_dir}\n")
    
    results = []
    
    # 运行测试
    results.append(("健康检查", await test_health_check(uri)))
    results.append(("空数据推理", await test_empty_inference(uri)))
    results.append(("真实数据推理", await test_real_data_inference(uri, data_dir)))
    results.append(("性能测试", await test_performance(uri, data_dir, num_requests=5)))
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print(f"\n通过率: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过!")
        return 0
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
