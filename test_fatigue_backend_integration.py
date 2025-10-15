"""测试疲劳度模型后端集成

此脚本用于测试疲劳度模型后端是否正常工作
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
    import numpy as np
    import cv2
except ImportError as e:
    print(f"错误: 缺少依赖库 - {e}")
    print("请安装: pip install websockets numpy opencv-python")
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
                    "rgb_frames": [],
                    "depth_frames": [],
                    "eyetrack_samples": [],
                    "elapsed_time": 0.0
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
                    print(f"   疲劳度分数: {predictions.get('fatigue_score')}")
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


async def test_mock_inference(uri: str):
    """测试模拟数据推理"""
    print(f"\n[3/4] 测试模拟数据推理...")
    try:
        # 生成模拟RGB图像 (至少75帧,因为模型需要这么多帧)
        num_frames = 75
        rgb_frames = []
        for i in range(num_frames):
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                rgb_b64 = base64.b64encode(buffer).decode("ascii")
                rgb_frames.append(rgb_b64)
        
        # 生成模拟深度图像
        depth_frames = []
        for i in range(num_frames):
            depth = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
            ok, buffer = cv2.imencode(".png", depth)
            if ok:
                depth_b64 = base64.b64encode(buffer).decode("ascii")
                depth_frames.append(depth_b64)
        
        # 生成模拟眼动数据 (眼动采样率通常是RGB的5倍)
        eyetrack_samples = []
        for i in range(num_frames * 5):
            sample = list(np.random.rand(8))
            eyetrack_samples.append(sample)
        
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=10) as ws:
            request = {
                "type": "inference_request",
                "request_id": "test-mock",
                "data": {
                    "rgb_frames": rgb_frames,
                    "depth_frames": depth_frames,
                    "eyetrack_samples": eyetrack_samples,
                    "elapsed_time": 10.0
                }
            }
            
            print(f"   发送推理请求...")
            print(f"   - RGB帧数: {len(rgb_frames)}")
            print(f"   - 深度帧数: {len(depth_frames)}")
            print(f"   - 眼动样本数: {len(eyetrack_samples)}")
            
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            
            if response.get("type") == "inference_response":
                result = response.get("result", {})
                status = result.get("status")
                
                if status == "success":
                    predictions = result.get("predictions", {})
                    print(f"✅ 模拟数据推理成功")
                    print(f"   疲劳度分数: {predictions.get('fatigue_score')}")
                    print(f"   预测类别: {predictions.get('prediction_class')}")
                    print(f"   推理耗时: {predictions.get('inference_time_ms')}ms")
                    print(f"   总耗时: {result.get('latency_ms')}ms")
                    return True
                else:
                    print(f"❌ 推理失败: {result.get('error')}")
                    return False
            else:
                print(f"❌ 推理失败: 未知响应类型")
                return False
    except Exception as e:
        print(f"❌ 模拟数据推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_performance(uri: str, num_requests: int = 10):
    """测试性能"""
    print(f"\n[4/4] 测试性能 ({num_requests}次请求)...")
    try:
        num_frames = 75  # 模型要求的最小帧数
        
        latencies = []
        fatigue_scores = []
        
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=30) as ws:
            for i in range(num_requests):
                # 每次请求生成不同的随机数据
                rgb_frames = []
                for j in range(num_frames):
                    # 使用不同的随机种子生成不同的图像
                    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    ok, buffer = cv2.imencode(".jpg", img)
                    if ok:
                        rgb_b64 = base64.b64encode(buffer).decode("ascii")
                        rgb_frames.append(rgb_b64)
                
                depth_frames = []
                for j in range(num_frames):
                    depth = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
                    ok, buffer = cv2.imencode(".png", depth)
                    if ok:
                        depth_b64 = base64.b64encode(buffer).decode("ascii")
                        depth_frames.append(depth_b64)
                
                # 每次生成不同的眼动数据
                eyetrack = [list(np.random.rand(8)) for _ in range(num_frames * 5)]
                
                request = {
                    "type": "inference_request",
                    "request_id": f"test-perf-{i}",
                    "data": {
                        "rgb_frames": rgb_frames,
                        "depth_frames": depth_frames,
                        "eyetrack_samples": eyetrack,
                        "elapsed_time": 5.0 + i * 0.5  # 不同的经过时间
                    }
                }
                
                start_time = time.time()
                await ws.send(json.dumps(request))
                response = json.loads(await ws.recv())
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000
                latencies.append(latency)
                
                # 获取疲劳度分数
                result = response.get("result", {})
                if result.get("status") == "success":
                    predictions = result.get("predictions", {})
                    fatigue_score = predictions.get("fatigue_score", 0)
                    fatigue_scores.append(fatigue_score)
                    print(f"   请求 {i+1}/{num_requests}: {latency:.0f}ms, 疲劳度: {fatigue_score:.2f}")
                else:
                    print(f"   请求 {i+1}/{num_requests}: {latency:.0f}ms, 失败")
        
        # 统计
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        avg_fatigue = sum(fatigue_scores) / len(fatigue_scores) if fatigue_scores else 0
        std_fatigue = np.std(fatigue_scores) if len(fatigue_scores) > 1 else 0
        
        print(f"✅ 性能测试完成")
        print(f"   平均延迟: {avg_latency:.2f}ms")
        print(f"   最小延迟: {min_latency:.2f}ms")
        print(f"   最大延迟: {max_latency:.2f}ms")
        print(f"   吞吐量: {1000 / avg_latency:.2f} 请求/秒")
        print(f"   平均疲劳度: {avg_fatigue:.2f}")
        print(f"   疲劳度标准差: {std_fatigue:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("=" * 70)
    print("疲劳度模型后端集成测试")
    print("=" * 70)
    
    uri = "ws://127.0.0.1:8767"
    print(f"\n测试目标: {uri}")
    print("请确保疲劳度模型后端已启动\n")
    
    results = []
    
    # 运行测试
    results.append(("健康检查", await test_health_check(uri)))
    results.append(("空数据推理", await test_empty_inference(uri)))
    results.append(("模拟数据推理", await test_mock_inference(uri)))
    results.append(("性能测试", await test_performance(uri, num_requests=10)))
    
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
