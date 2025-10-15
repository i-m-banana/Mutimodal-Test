"""测试统一推理后端（集成模式）

此脚本用于测试集成模式下的统一推理服务是否正常工作
直接通过主后端的WebSocket接口测试情绪和疲劳度模型
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


async def test_backend_connection(uri: str):
    """测试后端连接"""
    print(f"\n[1/5] 测试后端连接...")
    try:
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=5) as ws:
            # 先接收欢迎消息
            welcome_msg = await ws.recv()
            welcome_data = json.loads(welcome_msg)
            if welcome_data.get("type") != "welcome":
                print(f"⚠️  未收到预期的欢迎消息: {welcome_data.get('type')}")
            
            # 发送ping测试
            request = {
                "type": "ping",
                "timestamp": time.time()
            }
            await ws.send(json.dumps(request))
            
            # 循环接收消息直到收到pong
            while True:
                response = json.loads(await ws.recv())
                msg_type = response.get("type")
                
                if msg_type == "pong":
                    print(f"✅ 后端连接成功")
                    print(f"   延迟: {(time.time() - request['timestamp']) * 1000:.2f}ms")
                    return True
                elif msg_type == "event":
                    # 跳过事件消息
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
    except asyncio.TimeoutError:
        print(f"❌ 连接超时")
        return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


async def test_model_status(uri: str):
    """测试模型状态"""
    print(f"\n[2/5] 测试模型状态...")
    try:
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=5) as ws:
            # 先接收欢迎消息
            welcome_msg = await ws.recv()
            
            request = {
                "type": "get_model_status",
                "timestamp": time.time()
            }
            await ws.send(json.dumps(request))
            
            # 循环接收消息直到收到model_status
            while True:
                response = json.loads(await ws.recv())
                msg_type = response.get("type")
                
                if msg_type == "model_status":
                    models = response.get("models", {})
                    print(f"✅ 模型状态查询成功")
                    print(f"   运行中: {response.get('running', False)}")
                    print(f"   集成模型: {models.get('integrated_models', [])}")
                    print(f"   远程客户端: {models.get('remote_clients', [])}")
                    print(f"   总计: {models.get('total', 0)} 个模型")
                    return True
                elif msg_type == "event":
                    # 跳过事件消息
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
    except Exception as e:
        print(f"❌ 查询模型状态失败: {e}")
        return False


async def test_fatigue_inference(uri: str):
    """测试疲劳度推理"""
    print(f"\n[3/5] 测试疲劳度推理（模拟数据）...")
    try:
        # 生成较小的模拟数据（减少帧数和图像尺寸以避免消息过大）
        num_frames = 30  # 从75帧减少到30帧
        rgb_frames = []
        for i in range(num_frames):
            # 使用更小的图像尺寸 (112x112 而不是 224x224)
            img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
            ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # 降低质量
            if ok:
                rgb_b64 = base64.b64encode(buffer).decode("ascii")
                rgb_frames.append(rgb_b64)
        
        depth_frames = []
        for i in range(num_frames):
            depth = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
            ok, buffer = cv2.imencode(".png", depth)
            if ok:
                depth_b64 = base64.b64encode(buffer).decode("ascii")
                depth_frames.append(depth_b64)
        
        eyetrack_samples = [list(np.random.rand(8)) for _ in range(num_frames * 5)]
        
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=30) as ws:
            # 先接收欢迎消息
            welcome_msg = await ws.recv()
            
            request = {
                "type": "model_inference",
                "model_type": "fatigue",
                "request_id": "test-fatigue",
                "data": {
                    "rgb_frames": rgb_frames,
                    "depth_frames": depth_frames,
                    "eyetrack_samples": eyetrack_samples,
                    "elapsed_time": 10.0
                },
                "timestamp": time.time()
            }
            
            print(f"   发送推理请求...")
            print(f"   - RGB帧数: {len(rgb_frames)}")
            print(f"   - 深度帧数: {len(depth_frames)}")
            print(f"   - 眼动样本数: {len(eyetrack_samples)}")
            
            start_time = time.time()
            await ws.send(json.dumps(request))
            
            # 循环接收消息直到收到model_inference_result
            while True:
                response = json.loads(await ws.recv())
                msg_type = response.get("type")
                
                if msg_type == "model_inference_result":
                    latency = (time.time() - start_time) * 1000
                    result = response.get("result", {})
                    status = result.get("status")
                    
                    if status == "success":
                        predictions = result.get("predictions", {})
                        print(f"✅ 疲劳度推理成功")
                        print(f"   疲劳度分数: {predictions.get('fatigue_score'):.2f}")
                        print(f"   预测类别: {predictions.get('prediction_class')}")
                        print(f"   推理耗时: {predictions.get('inference_time_ms', 0):.2f}ms")
                        print(f"   总耗时: {latency:.2f}ms")
                        return True
                    else:
                        print(f"❌ 推理失败: {result.get('error')}")
                        return False
                elif msg_type == "event":
                    # 跳过事件消息
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
    except Exception as e:
        print(f"❌ 疲劳度推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_emotion_inference(uri: str, data_dir: Path):
    """测试情绪推理"""
    print(f"\n[4/5] 测试情绪推理（模拟数据）...")
    
    try:
        # 生成真实可用的视频和音频数据
        def create_real_video():
            """创建真实的视频文件并返回base64"""
            import tempfile
            import os
            
            # 创建临时视频文件
            with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # 使用 OpenCV 创建真实的视频（更小的尺寸和帧数）
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(tmp_path, fourcc, 10.0, (112, 112))  # 更小的分辨率
                
                # 只写入5帧（减少文件大小）
                for _ in range(5):
                    frame = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                    out.write(frame)
                
                out.release()
                
                # 读取并编码为base64
                with open(tmp_path, 'rb') as f:
                    video_data = f.read()
                
                return base64.b64encode(video_data).decode('utf-8')
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        
        def create_real_audio():
            """创建真实的音频文件并返回base64"""
            import tempfile
            import os
            import wave
            import struct
            
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # 创建真实的WAV文件（更短的时长）
                sample_rate = 16000
                duration = 1  # 只1秒
                num_samples = sample_rate * duration
                
                with wave.open(tmp_path, 'w') as wav_file:
                    wav_file.setnchannels(1)  # 单声道
                    wav_file.setsampwidth(2)  # 16位
                    wav_file.setframerate(sample_rate)
                    
                    # 生成简单的正弦波
                    for i in range(num_samples):
                        value = int(32767.0 * 0.3 * np.sin(2 * np.pi * 440 * i / sample_rate))
                        data = struct.pack('<h', value)
                        wav_file.writeframes(data)
                
                # 读取并编码为base64
                with open(tmp_path, 'rb') as f:
                    audio_data = f.read()
                
                return base64.b64encode(audio_data).decode('utf-8')
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
        
        # 准备3个模拟样本
        samples = []
        test_texts = [
            "我今天感觉很好，心情愉快",
            "工作压力有点大，需要休息",
            "这个项目很有挑战性"
        ]
        
        print(f"   正在生成真实视频和音频数据（较小尺寸）...")
        for i in range(3):
            video_b64 = create_real_video()
            audio_b64 = create_real_audio()
            text = test_texts[i]
            
            samples.append({
                "video_b64": video_b64,
                "audio_b64": audio_b64,
                "text": text,
                "question_index": i + 1
            })
            
            print(f"   样本 {i+1}: 视频={len(video_b64)/1024:.1f}KB, 音频={len(audio_b64)/1024:.1f}KB, 文本=\"{text}\"")
        
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=60) as ws:
            # 先接收欢迎消息
            welcome_msg = await ws.recv()
            
            request = {
                "type": "model_inference",
                "model_type": "emotion",
                "request_id": "test-emotion",
                "data": {
                    "samples": samples
                },
                "timestamp": time.time()
            }
            
            print(f"   发送推理请求 (共 {len(samples)} 个样本)...")
            start_time = time.time()
            await ws.send(json.dumps(request))
            
            # 循环接收消息直到收到model_inference_result
            while True:
                response = json.loads(await ws.recv())
                msg_type = response.get("type")
                
                if msg_type == "model_inference_result":
                    latency = (time.time() - start_time) * 1000
                    result = response.get("result", {})
                    
                    if result.get("status") == "success":
                        predictions = result.get("predictions", {})
                        print(f"✅ 情绪推理成功")
                        print(f"   情绪分数: {predictions.get('emotion_score', 0):.2f}")
                        print(f"   推理耗时: {predictions.get('inference_time_ms', 0):.2f}ms")
                        print(f"   总耗时: {latency:.2f}ms")
                        return True
                    else:
                        print(f"❌ 推理失败: {result.get('error')}")
                        return False
                elif msg_type == "event":
                    # 跳过事件消息
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
                
    except Exception as e:
        print(f"❌ 情绪推理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_concurrent_inference(uri: str):
    """测试并发推理"""
    print(f"\n[5/5] 测试并发推理（疲劳度模型）...")
    try:
        num_requests = 5
        
        async def single_request(idx: int):
            """单个请求"""
            # 生成较小的模拟数据
            num_frames = 30  # 减少帧数
            rgb_frames = []
            for i in range(num_frames):
                img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    rgb_b64 = base64.b64encode(buffer).decode("ascii")
                    rgb_frames.append(rgb_b64)
            
            depth_frames = []
            for i in range(num_frames):
                depth = np.random.randint(0, 255, (112, 112), dtype=np.uint8)
                ok, buffer = cv2.imencode(".png", depth)
                if ok:
                    depth_b64 = base64.b64encode(buffer).decode("ascii")
                    depth_frames.append(depth_b64)
            
            eyetrack_samples = [list(np.random.rand(8)) for _ in range(num_frames * 5)]
            
            async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=30) as ws:
                # 先接收欢迎消息
                welcome_msg = await ws.recv()
                
                request = {
                    "type": "model_inference",
                    "model_type": "fatigue",
                    "request_id": f"test-concurrent-{idx}",
                    "data": {
                        "rgb_frames": rgb_frames,
                        "depth_frames": depth_frames,
                        "eyetrack_samples": eyetrack_samples,
                        "elapsed_time": 5.0 + idx * 0.5
                    },
                    "timestamp": time.time()
                }
                
                start_time = time.time()
                await ws.send(json.dumps(request))
                
                # 循环接收消息直到收到model_inference_result
                while True:
                    response = json.loads(await ws.recv())
                    msg_type = response.get("type")
                    
                    if msg_type == "model_inference_result":
                        latency = (time.time() - start_time) * 1000
                        result = response.get("result", {})
                        
                        if result.get("status") == "success":
                            predictions = result.get("predictions", {})
                            fatigue_score = predictions.get("fatigue_score", 0)
                            return (idx, latency, fatigue_score, True)
                        else:
                            return (idx, latency, 0, False)
                    elif msg_type == "event":
                        # 跳过事件消息
                        continue
                    else:
                        return (idx, latency, 0, False)
        
        print(f"   发送 {num_requests} 个并发请求...")
        tasks = [single_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        successful = []
        failed = []
        for result in results:
            if isinstance(result, Exception):
                failed.append(result)
            elif result[3]:  # success
                successful.append(result)
            else:
                failed.append(result)
        
        print(f"✅ 并发测试完成")
        print(f"   成功: {len(successful)}/{num_requests}")
        print(f"   失败: {len(failed)}/{num_requests}")
        
        if successful:
            latencies = [r[1] for r in successful]
            scores = [r[2] for r in successful]
            print(f"   平均延迟: {sum(latencies)/len(latencies):.2f}ms")
            print(f"   最小延迟: {min(latencies):.2f}ms")
            print(f"   最大延迟: {max(latencies):.2f}ms")
            print(f"   平均疲劳度: {sum(scores)/len(scores):.2f}")
        
        return len(successful) > 0
        
    except Exception as e:
        print(f"❌ 并发测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """主测试函数"""
    print("=" * 70)
    print("统一推理后端集成测试（集成模式）")
    print("=" * 70)
    
    uri = "ws://127.0.0.1:8765"
    print(f"\n测试目标: {uri}")
    print("请确保主后端已启动 (python -m src.main)\n")
    
    results = []
    
    # 运行测试（不再需要真实数据目录）
    results.append(("后端连接", await test_backend_connection(uri)))
    results.append(("模型状态", await test_model_status(uri)))
    results.append(("疲劳度推理", await test_fatigue_inference(uri)))
    results.append(("情绪推理", await test_emotion_inference(uri, None)))  # 使用模拟数据，不需要路径
    results.append(("并发推理", await test_concurrent_inference(uri)))
    
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
