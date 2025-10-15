"""使用文件路径测试统一推理后端（文件模式）

此脚本通过发送文件路径而不是base64数据来测试后端
大大减少了网络传输数据量
"""

import asyncio
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

# 真实数据路径（使用最完整的数据集 - 包含3种模态）
REAL_DATA_DIR = Path(__file__).parent / "recordings" / "admin" / "20251014_221344"
FATIGUE_DIR = REAL_DATA_DIR / "fatigue"
EMOTION_DIR = REAL_DATA_DIR / "emotion"
EEG_DIR = REAL_DATA_DIR / "eeg"


def load_emotion_record(record_path):
    """加载情绪录制记录文件，获取问题和识别文本
    
    Returns:
        List[Dict]: 包含 question_index, question_text, recognized_text 等信息
    """
    import ast
    
    if not Path(record_path).exists():
        return []
    
    try:
        with open(record_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # record.txt 是 Python 列表格式
            records = ast.literal_eval(content)
            return records
    except Exception as e:
        print(f"⚠️  无法解析 record.txt: {e}")
        return []


async def test_backend_connection(uri: str):
    """测试后端连接"""
    print(f"\n[1/4] 测试后端连接...")
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
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False


async def test_model_status(uri: str):
    """测试模型状态"""
    print(f"\n[2/4] 测试模型状态...")
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
                    running = response.get("running", False)
                    models = response.get("models", {})
                    integrated_models = models.get("integrated_models", [])
                    
                    print(f"✅ 模型状态获取成功")
                    print(f"   运行中: {running}")
                    print(f"   集成模型: {integrated_models}")
                    return True
                elif msg_type == "event":
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
                    
    except Exception as e:
        print(f"❌ 获取模型状态失败: {e}")
        return False


async def test_fatigue_file_mode(uri: str, segment: int = 1):
    """测试疲劳度推理（文件路径模式，支持分段）
    
    Args:
        uri: WebSocket URI
        segment: 数据段编号（1 或 2），对应 rgb1/2.avi, depth1/2.avi, eyetrack1/2.json
    """
    print(f"\n[疲劳度 - 第{segment}段] 测试疲劳度推理（文件路径模式）...")
    
    # 使用绝对路径，根据段号选择文件
    rgb_path = str(FATIGUE_DIR / f"rgb{segment}.avi")
    depth_path = str(FATIGUE_DIR / f"depth{segment}.avi")
    eyetrack_path = str(FATIGUE_DIR / f"eyetrack{segment}.json")
    
    if not Path(rgb_path).exists():
        print(f"❌ RGB视频文件不存在: {rgb_path}")
        return False
    
    try:
        print(f"   发送文件路径...")
        print(f"   RGB: {rgb_path}")
        print(f"   Depth: {depth_path}")
        print(f"   Eyetrack: {eyetrack_path}")
        
        # 构建请求（文件模式）
        request = {
            "type": "model_inference",
            "model_type": "fatigue",  # 修正：使用 model_type 而不是 model
            "data": {
                "file_mode": True,
                "rgb_video_path": rgb_path,
                "depth_video_path": depth_path,
                "eyetrack_json_path": eyetrack_path,
                "max_frames": 30  # 读取30帧
            },
            "timestamp": time.time()
        }
        
        msg_json = json.dumps(request)
        print(f"   请求消息大小: {len(msg_json) / 1024:.2f} KB (仅路径信息)")
        
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=10) as ws:
            # 先接收欢迎消息
            welcome_msg = await ws.recv()
            
            start_time = time.time()
            await ws.send(msg_json)
            
            # 循环接收消息直到收到model_inference_result
            while True:
                response = json.loads(await ws.recv())
                msg_type = response.get("type")
                
                if msg_type == "model_inference_result":
                    latency = (time.time() - start_time) * 1000
                    result = response.get("result", {})
                    
                    if result.get("status") == "success":
                        predictions = result.get("predictions", {})
                        print(f"✅ 疲劳度推理成功")
                        print(f"   延迟: {latency:.2f}ms")
                        print(f"   疲劳度分数: {predictions.get('fatigue_score', 'N/A')}")
                        print(f"   疲劳度等级: {predictions.get('fatigue_class', 'N/A')}")
                        print(f"   RGB帧数: {predictions.get('num_rgb_frames', 'N/A')}")
                        print(f"   深度帧数: {predictions.get('num_depth_frames', 'N/A')}")
                        print(f"   眼动样本: {predictions.get('num_eyetrack_samples', 'N/A')}")
                        return True
                    else:
                        error_msg = result.get("error", "未知错误")
                        print(f"❌ 推理失败: {error_msg}")
                        return False
                elif msg_type == "event":
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
                    
    except Exception as e:
        print(f"❌ 疲劳度推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_emotion_file_mode(uri: str):
    """测试情绪推理（文件路径模式，使用真实文本）"""
    print(f"\n[4/4] 测试情绪推理（文件路径模式 + 真实文本）...")
    
    # 加载 record.txt 获取识别的文本
    record_path = EMOTION_DIR / "record.txt"
    records = load_emotion_record(record_path)
    
    if not records:
        print(f"⚠️  未找到 record.txt，使用空文本")
        test_index = 1
        text = ""
    else:
        # 使用第一个记录
        test_index = records[0].get('question_index', 1)
        text = records[0].get('recognized_text', '')
        question = records[0].get('question_text', '')
        print(f"   使用第 {test_index} 个样本")
        print(f"   问题: {question}")
        print(f"   识别文本: {text}")
    
    # 使用绝对路径
    video_path = str(EMOTION_DIR / f"{test_index}.avi")
    audio_path = str(EMOTION_DIR / f"{test_index}.wav")
    
    if not Path(video_path).exists():
        print(f"❌ 视频文件不存在: {video_path}")
        return False
    
    try:
        print(f"   发送文件路径...")
        print(f"   Video: {video_path}")
        print(f"   Audio: {audio_path}")
        
        # 构建请求（文件模式）
        request = {
            "type": "model_inference",
            "model_type": "emotion",  # 修正：使用 model_type 而不是 model
            "data": {
                "file_mode": True,
                "video_path": video_path,
                "audio_path": audio_path,
                "text": text  # 使用真实识别文本
            },
            "timestamp": time.time()
        }
        
        msg_json = json.dumps(request)
        print(f"   请求消息大小: {len(msg_json) / 1024:.2f} KB (路径 + 文本)")
        
        async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=30) as ws:
            # 先接收欢迎消息
            welcome_msg = await ws.recv()
            
            start_time = time.time()
            await ws.send(msg_json)
            
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
                        print(f"   延迟: {latency:.2f}ms")
                        print(f"   情绪分数: {predictions.get('emotion_score', 'N/A')}")
                        print(f"   预测: {predictions.get('prediction', 'N/A')}")
                        print(f"   概率: {predictions.get('probabilities', 'N/A')}")
                        print(f"   输入文本: {predictions.get('text_input', 'N/A')}")
                        return True
                    else:
                        error_msg = result.get("error", "未知错误")
                        print(f"❌ 推理失败: {error_msg}")
                        return False
                elif msg_type == "event":
                    continue
                else:
                    print(f"❌ 未知响应类型: {msg_type}")
                    return False
                    
    except Exception as e:
        print(f"❌ 情绪推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_emotion_file_mode_all(uri: str):
    """测试所有情绪问答（文件路径模式，使用真实识别文本）"""
    print(f"\n[情绪 - 全部问答] 测试所有情绪问答...")
    
    # 加载 record.txt
    record_path = EMOTION_DIR / "record.txt"
    records = load_emotion_record(record_path)
    
    if not records:
        print("❌ 无法加载情绪记录")
        return False
    
    total = len(records)
    success_count = 0
    
    for i, record in enumerate(records, 1):
        question_index = record.get('question_index', i)
        question_text = record.get('question_text', '')
        recognized_text = record.get('recognized_text', '')
        
        # 构建文件路径
        video_path = str(EMOTION_DIR / f"{question_index}.avi")
        audio_path = str(EMOTION_DIR / f"{question_index}.wav")
        
        if not Path(video_path).exists():
            print(f"   [{i}/{total}] ❌ 视频文件不存在: {video_path}")
            continue
        
        try:
            print(f"\n   [{i}/{total}] 问题 {question_index}: {question_text}")
            print(f"   识别文本: {recognized_text}")
            
            # 构建请求
            request = {
                "type": "model_inference",
                "model_type": "emotion",
                "data": {
                    "file_mode": True,
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "text": recognized_text
                },
                "timestamp": time.time()
            }
            
            async with websockets.connect(uri, max_size=100 * 1024 * 1024, open_timeout=30) as ws:
                # 接收欢迎消息
                await ws.recv()
                
                start_time = time.time()
                await ws.send(json.dumps(request))
                
                # 接收推理结果
                while True:
                    response = json.loads(await ws.recv())
                    msg_type = response.get("type")
                    
                    if msg_type == "model_inference_result":
                        latency = (time.time() - start_time) * 1000
                        result = response.get("result", {})
                        
                        if result.get("status") == "success":
                            predictions = result.get("predictions", {})
                            emotion_score = predictions.get('emotion_score', 'N/A')
                            print(f"   ✅ 情绪分数: {emotion_score} | 延迟: {latency:.0f}ms")
                            success_count += 1
                            break
                        else:
                            error_msg = result.get("error", "未知错误")
                            print(f"   ❌ 推理失败: {error_msg}")
                            break
                    elif msg_type == "event":
                        continue
                    else:
                        print(f"   ❌ 未知响应类型: {msg_type}")
                        break
                        
        except Exception as e:
            print(f"   ❌ 推理异常: {str(e)}")
    
    print(f"\n   总计: {success_count}/{total} 成功")
    return success_count == total


async def main():
    """主测试函数"""
    print("=" * 70)
    print("统一推理后端测试（文件路径模式）")
    print("=" * 70)
    
    # 检查数据目录
    if not REAL_DATA_DIR.exists():
        print(f"\n❌ 错误: 数据目录不存在: {REAL_DATA_DIR}")
        print("请确保已录制数据并保存在正确位置")
        return 1
    
    print(f"\n数据目录: {REAL_DATA_DIR}")
    print(f"   Fatigue 数据: {FATIGUE_DIR}")
    print(f"   Emotion 数据: {EMOTION_DIR}")
    
    uri = "ws://127.0.0.1:8765"
    print(f"\n测试目标: {uri}")
    print("请确保主后端已启动 (python -m src.main)\n")
    
    results = []
    
    # 运行测试（模拟实时检测流程）
    results.append(("后端连接", await test_backend_connection(uri)))
    results.append(("模型状态", await test_model_status(uri)))
    
    # 第一段疲劳度检测
    results.append(("疲劳度推理 - 第1段", await test_fatigue_file_mode(uri, segment=1)))
    
    # 情绪检测（所有问答）
    results.append(("情绪推理 - 全部问答", await test_emotion_file_mode_all(uri)))
    
    # 第二段疲劳度检测
    results.append(("疲劳度推理 - 第2段", await test_fatigue_file_mode(uri, segment=2)))
    
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
        print("\n💡 优势:")
        print("   - 消息大小: 仅 1-2 KB (vs 0.77-20 MB)")
        print("   - 无需编码/解码: 节省CPU和时间")
        print("   - 本地文件访问: 更快更可靠")
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
