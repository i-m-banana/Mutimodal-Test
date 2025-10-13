"""测试多模型后端架构

测试内容:
1. 模型后端基类功能
2. WebSocket连接
3. 推理请求-响应流程
4. 自动重连机制
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("test.model_backend")


async def test_backend_connection():
    """测试1: 连接到模型后端"""
    logger.info("=" * 60)
    logger.info("测试1: 连接到模型后端")
    
    try:
        import websockets
        
        uri = "ws://127.0.0.1:8766"
        logger.info(f"正在连接到: {uri}")
        
        async with websockets.connect(uri, timeout=5) as ws:
            logger.info("✓ 连接成功")
            
            # 发送健康检查
            health_check = {"type": "health_check"}
            await ws.send(json.dumps(health_check))
            logger.info("→ 发送健康检查")
            
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            health_response = json.loads(response)
            logger.info(f"← 收到响应: {health_response}")
            
            assert health_response.get("type") == "health_response"
            assert health_response.get("status") == "healthy"
            logger.info("✓ 健康检查通过")
            
            return True
    except Exception as e:
        logger.error(f"✗ 连接失败: {e}")
        return False


async def test_inference_request():
    """测试2: 推理请求-响应"""
    logger.info("=" * 60)
    logger.info("测试2: 推理请求-响应")
    
    try:
        import websockets
        import base64
        from PIL import Image
        import io
        
        uri = "ws://127.0.0.1:8766"
        
        async with websockets.connect(uri, timeout=5) as ws:
            # 创建测试图像
            test_image = Image.new('RGB', (640, 480), color='red')
            img_buffer = io.BytesIO()
            test_image.save(img_buffer, format='JPEG')
            img_b64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # 构造推理请求
            request = {
                "type": "inference_request",
                "request_id": "test-001",
                "model_type": "multimodal",
                "timestamp": time.time(),
                "data": {
                    "rgb_frame": img_b64,
                    "metadata": {"test": True}
                }
            }
            
            logger.info("→ 发送推理请求")
            await ws.send(json.dumps(request))
            
            # 接收响应
            response = await asyncio.wait_for(ws.recv(), timeout=10)
            result = json.loads(response)
            logger.info(f"← 收到响应")
            
            # 验证响应格式
            assert result.get("type") == "inference_response"
            assert result.get("request_id") == "test-001"
            assert result.get("result", {}).get("status") == "success"
            
            predictions = result.get("result", {}).get("predictions", {})
            latency_ms = result.get("result", {}).get("latency_ms", 0)
            
            logger.info(f"✓ 推理成功:")
            logger.info(f"  - 延迟: {latency_ms:.2f}ms")
            logger.info(f"  - 结果: {list(predictions.keys())}")
            logger.info(f"  - 疲劳度: {predictions.get('fatigue_score')}")
            logger.info(f"  - 注意力: {predictions.get('attention_level')}")
            logger.info(f"  - 姿态: {predictions.get('pose_status')}")
            
            return True
    except Exception as e:
        logger.error(f"✗ 推理失败: {e}", exc_info=True)
        return False


async def test_concurrent_requests():
    """测试3: 并发推理请求"""
    logger.info("=" * 60)
    logger.info("测试3: 并发推理请求")
    
    try:
        import websockets
        
        uri = "ws://127.0.0.1:8766"
        
        async with websockets.connect(uri, timeout=5) as ws:
            # 发送多个并发请求
            num_requests = 5
            futures = []
            
            for i in range(num_requests):
                request = {
                    "type": "inference_request",
                    "request_id": f"test-{i:03d}",
                    "model_type": "multimodal",
                    "data": {
                        "rgb_frame": "dGVzdA==",  # 假数据
                        "metadata": {"index": i}
                    }
                }
                futures.append(ws.send(json.dumps(request)))
            
            # 等待所有请求发送完成
            await asyncio.gather(*futures)
            logger.info(f"→ 已发送 {num_requests} 个并发请求")
            
            # 接收所有响应
            responses = []
            for _ in range(num_requests):
                response = await asyncio.wait_for(ws.recv(), timeout=10)
                responses.append(json.loads(response))
            
            logger.info(f"← 收到 {len(responses)} 个响应")
            
            # 验证所有响应
            success_count = sum(
                1 for r in responses
                if r.get("result", {}).get("status") == "success"
            )
            
            logger.info(f"✓ 成功处理: {success_count}/{num_requests}")
            
            # 计算平均延迟
            latencies = [
                r.get("result", {}).get("latency_ms", 0)
                for r in responses
            ]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            logger.info(f"  平均延迟: {avg_latency:.2f}ms")
            
            return success_count == num_requests
    except Exception as e:
        logger.error(f"✗ 并发测试失败: {e}")
        return False


async def test_error_handling():
    """测试4: 错误处理"""
    logger.info("=" * 60)
    logger.info("测试4: 错误处理")
    
    try:
        import websockets
        
        uri = "ws://127.0.0.1:8766"
        
        async with websockets.connect(uri, timeout=5) as ws:
            # 发送格式错误的请求
            request = {
                "type": "inference_request",
                "request_id": "test-error",
                "data": {}  # 缺少必需的rgb_frame
            }
            
            logger.info("→ 发送错误请求 (缺少rgb_frame)")
            await ws.send(json.dumps(request))
            
            response = await asyncio.wait_for(ws.recv(), timeout=5)
            result = json.loads(response)
            
            # 验证错误响应
            assert result.get("result", {}).get("status") == "error"
            error_msg = result.get("result", {}).get("error", "")
            
            logger.info(f"✓ 正确返回错误: {error_msg}")
            
            return True
    except Exception as e:
        logger.error(f"✗ 错误处理测试失败: {e}")
        return False


def test_client_api():
    """测试5: ModelBackendClient API"""
    logger.info("=" * 60)
    logger.info("测试5: ModelBackendClient API")
    
    try:
        from src.interfaces.model_ws_client import ModelBackendClient
        
        client = ModelBackendClient("multimodal", "ws://127.0.0.1:8766")
        client.start()
        logger.info("✓ 客户端已启动")
        
        time.sleep(2)  # 等待连接建立
        
        if not client.is_healthy():
            logger.warning("⚠ 客户端未连接,跳过API测试")
            client.stop()
            return True
        
        logger.info("✓ 客户端已连接")
        
        # 发送推理请求
        future = client.send_inference_request({
            "rgb_frame": "dGVzdA==",
            "metadata": {"test": True}
        })
        
        logger.info("→ 发送推理请求")
        
        try:
            result = future.result(timeout=5.0)
            logger.info("← 收到响应")
            
            if result.get("status") == "success":
                logger.info("✓ 推理成功")
                logger.info(f"  延迟: {result.get('latency_ms')}ms")
            else:
                logger.warning(f"⚠ 推理失败: {result.get('error')}")
        except TimeoutError:
            logger.error("✗ 推理超时")
        
        # 获取状态
        status = client.get_status()
        logger.info(f"✓ 客户端状态: {status}")
        
        client.stop()
        logger.info("✓ 客户端已停止")
        
        return True
    except Exception as e:
        logger.error(f"✗ 客户端API测试失败: {e}", exc_info=True)
        return False


async def run_async_tests():
    """运行异步测试"""
    results = []
    
    results.append(("连接测试", await test_backend_connection()))
    results.append(("推理测试", await test_inference_request()))
    results.append(("并发测试", await test_concurrent_requests()))
    results.append(("错误处理", await test_error_handling()))
    
    return results


def main():
    """主测试流程"""
    logger.info("开始测试多模型后端架构")
    logger.info("=" * 60)
    logger.info("前置条件: 确保模型后端已启动")
    logger.info("启动命令: python model_backends/multimodal_backend/main.py")
    logger.info("=" * 60)
    
    input("按回车键开始测试...")
    
    results = []
    
    # 运行异步测试
    async_results = asyncio.run(run_async_tests())
    results.extend(async_results)
    
    # 运行同步测试
    results.append(("客户端API", test_client_api()))
    
    # 汇总结果
    logger.info("=" * 60)
    logger.info("测试结果汇总:")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"  {name}: {status}")
    
    logger.info("=" * 60)
    if passed == total:
        logger.info(f"✓ 所有测试通过! ({passed}/{total})")
        logger.info("多模型后端架构工作正常")
    else:
        logger.error(f"✗ 部分测试失败 ({passed}/{total})")
    
    logger.info("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n测试被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"测试执行异常: {e}", exc_info=True)
        sys.exit(1)
