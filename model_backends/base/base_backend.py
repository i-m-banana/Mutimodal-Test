"""模型后端基类 - 标准化的模型服务接口

所有模型后端必须继承此基类,实现以下方法:
- initialize_model(): 加载模型
- process_inference(): 执行推理
- cleanup(): 清理资源
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

try:
    import websockets
    from websockets.server import serve, WebSocketServerProtocol
    from websockets.exceptions import ConnectionClosed
except ImportError as e:
    raise ImportError(
        "请安装 websockets 库: pip install websockets"
    ) from e


class BaseModelBackend(ABC):
    """所有模型后端的基类
    
    提供标准的WebSocket服务器和消息处理框架。
    子类只需要实现模型相关的三个抽象方法。
    
    Attributes:
        config: 配置字典
        model_type: 模型类型标识
        host: 监听地址
        port: 监听端口
        model: 模型实例 (由子类设置)
        main_backend_ws: 与主后端的WebSocket连接
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化模型后端
        
        Args:
            config: 配置字典,至少包含:
                - model_type: 模型类型 (如 'multimodal', 'emotion', 'eeg')
                - host: 监听地址 (默认 '127.0.0.1')
                - port: 监听端口 (如 8766)
        """
        self.config = config
        self.model_type = config.get("model_type", "unknown")
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 8766)
        self.logger = logging.getLogger(f"model.backend.{self.model_type}")
        self.model = None
        self.main_backend_ws: Optional[WebSocketServerProtocol] = None
        self._clients = set()
        
    @abstractmethod
    async def initialize_model(self) -> None:
        """加载和初始化AI模型
        
        在此方法中:
        1. 加载模型文件 (PyTorch .pt, TensorFlow .h5, ONNX .onnx等)
        2. 设置模型为评估模式
        3. 预热模型 (可选)
        4. 初始化预处理/后处理工具
        
        Example:
            ```python
            async def initialize_model(self):
                import torch
                self.model = torch.jit.load("model.pt")
                self.model.eval()
                self.logger.info("模型加载完成")
            ```
        
        Raises:
            Exception: 模型加载失败时抛出异常
        """
        pass
    
    @abstractmethod
    async def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行模型推理
        
        Args:
            data: 输入数据字典,可能包含:
                - rgb_frame: RGB图像 (base64编码)
                - depth_frame: 深度图像 (base64编码)
                - audio_features: 音频特征向量
                - eeg_signals: 脑电信号数据
                - metadata: 元数据 (时间戳等)
        
        Returns:
            推理结果字典,格式自定义,例如:
            ```python
            {
                "fatigue_score": 0.75,
                "attention_level": 0.82,
                "emotion": "happy",
                "confidence": 0.95
            }
            ```
        
        Raises:
            ValueError: 输入数据格式错误
            Exception: 推理过程出错
        
        Example:
            ```python
            async def process_inference(self, data):
                rgb_b64 = data.get("rgb_frame")
                # 1. 解码图像
                # 2. 预处理
                # 3. 模型推理
                # 4. 后处理
                return {"predictions": result}
            ```
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理模型和资源
        
        在此方法中:
        1. 释放模型占用的GPU/CPU内存
        2. 关闭文件句柄
        3. 清理临时文件
        
        Example:
            ```python
            async def cleanup(self):
                self.model = None
                import gc
                gc.collect()
                self.logger.info("资源已清理")
            ```
        """
        pass
    
    async def handle_client(self, websocket: WebSocketServerProtocol) -> None:
        """处理来自主后端的WebSocket连接
        
        此方法由框架自动调用,不需要子类重写。
        
        Args:
            websocket: WebSocket连接对象
        """
        self._clients.add(websocket)
        client_id = id(websocket)
        self.logger.info(f"客户端已连接 (ID: {client_id})")
        
        try:
            async for message in websocket:
                try:
                    request = json.loads(message)
                    
                    if request.get("type") == "inference_request":
                        response = await self._handle_inference_request(request)
                        await websocket.send(json.dumps(response))
                    
                    elif request.get("type") == "ping":
                        await websocket.send(json.dumps({
                            "type": "pong",
                            "timestamp": time.time()
                        }))
                    
                    elif request.get("type") == "health_check":
                        await websocket.send(json.dumps({
                            "type": "health_response",
                            "status": "healthy",
                            "model_type": self.model_type,
                            "model_loaded": self.model is not None
                        }))
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON解析失败: {e}")
                    error_response = {
                        "type": "error",
                        "error": "Invalid JSON format"
                    }
                    await websocket.send(json.dumps(error_response))
                    
                except Exception as e:
                    self.logger.error(f"处理消息失败: {e}", exc_info=True)
                    error_response = {
                        "type": "error",
                        "error": str(e)
                    }
                    try:
                        await websocket.send(json.dumps(error_response))
                    except:
                        pass
                    
        except ConnectionClosed:
            self.logger.info(f"客户端断开连接 (ID: {client_id})")
        except Exception as e:
            self.logger.error(f"连接处理异常: {e}", exc_info=True)
        finally:
            self._clients.discard(websocket)
    
    async def _handle_inference_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """处理推理请求并返回响应
        
        此方法由框架自动调用,不需要子类重写。
        
        Args:
            request: 推理请求消息
        
        Returns:
            推理响应消息
        """
        request_id = request.get("request_id", "unknown")
        
        start_time = time.time()
        try:
            # 调用子类实现的推理方法
            result = await self.process_inference(request.get("data", {}))
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                "type": "inference_response",
                "request_id": request_id,
                "model_type": self.model_type,
                "timestamp": time.time(),
                "result": {
                    "status": "success",
                    "predictions": result,
                    "latency_ms": round(latency_ms, 2)
                }
            }
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.logger.error(f"推理失败 (request_id={request_id}): {e}", exc_info=True)
            return {
                "type": "inference_response",
                "request_id": request_id,
                "model_type": self.model_type,
                "timestamp": time.time(),
                "result": {
                    "status": "error",
                    "error": str(e),
                    "latency_ms": round(latency_ms, 2)
                }
            }
    
    async def start(self) -> None:
        """启动模型后端服务器
        
        此方法由框架自动调用,不需要子类重写。
        """
        # 初始化模型
        self.logger.info(f"初始化 {self.model_type} 模型...")
        await self.initialize_model()
        
        # 启动WebSocket服务器
        self.logger.info(f"启动 {self.model_type} 模型后端: ws://{self.host}:{self.port}")
        
        # 增加max_size以支持大量图像数据 (默认1MB -> 100MB)
        async with serve(self.handle_client, self.host, self.port, max_size=100 * 1024 * 1024):
            self.logger.info(f"✅ {self.model_type} 模型后端已就绪,等待连接...")
            await asyncio.Future()  # 永久运行
    
    def run(self) -> None:
        """运行服务器 (阻塞式)
        
        这是模型后端的入口方法,在 if __name__ == "__main__" 中调用。
        
        Example:
            ```python
            if __name__ == "__main__":
                config = {
                    "model_type": "multimodal",
                    "host": "127.0.0.1",
                    "port": 8766
                }
                backend = MultimodalBackend(config)
                backend.run()
            ```
        """
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            self.logger.info("收到停止信号 (Ctrl+C)")
        except Exception as e:
            self.logger.error(f"服务器运行异常: {e}", exc_info=True)
        finally:
            self.logger.info("正在清理资源...")
            asyncio.run(self.cleanup())
            self.logger.info(f"{self.model_type} 模型后端已停止")
