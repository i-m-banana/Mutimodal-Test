"""模型后端WebSocket客户端

主后端使用此客户端连接到各个独立的模型后端进程。
"""

import asyncio
import json
import logging
import threading
import uuid
from concurrent.futures import Future
from typing import Any, Dict, Optional

try:
    import websockets
    from websockets.legacy.client import WebSocketClientProtocol
    HAS_WEBSOCKETS = True
except ImportError:
    websockets = None  # type: ignore
    WebSocketClientProtocol = None  # type: ignore
    HAS_WEBSOCKETS = False


class ModelBackendClient:
    """连接到单个模型后端的WebSocket客户端
    
    功能:
    - 建立WebSocket连接
    - 发送推理请求
    - 接收推理结果
    - 自动重连
    
    Attributes:
        model_type: 模型类型 (如 'multimodal', 'emotion', 'eeg')
        url: WebSocket URL
        connected: 是否已连接
    """
    
    def __init__(self, model_type: str, url: str, *, reconnect: bool = True, reconnect_interval: float = 5.0):
        """初始化模型后端客户端
        
        Args:
            model_type: 模型类型标识
            url: WebSocket服务器地址 (如 'ws://127.0.0.1:8766')
            reconnect: 是否自动重连
            reconnect_interval: 重连间隔(秒)
        """
        if not HAS_WEBSOCKETS:
            raise ImportError("请安装 websockets: pip install websockets")
        
        self.model_type = model_type
        self.url = url
        self.reconnect = reconnect
        self.reconnect_interval = reconnect_interval
        self.logger = logging.getLogger(f"model.client.{model_type}")
        
        self.ws: Optional[Any] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self.pending_requests: Dict[str, Future] = {}
        self.connected = False
        self._stop_event = threading.Event()
        
    def start(self) -> None:
        """启动客户端线程"""
        if self.thread and self.thread.is_alive():
            self.logger.warning(f"客户端已在运行")
            return
        
        self._stop_event.clear()
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_loop,
            name=f"model-client-{self.model_type}",
            daemon=True
        )
        self.thread.start()
        self.logger.info(f"客户端线程已启动")
    
    def _run_loop(self) -> None:
        """运行asyncio事件循环"""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._connect_loop())
        except Exception as e:
            self.logger.error(f"事件循环异常: {e}", exc_info=True)
    
    async def _connect_loop(self) -> None:
        """持续重连循环"""
        while not self._stop_event.is_set():
            try:
                self.logger.info(f"正在连接到模型后端: {self.url}")
                
                async with websockets.connect(self.url) as ws:
                    self.ws = ws
                    self.connected = True
                    self.logger.info(f"✅ 已连接到 {self.model_type} 模型后端")
                    
                    # 发送健康检查
                    await self._send_health_check()
                    
                    # 接收消息循环
                    await self._receive_loop()
                    
            except Exception as e:
                self.logger.error(f"连接失败: {e}")
                self.connected = False
                self.ws = None
                
                if not self.reconnect or self._stop_event.is_set():
                    break
                
                self.logger.info(f"{self.reconnect_interval}秒后重连...")
                await asyncio.sleep(self.reconnect_interval)
    
    async def _send_health_check(self) -> None:
        """发送健康检查"""
        try:
            health_check = {
                "type": "health_check",
                "timestamp": asyncio.get_event_loop().time()
            }
            await self.ws.send(json.dumps(health_check))
        except Exception as e:
            self.logger.warning(f"健康检查失败: {e}")
    
    async def _receive_loop(self) -> None:
        """接收消息循环"""
        try:
            async for message in self.ws:
                try:
                    response = json.loads(message)
                    await self._handle_response(response)
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON解析失败: {e}")
                except Exception as e:
                    self.logger.error(f"处理响应失败: {e}", exc_info=True)
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning(f"连接断开")
            self.connected = False
        except Exception as e:
            self.logger.error(f"接收循环异常: {e}", exc_info=True)
            self.connected = False
    
    async def _handle_response(self, response: Dict[str, Any]) -> None:
        """处理来自模型后端的响应"""
        response_type = response.get("type")
        
        if response_type == "inference_response":
            request_id = response.get("request_id")
            if request_id in self.pending_requests:
                future = self.pending_requests.pop(request_id)
                try:
                    future.set_result(response.get("result"))
                except Exception as e:
                    self.logger.error(f"设置Future结果失败: {e}")
            else:
                self.logger.warning(f"收到未知请求的响应: {request_id}")
        
        elif response_type == "health_response":
            status = response.get("status")
            self.logger.debug(f"健康检查响应: {status}")
        
        elif response_type == "pong":
            pass  # 心跳响应
        
        elif response_type == "error":
            error_msg = response.get("error", "Unknown error")
            self.logger.error(f"模型后端错误: {error_msg}")
        
        else:
            self.logger.warning(f"未知响应类型: {response_type}")
    
    def send_inference_request(
        self,
        data: Dict[str, Any],
        timeout: float = 5.0
    ) -> Future:
        """发送推理请求 (非阻塞)
        
        Args:
            data: 输入数据字典
            timeout: 超时时间(秒)
        
        Returns:
            Future对象,调用 future.result(timeout) 获取结果
        
        Raises:
            Exception: 客户端未连接
        
        Example:
            ```python
            future = client.send_inference_request({
                "rgb_frame": "base64_encoded_image",
                "metadata": {"timestamp": 123}
            })
            
            # 获取结果
            try:
                result = future.result(timeout=5.0)
                print(result['predictions'])
            except TimeoutError:
                print("推理超时")
            ```
        """
        request_id = str(uuid.uuid4())
        future: Future = Future()
        
        if not self.connected or not self.loop:
            future.set_exception(Exception(f"{self.model_type} 模型后端未连接"))
            return future
        
        self.pending_requests[request_id] = future
        
        request = {
            "type": "inference_request",
            "request_id": request_id,
            "model_type": self.model_type,
            "timestamp": self.loop.time(),
            "data": data
        }
        
        try:
            asyncio.run_coroutine_threadsafe(
                self.ws.send(json.dumps(request)),
                self.loop
            )
        except Exception as e:
            self.pending_requests.pop(request_id, None)
            future.set_exception(e)
        
        # 设置超时清理
        def timeout_cleanup():
            if request_id in self.pending_requests:
                self.pending_requests.pop(request_id)
                if not future.done():
                    future.set_exception(TimeoutError(f"推理超时 ({timeout}秒)"))
        
        if self.loop:
            self.loop.call_later(timeout, timeout_cleanup)
        
        return future
    
    def stop(self) -> None:
        """停止客户端"""
        self.logger.info(f"正在停止客户端...")
        self._stop_event.set()
        
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        
        self.connected = False
        self.logger.info(f"客户端已停止")
    
    def is_healthy(self) -> bool:
        """检查客户端是否健康"""
        return self.connected and self.ws is not None and not self.ws.closed
    
    def get_status(self) -> Dict[str, Any]:
        """获取客户端状态"""
        return {
            "model_type": self.model_type,
            "url": self.url,
            "connected": self.connected,
            "pending_requests": len(self.pending_requests),
            "thread_alive": self.thread.is_alive() if self.thread else False
        }
