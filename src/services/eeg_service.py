"""Backend EEG service - owns the BLE hardware connection and persistence.

This module exposes a websocket-friendly façade that the UI can call.  All
device IO happens here so the front-end stays simulator-only and the backend
process manages real EEG headsets.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None

from ..constants import EventTopic
from ..core.event_bus import Event, EventBus
from ..core.thread_pool import get_thread_pool
from ..devices import BleEEGDevice, HAS_EEG_HARDWARE, DeviceException


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


FORCE_SIMULATION = _env_flag("BACKEND_EEG_SIMULATION") or _env_flag("UI_EEG_SIMULATION")


class EEGService:
    """后端EEG服务 - 统一管理EEG硬件连接和数据采集生命周期"""

    def __init__(self, bus: EventBus, logger: logging.Logger) -> None:
        self.bus = bus
        self.logger = logger
        
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._recorder: Optional[EEGRecorder] = None
        self._running = False
        self._save_dir: Optional[str] = None

        self._thread_pool = get_thread_pool()
        self._inference_thread_name = "eeg-inference-poller"
        self._inference_stop = threading.Event()
        self._inference_active = False
        interval_env = os.getenv("BACKEND_EEG_POLL_INTERVAL")
        self._inference_interval = 3.0
        if interval_env:
            try:
                self._inference_interval = max(0.5, float(interval_env))
            except ValueError:
                self.logger.warning(
                    "Invalid BACKEND_EEG_POLL_INTERVAL '%s', fallback to 3.0s",
                    interval_env,
                )

    def _ensure_loop_thread(self) -> None:
        """确保事件循环线程运行（后端在独立线程维护异步BLE连接）。"""
        if self._thread and self._thread.is_alive():
            return

        def _run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=_run_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def start_recording(self, save_dir: str) -> Dict[str, Any]:
        """启动EEG采集"""
        if self._running:
            self._ensure_inference_polling()
            return {"status": "already-running", "save_dir": self._save_dir}

        self._ensure_loop_thread()
        self._save_dir = os.path.join(os.path.abspath(save_dir), 'eeg')
        os.makedirs(self._save_dir, exist_ok=True)

        def _create_and_start():
            self._recorder = EEGRecorder(
                save_dir=self._save_dir,
                bus=self.bus,
                logger=self.logger
            )
            self._running = True
            return self._recorder.start()

        fut = asyncio.run_coroutine_threadsafe(_create_and_start(), self._loop)
        try:
            fut.result(timeout=0.5)
            self.logger.info("EEG采集已启动，保存目录: %s", self._save_dir)
            self._ensure_inference_polling()
            return {"status": "started", "save_dir": self._save_dir}
        except Exception as exc:
            self.logger.error("EEG采集启动失败: %s", exc)
            return {"status": "error", "error": str(exc)}

    def stop_recording(self) -> Dict[str, Any]:
        """停止EEG采集"""
        if not self._running or self._recorder is None:
            return {"status": "not-running"}

        async def _stop():
            try:
                await self._recorder.stop()
            except Exception as exc:
                self.logger.error("停止EEG采集失败: %s", exc)

        fut = asyncio.run_coroutine_threadsafe(_stop(), self._loop)
        try:
            fut.result(timeout=5.0)
            self._running = False
            self._stop_inference_polling()
            file_paths = self.get_file_paths()
            self.logger.info("EEG采集已停止")
            return {"status": "stopped", "file_paths": file_paths}
        except Exception as exc:
            self.logger.error("停止EEG采集超时: %s", exc)
            return {"status": "error", "error": str(exc)}

    def _ensure_inference_polling(self) -> None:
        """启动后台线程，定期触发脑负荷推理请求。"""
        if self.bus is None or not self._running:
            return
        if self._inference_active:
            return

        existing = self._thread_pool.get_managed_thread(self._inference_thread_name)
        if existing and existing.is_alive():
            self._inference_active = True
            return

        self._inference_stop.clear()
        thread = self._thread_pool.register_managed_thread(
            self._inference_thread_name,
            self._inference_polling_loop,
            daemon=True,
        )
        self._inference_active = True
        thread.start()
        self.logger.info("✅ EEG推理轮询已启动 (间隔=%.2fs)", self._inference_interval)

    def _stop_inference_polling(self) -> None:
        """停止脑负荷推理轮询。"""
        if not self._inference_active:
            return

        self._inference_stop.set()
        try:
            self._thread_pool.unregister_managed_thread(
                self._inference_thread_name,
                timeout=2.0,
            )
        except Exception as exc:  # pragma: no cover - best effort cleanup
            self.logger.debug("停止EEG推理轮询失败: %s", exc)
        finally:
            self._inference_active = False
            self._inference_stop.clear()
            self.logger.info("⏹️ EEG推理轮询已停止")

    def _inference_polling_loop(self) -> None:
        self.logger.debug("EEG推理轮询线程已启动")
        try:
            while not self._inference_stop.is_set():
                if self._running:
                    self._thread_pool.submit_io_task(self._publish_inference_request)
                interval = self._inference_interval
                if interval <= 0:
                    interval = 1.0
                if self._inference_stop.wait(interval):
                    break
        finally:
            self._inference_active = False
            self.logger.debug("EEG推理轮询线程已停止")

    def _publish_inference_request(self) -> None:
        if not self._running or self.bus is None:
            return

        try:
            window_data = self.get_recent_window(seconds=2.0, sample_rate=500.0)
            ch1_data = window_data.get("ch1", [])
            ch2_data = window_data.get("ch2", [])
            sample_count = len(ch1_data)

            if sample_count < 500:
                return

            if not ch1_data or not ch2_data:
                return

            if len(ch1_data) != len(ch2_data):
                self.logger.warning(
                    "⚠️ EEG通道长度不匹配 %d≠%d",
                    len(ch1_data),
                    len(ch2_data),
                )
                return

            if np is not None:
                eeg_signal = np.column_stack([ch1_data, ch2_data]).tolist()
            else:
                eeg_signal = [[ch1_data[i], ch2_data[i]] for i in range(sample_count)]

            simulation_mode = bool(self._recorder.simulation_enabled) if self._recorder else bool(self.simulation_enabled)

            payload = {
                "mode": "memory",
                "eeg_signal": eeg_signal,
                "sample_count": sample_count,
                "timestamp": datetime.utcnow().isoformat(),
                "simulation_mode": simulation_mode,
            }

            self.bus.publish(Event(EventTopic.EEG_REQUEST, payload))
            self.logger.debug("📤EEG %d样本", sample_count)

        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug("EEG推理派发失败: %s", exc)

    def get_file_paths(self) -> Dict[str, str]:
        """获取采集的文件路径"""
        if not self._save_dir:
            return {}

        folder = os.path.abspath(self._save_dir)

        def latest(pattern):
            files = sorted(glob.glob(os.path.join(folder, pattern)))
            return files[-1] if files else ''

        return {
            'ch1_npy': latest('channel1_data_*.npy'),
            'ch2_npy': latest('channel2_data_*.npy'),
            'ch1_txt': latest('channel1_data_*.txt'),
            'ch2_txt': latest('channel2_data_*.txt'),
            'csv': latest('eeg_data_*.csv'),
            'metadata': latest('metadata_*.json'),
            'folder': folder,
        }

    def get_snapshot(self) -> Dict[str, Any]:
        """获取当前状态快照"""
        if not self._running or self._recorder is None:
            return {
                "status": "idle",
                "latest_sample": None,
                "recent_window": None,
            }

        return {
            "status": "running",
            "save_dir": self._save_dir,
            "latest_sample": self._recorder.get_latest_sample(),
            "recent_window": self._recorder.get_recent_window(),
        }

    def get_recent_window(self, seconds: float = 5.0, sample_rate: float = 500.0) -> Dict[str, Any]:
        """获取最近的时间窗口数据（代理到recorder）
        
        Args:
            seconds: 窗口时长（秒）
            sample_rate: 采样率（Hz）
        
        Returns:
            包含ch1、ch2、timestamps等数据的字典
        """
        if not self._running or self._recorder is None:
            return {'timestamps': [], 'ch1': [], 'ch2': [], 'ads_event': []}
        
        return self._recorder.get_recent_window(seconds, sample_rate)

    def diagnostics(self) -> Dict[str, Any]:
        """返回当前EEG模块的硬件可用性和设备连接状态。"""
        device_connected = False
        simulation_mode = FORCE_SIMULATION or not HAS_EEG_HARDWARE

        if self._recorder:
            # 读取 recorder 的真实运行模式
            simulation_mode = bool(self._recorder.simulation_enabled)

            if self._recorder.simulation_enabled:
                # 模拟模式下视为已连接，只要采集线程处于运行状态
                device_connected = self._running
            elif self._recorder.device:
                # 真实设备模式下，检查BLE连接状态
                try:
                    device_connected = bool(self._recorder.device.is_connected)
                except AttributeError:
                    # 兼容旧版本，没有 is_connected 属性
                    device_connected = self._running

        return {
            "hardware_driver_available": HAS_EEG_HARDWARE,
            "force_simulation": FORCE_SIMULATION,
            "simulation_mode": simulation_mode,
            "running": self._running,
            "device_connected": device_connected,
        }


class EEGRecorder:
    """EEG数据采集器，封装硬件读写与模拟模式切换"""

    def __init__(self, save_dir: str, bus: EventBus, logger: logging.Logger) -> None:
        self.save_dir = save_dir
        self.bus = bus
        self.logger = logger
        
        self.device: Optional[BleEEGDevice] = None
        self.running = False
        # 没有BLE驱动或显式开启模拟标志时走模拟路径，避免前端误触硬件
        self.simulation_enabled = FORCE_SIMULATION or not HAS_EEG_HARDWARE

        # 数据存储 - 在内存中累积所有数据
        self.eeg_ch1_data = []
        self.eeg_ch2_data = []
        self.timestamps = []
        self.sequence_numbers = []
        self.ads_events = []
        self.data_lock = threading.Lock()

        # 统计信息
        self.packets_received = 0
        self.sample_count = 0
        self.lost_packets = 0
        self.last_sequence = None
        
        self.save_interval_samples = 300000  # 每10分钟输出一次统计日志
        
        # 调试模式
        self.debug_mode = False

    async def start(self) -> None:
        """启动采集"""
        self.running = True
        
        if self.simulation_enabled:
            self.logger.info("🧪 EEG模块将在模拟模式下运行")
            asyncio.create_task(self._simulation_loop())
        else:
            try:
                # 后端实例化BLE封装器，确保硬件仅在服务端被访问
                self.device = BleEEGDevice()
            except DeviceException as exc:
                self.logger.error("❌ EEG设备驱动不可用: %s，切换到模拟模式", exc)
                self.simulation_enabled = True
                asyncio.create_task(self._simulation_loop())
                return

            # 🔥 连接重试机制 - 最多5次
            max_retries = 5
            retry_delay = 3.0
            
            for attempt in range(max_retries):
                try:
                    self.logger.info("🔗 正在连接EEG硬件 (BLE) -> %s (尝试 %d/%d)", 
                                   self.device.address, attempt + 1, max_retries)
                    await self.device.connect(self._on_data_received)
                    self.logger.info("✅ EEG设备连接成功")
                    # 发送启动命令
                    await self._send_start_command()
                    return
                except Exception as exc:
                    self.logger.warning("⚠️  EEG设备连接失败 (尝试 %d/%d): %s", 
                                      attempt + 1, max_retries, exc)
                    try:
                        await self.device.ensure_disconnected()
                    except:
                        pass
                    
                    if attempt < max_retries - 1:
                        self.logger.info("⏳ %d秒后重试连接...", retry_delay)
                        await asyncio.sleep(retry_delay)
                    else:
                        self.logger.error("❌ EEG设备连接失败，已达到最大重试次数")
                        self.logger.error("💡 提示: 请检查设备是否开机、蓝牙是否开启、设备地址是否正确")
                        self.logger.error("💡 如需使用模拟数据，请设置环境变量: set BACKEND_EEG_SIMULATION=1")
                        self.simulation_enabled = True
                        asyncio.create_task(self._simulation_loop())
                        try:
                            await self.device.ensure_disconnected()
                        finally:
                            self.device = None
                        return

    async def stop(self) -> None:
        """停止采集"""
        self.running = False
        
        if self.device and not self.simulation_enabled:
            try:
                # 发送停止命令
                await self._send_stop_command()
                await asyncio.sleep(0.5)
                await self.device.ensure_disconnected()
                self.logger.info("EEG设备已断开")
            except Exception as exc:
                self.logger.error("断开EEG设备失败: %s", exc)
            finally:
                self.device = None

        # 保存数据
        self._save_data()
    
    async def _send_start_command(self) -> None:
        """发送启动记录命令"""
        if not self.device:
            return
        try:
            # 命令格式: 48 4E 55 9F 03 00 00 00
            command = bytes([0x48, 0x4E, 0x55, 0x9F, 0x03, 0x00, 0x00, 0x00])
            await self.device.write(command)
            self.logger.info("▶️ 已发送启动记录命令")
        except Exception as exc:
            self.logger.warning("⚠️ 启动记录命令发送失败: %s", exc)
    
    async def _send_stop_command(self) -> None:
        """发送停止记录命令"""
        if not self.device:
            return
        try:
            # 命令格式: 48 4E 55 9F 04 00 00 00
            command = bytes([0x48, 0x4E, 0x55, 0x9F, 0x04, 0x00, 0x00, 0x00])
            await self.device.write(command)
            self.logger.info("⏹️ 已发送停止记录命令")
        except Exception as exc:
            self.logger.warning("⚠️ 停止记录命令发送失败: %s", exc)

    def _on_data_received(self, sender, data: bytearray) -> None:
        """BLE数据回调"""
        try:
            parsed = self._parse_eeg_packet(data)
            if parsed and 'samples' in parsed:
                now = time.time()
                timestamp_str = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                with self.data_lock:
                    for sample in parsed['samples']:
                        ch1_val = sample['ch1']
                        ch2_val = sample['ch2']
                        ads_event = sample.get('ads_event', 0)
                        
                        # 添加到数据列表
                        self.eeg_ch1_data.append(ch1_val)
                        self.eeg_ch2_data.append(ch2_val)
                        self.ads_events.append(ads_event)
                        self.timestamps.append(now)
                        self.sample_count += 1
                    
                    self.sequence_numbers.append(parsed['sequence'])
                    self.packets_received += 1
                    self.lost_packets += parsed.get('packet_loss', 0)

                # 定期输出统计
                if self.packets_received % 1500 == 0:
                    total_pkts = self.packets_received + self.lost_packets
                    loss_rate = (self.lost_packets / total_pkts) * 100 if total_pkts > 0 else 0
                    self.logger.info(
                        "📊 EEG统计: 收到%d包, 丢失%d包, 丢包率%.2f%%, 样本=%d",
                        self.packets_received, self.lost_packets, loss_rate, self.sample_count
                    )

                # 发送事件到UI（取第一个样本作为代表）
                if parsed['samples']:
                    sample = parsed['samples'][0]
                    self.bus.publish(Event(
                        topic=EventTopic.DETECTOR_RESULT,
                        payload={
                            "detector": "eeg",
                            "sample": {
                                "timestamp": now,
                                "ch1": sample['ch1'],
                                "ch2": sample['ch2'],
                            }
                        }
                    ))
        except Exception as exc:
            self.logger.debug("解析EEG数据失败: %s", exc)

    def _parse_eeg_packet(self, data: bytearray) -> Optional[Dict[str, Any]]:
        """解析EEG数据包（完整版本，支持PSG格式）"""
        if len(data) != 187:
            return None

        try:
            # 提取179字节的PSG数据部分 (跳过前8字节的命令头)
            psg_data = data[8:187]
            
            # 解析3字节序列号
            seq_high = psg_data[0]
            seq_mid = psg_data[1]
            seq_low = psg_data[2]
            sequence_number = (seq_high << 16) + (seq_mid << 8) + seq_low

            # 检测丢包
            packet_loss_count = 0
            if self.last_sequence is not None:
                expected_seq = (self.last_sequence + 1) % 16777216
                if sequence_number != expected_seq:
                    if sequence_number > expected_seq:
                        packet_loss_count = sequence_number - expected_seq
                    else:
                        packet_loss_count = (16777216 - expected_seq) + sequence_number

                    if packet_loss_count > 100:
                        self.logger.warning(
                            "⚠️  序列号异常跳跃: %d -> %d", 
                            self.last_sequence, sequence_number
                        )
                        packet_loss_count = min(packet_loss_count, 10)

            self.last_sequence = sequence_number

            # 解析脑电数据（10个样本）
            # 跳过序列号(3字节) + 电池(2字节) + 音频(60字节) = 65字节
            offset = 65
            samples = []

            if offset + 70 <= len(psg_data):
                for i in range(10):
                    group_offset = offset + i * 7
                    if group_offset + 7 <= len(psg_data):
                        # 通道1 (3字节)
                        ch1_hex = psg_data[group_offset:group_offset + 3].hex()
                        ch1_value = self._hex_to_microvolt(ch1_hex)

                        # 通道2 (3字节)
                        ch2_hex = psg_data[group_offset + 3:group_offset + 6].hex()
                        ch2_value = self._hex_to_microvolt(ch2_hex)

                        # ADS事件 (1字节)
                        ads_event = psg_data[group_offset + 6]

                        samples.append({
                            'ch1': ch1_value,
                            'ch2': ch2_value,
                            'ads_event': ads_event
                        })

            return {
                'sequence': sequence_number,
                'packet_loss': packet_loss_count,
                'samples': samples,
            }
        except Exception as exc:
            self.logger.debug("数据包解析错误: %s", exc)
            return None

    def _hex_to_microvolt(self, hex3: str, scale: float = 0.0447) -> float:
        """将16进制转换为微伏值"""
        b = bytes.fromhex(hex3)
        raw = b[0] | (b[1] << 8) | (b[2] << 16)
        if raw & 0x800000:
            raw -= 0x1000000
        return raw * scale

    async def _simulation_loop(self) -> None:
        """模拟数据生成"""
        import math
        phase = 0.0
        
        while self.running:
            now = time.time()
            ch1 = 50 * math.sin(phase) + 10 * np.random.randn() if np else 0
            ch2 = 30 * math.sin(phase + 1.0) + 8 * np.random.randn() if np else 0
            
            with self.data_lock:
                self.eeg_ch1_data.append(ch1)
                self.eeg_ch2_data.append(ch2)
                self.timestamps.append(now)
                self.sample_count += 1

            phase += 0.1
            await asyncio.sleep(0.002)  # 500Hz

    def get_latest_sample(self) -> Optional[Dict[str, Any]]:
        """获取最新样本"""
        with self.data_lock:
            if not self.eeg_ch1_data:
                return None
            return {
                'timestamp': self.timestamps[-1],
                'ch1': self.eeg_ch1_data[-1],
                'ch2': self.eeg_ch2_data[-1],
                'ads_event': self.ads_events[-1] if len(self.ads_events) == len(self.eeg_ch1_data) else None
            }

    def get_recent_window(self, seconds: float = 5.0, sample_rate: float = 500.0) -> Dict[str, Any]:
        """获取最近的时间窗口数据（优化：快速锁获取，避免阻塞BLE回调）"""
        # 快速获取切片边界，最小化锁持有时间
        with self.data_lock:
            n = len(self.eeg_ch1_data)
            if n == 0:
                return {'timestamps': [], 'ch1': [], 'ch2': [], 'ads_event': []}
            
            target_len = int(seconds * sample_rate)
            start = max(0, n - target_len)
            
            # 只复制索引，数据复制在锁外进行（列表切片是原子操作）
            # 注意：这里我们还是需要在锁内切片，但Python的列表切片是C实现，很快
            timestamps_slice = self.timestamps[start:]
            ch1_slice = self.eeg_ch1_data[start:]
            ch2_slice = self.eeg_ch2_data[start:]
            ads_slice = self.ads_events[start:n]
        
        # 在锁外返回结果（避免dict构造时持有锁）
        return {
            'timestamps': timestamps_slice,
            'ch1': ch1_slice,
            'ch2': ch2_slice,
            'ads_event': ads_slice
        }

    def _save_data(self) -> None:
        """保存采集的数据到文件"""
        if not self.eeg_ch1_data or np is None:
            self.logger.warning("没有数据可保存")
            return

        try:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存numpy数组
            ch1_npy_path = os.path.join(self.save_dir, f'channel1_data_{timestamp_str}.npy')
            ch2_npy_path = os.path.join(self.save_dir, f'channel2_data_{timestamp_str}.npy')
            np.save(ch1_npy_path, np.array(self.eeg_ch1_data))
            np.save(ch2_npy_path, np.array(self.eeg_ch2_data))
            
            # 保存TXT文件
            ch1_txt_path = os.path.join(self.save_dir, f'channel1_data_{timestamp_str}.txt')
            ch2_txt_path = os.path.join(self.save_dir, f'channel2_data_{timestamp_str}.txt')
            
            with open(ch1_txt_path, 'w') as f:
                for i, val in enumerate(self.eeg_ch1_data, 1):
                    f.write(f"{i}\t{val}\n")
            
            with open(ch2_txt_path, 'w') as f:
                for i, val in enumerate(self.eeg_ch2_data, 1):
                    f.write(f"{i}\t{val}\n")
            
            # 保存CSV文件
            csv_path = os.path.join(self.save_dir, f'eeg_data_{timestamp_str}.csv')
            with open(csv_path, 'w', newline='') as f:
                f.write("Sample,Timestamp,Channel1,Channel2,ADS_Event,Sequence\n")
                for i, (ch1, ch2, ts) in enumerate(zip(self.eeg_ch1_data, self.eeg_ch2_data, self.timestamps), 1):
                    timestamp_str_fmt = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    ads_evt = self.ads_events[i-1] if i <= len(self.ads_events) else 0
                    seq = self.sequence_numbers[i-1] if i <= len(self.sequence_numbers) else 0
                    f.write(f"{i},{timestamp_str_fmt},{ch1},{ch2},{ads_evt},{seq}\n")
            
            # 保存元数据
            metadata = {
                "timestamp": timestamp_str,
                "sample_count": self.sample_count,
                "packets_received": self.packets_received,
                "lost_packets": self.lost_packets,
                "start_time": datetime.fromtimestamp(self.timestamps[0]).isoformat() if self.timestamps else None,
                "end_time": datetime.fromtimestamp(self.timestamps[-1]).isoformat() if self.timestamps else None,
                "simulation_mode": self.simulation_enabled,
                "sample_rate": "50 packets/s, 10 samples/packet, 500 samples/s",
                "data_format": "24-bit signed integer",
                "csv_file": csv_path,
                "ch1_txt_file": ch1_txt_path,
                "ch2_txt_file": ch2_txt_path,
                "ch1_npy_file": ch1_npy_path,
                "ch2_npy_file": ch2_npy_path,
            }
            
            meta_path = os.path.join(self.save_dir, f'metadata_{timestamp_str}.json')
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 EEG数据已保存: {self.sample_count} 样本")
        except Exception as exc:
            self.logger.error(f"保存EEG数据失败: {exc}")


__all__ = ["EEGService"]
