import os
import glob
import time
import asyncio
import threading
import numpy as np
from datetime import datetime
from bleak import BleakClient

# 设备信息
DEVICE_ADDRESS = "F4:3C:7C:A6:29:E0"
TX_CHARACTERISTIC = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"  # Nordic UART TX
RX_CHARACTERISTIC = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"  # Nordic UART RX

_loop = None
_thread = None
_recorder = None
_running = False
_save_dir = None


class UnifiedEEGRecorder:
    """统合的脑电数据记录器 - 合并BLE桥接和UDP客户端功能"""

    def __init__(self):
        self.ble_client = None
        self.running = False
        self.event_loop = None

        # 脑电数据存储
        self.eeg_ch1_data = []
        self.eeg_ch2_data = []
        self.timestamps = []
        self.sequence_numbers = []
        self.ads_events = []
        # 线程安全
        self.data_lock = threading.Lock()

        # 统计信息
        self.packets_received = 0  # 已接收的数据包数量
        self.sample_count = 0  # 已累积的样本数量（通道值对）
        self.lost_packets = 0
        self.last_sequence = None

        # 数据保存配置（按样本数）
        self.save_interval_samples = 300000  # 10分钟保存一次（500样本/秒 * 600秒）
        self.output_dir = None  # 将由外部设置

        # 调试模式
        self.debug_mode = False

    def reset_stats(self):
        """重置统计信息"""
        self.last_sequence = None
        self.packets_received = 0
        self.sample_count = 0
        self.lost_packets = 0
        print("📊 统计信息已重置")

    def toggle_debug_mode(self):
        """切换调试模式"""
        self.debug_mode = not self.debug_mode
        status = "开启" if self.debug_mode else "关闭"
        print(f"🔍 调试模式已{status}")
        if self.debug_mode:
            print("   将显示详细的数据包解析信息")

    # -------- 实时与窗口访问接口 --------
    def get_latest_sample(self):
        """获取最新一对通道值 (ch1, ch2) 以及时间戳，若无则返回 None。"""
        with self.data_lock:
            if not self.eeg_ch1_data:
                return None
            return {
                'timestamp': self.timestamps[-1],
                'ch1': self.eeg_ch1_data[-1],
                'ch2': self.eeg_ch2_data[-1],
                'ads_event': self.ads_events[-1] if len(self.ads_events) == len(self.eeg_ch1_data) else None
            }

    def get_recent_window(self, seconds: float = 5.0, sample_rate_hz: float = 500.0):
        """获取过去 seconds 秒的脑电数据窗口，默认 5s*500Hz=2500 对。
        返回 dict: { 'timestamps': [...], 'ch1': [...], 'ch2': [...], 'ads_event': [...] }
        若数据不足则返回能获取到的全部。
        这里的sample_rate_hz仅用于统计，不能控制推送（采样）频率。
        """
        if seconds <= 0:
            seconds = 5.0
        if sample_rate_hz <= 0:
            sample_rate_hz = 500.0
        with self.data_lock:
            n = len(self.eeg_ch1_data)
            if n == 0:
                return {'timestamps': [], 'ch1': [], 'ch2': [], 'ads_event': []}
            target_len = int(seconds * sample_rate_hz)
            start = max(0, n - target_len)
            return {
                'timestamps': self.timestamps[start:],
                'ch1': self.eeg_ch1_data[start:],
                'ch2': self.eeg_ch2_data[start:],
                'ads_event': self.ads_events[start:len(self.eeg_ch1_data)]
            }

    def hex3_to_microvolt(self, hex3: str, scale: float = 0.0447):
        # 脑电数据调整
        b = bytes.fromhex(hex3)
        raw = b[0] | (b[1] << 8) | (b[2] << 16)
        if raw & 0x800000:
            raw -= 0x1000000
        return raw * scale

    def parse_eeg_packet(self, data):
        """解析PSG数据包，提取脑电数据"""
        try:
            if len(data) != 187:
                return None

            # 提取179字节的PSG数据部分 (跳过前8字节的命令头)
            psg_data = data[8:187]

            if len(psg_data) != 179:
                return None

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
                        print(f"⚠️  序列号异常跳跃: {self.last_sequence} -> {sequence_number}")
                        packet_loss_count = min(packet_loss_count, 10)

                    self.lost_packets += packet_loss_count

            self.last_sequence = sequence_number
            self.packets_received += 1

            # 解析脑电数据 - 根据实际数据包格式调整
            eeg_samples = []

            # 从载荷数据开始解析脑电数据
            # 跳过序列号(3字节) + 电池(2字节) + 音频(60字节) = 65字节
            offset = 65

            # 调试模式：显示数据包结构
            if self.debug_mode:
                print(f"🔍 数据包解析 (包{self.packets_received}):")
                print(f"   序列号: {sequence_number}")
                print(f"   载荷数据长度: {len(psg_data)}字节")
                print(f"   脑电数据偏移: {offset}字节")
                print(f"   载荷数据前16字节: {psg_data[:16].hex().upper()}")

            if offset + 70 <= len(psg_data):
                for i in range(10):
                    group_offset = offset + i * 7
                    if group_offset + 7 <= len(psg_data):
                        # 通道1 (3字节)
                        ch1_psg = psg_data[group_offset:group_offset + 3]
                        ch1_value = self.hex3_to_microvolt(ch1_psg.hex())

                        # 通道2 (3字节)
                        ch2_psg = psg_data[group_offset + 3:group_offset + 6]
                        ch2_value = self.hex3_to_microvolt(ch2_psg.hex())

                        # ADS事件 (1字节)
                        ads_event = psg_data[group_offset + 6]

                        eeg_samples.append({
                            'channel1': ch1_value,
                            'channel2': ch2_value,
                            'ads_event': ads_event
                        })

            # 存储脑电数据
            if eeg_samples:
                now = time.time()
                with self.data_lock:
                    for sample in eeg_samples:
                        self.eeg_ch1_data.append(sample['channel1'])
                        self.eeg_ch2_data.append(sample['channel2'])
                        self.ads_events.append(sample.get('ads_event', 0))
                        self.timestamps.append(now)
                        self.sequence_numbers.append(sequence_number)
                    # 增加样本计数（每包10个样本）
                    self.sample_count += len(eeg_samples)

                # 显示前几个脑电样本值（调试用）
                if self.packets_received <= 3 and self.debug_mode:  # 只在前3个包显示
                    print(f"🧠 脑电样本示例 (包{self.packets_received}):")
                    for i, sample in enumerate(eeg_samples[:3]):  # 只显示前3个样本
                        print(f"   样本{i + 1}: CH1={sample['channel1']}, CH2={sample['channel2']}")

            # 显示状态信息
            if self.packets_received % 1500 == 0:
                total_pkts = self.packets_received + self.lost_packets
                loss_rate = (self.lost_packets / total_pkts) * 100 if total_pkts > 0 else 0
                print(f"📊 包统计: 收到{self.packets_received}包, 丢失{self.lost_packets}包, 丢包率{loss_rate:.2f}%")
                print(f"🧠 样本统计: 已记录样本对={self.sample_count}")

            # # 定期保存数据（按样本数量）
            # if hasattr(self, 'save_interval_samples') and self.save_interval_samples > 0:
            #     if self.sample_count > 0 and self.sample_count % self.save_interval_samples == 0:
            #         self.save_data()

            return {
                'sequence_number': sequence_number,
                'packet_loss_count': packet_loss_count,
                'eeg_samples': len(eeg_samples)
            }

        except Exception as e:
            print(f"❌ 数据包解析错误: {e}")
            return None

    def save_data(self):
        """保存脑电数据到本地文件"""
        try:
            if not self.eeg_ch1_data:
                return

            if not self.output_dir:
                print("❌ 输出目录或会话ID未设置，无法保存数据")
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.output_dir, exist_ok=True)

            # 保存为numpy数组
            ch1_file = os.path.join(self.output_dir, f"channel1_data_{timestamp}.npy")
            ch2_file = os.path.join(self.output_dir, f"channel2_data_{timestamp}.npy")

            np.save(ch1_file, np.array(self.eeg_ch1_data))
            np.save(ch2_file, np.array(self.eeg_ch2_data))

            # 保存为文本文件（便于查看）
            ch1_txt = os.path.join(self.output_dir, f"channel1_data_{timestamp}.txt")
            ch2_txt = os.path.join(self.output_dir, f"channel2_data_{timestamp}.txt")

            with open(ch1_txt, 'w') as f:
                for i, value in enumerate(self.eeg_ch1_data):
                    f.write(f"{i + 1}\t{value}\n")

            with open(ch2_txt, 'w') as f:
                for i, value in enumerate(self.eeg_ch2_data):
                    f.write(f"{i + 1}\t{value}\n")

            # 保存为CSV格式（便于Excel等软件打开）
            csv_file = os.path.join(self.output_dir, f"eeg_data_{timestamp}.csv")
            with open(csv_file, 'w', newline='') as f:
                f.write("Sample,Timestamp,Channel1,Channel2,ADS_Event,Sequence\n")
                for i in range(len(self.eeg_ch1_data)):
                    timestamp_str = datetime.fromtimestamp(self.timestamps[i]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    seq_num = self.sequence_numbers[i] if i < len(self.sequence_numbers) else 0
                    ads = self.ads_events[i] if i < len(self.ads_events) else ''
                    f.write(f"{i + 1},{timestamp_str},{self.eeg_ch1_data[i]},{self.eeg_ch2_data[i]},{ads},{seq_num}\n")

            # 保存元数据
            metadata = {
                'timestamp': timestamp,
                'total_samples': len(self.eeg_ch1_data),
                'packets_received': self.packets_received,
                'sample_count': self.sample_count,
                'lost_packets': self.lost_packets,
                'sample_rate': '50 packets/s, 10 samples/packet, 500 samples/s',
                'data_format': '24-bit signed integer',
                'ads_event_present': True
            }

            import json
            metadata_file = os.path.join(self.output_dir, f"metadata_{timestamp}.json")
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"💾 数据已保存: {len(self.eeg_ch1_data)}个样本")
            print(f"   CH1: {ch1_file}")
            print(f"   CH2: {ch2_file}")
            print(f"   CSV: {csv_file}")
            print(f"   元数据: {metadata_file}")

        except Exception as e:
            print(f"❌ 数据保存失败: {e}")

    def create_ble_command(self, cmd_type, sub_cmd=0x00, data_size=0, data=b''):
        """创建原始BLE命令数据包"""
        if isinstance(data_size, int) and data_size < 256:
            size_low = data_size & 0xFF
            size_high = (data_size >> 8) & 0xFF
        else:
            size_low = 0x00
            size_high = 0x00

        command = bytes([0x48, 0x4E, 0x55, 0x9F, cmd_type, sub_cmd, size_low, size_high])
        if data:
            command += data
        return command

    async def send_ble_command(self, cmd_type, cmd_name, sub_cmd=0x00, data=b''):
        """发送BLE命令到设备"""
        try:
            if self.ble_client and self.ble_client.is_connected:
                command_data = self.create_ble_command(cmd_type, sub_cmd, len(data), data)
                await self.ble_client.write_gatt_char(RX_CHARACTERISTIC, command_data)
                print(f"📤 发送{cmd_name}命令: {command_data.hex().upper()}")
            else:
                print("❌ BLE未连接，无法发送命令")

        except Exception as e:
            print(f"❌ BLE命令发送失败: {e}")

    async def ble_data_callback(self, sender, data):
        """BLE数据回调，解析脑电数据"""
        try:
            if not self.running:
                return
            # 解析脑电数据包
            result = self.parse_eeg_packet(data)
            if result and self.packets_received % 500 == 0:
                print(f"📥 收到脑电数据包: 序列号{result['sequence_number']}, {result['eeg_samples']}个样本")

        except Exception as e:
            print(f"❌ BLE数据处理错误: {e}")

    async def start_recorder(self):
        """启动脑电记录器"""
        self.running = True

        try:
            # 保存当前事件循环
            self.event_loop = asyncio.get_event_loop()

            # 连接BLE设备
            print(f"🔗 正在连接BLE设备: {DEVICE_ADDRESS}")
            self.ble_client = BleakClient(DEVICE_ADDRESS)
            await self.ble_client.connect()
            print("✅ BLE设备连接成功!")

            # 启动BLE数据监听
            await self.ble_client.start_notify(TX_CHARACTERISTIC, self.ble_data_callback)
            print(f"📡 开始监听BLE数据: {TX_CHARACTERISTIC}")
            # 自动发送启动命令，无需用户输入
            try:
                await self.send_ble_command(0x03, '启动记录')
                print("▶️ 已自动发送启动记录命令")
            except Exception as e:
                print(f"⚠️ 启动记录命令发送失败: {e}")

            print("🧠 脑电记录器启动完成!")
            if self.output_dir:
                print(f"   📁 数据保存目录: {self.output_dir}")
            print(f"   💾 自动保存间隔: 每{self.save_interval_samples}个样本")
            print("-" * 50)

            # 保持运行，直到收到退出指令
            try:
                while self.running:
                    await asyncio.sleep(0.1)
            except KeyboardInterrupt:
                print("\n⏹️  正在停止记录器...")

        except Exception as e:
            print(f"❌ 记录器启动失败: {e}")

        finally:
            await self.stop_recorder()

    async def stop_recorder(self):
        """停止记录器"""
        if self.running == False:
            return
        self.running = False

        try:
            # 保存剩余数据
            if self.eeg_ch1_data:
                print("💾 保存剩余数据...")
                self.save_data()

            # 停止BLE
            if self.ble_client and self.ble_client.is_connected:
                # 自动发送停止命令
                try:
                    await self.send_ble_command(0x04, '停止记录')
                    await asyncio.sleep(0.2)
                    print("⏹️ 已自动发送停止记录命令")
                except Exception as e:
                    print(f"⚠️ 停止记录命令发送失败: {e}")
                await asyncio.sleep(0.5)
                await self.ble_client.stop_notify(TX_CHARACTERISTIC)
                await self.ble_client.disconnect()
                print("🔕 BLE连接已关闭")

            print("✅ 记录器已完全停止")

        except Exception as e:
            print(f"❌ 停止记录器时出错: {e}")


def _run_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def _ensure_loop_thread():
    global _loop, _thread
    if _loop is None or _thread is None or not _thread.is_alive():
        _loop = asyncio.new_event_loop()
        _thread = threading.Thread(target=_run_loop, args=(_loop,), daemon=True)
        _thread.start()


def start(save_dir: str) -> None:
    """Start EEG recording into the provided save_dir"""
    global _recorder, _running, _save_dir
    if _running:
        return
    _ensure_loop_thread()

    _save_dir = os.path.abspath(save_dir)
    _save_dir = os.path.join(_save_dir, 'eeg')

    def _create_and_start():
        global _recorder, _running
        _recorder = UnifiedEEGRecorder()
        # 直接使用传入目录作为输出目录，不再拆分父/子目录
        _recorder.output_dir = _save_dir

        _running = True
        return _recorder.start_recorder()

    fut = asyncio.run_coroutine_threadsafe(_create_and_start(), _loop)
    # Do not block Qt; allow async startup
    try:
        fut.result(timeout=0.1)
    except Exception:
        # Startup will continue in background
        pass


def stop() -> None:
    """Stop EEG recording if running."""
    global _recorder, _running
    if not _running or _recorder is None:
        return

    async def _stop():
        try:
            await _recorder.stop_recorder()
        finally:
            return

    fut = asyncio.run_coroutine_threadsafe(_stop(), _loop)
    try:
        fut.result(timeout=5.0)
    except Exception:
        pass
    _running = False


def get_file_paths() -> dict:
    # print(_save_dir)
    if not _save_dir:
        return {}
    folder = os.path.abspath(_save_dir)

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


def get_realtime_sample() -> dict | None:
    """获取最新一对通道值及时间戳。"""
    if not _running or _recorder is None:
        return None
    return _recorder.get_latest_sample()


def get_recent_5s() -> dict:
    """获取过去5秒（默认500Hz）的脑电窗口。"""
    if not _running or _recorder is None:
        return {'timestamps': [], 'ch1': [], 'ch2': []}
    return _recorder.get_recent_window(seconds=5.0, sample_rate_hz=500.0)
