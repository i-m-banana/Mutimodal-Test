import os
import time
import threading
import cv2
import pyaudio

class AVCollector:
    """
    轻量级音视频采集器（单例），供主程序直接调用。

    功能要点：
    - 采集启动/停止：启动摄像头后台抓帧线程；音频在录制时才打开输入流
    - 录制启动/停止：同时开始/停止音频与视频录制；录制完摄像头预览不断流
    - 获取当前帧：返回最新 BGR 帧（numpy array）用于 UI 展示
    - 获取所有音频/视频路径：返回当前会话内已保存段落的路径列表
    - 固定视频写入 FPS（默认 30fps），即使摄像头帧率偏低也不会回放“快进”
    """

    def __init__(self):
        # 基本状态
        self.session_dir = None
        self.camera_index = 0
        self._video_fps = 30.0

        # 摄像头
        self._cap = None
        self._grab_thread = None
        self._grab_running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()

        # 视频写入
        self._video_writer = None
        self._video_size = None
        self._video_filepath = None
        self._writer_thread = None
        self._writer_running = False

        # 音频
        self._pa = pyaudio.PyAudio()
        self._audio_stream = None
        self._audio_chunks = []
        self._audio_rate = 8000
        self._audio_channels = 1
        self._audio_format = pyaudio.paInt16
        self._audio_device_index = None  # 可由主程序传入
        self._audio_filepath = None
        self._audio_lock = threading.Lock()

        # 实时音频数据（用于电平显示）
        self._audio_level = 0
        self._audio_level_lock = threading.Lock()

        # 录制控制
        self._is_recording = False
        self._segment_index = 0
        self._audio_paths = []
        self._video_paths = []

    # ---------- 音视频采集与录制开关接口 ----------
    def start_collection(self, save_dir: str, camera_index: int = 0, video_fps: float = 30.0,
                         audio_rate: int = 8000, input_device_index: int = 2):
        """开始采集音视频帧"""
        # 初始化参数值
        self.session_dir = save_dir or 'recordings'
        os.makedirs(self.session_dir, exist_ok=True)
        self.camera_index = camera_index
        self._video_fps = float(video_fps) if video_fps and video_fps > 0 else 30.0
        self._audio_rate = int(audio_rate) if audio_rate else 8000

        # 打开摄像头
        if self._cap is None:
            self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap or not self._cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self._audio_device_index = input_device_index
        width = 640
        height = 480
        self._video_size = (width, height)

        # 启动摄像头抓帧线程 (用于页面展示）
        if not self._grab_running:
            self._grab_running = True
            self._grab_thread = threading.Thread(target=self._grab_loop, daemon=True)
            self._grab_thread.start()

        # 启动实时音频流（用于电平显示）
        self._start_realtime_audio_stream()

    def stop_collection(self):
        """停止采集音视频帧，释放av采集器的所有资源"""
        # 停止录制视频（若仍在录）
        if self._is_recording:
            self.stop_recording()

        # 停止实时音频流
        self._stop_realtime_audio_stream()

        # 停止抓帧
        self._grab_running = False
        if self._grab_thread:
            self._grab_thread.join(timeout=1.0)
        self._grab_thread = None

        # 释放摄像头
        try:
            if self._cap and self._cap.isOpened():
                self._cap.release()
        finally:
            self._cap = None

    def start_recording(self):
        """开始录制（点击麦克风）"""
        if self._is_recording:
            return
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("摄像头未启动，无法开始录制")

        self._segment_index += 1
        base = os.path.join(self.session_dir, 'emotion')
        os.makedirs(base, exist_ok=True)
        base = os.path.join(base,f"{self._segment_index}")
        self._audio_filepath = base + ".wav"
        self._video_filepath = base + ".avi"

        # 准备视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # print(self._video_size)
        self._video_writer = cv2.VideoWriter(self._video_filepath, fourcc, self._video_fps, self._video_size)
        if not self._video_writer.isOpened():
            self._video_writer = None
            raise RuntimeError("视频写入器打开失败")

        # 清空音频缓存（音频流已在实时运行，点击麦克风后缓存的数据用于落盘）
        with self._audio_lock:
            self._audio_chunks = []

        # 启动视频写入线程（固定节拍，每 1/fps 写最近帧）
        self._writer_running = True
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        self._is_recording = True

    def stop_recording(self):
        """停止录制（再次点击麦克风），将音视频保存到本地，不会影响音视频帧采集"""
        if not self._is_recording:
            return

        # 停止视频写入线程
        self._writer_running = False
        if self._writer_thread:
            self._writer_thread.join(timeout=1.0)
        self._writer_thread = None

        # 关闭视频写入器
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            finally:
                self._video_writer = None

        # 落盘音频数据（音频流继续运行用于实时电平显示）
        self._flush_wav()

        # 记录路径
        if self._audio_filepath and os.path.exists(self._audio_filepath):
            self._audio_paths.append(self._audio_filepath)
        if self._video_filepath and os.path.exists(self._video_filepath):
            self._video_paths.append(self._video_filepath)

        self._is_recording = False
        self._audio_filepath = None
        self._video_filepath = None

    def get_current_frame(self):
        """返回最新 BGR 帧（numpy array）; 若暂无则返回 None"""
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def get_current_audio_level(self):
        """返回当前音频电平 (0-100)"""
        with self._audio_level_lock:
            return self._audio_level

    def get_audio_paths(self):
        return list(self._audio_paths)

    def get_video_paths(self):
        return list(self._video_paths)

    # ---------- 实时音频流管理 ----------
    def _start_realtime_audio_stream(self):
        """启动实时音频采集流（用于电平显示，不录制）"""
        try:
            if self._audio_stream is None:
                self._audio_stream = self._pa.open(
                    format=self._audio_format,
                    channels=self._audio_channels,
                    rate=self._audio_rate,
                    input=True,
                    frames_per_buffer=1024,
                    input_device_index=self._audio_device_index,
                    stream_callback=self._audio_callback
                )
                self._audio_stream.start_stream()
        except Exception as e:
            print(f"启动实时音频流失败: {e}")

    def _stop_realtime_audio_stream(self):
        """停止实时音频流"""
        try:
            if self._audio_stream is not None:
                self._audio_stream.stop_stream()
                self._audio_stream.close()
                self._audio_stream = None
        except Exception as e:
            print(f"停止实时音频流失败: {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频流回调函数，用于实时处理音频数据"""
        # if status:
        #     print(f"音频流状态: {status}")

        # 计算音频电平
        import struct
        count = len(in_data) // 2
        if len(in_data) == count * 2:
            shorts = struct.unpack(f"{count}h", in_data)
            rms = (sum(n ** 2 for n in shorts) / count) ** 0.5 if count > 0 else 0
            level = min(100, int(rms / 30))

            with self._audio_level_lock:
                self._audio_level = level

        # 如果正在录制，将数据添加到录制缓存
        if self._is_recording:
            with self._audio_lock:
                self._audio_chunks.append(in_data)

        return (None, pyaudio.paContinue)

    def _flush_wav(self):
        """音频落盘"""
        if not self._audio_filepath:
            return
        import wave
        try:
            os.makedirs(os.path.dirname(self._audio_filepath), exist_ok=True)
            with wave.open(self._audio_filepath, 'wb') as wf:
                wf.setnchannels(self._audio_channels)
                wf.setsampwidth(self._pa.get_sample_size(self._audio_format))
                wf.setframerate(self._audio_rate)
                with self._audio_lock:
                    wf.writeframes(b''.join(self._audio_chunks))
            print(f"录音已保存至: {self._audio_filepath}")
        except Exception:
            pass

    # ---------- 内部线程 ----------
    def _grab_loop(self):
        while self._grab_running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.003)
                continue
            # 与 UI 一致：水平翻转
            frame = cv2.flip(frame, 1)
            with self._frame_lock:
                self._latest_frame = frame
            # 采集线程不写盘，仅提供最新帧；降低 CPU
            time.sleep(0.001)

    def _writer_loop(self):
        """固定节拍写入：每 1/fps 写一次最近帧。音频通过回调函数自动收集。"""
        frame_interval = 1.0 / max(1.0, float(self._video_fps))
        next_ts = time.monotonic()

        while self._writer_running and self._video_writer is not None:
            now = time.monotonic()
            if now < next_ts:
                time.sleep(max(0.0, next_ts - now))
                continue

            # 取最近帧写入；若暂无帧则跳过本次
            with self._frame_lock:
                bgr = self._latest_frame
            if bgr is not None:
                try:
                    self._video_writer.write(bgr)
                except Exception:
                    pass

            next_ts += frame_interval


# ----------------- 模块级单例与对外接口 -----------------
_collector = AVCollector()


def start_collection(save_dir: str, camera_index: int = 0, video_fps: float = 30.0,
                     audio_rate: int = 8000, input_device_index: int = None):
    """
    启动采集（仅开启摄像头抓帧线程），预览可用但不录制。
    - save_dir: 会话目录（主程序的 session_dir）
    - camera_index: 摄像头索引
    - video_fps: 目标写入帧率（录制用）
    - audio_rate: 音频采样率
    - input_device_index: PyAudio 输入设备索引（可为 None 自动选择默认）
    """
    _collector.start_collection(save_dir, camera_index, video_fps, audio_rate, input_device_index)
    return _collector


def stop_collection():
    """停止采集（会自动停止录制）。"""
    _collector.stop_recording()
    _collector.stop_collection()


def start_recording():
    """开始音视频录制（同时启动音频与视频写入）。"""
    _collector.start_recording()


def stop_recording():
    """停止音视频录制并保存文件。"""
    _collector.stop_recording()


def get_current_frame():
    """返回最新 BGR 帧（numpy array 或 None）。"""
    return _collector.get_current_frame()


def get_audio_paths():
    """返回当前会话内已保存的所有音频文件路径列表。"""
    return _collector.get_audio_paths()


def get_video_paths():
    """返回当前会话内已保存的所有视频文件路径列表。"""
    return _collector.get_video_paths()


def get_current_audio_level():
    """返回当前音频电平 (0-100)。"""
    return _collector.get_current_audio_level()
