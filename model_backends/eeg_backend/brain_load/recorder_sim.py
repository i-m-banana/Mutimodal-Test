# -*- coding: utf-8 -*-
"""
recorder_sim.py
用试次目录中的 .csv 模拟“在线记录器”，每次返回最近 2s 的窗口：
return {'timestamps': [...], 'ch1': [...], 'ch2': [...]}

用法：
from recorder_sim import start_simulator, get_recent_2s, stop_simulator
start_simulator(r"D:\...\raw_data\Data\1\15")    # 指向“第15次试次”的文件夹
win = get_recent_2s()  # -> dict; 空窗口表示结束
"""

import os, glob, json, re
import pandas as pd

# === 可按需调整的列名 ===
TS_COL, CH1_COL, CH2_COL = "Timestamp", "Channel1", "Channel2"
# =====================

FS = 500            # 采样率 Hz
WIN_SEC = 2.0       # 每次发送 2 秒
WIN_SAMPLES = int(FS * WIN_SEC)

_running = False
_recorder = None

_TIME_RE = re.compile(r'(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)')

def _pick_csv_in_trial(trial_dir: str) -> str:
    csvs = glob.glob(os.path.join(trial_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"试次目录无 csv：{trial_dir}")
    def fsize(p):
        try: return os.path.getsize(p)
        except: return -1
    eeg_like = [p for p in csvs if 'eeg' in os.path.basename(p).lower()]
    cand = eeg_like if eeg_like else csvs
    cand.sort(key=fsize, reverse=True)
    return cand[0]

def _extract_time_token(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'(Z|[+-]\d{2}:?\d{2})$', '', s)
    if 'T' in s:
        s = s.split('T', 1)[1]
    elif ' ' in s:
        s = s.split()[-1]
    return s

def time_to_seconds_any(x) -> float:
    try:
        return float(x)
    except Exception:
        pass
    s = _extract_time_token(str(x))
    m = _TIME_RE.search(s)
    if m:
        h = int(m.group('h')) if m.group('h') else 0
        mi = int(m.group('m'))
        se = float(m.group('s'))
        return h*3600 + mi*60 + se
    return float(s)  # 尝试最终转浮点

class RecorderSimulator:
    def __init__(self, trial_dir: str):
        csv_path = _pick_csv_in_trial(trial_dir)
        df = pd.read_csv(csv_path)
        need = {TS_COL, CH1_COL, CH2_COL}
        if not need.issubset(df.columns):
            raise ValueError(f"CSV 缺列：{need - set(df.columns)}\n文件={csv_path}\n现有列={list(df.columns)}")
        # 原样保留时间字符串，以便回传显示；另外保留秒（如需日志/计算）
        self.timestamps_str = df[TS_COL].astype(str).tolist()
        self.timestamps_sec = [time_to_seconds_any(x) for x in self.timestamps_str]
        self.ch1 = df[CH1_COL].astype(float).tolist()
        self.ch2 = df[CH2_COL].astype(float).tolist()
        self.N = len(self.ch1)
        self.ptr = 0  # 读指针

    def get_recent_window(self, seconds: float = 2.0, sample_rate_hz: float = 500.0) -> dict:
        """步长=seconds，返回该时长的窗口；读到末尾后返回空窗口。"""
        need = int(seconds * sample_rate_hz)
        if self.ptr + need > self.N:
            return {'timestamps': [], 'ch1': [], 'ch2': []}
        i0, i1 = self.ptr, self.ptr + need
        ts = self.timestamps_str[i0:i1]
        x1 = self.ch1[i0:i1]
        x2 = self.ch2[i0:i1]
        # 步长=窗口长度（非重叠）
        self.ptr = i1
        return {'timestamps': ts, 'ch1': x1, 'ch2': x2}

def start_simulator(trial_dir: str):
    global _running, _recorder
    _recorder = RecorderSimulator(trial_dir)
    _running = True

def stop_simulator():
    global _running, _recorder
    _running = False
    _recorder = None

def get_recent_2s() -> dict:
    """
    按你要求的接口：
    if not _running or _recorder is None:
        return {'timestamps': [], 'ch1': [], 'ch2': []}
    return _recorder.get_recent_window(seconds=2.0, sample_rate_hz=500.0)
    """
    if not _running or _recorder is None:
        return {'timestamps': [], 'ch1': [], 'ch2': []}
    return _recorder.get_recent_window(seconds=2.0, sample_rate_hz=500.0)
