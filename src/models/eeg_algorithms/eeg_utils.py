# -*- coding: utf-8 -*-
"""
eeg_utils.py
双通道(Fp1, Fp2) EEG读取、分窗、特征、以及按时间戳截段的工具函数。
"""

import os
import re
import json
import numpy as np
import pandas as pd
from scipy.signal import welch, butter, filtfilt, iirnotch, coherence, get_window
from scipy.stats import kurtosis

# =========================
# 基本参数（可按需修改）
# =========================
FS = 250               # 采样率 Hz（若CSV不含时间戳则用它推时间）
BANDPASS = (0.5, 45)   # 带通
NOTCH = 60             # 工频 60Hz（若你是 50Hz 地区改为 50）
NOTCH_Q = 30.0

# =========================
# 读 txt（旧数据） & 预处理
# =========================
def _robust_float_tokens(line):
    return re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)

def read_eeg_txt_two_channels(path):
    fp1, fp2 = [], []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            toks = _robust_float_tokens(line)
            if len(toks) >= 2:
                try:
                    a, b = float(toks[0]), float(toks[1])
                    fp1.append(a); fp2.append(b)
                except ValueError:
                    continue
    if len(fp1) == 0:
        raise ValueError(f"无法从文件中解析出两列数值: {path}")
    arr = np.stack([np.array(fp1, dtype=np.float64),
                    np.array(fp2, dtype=np.float64)], axis=1)
    return arr

# =========================
# 读 CSV（新数据）
# =========================
def read_eeg_csv_two_channels(
    path_csv: str,
    ch1_col: str = "Fp1",
    ch2_col: str = "Fp2",
    ts_col: str = "timestamp"
):
    """
    读取 CSV，返回:
      sig: (N,2) float64 -> [Fp1, Fp2]
      ts:  (N,)  float64 -> 秒级时间戳（UNIX epoch, float seconds）
    要求 CSV 含 ts_col + 两个通道列；若 ts 是毫秒/纳秒会自动转为秒。
    """
    df = pd.read_csv(path_csv)
    if ts_col not in df.columns:
        raise ValueError(f"CSV 缺少时间戳列 {ts_col}: {path_csv}")
    if ch1_col not in df.columns or ch2_col not in df.columns:
        raise ValueError(f"CSV 缺少通道列 {ch1_col}/{ch2_col}: {path_csv}")
    ts = df[ts_col].to_numpy(dtype=np.float64)

    # 自动单位识别：若很大，认为是毫秒或纳秒
    ts_median = float(np.median(ts))
    if ts_median > 1e12:          # 纳秒
        ts = ts / 1e9
    elif ts_median > 1e10:        # 毫秒
        ts = ts / 1e3
    # 否则视为秒

    sig = df[[ch1_col, ch2_col]].to_numpy(dtype=np.float64)
    return sig, ts

def read_triggers_json(path_json: str):
    """
    读取 triggers JSON（数组里含 'timestamp' 或 'time' 字段），
    返回升序的秒级时间戳 list[float]
    """
    with open(path_json, 'r', encoding='utf-8') as f:
        arr = json.load(f)
    ts_list = []
    for item in arr:
        if isinstance(item, dict):
            if 'timestamp' in item:
                ts_list.append(float(item['timestamp']))
            elif 'time' in item:
                ts_list.append(float(item['time']))
        elif isinstance(item, (int, float)):
            ts_list.append(float(item))
    ts_list = sorted(ts_list)
    return ts_list

# =========================
# 滤波/分窗/伪迹
# =========================
def butter_bandpass_filter(sig, fs=FS, band=BANDPASS, order=4):
    b, a = butter(order, [band[0]/(fs/2.0), band[1]/(fs/2.0)], btype='band')
    return filtfilt(b, a, sig, axis=0)

def notch_filter(sig, fs=FS, f0=NOTCH, Q=NOTCH_Q):
    b, a = iirnotch(w0=f0/(fs/2.0), Q=Q)
    return filtfilt(b, a, sig, axis=0)

def preprocess_eeg(arr_2ch, fs=FS):
    x = butter_bandpass_filter(arr_2ch, fs=fs)
    x = notch_filter(x, fs=fs)
    return x

def segment_windows(arr2, fs=FS, win_sec=2.0, step_sec=1.0):
    N = arr2.shape[0]
    win = int(win_sec * fs); step = int(step_sec * fs)
    out, starts = [], []
    for s in range(0, N - win + 1, step):
        out.append(arr2[s:s+win, :]); starts.append(s)
    return out, np.array(starts, dtype=np.int64)

def is_artifact_window(win2, fs=FS, amp_mad_thresh=5.0, kurt_thresh=8.0):
    for ch in range(win2.shape[1]):
        x = win2[:, ch]
        mad = np.median(np.abs(x - np.median(x))) + 1e-9
        amp_ok = (np.max(np.abs(x)) <= amp_mad_thresh * mad)
        kurt_ok = (kurtosis(x, fisher=True, bias=False) <= kurt_thresh)
        if not amp_ok or not kurt_ok:
            return True
    return False

# =========================
# 特征工程（与你之前算法一致）
# =========================
def bandpower_welch(x, fs, fmin, fmax):
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=int(1.0*fs),
                   noverlap=int(0.5*fs), detrend=False, scaling='density')
    idx = np.logical_and(f >= fmin, f <= fmax)
    return np.trapz(Pxx[idx], f[idx]) + 1e-12

def spectral_entropy(x, fs, fmin=1, fmax=45):
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=int(1.0*fs),
                   noverlap=int(0.5*fs), detrend=False, scaling='density')
    idx = np.logical_and(f >= fmin, f <= fmax)
    P = Pxx[idx]; P = P / (np.sum(P) + 1e-12)
    H = -np.sum(P * np.log(P + 1e-12))
    return float(H / (np.log(len(P) + 1e-12)))

def hjorth_params(x):
    x = np.asarray(x)
    dx = np.diff(x, prepend=x[0])
    var0 = np.var(x) + 1e-12
    var1 = np.var(dx) + 1e-12
    activity = var0
    mobility = np.sqrt(var1 / var0)
    ddx = np.diff(dx, prepend=dx[0])
    var2 = np.var(ddx) + 1e-12
    complexity = np.sqrt(var2 / var1) / (mobility + 1e-12)
    return float(activity), float(mobility), float(complexity)

from scipy.signal import coherence
def magnitude_coherence(x, y, fs):
    f, Cxy = coherence(x, y, fs=fs, window=get_window('hann', int(1.0*fs)),
                       nperseg=int(1.0*fs), noverlap=int(0.5*fs))
    def band_coh(lo, hi):
        idx = np.logical_and(f >= lo, f <= hi)
        return float(np.mean(Cxy[idx]) if np.any(idx) else 0.0)
    return band_coh(8,12), band_coh(13,30)

def extract_features_one_window(win2, fs=FS):
    x1, x2 = win2[:,0], win2[:,1]
    xdiff = x2 - x1
    bands = {'theta': (4,7), 'alpha': (8,12), 'beta': (13,30), 'gamma': (30,45)}
    feats, names = [], []

    def add_bp(sig, tag):
        for bname, (lo, hi) in bands.items():
            feats.append(np.log(bandpower_welch(sig, fs, lo, hi)))
            names.append(f'logbp_{tag}_{bname}')

    add_bp(x1,'fp1'); add_bp(x2,'fp2'); add_bp(xdiff,'diff')
    def get_logbp(sig, lo, hi): return np.log(bandpower_welch(sig, fs, lo, hi))
    for tag, sig in [('fp1',x1),('fp2',x2),('diff',xdiff)]:
        lt, la, lb = get_logbp(sig,4,7), get_logbp(sig,8,12), get_logbp(sig,13,30)
        feats += [lt-lb, la-lb, lb - np.log(np.exp(la)+np.exp(lt)+1e-12)]
        names += [f'ratio_{tag}_theta_beta',
                  f'ratio_{tag}_alpha_beta',
                  f'ratio_{tag}_beta_over_alpha_theta']
    FAA = get_logbp(x2,8,12) - get_logbp(x1,8,12)
    feats += [FAA]; names += ['FAA_alpha']
    coh_a, coh_b = magnitude_coherence(x1, x2, fs)
    feats += [coh_a, coh_b]; names += ['coh_alpha','coh_beta']
    for sig, tag in [(x1,'fp1'),(x2,'fp2'),(xdiff,'diff')]:
        act,mob,comp = hjorth_params(sig)
        feats += [act,mob,comp]; names += [f'hj_{tag}_act',f'hj_{tag}_mob',f'hj_{tag}_comp']
    feats += [spectral_entropy((x1+x2)/2.0, fs)]; names += ['spectral_entropy']
    return np.array(feats, dtype=np.float64), names

def extract_features_batch(windows, fs=FS, reject_artifacts=True):
    X, valid, feat_names = [], [], None
    for w in windows:
        if reject_artifacts and is_artifact_window(w, fs=fs):
            valid.append(False); continue
        f, names = extract_features_one_window(w, fs=fs)
        if feat_names is None: feat_names = names
        X.append(f); valid.append(True)
    X = np.array(X, dtype=np.float64) if len(X)>0 else np.zeros((0,0))
    return X, (feat_names or []), np.array(valid, dtype=bool)

# =========================
# 按时间戳截段
# =========================
def slice_by_time(sig_2ch: np.ndarray, ts: np.ndarray, t_start: float, t_end: float):
    """
    根据绝对时间 [t_start, t_end) 截出片段；返回 sig_slice, ts_slice
    """
    mask = (ts >= t_start) & (ts < t_end)
    if not np.any(mask):
        return np.zeros((0,2)), np.zeros((0,), dtype=np.float64)
    return sig_2ch[mask], ts[mask]
