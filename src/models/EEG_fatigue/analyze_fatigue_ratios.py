# -*- coding: utf-8 -*-
"""
analyze_fatigue_ratios.py  检查数据质量用，系统不需要用，无需集成
从试次目录中的 .csv 读取双通道 EEG（Timestamp, Channel1, Channel2z），
以 2 秒窗计算 θ/β、θ/α 比值（Fp1、Fp2 及其均值），输出 Excel 和趋势图 PNG。

特性：
- CSV 自动选择（优先文件名含 'eeg'，否则体积最大）
- 健壮时间解析：支持 'YYYY-MM-DD HH:MM:SS(.ffffff)' / '...T...' / 'HH:MM:SS(.fff)' / 'MM:SS(.fff)'
- 采样率默认 500 Hz；2 s 窗（可选重叠，step_sec 可改）
- 可选预处理：0.5–45 Hz 带通 + 60 Hz 陷波
- 结果保存：Excel（xlsxwriter/openpyxl 回退 CSV）+ PNG 趋势图
"""

import os, glob, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, iirnotch

# ===================== 配置 =====================
TRIAL_DIR = r"D:\zx\toll-box\code\EEG_fatigue\data\zyp1\20251101_014305\eeg"  # ← 改成你的试次目录
OUT_DIR   = r"D:\zx\toll-box\code\EEG_fatigue\fatigue_analysis"

# CSV 列名（按你的文件）
TS_COL  = "Timestamp"
CH1_COL = "Channel1"   # Fp1
CH2_COL = "Channel2"  # Fp2

# 采样 & 分窗
FS = 500.0
WIN_SEC  = 2.0
STEP_SEC = 2.0   # 如需 1s 步长重叠，改为 1.0
# 预处理
USE_PREPROCESS = True
BANDPASS = (0.5, 45.0)
NOTCH_F, NOTCH_Q = 60.0, 30.0
# =================================================


# --------- 工具：CSV 选择 + 时间解析 ----------
def pick_csv_in_trial(trial_dir: str) -> str:
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

_TIME_RE = re.compile(r'(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)')
def _extract_time_token(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r'(Z|[+-]\d{2}:?\d{2})$', '', s)  # 去时区后缀
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
    return float(s)  # 最后再尝试转浮点


# --------- 预处理 ----------
def notch_filter(sig2):
    b, a = iirnotch(NOTCH_F/(FS/2), NOTCH_Q)
    return filtfilt(b, a, sig2, axis=0)

def bandpass_filter(sig2):
    b, a = butter(4, [BANDPASS[0]/(FS/2), BANDPASS[1]/(FS/2)], btype='band')
    return filtfilt(b, a, sig2, axis=0)

def preprocess(sig2):
    x = bandpass_filter(sig2)
    x = notch_filter(x)
    return x

# --------- 功率 & 比值 ----------
def bandpower_welch(x, fs, band):
    fmin, fmax = band
    f, Pxx = welch(x, fs=fs, window='hann', nperseg=int(fs*2), noverlap=int(fs), detrend=False, scaling='density')
    idx = (f >= fmin) & (f <= fmax)
    return np.trapz(Pxx[idx], f[idx]) + 1e-12

def theta_alpha_beta_ratio(x, fs=FS):
    theta = bandpower_welch(x, fs, (4, 7))
    alpha = bandpower_welch(x, fs, (8, 13))
    beta  = bandpower_welch(x, fs, (13, 30))
    r_tb = theta / (beta  + 1e-12)
    r_ta = theta / (alpha + 1e-12)
    return float(r_tb), float(r_ta)

# --------- 分窗 ----------
def sliding_windows(sig2, tsec, win_sec=2.0, step_sec=2.0):
    n = sig2.shape[0]
    win = int(win_sec * FS)
    step = int(step_sec * FS)
    for s in range(0, max(0, n - win + 1), step):
        e = s + win
        yield s, e, sig2[s:e, :], tsec[s], tsec[e-1]

# --------- Excel 保存 ----------
def try_excel_writer(path_xlsx: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path_xlsx), exist_ok=True)
    try:
        with pd.ExcelWriter(path_xlsx, engine='xlsxwriter') as w:
            df.to_excel(w, index=False, sheet_name='ratios')
        print(f"[SAVE] {path_xlsx} ({len(df)} rows, xlsxwriter)")
    except Exception:
        try:
            with pd.ExcelWriter(path_xlsx, engine='openpyxl') as w:
                df.to_excel(w, index=False, sheet_name='ratios')
            print(f"[SAVE] {path_xlsx} ({len(df)} rows, openpyxl)")
        except Exception:
            csv_path = path_xlsx.replace('.xlsx', '.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"[SAVE] {csv_path} ({len(df)} rows, CSV fallback)")

# --------- 主函数 ----------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_path = pick_csv_in_trial(TRIAL_DIR)
    print(f"[LOAD] CSV = {csv_path}")

    df = pd.read_csv(csv_path)
    need = {TS_COL, CH1_COL, CH2_COL}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV 缺列：{need - set(df.columns)}，现有：{list(df.columns)}")

    # 读取并转成 numpy
    tsec = df[TS_COL].map(time_to_seconds_any).to_numpy(dtype=float)
    sig2 = df[[CH1_COL, CH2_COL]].to_numpy(dtype=float)

    # 预处理（可关）
    if USE_PREPROCESS:
        sig2 = preprocess(sig2)

    # 分窗并计算比值
    rows = []
    for k, (s, e, win2, t0, t1) in enumerate(sliding_windows(sig2, tsec, WIN_SEC, STEP_SEC), start=1):
        x1 = win2[:, 0]
        x2 = win2[:, 1]
        tb1, ta1 = theta_alpha_beta_ratio(x1, FS)
        tb2, ta2 = theta_alpha_beta_ratio(x2, FS)
        rows.append({
            "win_idx": k,
            "t_start_s": round(float(t0 - tsec[0]), 3),   # 相对起点秒
            "t_end_s":   round(float(t1 - tsec[0]), 3),
            "Fp1_theta_beta": tb1,
            "Fp1_theta_alpha": ta1,
            "Fp2_theta_beta": tb2,
            "Fp2_theta_alpha": ta2,
            "Mean_theta_beta":  (tb1 + tb2) / 2.0,
            "Mean_theta_alpha": (ta1 + ta2) / 2.0,
        })

    if not rows:
        print("[WARN] 无有效窗口（可能试次时长太短）。")
        return

    out_df = pd.DataFrame(rows)

    # 保存表格
    xlsx_path = os.path.join(OUT_DIR, "fatigue_ratios_trial15.xlsx")
    try_excel_writer(xlsx_path, out_df)

    # 画图：均值曲线（可按需改成 Fp1/Fp2 单独曲线）
    plt.figure(figsize=(10, 5))
    plt.plot(out_df["win_idx"], out_df["Mean_theta_beta"], label="Mean θ/β")
    plt.plot(out_df["win_idx"], out_df["Mean_theta_alpha"], label="Mean θ/α")
    plt.xlabel("Window Index")
    plt.ylabel("Ratio Value")
    plt.title("Fatigue Trend — Mean θ/β & θ/α (2s windows)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    png_path = os.path.join(OUT_DIR, "fatigue_ratios_trial15.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    print(f"[SAVE] {png_path}")

    # 简要统计提示
    tb_mean = out_df["Mean_theta_beta"].mean()
    ta_mean = out_df["Mean_theta_alpha"].mean()
    print(f"[STAT] Mean θ/β = {tb_mean:.4f}, θ/α = {ta_mean:.4f}")
    print("完成。")

if __name__ == "__main__":
    main()
