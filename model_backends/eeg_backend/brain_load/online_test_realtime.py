# -*- coding: utf-8 -*-
"""
online_test_realtime.py
从 recorder_sim.get_recent_2s() 接收“每2s的一段2s窗口”（500Hz×2s=1000点），
完成：预处理 -> 特征 -> scaler -> calibrator（已校准概率） -> 打分/EMA/迟滞，
实时打印并保存为 Excel（若无引擎回退 CSV）。

运行前：
1) 先启动模拟器，让它指向“第15次试次”的目录：
   from recorder_sim import start_simulator
   start_simulator(r"D:\zx\toll-box\code\brain_load\raw_data\Data\1\15")

2) 再运行本脚本，或在本脚本中直接调用 start_simulator 路径（已放在 main() 里）。
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch, coherence, get_window
from scipy.stats import kurtosis

from recorder_sim import start_simulator, stop_simulator, get_recent_2s

# ================= 配置 =================
FS = 500
BANDPASS = (0.5, 45.0)
NOTCH_F, NOTCH_Q = 60.0, 30.0

EMA_ALPHA = 0.7
TH_UP = 60.0
TH_DN = 50.0

# MODEL_DIR = r"D:\zx\toll-box\code\brain_load\models_subj1"
# OUT_DIR   = r"D:\zx\toll-box\code\brain_load\scores"
# TRIAL15_DIR = r"D:\zx\toll-box\code\brain_load\raw_data\Data\1\16"  # ← 修改成你的“15次试次”目录

print(os.getcwd())

MODEL_DIR = r"brain_load\models_subj1"
OUT_DIR   = r"brain_load\scores"
TRIAL15_DIR = r"brain_load\raw_data\Data\1\16"  # ← 修改成你的“15次试次”目录
# =======================================

# ---------- 预处理 ----------
def notch_filter(sig):
    b, a = iirnotch(NOTCH_F/(FS/2), NOTCH_Q)
    return filtfilt(b, a, sig, axis=0)

def bandpass_filter(sig):
    b, a = butter(4, [BANDPASS[0]/(FS/2), BANDPASS[1]/(FS/2)], btype='band')
    return filtfilt(b, a, sig, axis=0)

def preprocess(sig2):
    return notch_filter(bandpass_filter(sig2))

def is_artifact_window(win2, amp_mad_thresh=5.0, kurt_thresh=8.0):
    for ch in range(win2.shape[1]):
        x = win2[:, ch]
        mad = np.median(np.abs(x - np.median(x))) + 1e-9
        amp_ok = (np.max(np.abs(x)) <= amp_mad_thresh * mad)
        kurt_ok = (kurtosis(x, fisher=True, bias=False) <= kurt_thresh)
        if not amp_ok or not kurt_ok:
            return True
    return False

# ---------- 特征 ----------
def bandpower(x, lo, hi):
    f, Pxx = welch(x, fs=FS, window='hann', nperseg=int(1.0*FS),
                   noverlap=int(0.5*FS), detrend=False, scaling='density')
    idx = (f>=lo)&(f<=hi)
    return np.trapz(Pxx[idx], f[idx]) + 1e-12

def spectral_entropy(x):
    f, Pxx = welch(x, fs=FS, window='hann', nperseg=int(1.0*FS),
                   noverlap=int(0.5*FS), detrend=False, scaling='density')
    Pxx = Pxx[(f>=1)&(f<=45)]
    Pxx /= (np.sum(Pxx)+1e-12)
    H = -np.sum(Pxx*np.log(Pxx+1e-12))
    return H/np.log(len(Pxx)+1e-12)

def hjorth(x):
    dx = np.diff(x, prepend=x[0])
    ddx = np.diff(dx, prepend=dx[0])
    var0,var1,var2 = np.var(x),np.var(dx),np.var(ddx)
    act=var0
    mob=np.sqrt(var1/var0)
    comp=np.sqrt(var2/var1)/(mob+1e-12)
    return act,mob,comp

def coh(x1,x2):
    f, Cxy = coherence(x1,x2,fs=FS,window=get_window('hann',int(1.0*FS)),
                       nperseg=int(1.0*FS),noverlap=int(0.5*FS))
    def meanband(lo,hi):
        idx=(f>=lo)&(f<=hi)
        return np.mean(Cxy[idx]) if np.any(idx) else 0.0
    return meanband(8,12), meanband(13,30)

def extract_features_one_window(win2):
    x1,x2=win2[:,0],win2[:,1]; xd=x2-x1
    feats,names=[],[]
    # log bandpowers
    for sig,tag in [(x1,'fp1'),(x2,'fp2'),(xd,'diff')]:
        for bname,(lo,hi) in {'theta':(4,7),'alpha':(8,12),'beta':(13,30),'gamma':(30,45)}.items():
            feats.append(np.log(bandpower(sig,lo,hi))); names.append(f'logbp_{tag}_{bname}')
    # ratios
    def lbp(sig,lo,hi): return np.log(bandpower(sig,lo,hi))
    for sig,tag in [(x1,'fp1'),(x2,'fp2'),(xd,'diff')]:
        lt,la,lb=lbp(sig,4,7),lbp(sig,8,12),lbp(sig,13,30)
        feats += [lt-lb, la-lb, lb - np.log(np.exp(la)+np.exp(lt)+1e-12)]
        names += [f'ratio_{tag}_theta_beta', f'ratio_{tag}_alpha_beta', f'ratio_{tag}_beta_over_alpha_theta']
    # FAA
    FAA = lbp(x2,8,12) - lbp(x1,8,12)
    feats.append(FAA); names.append('FAA_alpha')
    # coherence
    ca,cb = coh(x1,x2); feats += [ca,cb]; names += ['coh_alpha','coh_beta']
    # Hjorth + spectrum entropy
    for sig,tag in [(x1,'fp1'),(x2,'fp2'),(xd,'diff')]:
        a,m,c = hjorth(sig); feats+=[a,m,c]; names += [f'hj_{tag}_act', f'hj_{tag}_mob', f'hj_{tag}_comp']
    feats.append(spectral_entropy((x1+x2)/2)); names.append('spectral_entropy')
    return np.array(feats, dtype=np.float64), names

# ---------- 保存工具 ----------
def try_excel_writer(path_xlsx: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path_xlsx), exist_ok=True)
    try:
        with pd.ExcelWriter(path_xlsx, engine='xlsxwriter') as w:
            df.to_excel(w, index=False, sheet_name='scores')
        print(f"[SAVE] {path_xlsx} ({len(df)} rows, xlsxwriter)")
        return
    except Exception as e1:
        try:
            with pd.ExcelWriter(path_xlsx, engine='openpyxl') as w:
                df.to_excel(w, index=False, sheet_name='scores')
            print(f"[SAVE] {path_xlsx} ({len(df)} rows, openpyxl)")
            return
        except Exception as e2:
            path_csv = path_xlsx.replace('.xlsx', '.csv')
            df.to_csv(path_csv, index=False, encoding='utf-8-sig')
            print(f"[SAVE] {path_csv} ({len(df)} rows, CSV fallback)")

# ---------- 主循环 ----------
def main():
    # 1) 启动模拟器（指向第15次试次的目录）
    start_simulator(TRIAL15_DIR)

    # 2) 加载模型
    scaler = joblib.load(os.path.join(MODEL_DIR, "mymodel_scaler.joblib"))
    calibrator = joblib.load(os.path.join(MODEL_DIR, "mymodel_calibrator.joblib"))

    # 3) 实时循环
    rows = []
    ema = None
    state = 'low'
    n_windows = 0

    while True:
        pkt = get_recent_2s()  # {'timestamps': [], 'ch1': [], 'ch2': []}
        if not pkt['timestamps']:   # 为空 -> 结束
            break

        # 组窗（1000×2）
        ch1 = np.asarray(pkt['ch1'], dtype=np.float64)
        ch2 = np.asarray(pkt['ch2'], dtype=np.float64)
        if ch1.shape[0] != ch2.shape[0] or ch1.shape[0] != 2*FS:
            # 尺寸不对（例如末尾不足2s），跳过
            continue
        win2 = np.stack([ch1, ch2], axis=1)

        # 预处理 + 伪迹
        win2 = preprocess(win2)
        if is_artifact_window(win2):
            # 伪迹窗可选择跳过或打标记，这里选择“跳过”
            continue

        # 特征
        feats, feat_names = extract_features_one_window(win2)
        X = feats.reshape(1, -1)
        Xn = scaler.transform(X)
        proba = float(calibrator.predict_proba(Xn)[:,1][0])
        score_raw = 100.0 * proba

        # EMA & 迟滞
        ema = score_raw if ema is None else (EMA_ALPHA*ema + (1-EMA_ALPHA)*score_raw)
        prev_state = state
        if state == 'low' and ema >= TH_UP:
            state = 'high'
        elif state == 'high' and ema <= TH_DN:
            state = 'low'

        t0, t1 = pkt['timestamps'][0], pkt['timestamps'][-1]
        n_windows += 1

        print(f"[ONLINE] #{n_windows:04d} "
              f"t=[{t0} ~ {t1}]  raw={score_raw:6.2f}  ema={ema:6.2f}  state={state}"
              f"{' (↑)' if state!=prev_state else ''}")

        rows.append({
            'win_index': n_windows,
            't_start': t0,
            't_end': t1,
            'score_raw': round(float(score_raw), 3),
            'score_ema': round(float(ema), 3),
            'state_hysteresis': state
        })

        # 如果你需要“真·每2秒刷一次”，解开下一行（现在默认尽快跑完）
        # time.sleep(2.0)

    # 4) 保存表格
    out_path = os.path.join(OUT_DIR, "EEG_online_scores.xlsx")
    df = pd.DataFrame(rows)
    try_excel_writer(out_path, df)

    stop_simulator()
    print("完成。")

if __name__ == "__main__":
    main()
