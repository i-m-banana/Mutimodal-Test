# -*- coding: utf-8 -*-
"""
eval_online_fatigue_quantile.py
在线评估：基于训练保存的 q_lo/q_hi 做概率→分数映射
- score = clip((prob - q_lo)/(q_hi - q_lo)*100, 0, 100)
- 若 q_hi==q_lo，则回退 score = clip(prob*100, 0, 100)

依赖：baseline_manager.py（用于被试级基线门控更新）
"""

import os, glob, json
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import butter, filtfilt, iirnotch, welch
import joblib

# === 模型参数存放路径 ===
MODEL_DIR     = r"D:\zx\toll-box\code\EEG_fatigue\models"
# === 在线数据存放路径（需读取时间戳和.csv) ===
TRIAL_DIR     = r"D:\zx\toll-box\code\EEG_fatigue\online_textdata\shh1\20251031_235314"
#此处需读取被试信息，调取models\baselines命名只需名字缩写
SUBJECT_BASE  = "shh"
QC_IS_LOWLOAD = True
# =====================

FS     = 500.0
BANDS  = dict(theta=(4,7), alpha=(8,12), beta=(13,30))
BP     = (0.5,45)
NOTCH  = 50.0
AMP_UV        = 120.0
AMP_BAD_RATIO = 0.30

def parse_iso(s): return datetime.fromisoformat(s)
def read_triggers(path):
    with open(path, "r", encoding="utf-8") as f: arr = json.load(f)
    return {int(x["call_index"]): parse_iso(x["datetime"]) for x in arr}
def read_eeg_csv(path):
    df = pd.read_csv(path); df["dt"]=pd.to_datetime(df["Timestamp"])
    return df["dt"].values, df["Channel1"].values.astype(float), df["Channel2"].values.astype(float)
def find_trial_files(trial_dir):
    eeg_dir=os.path.join(trial_dir,"eeg")
    if not os.path.isdir(eeg_dir): raise RuntimeError(f"未找到目录：{eeg_dir}")
    pj=os.path.join(eeg_dir,"part_timestamps.json")
    csvs=[p for p in glob.glob(os.path.join(eeg_dir,"*.csv")) if "kss" not in os.path.basename(p).lower()]
    if not (os.path.exists(pj) and csvs): raise RuntimeError("缺少 part_timestamps.json 或 eeg csv 文件")
    return pj, sorted(csvs)[-1]

def bandpass_filter(x, fs, lo, hi, order=4):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype="band"); return filtfilt(b,a,x)
def notch_filter(x, fs, f0=50.0, Q=30.0):
    b,a=iirnotch(w0=f0/(fs/2),Q=Q); return filtfilt(b,a,x)
def preprocess_pair(ch1, ch2, fs=FS):
    ch1=bandpass_filter(ch1-np.mean(ch1),fs,BP[0],BP[1]); ch1=notch_filter(ch1,fs,NOTCH)
    ch2=bandpass_filter(ch2-np.mean(ch2),fs,BP[0],BP[1]); ch2=notch_filter(ch2,fs,NOTCH)
    return ch1, ch2
def seg_by_time(dt,ch1,ch2,t0,t1):
    m=(dt>=np.datetime64(t0))&(dt<=np.datetime64(t1))
    if m.sum()<int(FS*10): return None,None
    return ch1[m], ch2[m]

def welch_bp(x, fs, band, nperseg=1024):
    f,Pxx=welch(x, fs=fs, nperseg=nperseg); lo,hi=band; m=(f>=lo)&(f<=hi)
    return np.trapz(Pxx[m], f[m])
def bandpowers_2ch(x1,x2,fs=FS):
    out={}
    for k,(lo,hi) in BANDS.items():
        out[k]=0.5*(welch_bp(x1,fs,(lo,hi))+welch_bp(x2,fs,(lo,hi)))
    return out
def ratios(p):
    return p["theta"]/max(p["beta"],1e-9), p["theta"]/max(p["alpha"],1e-9)

def robust_baseline_stats(b1,b2,fs=FS, mad_floor=1e-4):
    win=int(2*fs); step=win
    vals_tb, vals_ta = [], []
    for i in range(0, len(b1)-win+1, step):
        p = bandpowers_2ch(b1[i:i+win], b2[i:i+win], fs=fs)
        rtb,rta = ratios(p)
        vals_tb.append(rtb); vals_ta.append(rta)
    if not vals_tb: return None
    tb,ta = np.array(vals_tb), np.array(vals_ta)
    med_tb=float(np.median(tb)); mad_tb=float(max(mad_floor, np.median(np.abs(tb - med_tb))))
    med_ta=float(np.median(ta)); mad_ta=float(max(mad_floor, np.median(np.abs(ta - med_ta))))
    return dict(med_tb=med_tb, mad_tb=mad_tb, med_ta=med_ta, mad_ta=mad_ta)

def amplitude_bad_ratio(x1,x2,thr_uv=AMP_UV):
    amp=np.maximum(np.abs(x1), np.abs(x2))
    return float((amp>thr_uv).mean())

def prob_to_score_quantile(prob, q_lo, q_hi):
    if q_hi <= q_lo + 1e-9:
        return float(np.clip(prob*100.0, 0.0, 100.0))
    score = (prob - q_lo) / (q_hi - q_lo) * 100.0
    return float(np.clip(score, 0.0, 100.0))

def main():
    from baseline_manager import BaselineManager
    pack = joblib.load(os.path.join(MODEL_DIR, "fatigue_model.joblib"))
    bm   = BaselineManager(MODEL_DIR)

    pj, csv_path = find_trial_files(TRIAL_DIR)
    trig = read_triggers(pj)
    if not all(k in trig for k in (0,1,2,3)):
        raise RuntimeError("触发不完整，需要 0/1/2/3。")

    dt, ch1, ch2 = read_eeg_csv(csv_path)
    ch1, ch2 = preprocess_pair(ch1, ch2)

    # 1) 基线候选（0–1）
    b1,b2 = seg_by_time(dt,ch1,ch2,trig[0],trig[1])
    if b1 is None: raise RuntimeError("基线段过短或缺失。")
    cand = robust_baseline_stats(b1,b2)
    bad_ratio_base = amplitude_bad_ratio(b1,b2)
    qc = {"is_lowload": bool(QC_IS_LOWLOAD), "bad_ratio": bad_ratio_base}
    upd = bm.gated_update(SUBJECT_BASE, cand, qc=qc)
    stats = bm.get(SUBJECT_BASE) or cand

    # 2) SART 段（2–3）
    s1,s2 = seg_by_time(dt,ch1,ch2,trig[2],trig[3])
    if s1 is None: raise RuntimeError("SART 段缺失。")
    bad_ratio_task = amplitude_bad_ratio(s1,s2)

    p_task = bandpowers_2ch(s1,s2)
    Rt1,Rt2 = ratios(p_task)
    z1 = (Rt1 - stats["med_tb"]) / stats["mad_tb"]
    z2 = (Rt2 - stats["med_ta"]) / stats["mad_ta"]

    # 3) 概率 → 分位数映射分数
    w = pack.get("w",[0.6,0.4])
    F_raw = w[0]*z1 + w[1]*z2
    Fb = pack["base"].transform(np.array(F_raw).reshape(-1,1))
    prob = pack["clf"].predict_proba(Fb.reshape(-1,1))[:,1][0]

    q_lo = pack.get("q_lo", 0.0)
    q_hi = pack.get("q_hi", 1.0)
    score = prob_to_score_quantile(prob, q_lo, q_hi)

    out = dict(
        subject_base=SUBJECT_BASE,
        trial=os.path.basename(TRIAL_DIR),
        fatigue_score=round(score,2),
        prob=round(float(prob),4),
        q_lo=round(float(q_lo),4),
        q_hi=round(float(q_hi),4),
        z_theta_beta=round(z1,3), z_theta_alpha=round(z2,3),
        Rt_theta_beta=Rt1, Rt_theta_alpha=Rt2,
        bad_ratio_base=round(bad_ratio_base,3),
        bad_ratio_task=round(bad_ratio_task,3),
        baseline_update=upd.get("updated") if isinstance(upd,dict) else None,
        baseline_reason=upd.get("reason") if isinstance(upd,dict) else None,
    )

    print("="*72)
    print(f"🧠  Subject: {SUBJECT_BASE} | Trial: {os.path.basename(TRIAL_DIR)}")
    print(f"   Fatigue Score: {score:.2f} (0–100, 越高越疲劳)")
    print(f"   prob={prob:.4f}  q_lo={q_lo:.4f}  q_hi={q_hi:.4f}")
    print(f"   θ/β={Rt1:.3f}  θ/α={Rt2:.3f} | zθ/β={z1:.3f}  zθ/α={z2:.3f}")
    print(f"   BadRatio(base={bad_ratio_base:.2f}, task={bad_ratio_task:.2f})")
    print(f"   Baseline Update: {upd.get('updated')} ({upd.get('reason')})")
    print("="*72)

    save_path = os.path.join(TRIAL_DIR, "fatigue_report_online.csv")
    pd.DataFrame([out]).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved → {save_path}")

    return score

if __name__ == "__main__":
    final_score = main()
    print(f"\n✅ Final fatigue_score = {final_score:.2f}")
