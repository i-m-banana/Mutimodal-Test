# -*- coding: utf-8 -*-
"""
train_fatigue_calibrator_adv_quantile.py
从 data/ 训练疲劳标定器（逻辑回归），并保存“概率分位数映射”参数 q_lo/q_hi（5%/95%）
- 仅使用触发 0–1(基线30s) 与 2–3(SART)，忽略 4–5 朗读
- 打标：末位 1 → 正类，末位 0 → 负类；SART>=240s 且质控合格
- 被试级基线：聚合未排除试次 0–1 段（中位数+MAD）

输出：
  models/fatigue_model.joblib   （含 clf、base_scaler、q_lo、q_hi；另含 mm 作为备份）
  models/baselines/<subject_base>.json
  models/logs/train_summary.csv
"""

import os, json, glob
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import welch, butter, filtfilt, iirnotch
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
import joblib

# === 训练数据路径，0/1后缀自动识别分类 ===
DATA_ROOT = r"D:\Mutimodal-Test\models_data\eeg_training_data"
# === 训练好的数据模型存放路径，一个被试一套baseline ===
SAVE_DIR  = r"D:\zx\toll-box\code\EEG_fatigue\models"

# （可选）排除的试次目录（相对 DATA_ROOT 或绝对路径）
EXCLUDE_DIRS = set([
    # r"shh0\20251101_090000",
    # r"D:\zx\toll-box\code\EEG_fatigue\data\wsh1\20251029_223012",
])
EXCLUDE_LIST_FILE = None
# 例如 r"D:\zx\toll-box\code\EEG_fatigue\exclude_trials.txt"
# ===================

FS = 500.0
BANDS = dict(theta=(4,7), alpha=(8,12), beta=(13,30))
BP = (0.5,45)
NOTCH = 50.0

MIN_SART_SEC   = 240    # >= 4min
AMP_UV         = 120.0  # μV
AMP_BAD_RATIO  = 0.30

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "baselines"), exist_ok=True)

def _normalize_path(p):
    if not p: return None
    p = p.strip()
    if not p: return None
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(DATA_ROOT, p))

def _load_excludes():
    excl = set()
    for p in EXCLUDE_DIRS:
        q = _normalize_path(p)
        if q: excl.add(q)
    if EXCLUDE_LIST_FILE and os.path.exists(EXCLUDE_LIST_FILE):
        with open(EXCLUDE_LIST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                q = _normalize_path(line)
                if q: excl.add(q)
    return excl

def parse_iso(s): return datetime.fromisoformat(s)
def read_triggers(p):
    with open(p,"r",encoding="utf-8") as f: arr=json.load(f)
    return {int(x["call_index"]): parse_iso(x["datetime"]) for x in arr}

def read_eeg_csv(p):
    df = pd.read_csv(p)
    df["dt"] = pd.to_datetime(df["Timestamp"])
    return df["dt"].values, df["Channel1"].values.astype(float), df["Channel2"].values.astype(float)

def bandpass_filter(x, fs, lo, hi, order=4):
    b,a=butter(order,[lo/(fs/2),hi/(fs/2)],btype='band'); return filtfilt(b,a,x)
def notch_filter(x, fs, f0=50.0, Q=30.0):
    b,a=iirnotch(w0=f0/(fs/2),Q=Q); return filtfilt(b,a,x)
def preprocess_pair(ch1, ch2, fs=FS):
    ch1=bandpass_filter(ch1-np.mean(ch1),fs,BP[0],BP[1]); ch1=notch_filter(ch1,fs,NOTCH)
    ch2=bandpass_filter(ch2-np.mean(ch2),fs,BP[0],BP[1]); ch2=notch_filter(ch2,fs,NOTCH)
    return ch1, ch2

def seg_by_time(dt, ch1, ch2, t0, t1):
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

def list_trials_sorted(user_dir):
    arr=[]
    for name in os.listdir(user_dir):
        p=os.path.join(user_dir,name)
        if not os.path.isdir(p): continue
        try: ts=datetime.strptime(name,"%Y%m%d_%H%M%S")
        except: continue
        arr.append((ts,p))
    arr.sort(key=lambda x:x[0])
    return [td for _,td in arr]

def split_base_cat(user_code): return user_code[:-1], int(user_code[-1])

def find_trial_files(trial_dir):
    eeg_dir=os.path.join(trial_dir,"eeg")
    if not os.path.isdir(eeg_dir): return None,None
    pj=os.path.join(eeg_dir,"part_timestamps.json")
    csvs=[p for p in glob.glob(os.path.join(eeg_dir,"*.csv")) if "kss" not in os.path.basename(p).lower()]
    if os.path.exists(pj) and csvs:
        return pj, sorted(csvs)[-1]
    return None,None

def robust_baseline_stats_from_signal(b1,b2,fs=FS, mad_floor=1e-4):
    win=int(2*fs); step=win
    tb,ta=[],[]
    for i in range(0,len(b1)-win+1,step):
        p=bandpowers_2ch(b1[i:i+win], b2[i:i+win], fs=fs)
        rtb,rta=ratios(p); tb.append(rtb); ta.append(rta)
    if not tb: return None
    tb=np.array(tb); ta=np.array(ta)
    med_tb=float(np.median(tb)); med_ta=float(np.median(ta))
    mad_tb=float(max(mad_floor, np.median(np.abs(tb-med_tb))))
    mad_ta=float(max(mad_floor, np.median(np.abs(ta-med_ta))))
    return dict(med_tb=med_tb, mad_tb=mad_tb, med_ta=med_ta, mad_ta=mad_ta)

def main():
    EXCLUDES = _load_excludes()

    # 1) 被试级基线
    bases=set(u[:-1] for u in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT,u)))
    base_stats={}
    for b in sorted(bases):
        ch1_all, ch2_all = [], []
        for user_code in os.listdir(DATA_ROOT):
            udir=os.path.join(DATA_ROOT,user_code)
            if not os.path.isdir(udir) or not user_code.startswith(b): continue
            for tdir in list_trials_sorted(udir):
                if _normalize_path(tdir) in EXCLUDES:
                    continue
                pj, csv_path = find_trial_files(tdir)
                if not pj: continue
                trig=read_triggers(pj)
                if not all(k in trig for k in (0,1)): continue
                dt,c1,c2=read_eeg_csv(csv_path); c1,c2=preprocess_pair(c1,c2)
                b1,b2=seg_by_time(dt,c1,c2,trig[0],trig[1])
                if b1 is None: continue
                ch1_all.append(b1); ch2_all.append(b2)
        if ch1_all:
            c1=np.concatenate(ch1_all); c2=np.concatenate(ch2_all)
            stats=robust_baseline_stats_from_signal(c1,c2)
            if stats:
                base_stats[b]=stats
                with open(os.path.join(SAVE_DIR,"baselines",f"{b}.json"),"w",encoding="utf-8") as f:
                    json.dump(stats,f,ensure_ascii=False,indent=2)
        else:
            print(f"[WARN] 基名 {b} 暂无可聚合基线")

    # 2) 采样训练集
    rows=[]
    for user_code in os.listdir(DATA_ROOT):
        udir=os.path.join(DATA_ROOT,user_code)
        if not os.path.isdir(udir): continue
        base,cat=split_base_cat(user_code)
        for tdir in list_trials_sorted(udir):
            if _normalize_path(tdir) in EXCLUDES:
                continue
            pj, csv_path = find_trial_files(tdir)
            if not pj: continue
            try:
                trig=read_triggers(pj)
                if not all(k in trig for k in (0,1,2,3)): continue
                dt,c1,c2=read_eeg_csv(csv_path); c1,c2=preprocess_pair(c1,c2)

                # baseline（被试级优先，回退试次级）
                subj_stats = base_stats.get(base, None)
                if subj_stats is None:
                    b1,b2=seg_by_time(dt,c1,c2,trig[0],trig[1])
                    if b1 is None: continue
                    subj_stats = robust_baseline_stats_from_signal(b1,b2)

                s1,s2=seg_by_time(dt,c1,c2,trig[2],trig[3])
                if s1 is None: continue
                L=(trig[3]-trig[2]).total_seconds()
                if L < MIN_SART_SEC: continue

                amp=np.maximum(np.abs(s1), np.abs(s2))
                if (amp>AMP_UV).mean()>AMP_BAD_RATIO:
                    continue

                p_task=bandpowers_2ch(s1,s2); Rt1,Rt2=ratios(p_task)
                z1=(Rt1 - subj_stats["med_tb"])/subj_stats["mad_tb"]
                z2=(Rt2 - subj_stats["med_ta"])/subj_stats["mad_ta"]
                label=1 if cat==1 else 0

                rows.append(dict(user=user_code, base=base, trial=tdir, L=L, cat=cat,
                                 z_theta_beta=z1, z_theta_alpha=z2, label=label))
            except Exception as e:
                print("[WARN] skip", tdir, "->", e)
                continue

    df=pd.DataFrame(rows)
    df.to_csv(os.path.join(SAVE_DIR,"logs","train_summary.csv"),index=False,encoding="utf-8-sig")
    pos=int((df["label"]==1).sum()) if not df.empty else 0
    neg=int((df["label"]==0).sum()) if not df.empty else 0
    print("[DATA] train samples:", len(df), "pos=", pos, "neg=", neg)
    if df.empty or pos==0 or neg==0:
        raise RuntimeError("正负样本不足，请检查数据或排除列表。")

    # 3) 训练：F_raw → Robust → Logistic；然后保存 q_lo/q_hi（5%/95%）
    X=df[["z_theta_beta","z_theta_alpha"]].values
    F_raw=0.6*X[:,0] + 0.4*X[:,1]
    y=df["label"].values.astype(int)

    base_scaler=RobustScaler().fit(F_raw.reshape(-1,1))
    Fb=base_scaler.transform(F_raw.reshape(-1,1)).ravel()

    clf=LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(Fb.reshape(-1,1), y)

    prob = clf.predict_proba(Fb.reshape(-1,1))[:,1]
    q_lo, q_hi = np.quantile(prob, [0.05, 0.95])
    # 备份：MinMax(clip=True)，不强制使用
    mm = MinMaxScaler(feature_range=(0,100), clip=True).fit(prob.reshape(-1,1))

    joblib.dump(dict(model="logit_on_Fraw",
                     clf=clf, base=base_scaler,
                     w=[0.6,0.4],
                     q_lo=float(q_lo), q_hi=float(q_hi),
                     mm=mm),
                os.path.join(SAVE_DIR,"fatigue_model.joblib"))

    with open(os.path.join(SAVE_DIR,"fatigue_feature_def.json"),"w",encoding="utf-8") as f:
        json.dump(dict(fs=FS, bands=BANDS, bandpass=BP, notch=NOTCH,
                       meta=dict(kind="adv_subject_baseline_quantile", w=[0.6,0.4],
                                 q_lo=float(q_lo), q_hi=float(q_hi))),
                  f, ensure_ascii=False, indent=2)

    print("[OK] Saved model & quantiles:", os.path.join(SAVE_DIR,"fatigue_model.joblib"))
    print("[OK] Baselines in:", os.path.join(SAVE_DIR,"baselines"))
    print("[OK] Train log:", os.path.join(SAVE_DIR,"logs","train_summary.csv"))

if __name__ == "__main__":
    main()
