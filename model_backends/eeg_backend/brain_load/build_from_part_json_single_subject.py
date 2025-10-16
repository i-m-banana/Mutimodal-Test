# -*- coding: utf-8 -*-
"""
build_from_part_json_single_subject_v3.py
单被试：读取一个被试下的 1..15 个试次，
- 低负荷样本: trigger1 → trigger2
- 高负荷样本: trigger2 → trigger3
- 500 Hz，2s窗/1s步，带通(0.5-45Hz)+60Hz陷波，伪迹剔除，特征提取
- 试次目录内任意 .csv 均可（优先文件名含'eeg'，否则选体积最大）

修复：健壮的时间解析，支持
  'YYYY-MM-DD HH:MM:SS.ffffff'
  'YYYY-MM-DDTHH:MM:SS.ffffff'
  'HH:MM:SS(.fff)' / 'MM:SS(.fff)'
  纯数字秒(float/int)
"""

import os, glob, json, re
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch, welch, coherence, get_window
from scipy.stats import kurtosis

# ========== 配置 ==========
# 把这里改成你的被试路径（该目录下应有 1..15 子文件夹）
DATA_ROOT = r"D:\zx\toll-box\code\brain_load\raw_data\Data\3"
OUT_DIR   = r"D:\zx\toll-box\code\brain_load\processed_single_subject"

# CSV 必需列名（若不同，请在此处改）
TS_COL, CH1_COL, CH2_COL = "Timestamp", "Channel1", "Channel2"

# 采样 & 预处理
FS = 500
BANDPASS = (0.5, 45.0)
NOTCH_F, NOTCH_Q = 60.0, 30.0

# 分窗
WIN_SEC, STEP_SEC = 2.0, 1.0
WIN, STEP = int(WIN_SEC*FS), int(STEP_SEC*FS)

# 触发文件
TRIG_JSON_NAME = "part_timestamps.json"

# trigger 区间定义
IDX_LOW_START, IDX_LOW_END = 1, 2
IDX_HI_START,  IDX_HI_END  = 2, 3
# ==========================


# ---------- 健壮时间解析 ----------
_TIME_RE = re.compile(
    r'(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{1,2}(?:\.\d+)?)'
)

def _extract_time_token(s: str) -> str:
    """
    从 datetime 字符串里提取 time 部分：
    - 'YYYY-MM-DD HH:MM:SS.ffffff' -> 'HH:MM:SS.ffffff'
    - 'YYYY-MM-DDTHH:MM:SS.ffffff' -> 'HH:MM:SS.ffffff'
    - 已经是 'HH:MM:SS(.fff)' / 'MM:SS(.fff)' 直接返回
    - 去掉可能的时区后缀（Z / +08:00）
    """
    s = str(s).strip()
    # 去掉时区后缀
    s = re.sub(r'(Z|[+-]\d{2}:?\d{2})$', '', s)
    if 'T' in s:
        s = s.split('T', 1)[1]
    elif ' ' in s:
        s = s.split()[-1]
    return s

def time_to_seconds_any(x) -> float:
    """
    将各种可能格式的时间转换成秒（从 00:00:00 起算）。
    支持：
      - float/int（直接返回）
      - 'HH:MM:SS(.fff)'
      - 'MM:SS(.fff)'
      - 'YYYY-MM-DD HH:MM:SS(.fff)' / 'YYYY-MM-DDTHH:MM:SS(.fff)'
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = _extract_time_token(str(x))
    m = _TIME_RE.search(s)
    if m:
        h = m.group('h')
        h = int(h) if h is not None else 0
        m_ = int(m.group('m'))
        s_ = float(m.group('s'))
        return h*3600 + m_*60 + s_

    # 兜底：若就是纯数字字符串
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"无法解析时间：{x!r}（解析后：{s!r}）")


# ---------- 预处理/分窗/伪迹 ----------
def notch_filter(sig):
    b, a = iirnotch(NOTCH_F/(FS/2), NOTCH_Q)
    return filtfilt(b, a, sig, axis=0)

def bandpass_filter(sig):
    b, a = butter(4, [BANDPASS[0]/(FS/2), BANDPASS[1]/(FS/2)], btype='band')
    return filtfilt(b, a, sig, axis=0)

def preprocess(sig):
    return notch_filter(bandpass_filter(sig))

def segment_windows(sig):
    N = sig.shape[0]
    out, starts = [], []
    for s in range(0, max(0, N - WIN + 1), STEP):
        out.append(sig[s:s+WIN, :])
        starts.append(s)
    return out, np.array(starts, dtype=np.int64)

def is_artifact_window(win2):
    for ch in range(win2.shape[1]):
        x = win2[:, ch]
        mad = np.median(np.abs(x - np.median(x))) + 1e-9
        if np.max(np.abs(x)) > 5*mad or kurtosis(x, fisher=True, bias=False) > 8:
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

def feat_one(win2):
    x1,x2=win2[:,0],win2[:,1]; xd=x2-x1
    feats,names=[],[]
    for sig,tag in [(x1,'fp1'),(x2,'fp2'),(xd,'diff')]:
        for bname,(lo,hi) in {'theta':(4,7),'alpha':(8,12),'beta':(13,30),'gamma':(30,45)}.items():
            feats.append(np.log(bandpower(sig,lo,hi))); names.append(f'logbp_{tag}_{bname}')
    def lbp(sig,lo,hi): return np.log(bandpower(sig,lo,hi))
    for sig,tag in [(x1,'fp1'),(x2,'fp2'),(xd,'diff')]:
        lt,la,lb=lbp(sig,4,7),lbp(sig,8,12),lbp(sig,13,30)
        feats += [lt-lb, la-lb, lb - np.log(np.exp(la)+np.exp(lt)+1e-12)]
        names += [f'ratio_{tag}_theta_beta', f'ratio_{tag}_alpha_beta', f'ratio_{tag}_beta_over_alpha_theta']
    FAA = lbp(x2,8,12) - lbp(x1,8,12); feats.append(FAA); names.append('FAA_alpha')
    ca,cb = coh(x1,x2); feats += [ca,cb]; names += ['coh_alpha','coh_beta']
    for sig,tag in [(x1,'fp1'),(x2,'fp2'),(xd,'diff')]:
        a,m,c = hjorth(sig); feats += [a,m,c]; names += [f'hj_{tag}_act', f'hj_{tag}_mob', f'hj_{tag}_comp']
    feats.append(spectral_entropy((x1+x2)/2)); names.append('spectral_entropy')
    return np.array(feats), names

def feat_batch(windows):
    X, names = [], None
    for w in windows:
        if is_artifact_window(w): continue
        f, nm = feat_one(w)
        X.append(f); names = nm
    if not X: return np.zeros((0,0)), names or []
    return np.vstack(X), names


# ---------- I/O ----------
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

def read_csv_time(fcsv: str):
    df = pd.read_csv(fcsv)
    need = {TS_COL, CH1_COL, CH2_COL}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV 缺列：{need - set(df.columns)}\n文件={fcsv}\n现有列={list(df.columns)}")
    # 健壮时间解析
    tsec = df[TS_COL].map(time_to_seconds_any).to_numpy(dtype=float)
    sig  = df[[CH1_COL, CH2_COL]].to_numpy(dtype=float)
    return sig, tsec

def read_json_time(fjson: str, call_index: int) -> float:
    data = json.load(open(fjson, 'r', encoding='utf-8'))
    for it in data:
        if isinstance(it, dict) and it.get('call_index') == call_index:
            dt = it.get('datetime', it.get('time', ''))
            return time_to_seconds_any(dt)
    raise ValueError(f"{fjson} 未找到 call_index={call_index}")

def slice_by_time(sig, tsec, t1, t2):
    m = (tsec >= t1) & (tsec < t2)
    return sig[m], tsec[m]


# ---------- 主流程 ----------
def f_batch_label(sig, label):
    if sig.shape[0] < WIN:
        return np.zeros((0,0)), np.zeros((0,), dtype=np.int64)
    wins, _ = segment_windows(sig)
    X, nm = feat_batch(wins)
    y = np.full((X.shape[0],), int(label), dtype=np.int64)
    return X, y

def process_trial(trial_dir):
    fjson = os.path.join(trial_dir, TRIG_JSON_NAME)
    if not os.path.isfile(fjson):
        raise FileNotFoundError(f"缺少触发文件：{fjson}")
    fcsv = pick_csv_in_trial(trial_dir)

    sig_raw, tsec = read_csv_time(fcsv)
    sig = preprocess(sig_raw)

    # 低：1→2，高：2→3
    t1 = read_json_time(fjson, IDX_LOW_START)
    t2 = read_json_time(fjson, IDX_LOW_END)
    t3 = read_json_time(fjson, IDX_HI_END)

    lo, _ = slice_by_time(sig, tsec, t1, t2)
    hi, _ = slice_by_time(sig, tsec, t2, t3)

    Xl, yl = f_batch_label(lo, 0)
    Xh, yh = f_batch_label(hi, 1)

    if Xh.size and Xl.size:
        X = np.vstack([Xh, Xl]); y = np.concatenate([yh, yl])
    elif Xh.size:
        X, y = Xh, yh
    else:
        X, y = Xl, yl
    return X, y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    trial_dirs = [os.path.join(DATA_ROOT, str(i)) for i in range(1, 16)]
    train_dirs, test_dir = trial_dirs[:-1], trial_dirs[-1]

    allX, ally = [], []
    for td in tqdm(train_dirs, desc='build train (1–14)'):
        try:
            X, y = process_trial(td)
        except Exception as e:
            print(f"[ERROR] {td}: {e}")
            continue
        if X.size == 0:
            print(f"[SKIP] {td} 无有效窗")
            continue
        allX.append(X); ally.append(y)

    if not allX:
        print("[WARN] 没有生成任何训练样本。")
        return

    X = np.vstack(allX); y = np.concatenate(ally)
    np.savez_compressed(os.path.join(OUT_DIR, "subj_train_1to14.npz"), X=X, y=y)
    pd.DataFrame(X).assign(label=y).to_csv(os.path.join(OUT_DIR, "subj_train_1to14.csv"),
                                           index=False, encoding="utf-8-sig")
    print(f"[SAVE] train X={X.shape}, y={y.shape}")

    # 第15次试次保留作测试
    try:
        X15, y15 = process_trial(test_dir)
        np.savez_compressed(os.path.join(OUT_DIR, "subj_test_15.npz"), X=X15, y=y15)
        pd.DataFrame(X15).assign(label=y15).to_csv(os.path.join(OUT_DIR, "subj_test_15.csv"),
                                                   index=False, encoding="utf-8-sig")
        print(f"[SAVE] test15 X={X15.shape}, y={y15.shape}")
    except Exception as e:
        print(f"[WARN] 处理第15次试次失败：{e}")

    print("完成。")

if __name__ == "__main__":
    main()
