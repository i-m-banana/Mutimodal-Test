# -*- coding: utf-8 -*-
"""
online_inference.py
无标签上线测试：直接读 EEG 文件，逐窗输出 0-100 分数。
"""

import os
import joblib
import numpy as np
import pandas as pd
from eeg_utils import FS, preprocess_eeg, read_eeg_txt_two_channels, segment_windows, extract_features_batch

WIN_SEC = 2.0
STEP_SEC = 1.0
EMA_ALPHA = 0.7
TH_UP = 60.0
TH_DN = 50.0

def run_online(file_path, subject_id, scaler, calibrator, out_dir):
    # 读 & 预处理
    raw = read_eeg_txt_two_channels(file_path)
    raw = preprocess_eeg(raw, fs=FS)

    # 分窗
    wins, starts = segment_windows(raw, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC)

    # 提特征
    X, feat_names, mask = extract_features_batch(wins, fs=FS, reject_artifacts=True)
    starts = starts[mask]
    if X.shape[0] == 0:
        print(f"[WARN] 文件 {file_path} 没有有效窗")
        return

    # 全局 scaler
    Xn = scaler.transform(X)

    ema = None
    state = "low"
    rows = []

    for i in range(Xn.shape[0]):
        proba = calibrator.predict_proba(Xn[i:i+1])[:,1][0]
        score_raw = 100.0 * float(proba)
        ema = score_raw if ema is None else EMA_ALPHA*ema + (1-EMA_ALPHA)*score_raw

        # 迟滞判定
        prev = state
        if state == "low" and ema >= TH_UP:
            state = "high"
        elif state == "high" and ema <= TH_DN:
            state = "low"

        t_start = starts[i]/FS
        t_end = t_start + WIN_SEC

        print(f"[ONLINE] subj={subject_id} t=[{t_start:.2f},{t_end:.2f}] "
              f"raw={score_raw:.2f} ema={ema:.2f} state={state}")
        # {' (↑↓)' if state != prev else ''}
        rows.append({
            "subject": subject_id,
            "t_start_s": round(float(t_start),3),
            "t_end_s": round(float(t_end),3),
            "score_raw": round(score_raw,3),
            "score_ema": round(float(ema),3),
            "state": state
        })

    df = pd.DataFrame(rows)
    # 保存
    base = os.path.basename(file_path).replace(".txt","")
    out_path = os.path.join(out_dir, f"scores_{base}.xlsx")
    try:
        df.to_excel(out_path, index=False)
        print(f"[SAVE] {out_path}")
    except Exception:
        out_path = out_path.replace(".xlsx",".csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_path} (CSV fallback)")

def main():
    base_dir = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(base_dir, "mymodel_scaler.joblib"))
    calibrator = joblib.load(os.path.join(base_dir, "mymodel_calibrator.joblib"))

    out_dir = os.path.join(base_dir, "scores")
    os.makedirs(out_dir, exist_ok=True)

    # ==== 指定要测试的 EEG 文件 ====
    test_files = [
        r"D:\zx\toll-box\code\brain_load\raw_data\Stroop_Data\natural-15.txt",
        # 你也可以改成自己的 test_xxx.txt
    ]

    for f in test_files:
        run_online(f, subject_id="test", scaler=scaler, calibrator=calibrator, out_dir=out_dir)

if __name__ == "__main__":
    main()
