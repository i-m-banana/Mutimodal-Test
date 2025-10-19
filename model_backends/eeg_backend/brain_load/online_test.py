# -*- coding: utf-8 -*-
"""
online_test.py
被试14、15做“在线”测试：
1) 对每个 (subject, task) 先用前 30 秒的信号做个人基线（per-subject z-norm）。
2) 2s窗、1s步上线推理，输出经校准的 0–100 分（含 EMA 平滑与迟滞）。
3) 逐窗打印实时结果；同时保存为 Excel（自动选择 xlsxwriter/openpyxl；都没有则保存为 CSV）。
"""

import os
import time
import joblib
import numpy as np
import pandas as pd

from eeg_utils import (
    FS, preprocess_eeg, read_eeg_txt_two_channels, segment_windows,
    extract_features_batch, LABEL_MAP_4
)

ROOT = r"D:\zx\toll-box\code\brain_load\raw_data"
OUT_DIR = r"D:\zx\toll-box\code\brain_load\scores"

WIN_SEC = 2.0
STEP_SEC = 1.0
EMA_ALPHA = 0.7

# 迟滞阈值（如用个人分位点可自行替换）
TH_UP = 60.0   # 低->高
TH_DN = 50.0   # 高->低

LABEL_ORDER = ['natural', 'lowlevel', 'midlevel', 'highlevel']
TASKS = ['Arithmetic_Data', 'Stroop_Data']
SUBJECTS_TEST = [14, 15]


def try_excel_writer(path_xlsx: str, df: pd.DataFrame):
    """优先存 Excel；没有对应引擎时回退为 CSV。"""
    # 先尝试 xlsxwriter
    try:
        with pd.ExcelWriter(path_xlsx, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='scores')
        print(f"[SAVE] {path_xlsx}  共 {len(df)} 行 (xlsxwriter)")
        return
    except Exception as e1:
        # 尝试 openpyxl
        try:
            with pd.ExcelWriter(path_xlsx, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='scores')
            print(f"[SAVE] {path_xlsx}  共 {len(df)} 行 (openpyxl)")
            return
        except Exception as e2:
            # 回退为 CSV
            path_csv = path_xlsx.replace('.xlsx', '.csv')
            df.to_csv(path_csv, index=False, encoding='utf-8-sig')
            print(f"[SAVE] {path_csv}  共 {len(df)} 行 (fallback CSV)")
            print(f"[WARN] Excel 写入失败：{type(e1).__name__} / {type(e2).__name__}，已回退为 CSV。")


def collect_first_30s_windows_for_baseline(task: str, sid: int):
    """
    为 (task, sid) 采样前 30s 的窗用于基线估计。
    策略：按 LABEL_ORDER 依次找第一个存在的文件，从其开头取够 30s。
    若该文件不足 30s，就取它能提供的全部窗（t_end<=30s）。
    返回：Xn_base (N_base, D), starts_base (indices), feat_names
    """
    total_needed_end = 30.0
    feat_names = None
    X_base_list = []

    for lab in LABEL_ORDER:
        fpath = os.path.join(ROOT, task, f"{lab}-{sid}.txt")
        if not os.path.isfile(fpath):
            continue
        # 读 + 预处理
        raw = read_eeg_txt_two_channels(fpath)
        raw = preprocess_eeg(raw, fs=FS)

        # 分窗
        wins, starts = segment_windows(raw, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC)
        # 提特征
        X, fnames, mask = extract_features_batch(wins, fs=FS, reject_artifacts=True)
        if X.shape[0] == 0:
            continue
        feat_names = feat_names or fnames
        # 选取 t_end<=30s 的窗
        starts = starts[mask]
        t_start = starts / FS
        t_end = t_start + WIN_SEC
        sel = t_end <= total_needed_end
        if not np.any(sel):
            # 如果一个窗都不满足 t_end<=30s，至少取第一个窗做兜底
            sel = np.zeros_like(t_end, dtype=bool)
            sel[0] = True
        X_base_list.append(X[sel])

        # 只用第一个找到的文件做基线（足够简单稳定）
        break

    if len(X_base_list) == 0:
        return np.zeros((0, 0)), None  # 没有任何窗，外部做兜底
    X_base = np.vstack(X_base_list)
    return X_base, feat_names


def compute_subject_znorm_params(Xn_base: np.ndarray):
    """
    在“全局 scaler 之后”的特征空间上，计算个人基线的均值/方差。
    返回 (mu, std)。若无有效基线，返回 (0,1) 向量（不改变分布）。
    """
    if Xn_base.size == 0:
        return None, None
    mu = np.mean(Xn_base, axis=0)
    std = np.std(Xn_base, axis=0)
    std[std < 1e-6] = 1.0  # 防 0
    return mu, std


def run_online_for_subject_task(sid: int, task: str, calibrator, scaler):
    """
    先用前30s做个人基线（per-subject z-norm），再对该 task 下四类文件逐窗推理。
    返回 DataFrame。
    """
    rows_all = []

    # 1) —— 个人基线（在全局 scaler 之后再做一层 per-subject z-norm）
    # 收集基线窗（特征还未做全局 scaler）
    X_base_raw, feat_names = collect_first_30s_windows_for_baseline(task, sid)

    if X_base_raw.size == 0:
        print(f"[BASELINE][WARN] task={task}, subject={sid} 没有可用的基线窗，将不做个人 z-norm。")
        mu_s, std_s = None, None
    else:
        # 全局 scaler 变换后，计算个人均值/方差
        Xn_base = scaler.transform(X_base_raw)
        mu_s, std_s = compute_subject_znorm_params(Xn_base)
        print(f"[BASELINE] task={task}, subject={sid}  基线窗数={Xn_base.shape[0]}  "
              f"示例均值前3={mu_s[:3] if mu_s is not None else 'NA'}")

    # 2) —— 上线逐窗推理
    for lab in LABEL_ORDER:
        fpath = os.path.join(ROOT, task, f"{lab}-{sid}.txt")
        if not os.path.isfile(fpath):
            continue

        # 读+预处理
        raw = read_eeg_txt_two_channels(fpath)
        raw = preprocess_eeg(raw, fs=FS)

        # 分窗
        wins, starts = segment_windows(raw, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC)

        # 提特征
        X, feat_names2, mask = extract_features_batch(wins, fs=FS, reject_artifacts=True)
        starts = starts[mask]
        if X.shape[0] == 0:
            continue

        # 全局归一化
        Xn = scaler.transform(X)

        # 个人 z-norm（如果有基线）
        if mu_s is not None and std_s is not None:
            Xs = (Xn - mu_s) / (std_s + 1e-6)
        else:
            Xs = Xn

        # 在线模拟：逐窗预测 + EMA + 迟滞，并逐行打印
        ema = None
        state = 'low'
        for i in range(Xs.shape[0]):
            proba = calibrator.predict_proba(Xs[i:i+1])[:, 1][0]  # 已校准概率
            score_raw = 100.0 * float(proba)

            # EMA 平滑
            if ema is None:
                ema = score_raw
            else:
                ema = EMA_ALPHA * ema + (1.0 - EMA_ALPHA) * score_raw

            # 迟滞成离散（如果你要看状态）
            prev_state = state
            if state == 'low' and ema >= TH_UP:
                state = 'high'
            elif state == 'high' and ema <= TH_DN:
                state = 'low'

            t_start = starts[i] / FS
            t_end = t_start + WIN_SEC

            # —— 逐窗打印（你也可以改成仅打印某些字段）
            print(f"[ONLINE] task={task:<15} sub={sid:>2} lab={lab:<9} "
                  f"t=[{t_start:6.2f},{t_end:6.2f}]  "
                  f"score_raw={score_raw:6.2f}  score_ema={ema:6.2f}  state={state}"
                  f"{' (↑)' if state!=prev_state else ''}")

            rows_all.append({
                'task': task,
                'label_name': lab,
                'subject': sid,
                't_start_s': round(float(t_start), 3),
                't_end_s': round(float(t_end), 3),
                'score_raw': round(score_raw, 3),
                'score_ema': round(float(ema), 3),
                'state_hysteresis': state
            })

            # 如果希望“真·每秒刷新”，解开下一行
            # time.sleep(STEP_SEC)

    df = pd.DataFrame(rows_all)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载模型（全局 scaler + 概率校准器）
    base_dir = os.path.dirname(__file__)
    scaler = joblib.load(os.path.join(base_dir, "mymodel_scaler.joblib"))
    calibrator = joblib.load(os.path.join(base_dir, "mymodel_calibrator.joblib"))

    for task in TASKS:
        for sid in SUBJECTS_TEST:
            print(f"\n========== [RUN] task={task}, subject={sid} ==========")
            df = run_online_for_subject_task(sid, task, calibrator, scaler)
            if df.empty:
                print(f"[WARN] 无有效数据: task={task}, subject={sid}")
                continue

            out_xlsx = os.path.join(OUT_DIR, f"scores_{task}_subj{sid}.xlsx")
            try_excel_writer(out_xlsx, df)

    print("\n完成。")


if __name__ == "__main__":
    main()
