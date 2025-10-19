# -*- coding: utf-8 -*-
"""
train_offline.py
使用 1-14 试次“处理后的数据”（npz: X,y）离线训练。
- 模型: RobustScaler + LogisticRegression(L2, class_weight=balanced) +
        CalibratedClassifierCV(sigmoid, cv=5)
- 兼容 sklearn 新旧版本的 CalibratedClassifierCV 参数(estimator/base_estimator)
- 输出: mymodel_scaler.joblib, mymodel_clf.joblib(可选), mymodel_calibrator.joblib, feature_names.json
"""

import os
import json
import argparse
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, f1_score, classification_report,
                             accuracy_score, precision_score, recall_score)


# -------- 默认路径（按需修改） --------
DEFAULT_NPZ = r"D:\zx\toll-box\code\brain_load\processed_single_subject\subj_train_1to14.npz"
DEFAULT_OUT = r"D:\zx\toll-box\code\brain_load\models_subj1"
# ------------------------------------


def save_feature_names(path_json, names):
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump({"feature_names": list(names)}, f, ensure_ascii=False, indent=2)


def make_calibrator(clf, method="sigmoid", cv=5):
    """
    兼容 sklearn 新旧版本：
      - 新版本: CalibratedClassifierCV(estimator=..., ...)
      - 旧版本: CalibratedClassifierCV(base_estimator=..., ...)
    """
    try:
        return CalibratedClassifierCV(estimator=clf, method=method, cv=cv)
    except TypeError:
        return CalibratedClassifierCV(base_estimator=clf, method=method, cv=cv)


def get_inner_estimator(calib):
    """
    兼容地取出校准器内部的基分类器（可选保存，便于排查）。
    新版本属性名为 estimator，旧版本为 base_estimator。
    """
    return getattr(calib, "estimator", None) or getattr(calib, "base_estimator", None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default=DEFAULT_NPZ,
                        help="包含 X,y 的 npz 路径（train_only_1to14.npz）")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT,
                        help="模型输出目录")
    parser.add_argument("--cv", type=int, default=5,
                        help="CalibratedClassifierCV 的折数，默认 5")
    parser.add_argument("--C", type=float, default=1.0,
                        help="LogisticRegression 的正则强度 C，默认 1.0")
    parser.add_argument("--max_iter", type=int, default=500,
                        help="LogisticRegression 最大迭代次数，默认 500")
    args = parser.parse_args()

    npz_path = args.npz
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # 1) 读取处理后的训练数据
    data = np.load(npz_path, allow_pickle=True)
    if "X" not in data or "y" not in data:
        raise RuntimeError(f"{npz_path} 不包含 'X' 或 'y'。")
    X = data["X"]
    y = data["y"].astype(int)

    # 特征名（可选）
    if "feature_names" in data:
        feat_names = [str(n) for n in data["feature_names"].tolist()]
    else:
        feat_names = [f"f{i}" for i in range(X.shape[1])]
    print(f"[DATA] X={X.shape}, y={y.shape}, 正类占比={y.mean():.3f}")

    # 2) 归一化
    scaler = RobustScaler().fit(X)
    Xn = scaler.transform(X)

    # 3) 训练逻辑回归（基础分类器）
    base_clf = LogisticRegression(
        penalty="l2", C=args.C, class_weight="balanced",
        max_iter=args.max_iter, solver="lbfgs"
    )

    # 4) 概率校准（Platt sigmoid + CV）—— 兼容写法
    calib = make_calibrator(base_clf, method="sigmoid", cv=args.cv)
    calib.fit(Xn, y)

    # 5) 训练集上做一个 sanity check（便于确认训练是否正常）
    proba = calib.predict_proba(Xn)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y, proba)
    f1 = f1_score(y, y_pred)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)

    print(f"[TRAIN] AUC={auc:.3f}, F1={f1:.3f}, ACC={acc:.3f}, P={prec:.3f}, R={rec:.3f}")
    print(classification_report(y, y_pred, digits=3))

    # 6) 保存
    joblib.dump(scaler, os.path.join(out_dir, "mymodel_scaler.joblib"))
    base_est = get_inner_estimator(calib)
    if base_est is not None:
        joblib.dump(base_est, os.path.join(out_dir, "mymodel_clf.joblib"))
    joblib.dump(calib, os.path.join(out_dir, "mymodel_calibrator.joblib"))
    save_feature_names(os.path.join(out_dir, "feature_names.json"), feat_names)

    print("已保存：")
    print(" -", os.path.join(out_dir, "mymodel_scaler.joblib"))
    if base_est is not None:
        print(" -", os.path.join(out_dir, "mymodel_clf.joblib"))
    print(" -", os.path.join(out_dir, "mymodel_calibrator.joblib"))
    print(" -", os.path.join(out_dir, "feature_names.json"))


if __name__ == "__main__":
    main()


# # -*- coding: utf-8 -*-
# """
# train_offline.py
# 使用被试 1-13（Arithmetic + Stroop）训练双通道二分类模型（低 vs 高）；
# 保存 scaler、分类器、概率校准器和特征名。
# """
#
# import os
# import joblib
# import numpy as np
# from sklearn.preprocessing import RobustScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import roc_auc_score, f1_score, classification_report
# from eeg_utils import (FS, preprocess_eeg, read_eeg_txt_two_channels, segment_windows,
#                        extract_features_batch, iter_task_files,
#                        LABEL_MAP_4, LOW_SET, HIGH_SET, save_feature_names)
#
# ROOT = r"D:\zx\toll-box\code\brain_load\raw_data"
# OUT_DIR = r"D:\zx\toll-box\code\brain_load"
#
# WIN_SEC = 2.0
# STEP_SEC = 1.0
#
# def build_dataset(subject_train=range(1,14)):
#     X_all = []
#     y_all = []
#     for task, lab_name, sid, fpath in iter_task_files(ROOT):
#         if sid not in subject_train:
#             continue
#         # 读与预处理
#         raw = read_eeg_txt_two_channels(fpath)
#         raw = preprocess_eeg(raw, fs=FS)
#
#         # 分窗
#         wins, _ = segment_windows(raw, fs=FS, win_sec=WIN_SEC, step_sec=STEP_SEC)
#
#         # 提特征（带伪迹剔除）
#         X, feat_names, mask = extract_features_batch(wins, fs=FS, reject_artifacts=True)
#         if X.shape[0] == 0:
#             continue
#
#         # 标签：低(0)=natural/lowlevel，高(1)=midlevel/highlevel
#         y4 = LABEL_MAP_4[lab_name]
#         y_bin = 1 if y4 in HIGH_SET else 0
#         y = np.full((X.shape[0],), y_bin, dtype=np.int64)
#
#         X_all.append(X)
#         y_all.append(y)
#
#     if len(X_all) == 0:
#         raise RuntimeError("训练集为空，请检查路径与文件。")
#     X_all = np.vstack(X_all)
#     y_all = np.concatenate(y_all)
#     return X_all, y_all, feat_names
#
#
# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     print(f"[SCAN] ROOT={ROOT}")
#
#     X, y, feat_names = build_dataset(subject_train=range(1,14))
#     print(f"[DATA] X={X.shape}, y={y.shape}, 正类占比={y.mean():.3f}")
#
#     # 全局鲁棒归一化（对异常更稳）
#     scaler = RobustScaler().fit(X)
#     Xn = scaler.transform(X)
#
#     # 线性逻辑回归（稳定、可解释、输出概率）
#     clf = LogisticRegression(
#         penalty='l2', C=1.0, class_weight='balanced', max_iter=500, solver='lbfgs'
#     ).fit(Xn, y)
#
#     # 概率校准（Platt sigmoid）
#     calib = CalibratedClassifierCV(clf, method='sigmoid', cv=5)  # 用CV做校准更稳
#     calib.fit(Xn, y)
#
#     # 简单训练集指标（仅做 sanity check）
#     proba = calib.predict_proba(Xn)[:,1]
#     auc = roc_auc_score(y, proba)
#     f1  = f1_score(y, (proba>=0.5).astype(int))
#     print(f"[TRAIN] AUC={auc:.3f}, F1={f1:.3f}")
#     print(classification_report(y, (proba>=0.5).astype(int), digits=3))
#
#     # 保存
#     joblib.dump(scaler, os.path.join(OUT_DIR, "mymodel_scaler.joblib"))
#     joblib.dump(clf,    os.path.join(OUT_DIR, "mymodel_clf.joblib"))
#     joblib.dump(calib,  os.path.join(OUT_DIR, "mymodel_calibrator.joblib"))
#     save_feature_names(os.path.join(OUT_DIR, "feature_names.json"), feat_names)
#     print(f"已保存: mymodel_scaler.joblib, mymodel_clf.joblib, mymodel_calibrator.joblib, feature_names.json")
#
# if __name__ == "__main__":
#     main()
