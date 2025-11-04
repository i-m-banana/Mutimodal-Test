"""EEG疲劳度模型 - 直接集成版本

基于脑电信号的疲劳度评估
使用 EEG_fatigue 模块中的在线评估算法
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# 抑制sklearn版本警告和scipy频谱警告
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.signal")

from .base_inference_model import BaseInferenceModel

# 添加 EEG_fatigue 路径
_EEG_FATIGUE_PATH = Path(__file__).parent / "EEG_fatigue"
sys.path.insert(0, str(_EEG_FATIGUE_PATH))

try:
    import joblib
    from scipy.signal import butter, filtfilt, iirnotch, welch
    from baseline_manager import BaselineManager
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    _import_error = e


class EEGFatigueModel(BaseInferenceModel):
    """EEG疲劳度模型（集成版本）
    
    基于双通道EEG信号（Fp1, Fp2）评估疲劳度
    使用频带能量比值和机器学习模型
    
    功能:
    - 处理双通道EEG信号
    - 提取θ/β、θ/α比值
    - 输出疲劳度分数 (0-100)
    - 支持被试级自适应基线更新
    """
    
    # 信号处理参数
    FS = 500.0                  # 采样率
    BANDS = {
        "theta": (4, 7),        # θ波
        "alpha": (8, 12),       # α波
        "beta": (13, 30)        # β波
    }
    BP = (0.5, 45)              # 带通滤波范围
    NOTCH = 50.0                # 陷波频率
    AMP_UV = 120.0              # 幅值阈值(μV)
    AMP_BAD_RATIO = 0.30        # 坏信号比例阈值
    
    # 窗口参数
    WIN_SEC = 2.0               # 窗口长度(秒)
    
    def initialize(self) -> None:
        """初始化EEG疲劳度模型"""
        if not HAS_DEPS:
            raise RuntimeError(f"无法加载依赖: {_import_error}")
        
        # 确定模型路径 - 模型文件已移动到 models_data/eeg_fatigue_models
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models_data" / "eeg_fatigue_models"
        
        model_path = models_dir / "fatigue_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 加载模型
        self.logger.info("加载EEG疲劳度模型...")
        model_pack = joblib.load(str(model_path))
        
        # 模型文件中的键名是 "base" 而不是 "scaler"
        self.scaler = model_pack.get("base") or model_pack.get("scaler")
        if self.scaler is None:
            raise KeyError("模型文件中缺少 'base' 或 'scaler' 键")
        
        self.clf = model_pack["clf"]
        self.q_lo = model_pack.get("q_lo", 0.0)
        self.q_hi = model_pack.get("q_hi", 1.0)
        
        # 权重 w: [theta_beta权重, theta_alpha权重]，用于加权求和
        self.w = model_pack.get("w", [0.6, 0.4])  # 默认0.6*tb + 0.4*ta
        
        # 特征名称（默认为theta/beta和theta/alpha比值）
        self.feature_names = ["theta_beta_ratio", "theta_alpha_ratio"]
        
        self.logger.info(f"  ✓ 特征数量: {len(self.feature_names)}")
        self.logger.info(f"  ✓ 分位数范围: [{self.q_lo:.3f}, {self.q_hi:.3f}]")
        self.logger.info(f"  ✓ 缩放器类型: {type(self.scaler).__name__}")
        
        # 初始化基线管理器
        self.baseline_manager = BaselineManager(str(models_dir))
        self.logger.info("  ✓ 基线管理器初始化完成")
        
        self.logger.info("✅ EEG疲劳度模型初始化完成")
    
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行EEG疲劳度推理
        
        支持三种输入模式：
        1. 内存模式（推荐）：
           - memory_mode: bool = True
           - eeg_signal: np.ndarray - EEG信号数组 [N, 2]
           - sampling_rate: float = 500.0
           - subject_id: str = "unknown"
           - trigger_times: List[datetime] - 可选，触发时间点
           
        2. 文件路径模式：
           - file_mode: bool = True
           - eeg_file_path: str - EEG数据文件路径
           - sampling_rate: float = 500.0
           - subject_id: str = "unknown"
           
        3. 会话目录模式（完整评估）：
           - session_dir: str - 会话目录路径
           - subject_base: str - 被试标识（如"shh"）
           - qc_is_lowload: bool = True - 是否低负荷状态
        
        Args:
            data: 输入数据字典
        
        Returns:
            推理结果:
                - status: "success" | "no-data" | "error"
                - eeg_fatigue_score: 疲劳度分数 (0-100)
                - window_results: 各窗口详细结果
                - baseline_updated: 是否更新了基线
        """
        # 优先使用会话目录模式（完整评估）
        if "session_dir" in data:
            print("会话目录模式------------------------------------")
            return self._infer_from_session(data)
        # 内存模式
        elif data.get("memory_mode") == True:
            print("内存模式------------------------------------")
            return self._infer_from_memory(data)
        # 文件路径模式
        elif data.get("file_mode") == True:
            print("文件路径模式------------------------------------")
            return self._infer_from_file(data)
        else:
            print("未指定有效的输入模式------------------------------------")
            return {
                "status": "error",
                "error": "未指定有效的输入模式",
                "eeg_fatigue_score": 0.0
            }
    
    def _bandpass_filter(self, x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
        """带通滤波"""
        b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        return filtfilt(b, a, x)
    
    def _notch_filter(self, x: np.ndarray, fs: float, f0: float = 50.0, Q: float = 30.0) -> np.ndarray:
        """陷波滤波"""
        b, a = iirnotch(w0=f0 / (fs / 2), Q=Q)
        return filtfilt(b, a, x)
    
    def _preprocess_pair(self, ch1: np.ndarray, ch2: np.ndarray, fs: float) -> tuple:
        """预处理双通道信号"""
        # 去均值 + 带通滤波 + 陷波
        ch1 = self._bandpass_filter(ch1 - np.mean(ch1), fs, self.BP[0], self.BP[1])
        ch1 = self._notch_filter(ch1, fs, self.NOTCH)
        
        ch2 = self._bandpass_filter(ch2 - np.mean(ch2), fs, self.BP[0], self.BP[1])
        ch2 = self._notch_filter(ch2, fs, self.NOTCH)
        
        return ch1, ch2
    
    def _welch_bp(self, x: np.ndarray, fs: float, band: tuple, nperseg: int = 1024) -> float:
        """计算频带能量"""
        f, Pxx = welch(x, fs=fs, nperseg=nperseg)
        lo, hi = band
        m = (f >= lo) & (f <= hi)
        return np.trapz(Pxx[m], f[m])
    
    def _bandpowers_2ch(self, x1: np.ndarray, x2: np.ndarray, fs: float) -> Dict[str, float]:
        """计算双通道频带能量"""
        out = {}
        for k, (lo, hi) in self.BANDS.items():
            out[k] = 0.5 * (self._welch_bp(x1, fs, (lo, hi)) + self._welch_bp(x2, fs, (lo, hi)))
        return out
    
    def _compute_ratios(self, powers: Dict[str, float]) -> tuple:
        """计算θ/β和θ/α比值"""
        tb = powers["theta"] / max(powers["beta"], 1e-9)
        ta = powers["theta"] / max(powers["alpha"], 1e-9)
        return tb, ta
    
    def _normalize_with_baseline(self, tb: float, ta: float, subject_id: str) -> tuple:
        """使用被试基线归一化比值为z-score
        
        Args:
            tb: θ/β比值
            ta: θ/α比值
            subject_id: 被试ID（如 "zyp", "shh", "wsh"）
        
        Returns:
            (z_tb, z_ta): 归一化后的z-score
        """
        # 从subject_id提取base（去掉末尾数字，如 "zyp0" -> "zyp"）
        subject_base = ''.join(c for c in subject_id if not c.isdigit())
        
        # 获取个性化基线
        baseline = self.baseline_manager.get(subject_base)
        
        if baseline is None:
            # 没有基线时，返回原始值（不进行归一化）
            self.logger.warning(f"⚠️ 未找到被试 {subject_base} 的基线，使用原始比值")
            return tb, ta
        
        # 使用基线进行z-score归一化
        z_tb = (tb - baseline["med_tb"]) / baseline["mad_tb"]
        z_ta = (ta - baseline["med_ta"]) / baseline["mad_ta"]
        
        return z_tb, z_ta
    
    def _robust_baseline_stats(self, b1: np.ndarray, b2: np.ndarray, fs: float, mad_floor: float = 1e-4) -> Optional[Dict]:
        """计算鲁棒基线统计量"""
        win = int(2 * fs)
        step = win
        vals_tb, vals_ta = [], []
        
        for i in range(0, len(b1) - win + 1, step):
            p = self._bandpowers_2ch(b1[i:i+win], b2[i:i+win], fs=fs)
            rtb, rta = self._compute_ratios(p)
            vals_tb.append(rtb)
            vals_ta.append(rta)
        
        if not vals_tb:
            return None
        
        tb, ta = np.array(vals_tb), np.array(vals_ta)
        med_tb = float(np.median(tb))
        mad_tb = float(max(mad_floor, np.median(np.abs(tb - med_tb))))
        med_ta = float(np.median(ta))
        mad_ta = float(max(mad_floor, np.median(np.abs(ta - med_ta))))
        
        return {
            "med_tb": med_tb,
            "mad_tb": mad_tb,
            "med_ta": med_ta,
            "mad_ta": mad_ta
        }
    
    def _amplitude_bad_ratio(self, x1: np.ndarray, x2: np.ndarray, thr_uv: float) -> float:
        """计算幅值超限比例"""
        amp = np.maximum(np.abs(x1), np.abs(x2))
        return float((amp > thr_uv).mean())
    
    def _prob_to_score_quantile(self, prob: float) -> float:
        """概率转分数（使用分位数映射）"""
        if self.q_hi <= self.q_lo + 1e-9:
            return float(np.clip(prob * 100.0, 0.0, 100.0))
        score = (prob - self.q_lo) / (self.q_hi - self.q_lo) * 100.0
        return float(np.clip(score, 0.0, 100.0))
    
    def _infer_from_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从内存中的numpy数组直接推理
        
        注意：只分析已经过滤好的数据段（调用者负责时间戳过滤）
        """
        start_time = time.time()
        
        eeg_signal = data.get("eeg_signal")
        sampling_rate = data.get("sampling_rate", self.FS)
        subject_id = data.get("subject_id", "unknown")
        
        if eeg_signal is None:
            return {
                "status": "error",
                "error": "未提供EEG信号",
                "eeg_fatigue_score": 0.0
            }
        
        try:
            # 确保是numpy数组
            if not isinstance(eeg_signal, np.ndarray):
                eeg_signal = np.array(eeg_signal, dtype=np.float64)
            
            # 检查数据
            if eeg_signal.size == 0:
                raise ValueError("EEG信号为空")
            
            if eeg_signal.ndim != 2 or eeg_signal.shape[1] != 2:
                raise ValueError(f"EEG信号必须是双通道 [N, 2], 当前: {eeg_signal.shape}")
            
            ch1, ch2 = eeg_signal[:, 0], eeg_signal[:, 1]
            
            # 预处理
            ch1, ch2 = self._preprocess_pair(ch1, ch2, sampling_rate)
            
            # 分窗处理
            win = int(self.WIN_SEC * sampling_rate)
            step = win
            window_results = []
            print(f"EEG signal length: {len(ch1)}, window size: {win}, step size: {step},winow count:{len(ch1) - win + 1}")
            for i in range(0, len(ch1) - win + 1, step):
                print(f"zxProcessing window {i // step + 1} of {len(ch1) // step}")
                # 提取窗口
                w1, w2 = ch1[i:i+win], ch2[i:i+win]
                
                # 计算频带能量
                powers = self._bandpowers_2ch(w1, w2, sampling_rate)
                tb, ta = self._compute_ratios(powers)
                
                # 使用个性化基线归一化（z-score）
                z_tb, z_ta = self._normalize_with_baseline(tb, ta, subject_id)
                
                # 先加权求和再标准化（与训练时一致）
                F_raw = self.w[0] * z_tb + self.w[1] * z_ta  # 0.6*z_tb + 0.4*z_ta
                F_scaled = self.scaler.transform(np.array([[F_raw]]))  # (1,1) => (1,1)
                
                # 预测
                prob = self.clf.predict_proba(F_scaled)[:, 1][0]
                score = self._prob_to_score_quantile(prob)
                
                t_start = i / sampling_rate
                t_end = t_start + self.WIN_SEC
                
                window_results.append({
                    "window_index": len(window_results),
                    "t_start_s": round(t_start, 3),
                    "t_end_s": round(t_end, 3),
                    "theta_beta_ratio": round(tb, 4),
                    "theta_alpha_ratio": round(ta, 4),
                    "z_theta_beta": round(z_tb, 4),
                    "z_theta_alpha": round(z_ta, 4),
                    "probability": round(prob, 4),
                    "score": round(score, 2)
                })
            
            if not window_results:
                return {
                    "status": "no-data",
                    "error": "信号太短，无法分窗",
                    "eeg_fatigue_score": 0.0,
                    "window_results": []
                }
            
            # 计算平均分数
            avg_score = np.mean([w["score"] for w in window_results])
            
            inference_time = (time.time() - start_time) * 1000
            
            # 单行输出推理结果
            fatigue_level = "清醒😊" if avg_score < 30 else "轻度疲劳😐" if avg_score < 60 else "重度疲劳😴"
            self.logger.info(
                f"🧠💤 EEG疲劳度: {round(avg_score, 2)} ({fatigue_level}, "
                f"{len(window_results)}窗口, {round(inference_time, 1)}ms)"
            )
            print(f"🧠💤 -------zx的EEG疲劳度: {round(avg_score, 2)} ({fatigue_level}, ")
            print(f"{len(window_results)}窗口, {round(inference_time, 1)}ms)-------")
            return {
                "status": "success",
                "eeg_fatigue_score": round(avg_score, 2),
                "window_results": window_results,
                "num_windows": len(window_results),
                "subject_id": subject_id,
                "inference_mode": "memory",
                "inference_time_ms": round(inference_time, 1)
            }
            
        except Exception as e:
            self.logger.error(f"内存推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "eeg_fatigue_score": 0.0
            }
        finally:
            eeg_signal = None
            gc.collect()
    
    def _infer_from_file(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从文件路径读取数据并推理
        
        支持时间戳过滤：自动从CSV文件的ADS_Event列读取时间戳标记
        """
        eeg_file_path = data.get("eeg_file_path")
        sampling_rate = data.get("sampling_rate", self.FS)
        subject_id = data.get("subject_id", "unknown")
        
        if not eeg_file_path:
            return {
                "status": "error",
                "error": "缺少必需的EEG文件路径",
                "eeg_fatigue_score": 0.0
            }
        
        # 验证文件存在
        if not Path(eeg_file_path).exists():
            return {
                "status": "error",
                "error": f"EEG文件不存在: {eeg_file_path}",
                "eeg_fatigue_score": 0.0
            }
        
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🧠💤 EEG疲劳度分析 - 文件模式")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"📂 文件路径: {Path(eeg_file_path).name}")
            
            # 读取信号（CSV格式可能包含: Sample, Timestamp, Channel1, Channel2, ADS_Event, Sequence）
            import pandas as pd
            df = pd.read_csv(eeg_file_path)
            ch1 = df["Channel1"].values.astype(float)
            ch2 = df["Channel2"].values.astype(float)
            eeg_signal = np.column_stack([ch1, ch2])
            
            # 尝试从同目录的 part_timestamps.json 读取时间范围并过滤
            timestamps_file = Path(eeg_file_path).parent / "part_timestamps.json"
            if timestamps_file.exists():
                import json
                with open(timestamps_file, 'r', encoding='utf-8') as f:
                    timestamps = json.load(f)
                
                # part_timestamps.json 格式: [{"datetime": "ISO格式", "call_index": 0}, ...]
                # call_index 1=SART任务开始, 3=SART任务结束
                # 根据时间范围过滤数据
                if isinstance(timestamps, list) and len(timestamps) > 0:
                    # 读取CSV中的时间戳列
                    if "Timestamp" in df.columns:
                        # 将 CSV 的时间戳字符串转换为 datetime64
                        csv_datetimes = pd.to_datetime(df["Timestamp"])
                        
                        # 查找 call_index=1 和 call_index=3 的时间点
                        ts_map = {t.get("call_index"): t.get("datetime") for t in timestamps if t.get("datetime")}
                        
                        if 1 in ts_map and 3 in ts_map:
                            # 转换为 datetime64
                            t1 = np.datetime64(ts_map[1])
                            t3 = np.datetime64(ts_map[3])
                            
                            # 根据时间范围过滤信号（只保留时间戳1到时间戳3之间的数据）
                            mask = (csv_datetimes >= t1) & (csv_datetimes <= t3)
                            num_samples = mask.sum()
                            
                            if num_samples > 500:  # 至少1秒数据
                                eeg_signal = eeg_signal[mask]
                                duration = num_samples / sampling_rate
                                self.logger.info(
                                    f"  ✓ 时间戳过滤: call_index 1→3"
                                )
                                self.logger.info(
                                    f"  ⏱️ 过滤后保留 {num_samples} 个样本 ({duration:.1f}秒)"
                                )
                            else:
                                self.logger.warning(
                                    f"  ⚠️ 时间范围内样本太少({num_samples})，使用全部数据"
                                )
                        else:
                            self.logger.warning(
                                f"  ⚠️ part_timestamps.json 中缺少 call_index 1或3"
                            )
                    else:
                        self.logger.warning(f"  ⚠️ CSV中缺少Timestamp列，无法应用时间过滤")
            
            self.logger.info(f"  ✓ 信号读取完成: shape={eeg_signal.shape}")
            
            # 使用内存模式处理
            return self._infer_from_memory({
                "memory_mode": True,
                "eeg_signal": eeg_signal,
                "sampling_rate": sampling_rate,
                "subject_id": subject_id
            })
            
        except Exception as e:
            self.logger.error(f"从文件推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "eeg_fatigue_score": 0.0
            }
    
    def _infer_from_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从会话目录读取完整数据并评估（支持基线更新）
        
        Args:
            data: {
                "session_dir": str - 会话目录路径
                "subject_base": str - 被试标识（如"shh", "zyp", "wsh"）
                "qc_is_lowload": bool - 是否低负荷状态（默认True）
                "update_baseline": bool - 是否更新基线（默认True）
            }
        
        Returns:
            推理结果 + 基线更新信息
        """
        session_dir = Path(data.get("session_dir", ""))
        subject_base = data.get("subject_base", "unknown")
        qc_is_lowload = data.get("qc_is_lowload", True)
        update_baseline = data.get("update_baseline", True)
        
        if not session_dir.exists():
            return {
                "status": "error",
                "error": f"会话目录不存在: {session_dir}",
                "eeg_fatigue_score": 0.0
            }
        
        try:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"🧠💤 EEG疲劳度分析 - 会话模式（含基线更新）")
            self.logger.info(f"{'='*60}")
            self.logger.info(f"📂 会话目录: {session_dir.name}")
            self.logger.info(f"👤 被试标识: {subject_base}")
            
            # 1. 查找 EEG 数据文件
            eeg_dir = session_dir / "eeg"
            if not eeg_dir.exists():
                return {
                    "status": "error",
                    "error": f"EEG目录不存在: {eeg_dir}",
                    "eeg_fatigue_score": 0.0
                }
            
            # 查找 EEG CSV 文件
            # 支持两种命名格式：
            # 1. 新格式: eeg_data_YYYYMMDD_HHMMSS.csv
            # 2. 旧格式: part3.csv, part1.csv
            eeg_csv = None
            
            # 优先查找新格式的文件（按时间戳排序，取最新的）
            eeg_csv_files = sorted(eeg_dir.glob("eeg_data_*.csv"), reverse=True)
            if eeg_csv_files:
                eeg_csv = eeg_csv_files[0]  # 取最新的文件
                self.logger.info(f"  ✓ 找到EEG数据文件（新格式）: {eeg_csv.name}")
            else:
                # 回退到旧格式
                for part_name in ["part3.csv", "part1.csv"]:
                    candidate = eeg_dir / part_name
                    if candidate.exists():
                        eeg_csv = candidate
                        self.logger.info(f"  ✓ 找到EEG数据文件（旧格式）: {eeg_csv.name}")
                        break
            
            if not eeg_csv:
                return {
                    "status": "error",
                    "error": "未找到EEG数据文件（eeg_data_*.csv 或 part*.csv）",
                    "eeg_fatigue_score": 0.0
                }
            
            # 2. 读取时间戳文件
            timestamps_file = eeg_dir / "part_timestamps.json"
            if not timestamps_file.exists():
                self.logger.warning("  ⚠️ 未找到 part_timestamps.json，无法进行基线更新")
                update_baseline = False
            
            # 3. 读取EEG信号
            import pandas as pd
            df = pd.read_csv(eeg_csv)
            ch1 = df["Channel1"].values.astype(float)
            ch2 = df["Channel2"].values.astype(float)
            
            # 4. 预处理
            ch1, ch2 = self._preprocess_pair(ch1, ch2, self.FS)
            
            # 5. 如果支持基线更新，提取基线段（call_index 0→1）
            baseline_updated = False
            baseline_reason = "未尝试更新"
            
            if update_baseline and timestamps_file.exists():
                try:
                    with open(timestamps_file, 'r', encoding='utf-8') as f:
                        timestamps = json.load(f)
                    
                    # 提取时间戳映射
                    ts_map = {t.get("call_index"): t.get("datetime") for t in timestamps if t.get("datetime")}
                    
                    if 0 in ts_map and 1 in ts_map and "Timestamp" in df.columns:
                        # 提取基线段（call_index 0→1）
                        csv_datetimes = pd.to_datetime(df["Timestamp"])
                        t0 = np.datetime64(ts_map[0])
                        t1 = np.datetime64(ts_map[1])
                        
                        mask_baseline = (csv_datetimes >= t0) & (csv_datetimes <= t1)
                        num_baseline = mask_baseline.sum()
                        
                        if num_baseline > 1000:  # 至少2秒数据
                            b1 = ch1[mask_baseline]
                            b2 = ch2[mask_baseline]
                            
                            # 计算候选基线统计量
                            candidate = self._robust_baseline_stats(b1, b2, self.FS)
                            
                            if candidate:
                                # 计算幅值超限比例
                                bad_ratio = self._amplitude_bad_ratio(b1, b2, self.AMP_UV)
                                qc = {"is_lowload": qc_is_lowload, "bad_ratio": bad_ratio}
                                
                                # 门控更新
                                update_result = self.baseline_manager.gated_update(
                                    subject_base, candidate, qc=qc
                                )
                                
                                baseline_updated = update_result.get("updated", False)
                                baseline_reason = update_result.get("reason", "未知")
                                
                                self.logger.info(
                                    f"  🔄 基线更新: {'✅成功' if baseline_updated else '❌跳过'} "
                                    f"({baseline_reason})"
                                )
                                self.logger.info(
                                    f"     基线段: {num_baseline}样本 ({num_baseline/self.FS:.1f}秒), "
                                    f"坏信号率={bad_ratio:.2%}"
                                )
                        else:
                            baseline_reason = f"基线段太短({num_baseline}样本)"
                            self.logger.warning(f"  ⚠️ {baseline_reason}")
                    else:
                        baseline_reason = "时间戳数据不完整"
                        self.logger.warning(f"  ⚠️ {baseline_reason}")
                        
                except Exception as e:
                    baseline_reason = f"基线更新失败: {str(e)}"
                    self.logger.error(f"  ❌ {baseline_reason}", exc_info=True)
            
            # 6. 提取任务段（call_index 1→3）进行疲劳度推理
            task_signal = None
            if timestamps_file.exists():
                try:
                    with open(timestamps_file, 'r', encoding='utf-8') as f:
                        timestamps = json.load(f)
                    
                    ts_map = {t.get("call_index"): t.get("datetime") for t in timestamps if t.get("datetime")}
                    
                    if 1 in ts_map and 3 in ts_map and "Timestamp" in df.columns:
                        csv_datetimes = pd.to_datetime(df["Timestamp"])
                        t1 = np.datetime64(ts_map[1])
                        t3 = np.datetime64(ts_map[3])
                        
                        mask_task = (csv_datetimes >= t1) & (csv_datetimes <= t3)
                        num_task = mask_task.sum()
                        
                        if num_task > 500:
                            task_signal = np.column_stack([ch1[mask_task], ch2[mask_task]])
                            self.logger.info(
                                f"  ✓ 任务段: {num_task}样本 ({num_task/self.FS:.1f}秒)"
                            )
                except Exception as e:
                    self.logger.warning(f"  ⚠️ 任务段提取失败: {e}")
            
            # 如果没有成功提取任务段，使用全部信号
            if task_signal is None:
                task_signal = np.column_stack([ch1, ch2])
                self.logger.info(f"  ✓ 使用全部信号: {len(ch1)}样本")
            
            # 7. 执行疲劳度推理
            result = self._infer_from_memory({
                "memory_mode": True,
                "eeg_signal": task_signal,
                "sampling_rate": self.FS,
                "subject_id": subject_base
            })
            
            # 8. 添加基线更新信息
            if result.get("status") == "success":
                result["baseline_updated"] = baseline_updated
                result["baseline_reason"] = baseline_reason
                result["inference_mode"] = "session"
            
            return result
            
        except Exception as e:
            self.logger.error(f"会话模式推理失败: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "eeg_fatigue_score": 0.0
            }
    
    def cleanup(self) -> None:
        """清理模型资源"""
        if hasattr(self, 'scaler'):
            self.scaler = None
        if hasattr(self, 'clf'):
            self.clf = None
        if hasattr(self, 'baseline_manager'):
            self.baseline_manager = None
        
        gc.collect()
        self.logger.info("EEG疲劳度模型资源已清理")


__all__ = ["EEGFatigueModel"]
