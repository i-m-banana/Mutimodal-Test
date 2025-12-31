"""EEG脑负荷模型 - 直接集成版本

集成 eeg_algorithms/ 中的算法库进行在线推理
"""

import gc
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base_inference_model import BaseInferenceModel

try:
    import joblib
    from .eeg_algorithms.eeg_utils import (
        FS,
        preprocess_eeg,
        segment_windows,
        extract_features_batch
    )
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    _import_error = e


class EEGModel(BaseInferenceModel):
    """EEG脑负荷模型（集成版本）
    
    直接在后端进程中运行，无需独立进程
    
    功能:
    - 处理双通道EEG信号 (Fp1, Fp2)
    - 提取时频域特征
    - 输出脑负荷分数 (0-100)
    """
    
    # 模型参数
    WIN_SEC = 2.0      # 窗口长度(秒)
    STEP_SEC = 1.0     # 滑动步长(秒)
    EMA_ALPHA = 0.7    # 指数移动平均系数
    TH_UP = 60.0       # 高负荷阈值
    TH_DN = 50.0       # 低负荷阈值
    
    def initialize(self) -> None:
        """初始化EEG脑负荷模型"""
        if not HAS_DEPS:
            raise RuntimeError(f"无法加载依赖: {_import_error}")
        
        # 确定模型路径 - 从根目录的models_data文件夹加载
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models_data" / "eeg_models"
        
        self.scaler_path = models_dir / "mymodel_scaler.joblib"
        self.calibrator_path = models_dir / "mymodel_calibrator.joblib"
        
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler文件不存在: {self.scaler_path}")
        if not self.calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator文件不存在: {self.calibrator_path}")
        
        # 加载scaler和calibrator
        self.logger.debug("加载EEG模型组件...")
        self.scaler = joblib.load(str(self.scaler_path))
        self.logger.debug("  ✓ Scaler加载完成")
        
        self.calibrator = joblib.load(str(self.calibrator_path))
        self.logger.debug("  ✓ Calibrator加载完成")
        
        # 初始化EMA状态
        self.ema = None
        self.state = "low"
        
        self.logger.debug("✅ EEG脑负荷模型初始化完成")
    
    def infer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行EEG脑负荷推理（仅支持内存模式）
        
        Args:
            data: 输入数据字典
                - memory_mode: bool = True
                - eeg_signal: np.ndarray - EEG信号数组 [N, 2]
                - sampling_rate: int = 250
                - subject_id: str = "unknown"
        
        Returns:
            推理结果:
                - status: "success" | "no-data" | "error"
                    - no-data: 输入有效但当前时间窗内没有可用窗口（太短或全部判为伪迹），建议上层跳过发布
                - brain_load_score: 脑负荷分数 (0-100)
                - state: 状态 ("low"/"high")
                - window_results: 各窗口详细结果
        """
        return self._infer_from_memory(data)
    
    def _infer_from_memory(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """从内存中的numpy数组直接推理（零I/O开销）"""
        start_time = time.time()
        
        eeg_signal = data.get("eeg_signal")
        sampling_rate = data.get("sampling_rate", FS)
        subject_id = data.get("subject_id", "unknown")
        
        if eeg_signal is None:
            return {
                "status": "error",
                "error": "未提供EEG信号",
                "brain_load_score": 0.0,
                "state": self.state
            }
        
        try:
            # 确保是numpy数组
            if not isinstance(eeg_signal, np.ndarray):
                eeg_signal = np.array(eeg_signal, dtype=np.float64)
            
            # 检查数据是否为空
            if eeg_signal.size == 0:
                raise ValueError(f"EEG信号为空，无法进行推理")
            
            # 检查维度
            if eeg_signal.ndim != 2:
                raise ValueError(f"EEG信号必须是2维数组 [n_samples, 2], 当前维度: {eeg_signal.ndim}")
            
            if eeg_signal.shape[1] != 2:
                raise ValueError(f"EEG信号必须是双通道 (Fp1, Fp2), 当前: {eeg_signal.shape[1]}通道")
            
            # 预处理
            raw = preprocess_eeg(eeg_signal, fs=sampling_rate)
            
            # 分窗
            wins, starts = segment_windows(
                raw,
                fs=sampling_rate,
                win_sec=self.WIN_SEC,
                step_sec=self.STEP_SEC
            )
            
            if len(wins) == 0:
                # 无有效窗口：标记为 no-data，避免上层将其视为有效0分
                return {
                    "status": "no-data",
                    "error": "信号太短，无法分窗",
                    "brain_load_score": 0.0,
                    "state": self.state,
                    "window_results": [],
                    "num_windows": 0,
                    "inference_mode": "memory"
                }
            
            # 提取特征
            X, feat_names, mask = extract_features_batch(
                wins,
                fs=sampling_rate,
                reject_artifacts=True
            )
            starts = starts[mask]
            
            if X.shape[0] == 0:
                # 全部伪迹：标记为 no-data，避免上层将其视为有效0分
                return {
                    "status": "no-data",
                    "error": "所有窗口都被判定为伪迹",
                    "brain_load_score": 0.0,
                    "state": self.state,
                    "window_results": [],
                    "num_windows": 0,
                    "inference_mode": "memory"
                }
            
            # 特征标准化
            Xn = self.scaler.transform(X)
            
            # 推理
            window_results = []
            
            for i in range(Xn.shape[0]):
                # 预测概率
                proba = self.calibrator.predict_proba(Xn[i:i+1])[:, 1][0]
                score_raw = 100.0 * float(proba)
                
                # 指数移动平均
                if self.ema is None:
                    self.ema = score_raw
                else:
                    self.ema = self.EMA_ALPHA * self.ema + (1 - self.EMA_ALPHA) * score_raw
                
                # 迟滞判定
                prev_state = self.state
                if self.state == "low" and self.ema >= self.TH_UP:
                    self.state = "high"
                elif self.state == "high" and self.ema <= self.TH_DN:
                    self.state = "low"
                
                # 时间范围
                t_start = starts[i] / sampling_rate
                t_end = t_start + self.WIN_SEC
                
                window_results.append({
                    "window_index": i,
                    "t_start_s": round(float(t_start), 3),
                    "t_end_s": round(float(t_end), 3),
                    "score_raw": round(score_raw, 3),
                    "score_ema": round(float(self.ema), 3),
                    "state": self.state,
                    "state_changed": prev_state != self.state
                })
            
            
            inference_time = (time.time() - start_time) * 1000
            
            # 最终分数使用EMA
            final_score = float(self.ema) if self.ema is not None else 0.0
            
            # 单行输出推理结果（debug级别）
            load_level = "正常😊" if final_score < 30 else "轻度负荷😐" if final_score < 60 else "重度负荷🔥"
            self.logger.debug(
                f"🧠 脑负荷: {round(final_score, 2)} ({load_level}, {self.state}, "
                f"{len(window_results)}窗口, {round(inference_time, 1)}ms)"
            )
            
            return {
                "status": "success",
                "brain_load_score": round(final_score, 2),
                "state": self.state,
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
                "brain_load_score": 0.0,
                "state": self.state
            }
    
    def cleanup(self) -> None:
        """清理模型资源"""
        if hasattr(self, 'scaler'):
            self.scaler = None
        if hasattr(self, 'calibrator'):
            self.calibrator = None
        
        # 重置状态
        self.ema = None
        self.state = "low"
        
        # 强制垃圾回收
        gc.collect()


__all__ = ["EEGModel"]
