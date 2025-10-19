"""EEG脑负荷模型后端

使用brain_load中的脑负荷推理模型
处理双通道EEG信号,输出脑负荷分数
"""

import base64
import io
import logging
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
import tempfile
import os

# 添加base路径和brain_load路径
base_path = Path(__file__).parent.parent / "base"
brain_load_path = Path(__file__).parent / "brain_load"

sys.path.insert(0, str(base_path))
sys.path.insert(0, str(brain_load_path))

from base_backend import BaseModelBackend

try:
    import joblib
    from eeg_utils import (
        FS, preprocess_eeg, segment_windows, 
        extract_features_batch, read_eeg_txt_two_channels
    )
except ImportError as e:
    print("错误: 缺少依赖库")
    print(f"详细错误: {e}")
    print("请安装: pip install joblib numpy scipy pandas")
    sys.exit(1)


class EEGBackend(BaseModelBackend):
    """EEG脑负荷模型后端实现
    
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
    
    async def initialize_model(self) -> None:
        """加载EEG脑负荷模型"""
        self.logger.info("正在加载EEG脑负荷模型...")
        
        # 确定模型路径 - 从根目录的models_data文件夹加载
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models_data" / "eeg_models"
        
        self.scaler_path = models_dir / "mymodel_scaler.joblib"
        self.calibrator_path = models_dir / "mymodel_calibrator.joblib"
        
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler文件不存在: {self.scaler_path}")
        if not self.calibrator_path.exists():
            raise FileNotFoundError(f"Calibrator文件不存在: {self.calibrator_path}")
        
        self.logger.info("加载模型组件...")
        print("="*60)
        print("📦 加载EEG脑负荷模型...")
        print("="*60)
        
        # 加载scaler和calibrator
        print(f"  [1/2] 加载特征缩放器 (scaler)...")
        self.scaler = joblib.load(str(self.scaler_path))
        print(f"  ✓ Scaler加载完成")
        
        print(f"  [2/2] 加载分类器 (calibrator)...")
        self.calibrator = joblib.load(str(self.calibrator_path))
        print(f"  ✓ Calibrator加载完成")
        
        print("="*60)
        self.logger.info("✅ EEG脑负荷模型加载完成")
        
        # 初始化EMA状态
        self.ema = None
        self.state = "low"
    
    async def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行EEG脑负荷推理
        
        Args:
            data: 输入数据,包含:
                - eeg_signal: EEG信号数据 (2D数组 [N, 2] 或 base64编码的CSV)
                - sampling_rate: 采样率 (默认250Hz)
                - subject_id: 被试ID (可选)
        
        Returns:
            推理结果:
                - brain_load_score: 脑负荷分数 (0-100)
                - state: 状态 ("low"/"high")
                - window_results: 各窗口详细结果
        """
        start_time = time.time()
        
        # 获取输入参数
        eeg_signal = data.get("eeg_signal", None)
        sampling_rate = data.get("sampling_rate", FS)
        subject_id = data.get("subject_id", "unknown")
        
        # 打印输入信息
        print(f"\n{'='*60}")
        print(f"📥 接收到EEG推理请求")
        print(f"{'='*60}")
        print(f"  被试ID: {subject_id}")
        print(f"  采样率: {sampling_rate}Hz")
        
        if eeg_signal is None:
            print(f"⚠️  未提供EEG信号,返回默认值")
            print(f"{'='*60}\n")
            return {
                "brain_load_score": 0.0,
                "state": "low",
                "window_results": [],
                "message": "未提供EEG信号"
            }
        
        try:
            # 解析EEG信号
            if isinstance(eeg_signal, str):
                # Base64编码的CSV/TXT文件
                print(f"  解码EEG信号文件...")
                signal_bytes = base64.b64decode(eeg_signal)
                
                # 保存到临时文件
                with tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False) as tmp:
                    tmp.write(signal_bytes)
                    tmp_path = tmp.name
                
                try:
                    # 读取信号
                    raw = read_eeg_txt_two_channels(tmp_path)
                finally:
                    # 清理临时文件
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                        
            elif isinstance(eeg_signal, list):
                # 直接提供的数组
                raw = np.array(eeg_signal, dtype=np.float64)
            else:
                raise ValueError(f"不支持的EEG信号格式: {type(eeg_signal)}")
            
            print(f"  ✓ 信号解析完成: shape={raw.shape}")
            
            if raw.shape[1] != 2:
                raise ValueError(f"EEG信号必须是双通道 (Fp1, Fp2), 当前: {raw.shape[1]}通道")
            
            # 预处理
            print(f"  🔧 信号预处理...")
            raw = preprocess_eeg(raw, fs=sampling_rate)
            print(f"  ✓ 预处理完成: 带通滤波 + 陷波器")
            
            # 分窗
            print(f"  📊 信号分窗...")
            wins, starts = segment_windows(
                raw, 
                fs=sampling_rate, 
                win_sec=self.WIN_SEC, 
                step_sec=self.STEP_SEC
            )
            print(f"  ✓ 分窗完成: {len(wins)}个窗口")
            
            if len(wins) == 0:
                print(f"⚠️  信号太短,无法分窗")
                print(f"{'='*60}\n")
                return {
                    "brain_load_score": 0.0,
                    "state": self.state,
                    "window_results": [],
                    "message": "信号太短,无法分窗"
                }
            
            # 提取特征
            print(f"  🔍 提取特征...")
            X, feat_names, mask = extract_features_batch(
                wins, 
                fs=sampling_rate, 
                reject_artifacts=True
            )
            starts = starts[mask]
            
            if X.shape[0] == 0:
                print(f"⚠️  所有窗口都被判定为伪迹")
                print(f"{'='*60}\n")
                return {
                    "brain_load_score": 0.0,
                    "state": self.state,
                    "window_results": [],
                    "message": "所有窗口都被判定为伪迹"
                }
            
            print(f"  ✓ 特征提取完成: {X.shape[0]}个有效窗口, {X.shape[1]}维特征")
            
            # 特征标准化
            print(f"  📐 特征标准化...")
            Xn = self.scaler.transform(X)
            print(f"  ✓ 标准化完成")
            
            # 推理
            print(f"  🧠 模型推理中...")
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
                
                state_change = " (↑)" if (prev_state == "low" and self.state == "high") else \
                              (" (↓)" if (prev_state == "high" and self.state == "low") else "")
                
                print(f"    窗口 {i+1}/{Xn.shape[0]}: "
                      f"t=[{t_start:.2f}s, {t_end:.2f}s], "
                      f"raw={score_raw:.2f}, "
                      f"ema={self.ema:.2f}, "
                      f"state={self.state}{state_change}")
                
                window_results.append({
                    "window_index": i,
                    "t_start_s": round(float(t_start), 3),
                    "t_end_s": round(float(t_end), 3),
                    "score_raw": round(score_raw, 3),
                    "score_ema": round(float(self.ema), 3),
                    "state": self.state,
                    "state_changed": prev_state != self.state
                })
            
            print(f"  ✓ 推理完成")
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}", exc_info=True)
            print(f"❌ 推理失败: {e}")
            print(f"{'='*60}\n")
            raise
        
        inference_time = time.time() - start_time
        
        # 最终分数使用EMA
        final_score = float(self.ema) if self.ema is not None else 0.0
        
        print(f"\n📈 推理结果:")
        print(f"  脑负荷分数: {final_score:.2f}/100")
        print(f"  当前状态: {self.state}")
        print(f"  有效窗口数: {len(window_results)}")
        print(f"  总耗时: {inference_time*1000:.0f}ms")
        print(f"{'='*60}\n")
        
        # 记录推理日志
        self.logger.info(
            f"EEG推理完成: 分数={round(final_score, 2)}, "
            f"状态={self.state}, "
            f"窗口数={len(window_results)}, "
            f"耗时={inference_time*1000:.0f}ms"
        )
        
        result = {
            "brain_load_score": round(final_score, 2),
            "state": self.state,
            "window_results": window_results,
            "inference_time_ms": round(inference_time * 1000, 2),
            "num_windows": len(window_results),
            "subject_id": subject_id
        }
        
        self.logger.debug(
            f"推理完成: 脑负荷分数={result['brain_load_score']}, "
            f"状态={result['state']}, "
            f"窗口数={result['num_windows']}"
        )
        
        return result
    
    async def cleanup(self) -> None:
        """清理模型资源"""
        self.logger.info("正在清理EEG模型资源...")
        
        # 释放模型
        self.scaler = None
        self.calibrator = None
        
        # 重置状态
        self.ema = None
        self.state = "low"
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        self.logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    # 配置日志 - 同时输出到控制台和文件
    log_dir = Path(__file__).parent.parent.parent / "logs" / "model"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eeg_backend.log"
    
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器（追加模式）
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    logging.info(f"📝 日志保存到: {log_file}")
    
    # 模型后端配置
    config = {
        "model_type": "eeg",
        "host": "127.0.0.1",
        "port": 8769  # 使用独立端口
    }
    
    # 创建并运行后端
    backend = EEGBackend(config)
    
    print("\n" + "=" * 70)
    print("EEG脑负荷模型后端")
    print("=" * 70)
    print(f"监听地址: ws://{config['host']}:{config['port']}")
    print(f"模型类型: {config['model_type']}")
    print("按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    backend.run()


if __name__ == "__main__":
    main()
