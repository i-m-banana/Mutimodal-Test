"""疲劳度模型后端

使用emotion_fatigue_infer中的疲劳度推理模型
"""

import base64
import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
import time

# 添加base路径和emotion_fatigue_infer路径
base_path = Path(__file__).parent.parent / "base"
emotion_fatigue_path = Path(__file__).parent.parent / "emotion_fatigue_infer"
fatigue_path = emotion_fatigue_path / "fatigue"

sys.path.insert(0, str(base_path))
sys.path.insert(0, str(emotion_fatigue_path))
sys.path.insert(0, str(fatigue_path))  # 添加fatigue目录,以便导入facewap

from base_backend import BaseModelBackend

try:
    import torch
    import numpy as np
    from PIL import Image
    from fatigue.infer_multimodal import (
        FatigueFaceOnlyCNN,
        extract_eye_features_from_samples,
        extract_face_features_from_frames
    )
except ImportError as e:
    print("错误: 缺少依赖库")
    print(f"详细错误: {e}")
    print("请安装: pip install torch numpy pillow")
    sys.exit(1)


class FatigueBackend(BaseModelBackend):
    """疲劳度模型后端实现
    
    功能:
    - 处理RGB图像、深度图像和眼动数据
    - 输出疲劳度分数 (0-100)
    """
    
    async def initialize_model(self) -> None:
        """加载疲劳度模型"""
        self.logger.info("正在加载疲劳度模型...")
        
        # 确定模型路径
        model_dir = Path(__file__).parent.parent / "emotion_fatigue_infer" / "fatigue"
        self.model_path = model_dir / "fatigue_best_model.pt"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = FatigueFaceOnlyCNN().to(self.device)
        self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
        self.model.eval()
        
        self.logger.info("✅ 疲劳度模型加载完成")
    
    async def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行疲劳度推理
        
        Args:
            data: 输入数据,包含:
                - rgb_frames: RGB图像列表 (base64编码)
                - depth_frames: 深度图像列表 (base64编码)
                - eyetrack_samples: 眼动数据列表 (8维特征列表)
                - elapsed_time: 采集时长(秒)
        
        Returns:
            推理结果:
                - fatigue_score: 疲劳度分数 (0-100)
                - prediction_class: 预测类别 (0或1)
                - elapsed_time: 采集时长
        """
        start_time = time.time()
        
        rgb_b64_list = data.get("rgb_frames", [])
        depth_b64_list = data.get("depth_frames", [])
        eyetrack_samples = data.get("eyetrack_samples", [])
        elapsed = data.get("elapsed_time", 0.0)
        
        # 打印输入信息
        print(f"\n{'='*60}")
        print(f"📥 接收到推理请求")
        print(f"{'='*60}")
        print(f"  RGB帧数: {len(rgb_b64_list)}")
        print(f"  深度帧数: {len(depth_b64_list)}")
        print(f"  眼动样本数: {len(eyetrack_samples)}")
        print(f"  采集时长: {elapsed:.2f}秒")
        
        if not rgb_b64_list or not depth_b64_list:
            # 返回默认分数
            print(f"⚠️  数据不足,返回默认值")
            print(f"{'='*60}\n")
            return {
                "fatigue_score": 0.0,
                "prediction_class": 0,
                "elapsed_time": elapsed,
                "message": "数据不足,返回默认值"
            }
        
        # 1. 解码RGB和深度图像
        rgb_frames = []
        depth_frames = []
        
        try:
            print(f"📸 解码图像帧...")
            for rgb_b64 in rgb_b64_list:
                rgb_bytes = base64.b64decode(rgb_b64)
                rgb_image = Image.open(io.BytesIO(rgb_bytes)).convert("RGB")
                rgb_array = np.array(rgb_image)
                rgb_frames.append(rgb_array)
            
            for depth_b64 in depth_b64_list:
                depth_bytes = base64.b64decode(depth_b64)
                depth_image = Image.open(io.BytesIO(depth_bytes)).convert("L")
                depth_array = np.array(depth_image)
                depth_frames.append(depth_array)
            print(f"  ✓ 解码完成: RGB={len(rgb_frames)}帧, Depth={len(depth_frames)}帧")
        except Exception as e:
            self.logger.error(f"图像解码失败: {e}")
            print(f"❌ 图像解码失败: {e}")
            print(f"{'='*60}\n")
            raise ValueError(f"图像解码失败: {e}")
        
        # 2. 提取特征
        try:
            frames = min(len(rgb_frames), len(depth_frames))
            print(f"🔍 提取特征...")
            print(f"  使用帧数: {frames}")
            
            # 提取面部特征
            face_feat = extract_face_features_from_frames(
                rgb_frames, 
                depth_frames, 
                frames=frames
            ).to(self.device)  # [1, 42, frames]
            print(f"  ✓ 面部特征: {face_feat.shape}")
            
            # 提取眼动特征
            eye_feat = extract_eye_features_from_samples(eyetrack_samples).to(self.device)  # [1, 8, T]
            print(f"  ✓ 眼动特征: {eye_feat.shape}")
            
        except Exception as e:
            self.logger.error(f"特征提取失败: {e}")
            print(f"❌ 特征提取失败: {e}")
            print(f"{'='*60}\n")
            raise ValueError(f"特征提取失败: {e}")
        
        # 3. 模型推理
        try:
            print(f"🧠 模型推理中...")
            print(f"  输入形状: eye_feat={eye_feat.shape}, face_feat={face_feat.shape}")
            
            with torch.no_grad():
                output = self.model(eye_feat, face_feat)  # [1, num_classes] - 已经是概率(softmax后)
                print(f"  模型原始输出: {output}")
                print(f"  输出形状: {output.shape}")
                
                probs = output.cpu().numpy()[0]  # [num_classes] - 直接使用,不再softmax
                num_classes = output.shape[1]
                
                # 分数加权拟合: sum_i(prob_i * (i * 100 / (num_classes-1)))
                scores = np.linspace(0, 100, num_classes)
                score = float(np.dot(probs, scores))
                pred = int(np.argmax(probs))
                
            print(f"  ✓ 推理完成")
            print(f"  类别概率: {[f'{p:.3f}' for p in probs]}")
            print(f"  分数权重: {scores}")
            print(f"  加权计算: {' + '.join([f'{p:.3f}*{s:.1f}' for p, s in zip(probs, scores)])}")
                
        except Exception as e:
            self.logger.error(f"模型推理失败: {e}")
            print(f"❌ 模型推理失败: {e}")
            print(f"{'='*60}\n")
            raise ValueError(f"模型推理失败: {e}")
        
        inference_time = time.time() - start_time
        
        # 打印结果
        print(f"\n📊 推理结果:")
        print(f"  疲劳度分数: {round(score, 2)}/100")
        print(f"  预测类别: {pred}")
        print(f"  推理耗时: {inference_time*1000:.2f}ms")
        print(f"{'='*60}\n")
        
        # 记录推理日志
        self.logger.info(
            f"疲劳度推理完成: 分数={round(score, 2)}, 类别={pred}, "
            f"耗时={inference_time*1000:.0f}ms, "
            f"RGB帧数={len(rgb_frames)}, 深度帧数={len(depth_frames)}, 眼动样本数={len(eyetrack_samples)}"
        )
        
        result = {
            "fatigue_score": round(score, 2),
            "prediction_class": pred,
            "elapsed_time": round(elapsed, 2),
            "inference_time_ms": round(inference_time * 1000, 2),
            "num_rgb_frames": len(rgb_frames),
            "num_depth_frames": len(depth_frames),
            "num_eyetrack_samples": len(eyetrack_samples)
        }
        
        self.logger.debug(
            f"推理完成: 疲劳度分数={result['fatigue_score']}, "
            f"预测类别={result['prediction_class']}, "
            f"推理耗时={result['inference_time_ms']}ms"
        )
        
        return result
    
    async def cleanup(self) -> None:
        """清理模型资源"""
        self.logger.info("正在清理疲劳度模型资源...")
        
        # 释放模型
        self.model = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        self.logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    # 配置日志 - 同时输出到控制台和文件
    log_dir = Path(__file__).parent.parent.parent / "logs" / "model"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "fatigue_backend.log"
    
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
        "model_type": "fatigue",
        "host": "127.0.0.1",
        "port": 8767  # 使用不同端口,避免与multimodal后端冲突
    }
    
    # 创建并运行后端
    backend = FatigueBackend(config)
    
    print("\n" + "=" * 70)
    print("疲劳度模型后端")
    print("=" * 70)
    print(f"监听地址: ws://{config['host']}:{config['port']}")
    print(f"模型类型: {config['model_type']}")
    print("按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    backend.run()


if __name__ == "__main__":
    main()
