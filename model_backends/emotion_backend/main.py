"""情绪模型后端

使用emotion_fatigue_infer中的情绪推理模型
处理视频、音频和文本数据,输出情绪分数
"""

import base64
import io
import logging
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List
import time
import tempfile
import os

# 添加base路径和emotion_fatigue_infer路径
base_path = Path(__file__).parent.parent / "base"
emotion_fatigue_path = Path(__file__).parent.parent / "emotion_fatigue_infer"
emotion_path = emotion_fatigue_path / "emotion"

sys.path.insert(0, str(base_path))
sys.path.insert(0, str(emotion_fatigue_path))
sys.path.insert(0, str(emotion_path))

from base_backend import BaseModelBackend

try:
    import torch
    import numpy as np
    import cv2
    import soundfile as sf
    from transformers import VivitImageProcessor, Wav2Vec2Processor, AutoTokenizer, AutoModel, Wav2Vec2Model
    from emotion.inference_standalone_all import (
        SimpleMultimodalClassifier,
        extract_vision_feature,
        extract_audio_feature,
        extract_text_feature
    )
except ImportError as e:
    print("错误: 缺少依赖库")
    print(f"详细错误: {e}")
    print("请安装: pip install torch numpy opencv-python soundfile transformers")
    sys.exit(1)


class EmotionBackend(BaseModelBackend):
    """情绪模型后端实现
    
    功能:
    - 处理视频、音频和文本数据
    - 提取视觉、音频、文本特征
    - 输出情绪分数 (0-100)
    """
    
    async def initialize_model(self) -> None:
        """加载情绪模型"""
        self.logger.info("正在加载情绪模型...")
        
        # 确定模型路径 - 从根目录的models_data文件夹加载
        project_root = Path(__file__).parent.parent.parent
        models_dir = project_root / "models_data" / "emotion_models"
        self.model_path = models_dir / "best_model.pt"
        # 预训练模型也移到models_data
        pretrained_models_dir = project_root / "models_data" / "emotion_pretrained_models"
        self.model_dir = pretrained_models_dir
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # 设置设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"使用设备: {self.device}")
        
        # 加载预处理器和特征提取模型
        print("="*60)
        print("📦 加载预训练模型...")
        print("="*60)
        
        self.logger.info("加载视觉处理器...")
        print("  [1/6] 加载视觉处理器 (TIMESFORMER)...")
        self.vision_processor = VivitImageProcessor.from_pretrained(str(self.model_dir / "TIMESFORMER"))
        print("  ✓ 视觉处理器加载完成")
        
        self.logger.info("加载音频处理器...")
        print("  [2/6] 加载音频处理器 (WAV2VEC2)...")
        self.audio_processor = Wav2Vec2Processor.from_pretrained(str(self.model_dir / "WAV2VEC2"))
        print("  ✓ 音频处理器加载完成")
        
        self.logger.info("加载文本分词器...")
        print("  [3/6] 加载文本分词器 (ROBBERTA)...")
        self.text_tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir / "ROBBERTA"))
        print("  ✓ 文本分词器加载完成")
        
        self.logger.info("加载视觉模型...")
        print("  [4/6] 加载视觉模型 (TIMESFORMER)...")
        self.vision_model = AutoModel.from_pretrained(
            str(self.model_dir / "TIMESFORMER"), 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.vision_model.eval()
        print("  ✓ 视觉模型加载完成")
        
        self.logger.info("加载音频模型...")
        print("  [5/6] 加载音频模型 (WAV2VEC2)...")
        self.audio_model = Wav2Vec2Model.from_pretrained(
            str(self.model_dir / "WAV2VEC2"), 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.audio_model.eval()
        print("  ✓ 音频模型加载完成")
        
        self.logger.info("加载文本模型...")
        print("  [6/6] 加载文本模型 (ROBBERTA)...")
        self.text_model = AutoModel.from_pretrained(
            str(self.model_dir / "ROBBERTA"), 
            ignore_mismatched_sizes=True
        ).to(self.device)
        self.text_model.eval()
        print("  ✓ 文本模型加载完成")
        print("="*60)
        
        # 预热:提取一次特征以确定维度
        self.logger.info("预热模型...")
        print("\n🔥 预热模型(提取特征维度)...")
        dummy_video = np.zeros((224, 224, 3), dtype=np.uint8)
        dummy_audio = np.zeros(16000, dtype=np.float32)
        dummy_text = "test"
        
        # 创建临时视频文件
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as tmp_video:
            tmp_video_path = tmp_video.name
            out = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (224, 224))
            for _ in range(8):
                out.write(dummy_video)
            out.release()
        
        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio_path = tmp_audio.name
            sf.write(tmp_audio_path, dummy_audio, 16000)
        
        try:
            vision_feat = extract_vision_feature(tmp_video_path, self.vision_processor, self.vision_model, self.device)
            audio_feat = extract_audio_feature(tmp_audio_path, self.audio_processor, self.audio_model, self.device)
            text_feat = extract_text_feature(dummy_text, self.text_tokenizer, self.text_model, self.device)
            
            print(f"  特征维度: 视觉={vision_feat.shape}, 音频={audio_feat.shape}, 文本={text_feat.shape}")
            
            # 初始化分类器
            print(f"  初始化分类器: hidden_dim=512, num_classes=2")
            self.model = SimpleMultimodalClassifier(
                vision_feat_dim=vision_feat.shape[-1],
                audio_feat_dim=audio_feat.shape[-1],
                text_feat_dim=text_feat.shape[-1],
                hidden_dim=512,
                num_classes=2,
            ).to(self.device)
            
            # 加载权重
            print(f"  加载模型权重: {self.model_path.name}")
            self.model.load_state_dict(torch.load(str(self.model_path), map_location=self.device))
            self.model.eval()
            print(f"  ✓ 预热完成")
            
        finally:
            # 清理临时文件
            if os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
        
        print("="*60)
        self.logger.info("✅ 情绪模型加载完成")
    
    async def process_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """执行情绪推理
        
        Args:
            data: 输入数据,包含:
                - samples: 样本列表,每个样本包含:
                    - video_b64: 视频文件base64编码
                    - audio_b64: 音频文件base64编码
                    - text: 识别的文本
                    - question_index: 问题编号 (可选)
        
        Returns:
            推理结果:
                - emotion_score: 情绪分数 (0-100)
                - sample_scores: 每个样本的分数
                - predictions: 每个样本的预测类别
        """
        start_time = time.time()
        
        samples = data.get("samples", [])
        
        # 打印输入信息
        print(f"\n{'='*60}")
        print(f"📥 接收到情绪推理请求")
        print(f"{'='*60}")
        print(f"  样本数量: {len(samples)}")
        
        if not samples:
            print(f"⚠️  数据不足,返回默认值")
            print(f"{'='*60}\n")
            return {
                "emotion_score": 0.0,
                "sample_scores": [],
                "predictions": [],
                "message": "数据不足,返回默认值"
            }
        
        # 创建临时文件存储视频和音频
        temp_files = []
        try:
            logits_list = []
            sample_results = []
            
            for idx, sample in enumerate(samples):
                video_b64 = sample.get("video_b64", "")
                audio_b64 = sample.get("audio_b64", "")
                text = sample.get("text", "")
                question_index = sample.get("question_index", idx + 1)
                
                print(f"\n🎬 处理样本 {idx+1}/{len(samples)} (问题 {question_index})")
                print(f"  原始文本: \"{text}\"")
                print(f"  视频大小: {len(video_b64)/1024:.1f}KB")
                print(f"  音频大小: {len(audio_b64)/1024:.1f}KB")
                
                # 解码并保存视频
                video_bytes = base64.b64decode(video_b64)
                tmp_video = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
                tmp_video.write(video_bytes)
                tmp_video.close()
                temp_files.append(tmp_video.name)
                
                # 解码并保存音频
                audio_bytes = base64.b64decode(audio_b64)
                tmp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                tmp_audio.write(audio_bytes)
                tmp_audio.close()
                temp_files.append(tmp_audio.name)
                
                # 提取特征
                print(f"  🔍 提取特征...")
                sample_start = time.time()
                v_feat = extract_vision_feature(tmp_video.name, self.vision_processor, self.vision_model, self.device)
                print(f"    ✓ 视觉特征: {v_feat.shape}")
                
                a_feat = extract_audio_feature(tmp_audio.name, self.audio_processor, self.audio_model, self.device)
                print(f"    ✓ 音频特征: {a_feat.shape}")
                
                # 显示文本预处理
                print(f"  📝 文本预处理:")
                print(f"    原始文本: \"{text}\"")
                # 简单的预处理(与preprocess_text一致)
                preprocessed_text = " ".join([
                    '@user' if t.startswith('@') and len(t) > 1 else 
                    'http' if t.startswith('http') else t
                    for t in str(text).split(" ")
                ])
                print(f"    预处理后: \"{preprocessed_text}\"")
                
                t_feat = extract_text_feature(text, self.text_tokenizer, self.text_model, self.device)
                print(f"    ✓ 文本特征: {t_feat.shape}")
                
                feat_time = time.time() - sample_start
                print(f"  ⏱️  特征提取耗时: {feat_time*1000:.0f}ms")
                
                # 推理
                print(f"  🧠 模型推理中...")
                print(f"    输入形状: v={v_feat.shape}, a={a_feat.shape}, t={t_feat.shape}")
                
                with torch.no_grad():
                    logits = self.model(v_feat.unsqueeze(0), a_feat.unsqueeze(0), t_feat.unsqueeze(0))  # [1, 2]
                    print(f"    模型logits输出: {logits.cpu().numpy()}")
                    
                    probs = torch.softmax(logits, dim=1)
                    print(f"    Softmax概率: {probs.cpu().numpy()}")
                    
                    logits_list.append(logits.squeeze(0).cpu().numpy())
                    pred = torch.argmax(logits, dim=1).item()
                    prob_values = probs.squeeze(0).cpu().numpy()
                
                sample_elapsed = time.time() - sample_start
                
                print(f"  📊 样本结果:")
                print(f"    预测类别: {pred} ({'积极' if pred == 1 else '消极'})")
                print(f"    类别概率: [{prob_values[0]:.3f}, {prob_values[1]:.3f}]")
                print(f"    logits值: [{logits.squeeze(0).cpu().numpy()[0]:.3f}, {logits.squeeze(0).cpu().numpy()[1]:.3f}]")
                print(f"    总耗时: {sample_elapsed*1000:.0f}ms")
                
                sample_results.append({
                    "question_index": question_index,
                    "prediction": pred,
                    "probabilities": prob_values.tolist(),
                    "inference_time_ms": round(sample_elapsed * 1000, 2)
                })
                
                self.logger.debug(f"样本 {question_index} 推理完成: 类别={pred}, 耗时={sample_elapsed:.4f}秒")
            
            # 计算最终分数
            print(f"\n📈 计算最终情绪分数...")
            logits_arr = np.array(logits_list)  # [N, 2]
            probs = torch.softmax(torch.tensor(logits_arr), dim=1).numpy()  # [N, 2]
            pos_probs = probs[:, 1]  # [N]
            
            min_prob, max_prob = pos_probs.min(), pos_probs.max()
            if max_prob - min_prob < 1e-6:
                scores = np.full_like(pos_probs, 50.0)
            else:
                scores = (pos_probs - min_prob) / (max_prob - min_prob) * 100
            
            final_score = float(np.mean(scores))
            print(f"  样本分数: {[f'{s:.2f}' for s in scores]}")
            print(f"  最终分数: {final_score:.2f}/100")
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            print(f"❌ 推理失败: {e}")
            print(f"{'='*60}\n")
            raise
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        
        inference_time = time.time() - start_time
        
        print(f"\n✅ 情绪推理完成")
        print(f"  总耗时: {inference_time*1000:.0f}ms")
        print(f"  平均每样本: {inference_time*1000/len(samples):.0f}ms")
        print(f"{'='*60}\n")
        
        # 记录推理日志
        self.logger.info(
            f"情绪推理完成: 分数={round(final_score, 2)}, "
            f"样本数={len(samples)}, "
            f"总耗时={inference_time*1000:.0f}ms, "
            f"平均={inference_time*1000/len(samples):.0f}ms/样本"
        )
        
        result = {
            "emotion_score": round(final_score, 2),
            "sample_scores": scores.tolist(),
            "sample_results": sample_results,
            "inference_time_ms": round(inference_time * 1000, 2),
            "num_samples": len(samples)
        }
        
        self.logger.debug(
            f"推理完成: 情绪分数={result['emotion_score']}, "
            f"样本数={result['num_samples']}, "
            f"总耗时={result['inference_time_ms']}ms"
        )
        
        return result
    
    async def cleanup(self) -> None:
        """清理模型资源"""
        self.logger.info("正在清理情绪模型资源...")
        
        # 释放模型
        self.model = None
        self.vision_model = None
        self.audio_model = None
        self.text_model = None
        self.vision_processor = None
        self.audio_processor = None
        self.text_tokenizer = None
        
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
    log_file = log_dir / "emotion_backend.log"
    
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
        "model_type": "emotion",
        "host": "127.0.0.1",
        "port": 8768  # 使用不同端口,避免与其他后端冲突
    }
    
    # 创建并运行后端
    backend = EmotionBackend(config)
    
    print("\n" + "=" * 70)
    print("情绪模型后端")
    print("=" * 70)
    print(f"监听地址: ws://{config['host']}:{config['port']}")
    print(f"模型类型: {config['model_type']}")
    print("按 Ctrl+C 停止服务")
    print("=" * 70 + "\n")
    
    backend.run()


if __name__ == "__main__":
    main()
