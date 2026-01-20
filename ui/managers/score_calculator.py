"""分数计算和数据准备管理器

负责处理测试分数的计算、准备和存储，包括：
- 疲劳度和脑负荷平均分数计算
- 测试结果数据准备（用于分数页面展示）
- 历史分数的保存和加载
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..app import config

logger = config.logger


class ScoreCalculator:
    """测试分数计算和数据准备管理器
    
    封装所有分数计算、数据准备和CSV存储逻辑。
    """
    
    def __init__(self, scores_csv_file: str):
        """初始化分数计算器
        
        Args:
            scores_csv_file: 分数CSV文件路径
        """
        self.scores_csv_file = scores_csv_file
        self.history_scores: List[int] = []
        
    def calculate_average_scores(
        self, 
        fatigue_result: Optional[float],
        brain_load_scores: List[float]
    ) -> Dict[str, Optional[float]]:
        """计算疲劳度和脑负荷的平均分数
        
        ✅ 疲劳度数据来源（离线评估）：
        - 使用后端离线评估的融合结果（RGB + EEG）
        - 评估时机：朗读阶段结束后触发一次性评估
        - 评估范围：基线校准 + SART实验 + 文本朗读
        
        ✅ 脑负荷数据来源（实时推理）：
        - 使用EEG实时推理结果
        - 推理时机：EEG采集期间持续推理
        - 累积方式：接收多次推理结果并计算平均值
        - 覆盖范围：整个测试流程（基线校准 + SART + 朗读 + 舒尔特）
        
        Args:
            fatigue_result: 疲劳度评估结果（0-90）
            brain_load_scores: 脑负荷分数列表（实时推理）
            
        Returns:
            包含平均分数的字典:
            {
                "fatigue_avg": 疲劳度分数 (0-90, 离线评估一次),
                "brain_load_avg": 脑负荷分数 (0-100, 实时推理平均值),
                "fatigue_count": 1 (离线评估只返回一次),
                "brain_load_count": N (实时推理累积次数)
            }
        """
        result = {
            "fatigue_avg": None,
            "brain_load_avg": None,
            "fatigue_count": 0,
            "brain_load_count": 0
        }
        
        # 使用离线疲劳度评估结果
        if fatigue_result is not None:
            result["fatigue_avg"] = fatigue_result
            result["fatigue_count"] = 1
            logger.debug(
                f"疲劳度评估结果: {result['fatigue_avg']:.2f}/90 (离线评估)"
            )
        else:
            logger.warning("未接收到疲劳度评估结果")
        
        # 计算脑负荷平均值
        if brain_load_scores:
            result["brain_load_avg"] = sum(brain_load_scores) / len(brain_load_scores)
            result["brain_load_count"] = len(brain_load_scores)
            logger.debug(
                f"脑负荷平均分数: {result['brain_load_avg']:.2f} "
                f"(基于 {result['brain_load_count']} 个样本)"
            )
        else:
            logger.warning("没有收集到脑负荷分数数据")
        
        return result
    
    def prepare_score_data(
        self,
        fatigue_result: Optional[float],
        brain_load_scores: List[float],
        emotion_score: Optional[float],
        schulte_accuracy: Optional[float],
        schulte_total_score: Optional[float],
        bp_results: Optional[Dict[str, Any]],
        stage_completed: Dict[str, bool]
    ) -> Dict[str, Any]:
        """准备传递给分数展示页面的所有数据
        
        Args:
            fatigue_result: 疲劳度评估结果
            brain_load_scores: 脑负荷分数列表
            emotion_score: 情绪分数
            schulte_accuracy: 舒尔特准确率
            schulte_total_score: 舒尔特综合得分
            bp_results: 血压测量结果字典
            stage_completed: 阶段完成状态字典
            
        Returns:
            包含所有测试结果的字典
        """
        # 计算平均分数
        avg_scores = self.calculate_average_scores(fatigue_result, brain_load_scores)
        
        # 准备数据
        score_data = {
            # 疲劳检测 (平均值) - 仅朗读录音阶段测试
            "疲劳检测": avg_scores["fatigue_avg"] if avg_scores["fatigue_avg"] is not None else 0,
            
            # 情绪分数 - 仅朗读录音阶段测试
            "情绪": emotion_score if emotion_score is not None else 0,
            
            # 脑负荷 (平均值) - 贯穿整个流程
            "脑负荷": avg_scores["brain_load_avg"] if avg_scores["brain_load_avg"] is not None else 0,
            
            # 舒尔特准确率 - 舒尔特测试阶段
            "舒尔特准确率": schulte_accuracy if schulte_accuracy is not None else 0,
            
            # 血压数据 - 血压测试阶段
            "收缩压": bp_results.get("systolic", 0) if bp_results else 0,
            "舒张压": bp_results.get("diastolic", 0) if bp_results else 0,
            "脉搏": bp_results.get("pulse", 0) if bp_results else 0,
            
            # 舒尔特综合得分
            "舒尔特综合得分": schulte_total_score if schulte_total_score is not None else 0,
            
            # 阶段完成状态(用于控制分数页面显示)
            "_stage_completed": {
                "多模态疲劳检测": stage_completed.get('多模态疲劳检测', False),
                "情绪检测": stage_completed.get('情绪检测', False),
                "血压脉搏检测": stage_completed.get('血压脉搏检测', False),
                "舒尔特专注度检测": stage_completed.get('舒尔特专注度检测', False),
            },
            
            # 元数据
            "_metadata": {
                "fatigue_sample_count": avg_scores["fatigue_count"],
                "brain_load_sample_count": avg_scores["brain_load_count"],
                "has_emotion_score": emotion_score is not None,
                "has_schulte_result": schulte_accuracy is not None,
                "has_bp_result": bp_results is not None and bp_results.get('systolic') is not None,
            }
        }
        
        logger.debug(f"准备分数数据完成: {score_data}")
        logger.debug(f"阶段完成状态: {score_data['_stage_completed']}")
        return score_data
    
    def save_score_to_csv(self, score: Optional[float]) -> None:
        """保存分数到CSV文件
        
        Args:
            score: 要保存的分数
        """
        try:
            if score is not None:
                with open(self.scores_csv_file, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), score])
                self.history_scores.append(int(score))
                logger.debug(f"分数已保存到CSV文件: {score}")
            else:
                logger.warning("分数尚未计算，跳过CSV保存")
        except Exception as e:
            logger.error(f"保存分数时出错: {e}")
    
    def load_history_scores(self) -> List[int]:
        """从CSV文件加载历史分数
        
        Returns:
            历史分数列表
        """
        self.history_scores = []
        if not os.path.exists(self.scores_csv_file):
            return self.history_scores
        
        try:
            with open(self.scores_csv_file, 'r', encoding='utf-8') as f:
                for row in csv.reader(f):
                    if len(row) >= 2:
                        self.history_scores.append(int(row[1]))
            logger.debug(f"已加载 {len(self.history_scores)} 条历史分数")
        except Exception as e:
            logger.error(f"读取历史分数时出错: {e}")
        
        return self.history_scores
    
    def get_history_scores(self) -> List[int]:
        """获取历史分数列表
        
        Returns:
            历史分数列表
        """
        return list(self.history_scores)


__all__ = ["ScoreCalculator"]
