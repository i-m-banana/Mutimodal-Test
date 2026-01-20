"""数据库操作管理器

负责处理测试数据的数据库持久化操作，包括：
- 音视频路径保存
- 多模态数据路径保存（RGB/Depth/Eyetrack）
- EEG数据路径保存
- 推理结果保存（疲劳/脑负荷/情绪分数）
"""

from typing import Optional, Dict, Any, Callable, List
import time


class TestDBManager:
    """测试数据库操作管理器
    
    封装所有与数据库交互的逻辑，提供简单的API用于保存各类测试数据。
    """
    
    def __init__(self, db_service, db_disabled: bool = False):
        """
        初始化数据库管理器
        
        Args:
            db_service: 数据库服务实例
            db_disabled: 是否禁用数据库操作
        """
        self.db_service = db_service
        self.db_disabled = db_disabled
        self._logger = None
        self._config = None
    
    @property
    def logger(self):
        """延迟导入 logger 以避免循环导入"""
        if self._logger is None:
            from ..app import config
            self._logger = config.logger
        return self._logger
    
    @property
    def config(self):
        """延迟导入 config 以避免循环导入"""
        if self._config is None:
            from ..app import config
            self._config = config
        return self._config
    
    @property
    def row_id(self) -> Optional[int]:
        """获取当前数据库记录ID"""
        if self.db_service:
            return self.db_service.row_id
        return None
    
    def queue_update(self, update_payload: dict, context: str) -> None:
        """队列化数据库更新操作
        
        Args:
            update_payload: 要更新的字段字典
            context: 操作上下文描述（用于错误日志）
        """
        if self.db_disabled:
            self.logger.debug(f"数据库已禁用，跳过更新: {context}")
            return
            
        self.db_service.update_test_record(update_payload, context)
    
    def queue_update_with_callback(
        self, 
        update_payload: dict, 
        context: str, 
        on_success: Optional[Callable[[dict], None]] = None
    ) -> None:
        """队列化数据库更新操作（带成功回调）
        
        Args:
            update_payload: 要更新的字段字典
            context: 操作上下文描述
            on_success: 成功回调函数
        """
        if self.db_disabled:
            self.logger.debug(f"数据库已禁用，跳过更新: {context}")
            return
            
        self.db_service.update_test_record_with_callback(
            update_payload, 
            on_success, 
            context
        )
    
    def handle_failure(self, error: Exception, context: str) -> None:
        """处理数据库操作失败
        
        Args:
            error: 异常对象
            context: 操作上下文描述
        """
        self.logger.error(f"{context}: {error}", exc_info=True)
    
    def persist_av_paths(self, video_paths: List[str], audio_paths: List[str]) -> None:
        """保存音视频路径到数据库
        
        Args:
            video_paths: 视频文件路径列表
            audio_paths: 音频文件路径列表
        """
        if self.db_disabled:
            return
        
        try:
            update_payload = {
                "video": list(video_paths),
                "audio": list(audio_paths),
            }
            
            self.logger.debug(f"准备保存音视频路径: {len(video_paths)} 视频, {len(audio_paths)} 音频")
            
            self.queue_update(update_payload, "保存音视频路径失败")
            
            self.logger.info("✅ 音视频路径已加入数据库更新队列")
            
        except Exception as e:
            self.logger.exception(f"❌ 保存音视频路径时发生异常: {e}")
    
    def persist_multimodal_paths(self, clear_recognition_cache: bool = True) -> None:
        """保存多模态数据文件路径到数据库（RGB/Depth/Eyetrack）
        
        Args:
            clear_recognition_cache: 是否清理语音识别缓存
        """
        if self.db_disabled:
            return
            
        try:
            if not self.config.HAS_MULTIMODAL:
                self.logger.warning("多模态数据采集模块不可用，跳过数据库写入。")
                return
            
            from ..services.backend_proxy import get_multimodal_file_paths
            file_paths_result = get_multimodal_file_paths()
            file_paths = file_paths_result.get("paths", {}) if isinstance(file_paths_result, dict) else {}
            
            if not file_paths:
                self.logger.warning("未获取到多模态数据文件路径")
                return
            
            # 根据需要清理语音识别结果缓存
            if clear_recognition_cache:
                try:
                    self.config.clear_recognition_results()
                    self.logger.debug("已清理语音识别结果缓存")
                except Exception as e:
                    self.logger.debug(f"清理语音识别结果失败: {e}")
            
            update_payload = {}
            if file_paths.get('rgb'):
                update_payload['rgb'] = file_paths.get('rgb')
            if file_paths.get('depth'):
                update_payload['depth'] = file_paths.get('depth')
            if file_paths.get('eyetrack'):
                update_payload['tobii'] = file_paths.get('eyetrack')
            
            if not update_payload:
                self.logger.debug("多模态文件路径为空，跳过数据库更新。")
                return
            
            self.queue_update(update_payload, "更新多模态数据路径到数据库失败")
            
        except Exception as e:
            self.logger.error(f"写入多模态数据路径到数据库失败: {e}")
    
    def persist_eeg_paths(self, eeg_paths: dict, wait_for_row: bool = True) -> None:
        """保存EEG数据文件路径到数据库
        
        Args:
            eeg_paths: EEG路径字典，支持多种格式
            wait_for_row: 是否等待数据库行创建
        """
        if self.db_disabled:
            self.logger.debug("数据库已禁用，跳过 EEG 路径保存")
            return
        
        try:
            # 提取路径（兼容多种格式）
            update_payload = {}
            
            # 格式 1: {'ch1_txt': 'path1', 'ch2_txt': 'path2'}
            if 'ch1_txt' in eeg_paths or 'ch2_txt' in eeg_paths:
                if eeg_paths.get('ch1_txt'):
                    update_payload['eeg1'] = eeg_paths['ch1_txt']
                if eeg_paths.get('ch2_txt'):
                    update_payload['eeg2'] = eeg_paths['ch2_txt']
            
            # 格式 2: {'eeg_json_path': 'path1', 'eeg_csv_path': 'path2'}
            elif 'eeg_json_path' in eeg_paths or 'eeg_csv_path' in eeg_paths:
                if eeg_paths.get('eeg_json_path'):
                    update_payload['eeg1'] = eeg_paths['eeg_json_path']
                if eeg_paths.get('eeg_csv_path'):
                    update_payload['eeg2'] = eeg_paths['eeg_csv_path']
            
            # 格式 3: 列表形式 ['path1', 'path2']
            elif isinstance(eeg_paths, list):
                if len(eeg_paths) > 0 and eeg_paths[0]:
                    update_payload['eeg1'] = eeg_paths[0]
                if len(eeg_paths) > 1 and eeg_paths[1]:
                    update_payload['eeg2'] = eeg_paths[1]
            
            if not update_payload:
                self.logger.warning(f"⚠️ EEG 路径为空或格式不支持: {eeg_paths}")
                return
            
            # 如果需要等待数据库行创建
            if wait_for_row and not self.row_id:
                self.logger.info("⏳ 等待数据库行创建...")
                max_wait = 30  # 最多等待 3 秒 (30 * 0.1s)
                wait_count = 0
                while not self.row_id and wait_count < max_wait:
                    time.sleep(0.1)
                    wait_count += 1
                
                if not self.row_id:
                    self.logger.error("❌ 等待数据库行创建超时，EEG 路径将被加入待处理队列")
                    self.queue_update(update_payload, "保存 EEG 路径失败（等待超时）")
                    return
                else:
                    self.logger.info(f"✅ 数据库行已创建 (row_id={self.row_id})")
            
            # 使用排队机制
            self.queue_update(update_payload, "写入EEG路径到数据库失败")
            self.logger.info(f"✅ EEG 路径已加入数据库更新队列 (row_id={self.row_id}): {update_payload}")
            
        except Exception as e:
            self.logger.exception(f"❌ 保存 EEG 路径时发生异常: {e}")
            self.handle_failure(e, "写入EEG路径到数据库失败")
    
    def persist_inference_scores(
        self, 
        score_data: Dict[str, Any],
        on_success: Optional[Callable[[dict], None]] = None
    ) -> None:
        """保存推理结果到数据库（疲劳/脑负荷/情绪）
        
        Args:
            score_data: 包含所有分数的字典
            on_success: 成功回调函数
        """
        if self.db_disabled:
            self.logger.debug("数据库已禁用,跳过保存推理结果")
            return
        
        try:
            # 提取推理结果
            update_payload = {
                "fatigue_score": score_data.get("疲劳检测", 0),
                "brain_load_score": score_data.get("脑负荷", 0),
                "emotion_score": score_data.get("情绪", 0),
            }
            
            # 过滤掉0值(表示没有数据)
            update_payload = {k: v for k, v in update_payload.items() if v > 0}
            
            if not update_payload:
                self.logger.debug("没有有效的推理结果需要保存到数据库")
                return
            
            # 如果提供了回调，使用带回调的更新
            if on_success:
                self.queue_update_with_callback(
                    update_payload,
                    "保存推理结果到数据库失败",
                    on_success=on_success
                )
            else:
                self.queue_update(update_payload, "保存推理结果到数据库失败")
            
            self.logger.info(f"📊 推理结果已加入数据库更新队列: {update_payload}")
            
        except Exception as e:
            self.logger.error(f"保存推理结果到数据库失败: {e}", exc_info=True)


__all__ = ["TestDBManager"]
