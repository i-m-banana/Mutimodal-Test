"""
Database access layer for the backend service.
Provides database operations for the test data table.
"""

import datetime
import json
import os
import pymysql
from pymysql.cursors import DictCursor
from typing import List, Dict, Optional, Any


class MysqlDB:
    """MySQL wrapper providing basic CRUD operations."""

    def __init__(self, host="localhost", user="root", password="123456", database="tired"):
        # Database connection configuration
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'cursorclass': DictCursor,
        }

    def _connect(self):
        """Get database connection."""
        return pymysql.connect(**self.config)

    def query_all(self, table: str) -> List[Dict]:
        """Query all data from specified table."""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table}")
                return cursor.fetchall()
        finally:
            conn.close()

    def insert(self, data: Dict, table: str, show_all: bool = False) -> int | Optional[List[Dict]]:
        """Insert a row of data.
        - show_all: True returns all table data, False returns auto-increment primary key id
        """
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                columns = ", ".join(data.keys())
                placeholders = ", ".join(["%s"] * len(data))
                sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
                cursor.execute(sql, tuple(data.values()))
                conn.commit()
                return self.query_all(table) if show_all else cursor.lastrowid
        finally:
            conn.close()

    def update(self, id: int, data: Dict, table: str, show_all: bool = False) -> Optional[List[Dict]]:
        """Update a row by id."""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                set_clause = ", ".join([f"{k}=%s" for k in data.keys()])
                sql = f"UPDATE {table} SET {set_clause} WHERE id = %s"
                cursor.execute(sql, (*data.values(), id))
                conn.commit()
                return self.query_all(table) if show_all else None
        finally:
            conn.close()

    def delete(self, id: int, table: str, show_all: bool = False) -> Optional[List[Dict]]:
        """Delete a row by id."""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                sql = f"DELETE FROM {table} WHERE id = %s"
                cursor.execute(sql, (id,))
                conn.commit()
                return self.query_all(table) if show_all else cursor.lastrowid
        finally:
            conn.close()


class TestTableStore:
    """High-level interface for interacting with the `test` table.
    JSON handling:
    - Writing: Converts list/dict to JSON string; other types remain unchanged
    - Reading: Provides `_parse_json_string` to convert JSON string back to list/dict (used by caller as needed)
    """

    def __init__(self, *, host: str, user: str, password: str, database: str):
        self.db = MysqlDB(host=host, user=user, password=password, database=database)

    @staticmethod
    def _ensure_json_string(value: Optional[object]) -> Optional[str]:
        """Only convert list/dict to JSON string; other types remain unchanged."""
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value  # type: ignore[return-value]

    @staticmethod
    def _parse_json_string(value: Optional[object]) -> Optional[object]:
        """Try to parse JSON string into Python object (list/dict); return unchanged if parsing fails."""
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value

    def insert_row(
            self,
            *,
            name: str,
            dt: Optional[datetime.datetime] = None,
            score: Optional[int] = None,
            audio: Optional[object] = None,
            video: Optional[object] = None,
            record_text: Optional[object] = None,
            rgb: Optional[str] = None,
            depth: Optional[str] = None,
            tobii: Optional[str] = None,
            blood: Optional[str] = None,
            eeg1: Optional[str] = None,
            eeg2: Optional[str] = None,
            ptime: Optional[object] = None,
            accuracy: Optional[float] = None,
            elapsed: Optional[float] = None,
            fatigue_score: Optional[float] = None,
            brain_load_score: Optional[float] = None,
            emotion_score: Optional[float] = None,
    ) -> int:
        """Insert a record.
        JSON fields: record, audio, video, ptime support direct list/dict, will be auto-serialized to JSON string.
        Inference scores: fatigue_score, brain_load_score, emotion_score (0-100)
        """
        record_json = self._ensure_json_string(record_text)
        audio_json = self._ensure_json_string(audio)
        video_json = self._ensure_json_string(video)
        row = {
            "name": name,
            "datetime": dt or datetime.datetime.now(),
            "score": score,
            "audio": audio_json,
            "video": video_json,
            "record": record_json,
            "rgb": rgb,
            "depth": depth,
            "tobii": tobii,
            "blood": blood,
            "eeg1": eeg1,
            "eeg2": eeg2,
            "ptime": ptime,
            "accuracy": accuracy,
            "elapsed": elapsed,
            "fatigue_score": fatigue_score,
            "brain_load_score": brain_load_score,
            "emotion_score": emotion_score,
        }
        return int(self.db.insert(row, "test", show_all=False))

    def update_values(
            self,
            *,
            row_id: int,
            audio: Optional[object] = None,
            video: Optional[object] = None,
            record_text: Optional[object] = None,
            rgb: Optional[str] = None,
            depth: Optional[str] = None,
            tobii: Optional[str] = None,
            blood: Optional[str] = None,
            eeg1: Optional[str] = None,
            eeg2: Optional[str] = None,
            ptime: Optional[object] = None,
            accuracy: Optional[float] = None,
            elapsed: Optional[float] = None,
            score: Optional[float] = None,
            fatigue_score: Optional[float] = None,
            brain_load_score: Optional[float] = None,
            emotion_score: Optional[float] = None,
    ) -> None:
        """Update specified id record fields as needed.
        - For JSON columns (audio, video, ptime, record), if list/dict is passed, will be auto-serialized to JSON string.
        - Fields not provided will not be updated.
        - Inference scores: fatigue_score, brain_load_score, emotion_score (0-100)
        """
        payload: Dict[str, Optional[object]] = {}
        if audio is not None:
            payload["audio"] = self._ensure_json_string(audio)
        if video is not None:
            payload["video"] = self._ensure_json_string(video)
        if record_text is not None:
            payload["record"] = self._ensure_json_string(record_text)
        if rgb is not None:
            payload["rgb"] = rgb
        if depth is not None:
            payload["depth"] = depth
        if tobii is not None:
            payload["tobii"] = tobii
        if blood is not None:
            payload["blood"] = blood
        if eeg1 is not None:
            payload["eeg1"] = eeg1
        if eeg2 is not None:
            payload["eeg2"] = eeg2
        if ptime is not None:
            payload["ptime"] = ptime
        if accuracy is not None:
            payload["accuracy"] = accuracy
        if elapsed is not None:
            payload["elapsed"] = elapsed
        if score is not None:
            payload["score"] = score
        if fatigue_score is not None:
            payload["fatigue_score"] = fatigue_score
        if brain_load_score is not None:
            payload["brain_load_score"] = brain_load_score
        if emotion_score is not None:
            payload["emotion_score"] = emotion_score
        if not payload:
            return
        self.db.update(id=row_id, data=payload, table="test", show_all=False)

    def get_user_history_data(self, username: str, limit: int = 30) -> Dict[str, Any]:
        """Get historical data for specified user, for score display page
        
        Args:
            username: Username
            limit: Maximum number of historical records to return (default: 30, 0 for all)
            
        Returns:
            Dictionary containing current data and historical data
        """
        try:
            # Query all historical records for this user
            all_records = self.db.query_all("test")
            user_records = [record for record in all_records if record.get('name') == username]
            
            # Sort by time, get latest records
            user_records.sort(key=lambda x: x.get('datetime', ''), reverse=True)
            
            # Limit the number of records if specified
            if limit > 0 and len(user_records) > limit:
                user_records = user_records[:limit]
            
            if not user_records:
                # No records found
                return self._empty_history_data()
            
            # Extract historical data and dates
            history_dates = []
            fatigue_history = []
            emotion_history = []
            brain_load_history = []
            systolic_history = []
            diastolic_history = []
            pulse_history = []
            accuracy_history = []
            score_history = []
            
            for record in user_records:
                # Date
                dt = record.get('datetime')
                if dt:
                    history_dates.append(str(dt))
                
                # 从数据库字段读取推理结果
                fatigue_score = record.get('fatigue_score')
                emotion_score = record.get('emotion_score')
                brain_load_score = record.get('brain_load_score')
                
                # 添加到历史列表(如果为None则用0)
                fatigue_history.append(float(fatigue_score) if fatigue_score is not None else 0.0)
                emotion_history.append(float(emotion_score) if emotion_score is not None else 0.0)
                brain_load_history.append(float(brain_load_score) if brain_load_score is not None else 0.0)
                
                # Blood pressure and pulse
                blood_data = record.get('blood', '')
                if blood_data and '/' in blood_data:
                    try:
                        parts = blood_data.split('/')
                        systolic_history.append(int(parts[0]))
                        diastolic_history.append(int(parts[1]))
                        pulse_history.append(int(parts[2]) if len(parts) > 2 else 0)
                    except (ValueError, IndexError):
                        systolic_history.append(0)
                        diastolic_history.append(0)
                        pulse_history.append(0)
                else:
                    systolic_history.append(0)
                    diastolic_history.append(0)
                    pulse_history.append(0)
                
                # Schulte accuracy
                accuracy = record.get('accuracy')
                if accuracy is not None and accuracy != 0:
                    try:
                        accuracy_history.append(float(accuracy))
                    except (ValueError, TypeError):
                        accuracy_history.append(0)
                else:
                    accuracy_history.append(0)
                
                # Schulte comprehensive score
                score = record.get('score')
                if score is not None and score != 0:
                    try:
                        score_history.append(int(score))
                    except (ValueError, TypeError):
                        score_history.append(0)
                else:
                    score_history.append(0)
            
            # Reverse order so latest is at end (for chart display)
            history_dates = history_dates[::-1]
            fatigue_history = fatigue_history[::-1]
            emotion_history = emotion_history[::-1]
            brain_load_history = brain_load_history[::-1]
            systolic_history = systolic_history[::-1]
            diastolic_history = diastolic_history[::-1]
            pulse_history = pulse_history[::-1]
            accuracy_history = accuracy_history[::-1]
            score_history = score_history[::-1]
            
            # Get current (latest) values
            latest = user_records[0]
            blood_data = latest.get('blood', '')
            systolic, diastolic, pulse = 0, 0, 0
            if blood_data and '/' in blood_data:
                try:
                    parts = blood_data.split('/')
                    systolic = int(parts[0])
                    diastolic = int(parts[1])
                    pulse = int(parts[2]) if len(parts) > 2 else 0
                except (ValueError, IndexError):
                    pass
            
            # 计算数据有效性(非零值的数量)
            def count_valid(values):
                return sum(1 for v in values if v is not None and v > 0)
            
            data_validity = {
                "疲劳检测": count_valid(fatigue_history),
                "情绪": count_valid(emotion_history),
                "脑负荷": count_valid(brain_load_history),
                "收缩压": count_valid(systolic_history),
                "舒张压": count_valid(diastolic_history),
                "脉搏": count_valid(pulse_history),
                "舒尔特准确率": count_valid(accuracy_history),
                "舒尔特综合得分": count_valid(score_history),
            }
            
            return {
                "疲劳检测": fatigue_history[-1] if fatigue_history else 0,
                "情绪": emotion_history[-1] if emotion_history else 0,
                "脑负荷": brain_load_history[-1] if brain_load_history else 0,
                "收缩压": systolic,
                "舒张压": diastolic,
                "脉搏": pulse,
                "血压脉搏": systolic,  # For compatibility
                "舒尔特准确率": accuracy_history[-1] if accuracy_history else 0,
                "舒尔特综合得分": score_history[-1] if score_history else 0,
                "历史": {
                    "疲劳检测": fatigue_history,
                    "情绪": emotion_history,
                    "脑负荷": brain_load_history,
                    "收缩压": systolic_history,
                    "舒张压": diastolic_history,
                    "脉搏": pulse_history,
                    "血压脉搏": systolic_history,  # For compatibility
                    "舒尔特准确率": accuracy_history,
                    "舒尔特综合得分": score_history,
                },
                "历史日期": history_dates,
                "数据有效性": data_validity,  # 添加有效性信息
            }
            
        except Exception:
            # If query fails, return empty data
            return self._empty_history_data()
    
    def _empty_history_data(self) -> Dict[str, Any]:
        """Return empty history data structure"""
        return {
            "疲劳检测": 0,
            "情绪": 0,
            "脑负荷": 0,
            "收缩压": 0,
            "舒张压": 0,
            "脉搏": 0,
            "血压脉搏": 0,
            "舒尔特准确率": 0,
            "舒尔特综合得分": 0,
            "历史": {
                "疲劳检测": [0],
                "情绪": [0],
                "脑负荷": [0],
                "收缩压": [0],
                "舒张压": [0],
                "脉搏": [0],
                "血压脉搏": [0],
                "舒尔特准确率": [0],
                "舒尔特综合得分": [0],
            },
            "历史日期": [],
            "数据有效性": {
                "疲劳检测": 0,
                "情绪": 0,
                "脑负荷": 0,
                "收缩压": 0,
                "舒张压": 0,
                "脉搏": 0,
                "血压脉搏": 0,
                "舒尔特准确率": 0,
                "舒尔特综合得分": 0,
            },
        }


def build_store_dir(root: str, username: str, start_time: Optional[str] = None) -> str:
    """Build storage directory, path format: root/username/timestamp."""
    ts = start_time or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    store_dir = os.path.join(root, username, ts)
    os.makedirs(store_dir, exist_ok=True)
    return store_dir


__all__ = ["MysqlDB", "TestTableStore", "build_store_dir"]
