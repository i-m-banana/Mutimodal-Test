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
    ) -> int:
        """Insert a record.
        JSON fields: record, audio, video, ptime support direct list/dict, will be auto-serialized to JSON string.
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
    ) -> None:
        """Update specified id record fields as needed.
        - For JSON columns (audio, video, ptime, record), if list/dict is passed, will be auto-serialized to JSON string.
        - Fields not provided will not be updated.
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
        if not payload:
            return
        self.db.update(id=row_id, data=payload, table="test", show_all=False)

    def get_user_history_data(self, username: str) -> Dict[str, Any]:
        """Get historical data for specified user, for score display page
        
        Args:
            username: Username
            
        Returns:
            Dictionary containing current data and historical data
        """
        try:
            # Query all historical records for this user
            all_records = self.db.query_all("test")
            user_records = [record for record in all_records if record.get('name') == username]
            
            # Sort by time, get latest records
            user_records.sort(key=lambda x: x.get('datetime', ''), reverse=True)
            
            # Extract historical data
            blood_pressure_history = []
            accuracy_history = []
            score_history = []
            
            for record in user_records:
                # Blood pressure (only take systolic)
                blood_data = record.get('blood', '')
                if blood_data and '/' in blood_data:
                    try:
                        systolic = int(blood_data.split('/')[0])
                        blood_pressure_history.append(systolic)
                    except (ValueError, IndexError):
                        pass
                
                # Schulte accuracy
                accuracy = record.get('accuracy')
                if accuracy is not None:
                    accuracy_history.append(float(accuracy))
                
                # Schulte comprehensive score
                score = record.get('score')
                if score is not None:
                    score_history.append(int(score))
            
            # Get current latest data
            current_blood = blood_pressure_history[0] if blood_pressure_history else 0
            current_accuracy = accuracy_history[0] if accuracy_history else 0
            current_score = score_history[0] if score_history else 0
            
            return {
                "血压脉搏": current_blood,
                "舒尔特准确率": current_accuracy,
                "舒尔特综合得分": current_score,
                "历史": {
                    "血压脉搏": blood_pressure_history[::-1] if blood_pressure_history else [0],  # Reverse order, latest at end
                    "舒尔特准确率": accuracy_history[::-1] if accuracy_history else [0],  # Reverse order, latest at end
                    "舒尔特综合得分": score_history[::-1] if score_history else [0],  # Reverse order, latest at end
                }
            }
            
        except Exception:
            # If query fails, return empty data
            return {
                "血压脉搏": 0,
                "舒尔特准确率": 0,
                "舒尔特综合得分": 0,
                "历史": {
                    "血压脉搏": [0],
                    "舒尔特准确率": [0],
                    "舒尔特综合得分": [0],
                }
            }


def build_store_dir(root: str, username: str, start_time: Optional[str] = None) -> str:
    """Build storage directory, path format: root/username/timestamp."""
    ts = start_time or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    store_dir = os.path.join(root, username, ts)
    os.makedirs(store_dir, exist_ok=True)
    return store_dir


__all__ = ["MysqlDB", "TestTableStore", "build_store_dir"]
