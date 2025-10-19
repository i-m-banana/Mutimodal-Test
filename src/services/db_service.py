"""Database command handlers exposed to the UI command router."""



import logging

import datetime
import json
import pymysql
from pymysql.cursors import DictCursor

def _env_flag(name: str) -> bool:
    import os
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}

class DatabaseUnavailable(RuntimeError):
    """Raised when the database layer is disabled or missing."""

class MysqlDB:
    """MySQL wrapper providing basic CRUD operations."""
    def __init__(self, host="localhost", user="root", password="123456", database="tired"):
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'cursorclass': DictCursor,
        }
    def _connect(self):
        return pymysql.connect(**self.config)
    def query_all(self, table: str):
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table}")
                return cursor.fetchall()
        finally:
            conn.close()
    def insert(self, data: dict, table: str, show_all: bool = False):
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
    def update(self, id: int, data: dict, table: str, show_all: bool = False):
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
    def delete(self, id: int, table: str, show_all: bool = False):
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
    def __init__(self, *, host: str, user: str, password: str, database: str):
        self.db = MysqlDB(host=host, user=user, password=password, database=database)
    @staticmethod
    def _ensure_json_string(value):
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value
    @staticmethod
    def _parse_json_string(value):
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
    def insert_row(self, *, name, dt=None, score=None, audio=None, video=None, record_text=None, rgb=None, depth=None, tobii=None, blood=None, eeg1=None, eeg2=None, ptime=None, accuracy=None, elapsed=None, fatigue_score=None, brain_load_score=None, emotion_score=None):
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
    def update_values(self, *, row_id, audio=None, video=None, record_text=None, rgb=None, depth=None, tobii=None, blood=None, eeg1=None, eeg2=None, ptime=None, accuracy=None, elapsed=None, score=None, fatigue_score=None, brain_load_score=None, emotion_score=None):
        payload = {}
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
    def get_user_history_data(self, username, limit=30):
        try:
            all_records = self.db.query_all("test")
            user_records = [record for record in all_records if record.get('name') == username]
            user_records.sort(key=lambda x: x.get('datetime', ''), reverse=True)
            if limit > 0 and len(user_records) > limit:
                user_records = user_records[:limit]
            if not user_records:
                return self._empty_history_data()
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
                dt = record.get('datetime')
                if dt:
                    history_dates.append(str(dt))
                fatigue_score = record.get('fatigue_score')
                emotion_score = record.get('emotion_score')
                brain_load_score = record.get('brain_load_score')
                fatigue_history.append(float(fatigue_score) if fatigue_score is not None else 0.0)
                emotion_history.append(float(emotion_score) if emotion_score is not None else 0.0)
                brain_load_history.append(float(brain_load_score) if brain_load_score is not None else 0.0)
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
                accuracy = record.get('accuracy')
                if accuracy is not None and accuracy != 0:
                    try:
                        accuracy_history.append(float(accuracy))
                    except (ValueError, TypeError):
                        accuracy_history.append(0)
                else:
                    accuracy_history.append(0)
                score = record.get('score')
                if score is not None and score != 0:
                    try:
                        score_history.append(int(score))
                    except (ValueError, TypeError):
                        score_history.append(0)
                else:
                    score_history.append(0)
            history_dates = history_dates[::-1]
            fatigue_history = fatigue_history[::-1]
            emotion_history = emotion_history[::-1]
            brain_load_history = brain_load_history[::-1]
            systolic_history = systolic_history[::-1]
            diastolic_history = diastolic_history[::-1]
            pulse_history = pulse_history[::-1]
            accuracy_history = accuracy_history[::-1]
            score_history = score_history[::-1]
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
                "数据有效性": data_validity,
            }
        except Exception:
            return self._empty_history_data()
    def _empty_history_data(self):
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

class DatabaseService:
    _INSERT_FIELDS = {
        "name",
        "dt",
        "score",
        "audio",
        "video",
        "record_text",
        "rgb",
        "depth",
        "tobii",
        "blood",
        "eeg1",
        "eeg2",
        "ptime",
        "accuracy",
        "elapsed",
        "fatigue_score",
        "brain_load_score",
        "emotion_score",
    }
    _UPDATE_FIELDS = {
        "audio",
        "video",
        "record_text",
        "rgb",
        "depth",
        "tobii",
        "blood",
        "eeg1",
        "eeg2",
        "ptime",
        "accuracy",
        "elapsed",
        "score",
        "fatigue_score",
        "brain_load_score",
        "emotion_score",
    }
    def __init__(self, logger=None):
        import os
        self.logger = logger or logging.getLogger("service.db")
        self._store = None
        self._db_disabled = _env_flag("UI_SKIP_DATABASE")
        self._config = {
            "host": os.getenv("UI_DB_HOST", "localhost"),
            "user": os.getenv("UI_DB_USER", "root"),
            "password": os.getenv("UI_DB_PASSWORD", "123456"),
            "database": os.getenv("UI_DB_NAME", "tired"),
        }
        if self._db_disabled:
            self.logger.warning("Database commands will be disabled (UI_SKIP_DATABASE=1)")
    def _ensure_store(self):
        if self._db_disabled:
            raise DatabaseUnavailable("database access disabled (UI_SKIP_DATABASE=1 or module missing)")
        if self._store is None:
            try:
                self._store = TestTableStore(**self._config)
                self.logger.info(
                    "Database connection ready -> %s/%s",
                    self._config["host"],
                    self._config["database"],
                )
            except Exception as exc:
                self.logger.error("Failed to connect to database: %s", exc)
                raise DatabaseUnavailable(f"database connection failed: {exc}") from exc
        return self._store
    def insert_test_record(self, payload):
        store = self._ensure_store()
        fields = {key: payload.get(key) for key in self._INSERT_FIELDS if key in payload}
        if not fields.get("name"):
            raise ValueError("'name' is required to insert a test record")
        try:
            row_id = store.insert_row(**fields)
            self.logger.debug("Inserted test record for %s with id %s", fields.get("name"), row_id)
            return {"row_id": row_id}
        except Exception:
            self.logger.exception("Insert test record failed")
            raise
    def update_test_record(self, payload):
        store = self._ensure_store()
        row_id = payload.get("row_id")
        if not row_id:
            raise ValueError("'row_id' is required to update a test record")
        updates = {key: payload.get(key) for key in self._UPDATE_FIELDS if key in payload}
        if not updates:
            self.logger.debug("No update fields supplied for row %s", row_id)
            return {"row_id": row_id, "updated": False}
        try:
            store.update_values(row_id=row_id, **updates)
            self.logger.debug("Updated test record %s with keys %s", row_id, list(updates.keys()))
            return {"row_id": row_id, "updated": True}
        except Exception:
            self.logger.exception("Update test record %s failed", row_id)
            raise
    def get_user_history(self, payload):
        store = self._ensure_store()
        username = payload.get("name") or payload.get("username")
        if not username:
            raise ValueError("'name' is required to fetch user history")
        limit = payload.get("limit", 30)
        if not isinstance(limit, int) or limit < 0:
            limit = 30
        try:
            history = store.get_user_history_data(username, limit=limit)
            return {"history": history}
        except Exception:
            self.logger.exception("Fetching history for %s failed", username)
            raise
    def diagnostics(self):
        return {
            "disabled": self._db_disabled,
            "configured_host": self._config.get("host"),
            "configured_database": self._config.get("database"),
        }

__all__ = ["DatabaseService", "DatabaseUnavailable"]
