"""
脚本提供以下工具：
- MysqlDB：提供增删改查功能
- TestTableStore：test表的记录新增和更改
- build_store_dir：构建"root/username/时间戳"存储目录
"""


import datetime
import json
import os
import pymysql
from pymysql.cursors import DictCursor
from typing import List, Dict, Optional


class MysqlDB:
    """MySQL 简易封装：提供增删改查基础能力。"""

    def __init__(self, host="localhost", user="root", password="", database="test_db"):
        # 数据库连接配置
        self.config = {
            'host': host,
            'user': user,
            'password': password,
            'database': database,
            'charset': 'utf8mb4',
            'cursorclass': DictCursor,
        }

    def _connect(self):
        """获取数据库连接。"""
        return pymysql.connect(**self.config)

    def query_all(self, table) -> List[Dict]:
        """查询指定表的全部数据。"""
        conn = self._connect()
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table}")
                return cursor.fetchall()
        finally:
            conn.close()

    def insert(self, data: Dict, table, show_all=False) -> int | Optional[List[Dict]]:
        """插入一行数据。
        - show_all: True 时返回整表数据，False 返回自增主键 id
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

    def update(self, id: int, data: Dict, table, show_all=False) -> Optional[List[Dict]]:
        """按 id 更新一行数据。"""
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

    def delete(self, id: int, table, show_all=False) -> Optional[List[Dict]]:
        """按 id 删除一行数据。"""
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
    """与 `test` 表交互的高层接口。
    对JSON处理：
    - 写入时：将 list/dict 转为 JSON 字符串；其他类型保持不变
    - 读取时：提供 `_parse_json_string` 将 JSON 字符串转回 list/dict（由调用方按需使用）
    """

    def __init__(self, *, host: str, user: str, password: str, database: str):
        self.db = MysqlDB(host=host, user=user, password=password, database=database)

    @staticmethod
    def _ensure_json_string(value: Optional[object]) -> Optional[str]:
        """仅将 list/dict 转为 JSON 字符串；其他类型保持原样。"""
        if value is None:
            return None
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        return value  # type: ignore[return-value]

    @staticmethod
    def _parse_json_string(value: Optional[object]) -> Optional[object]:
        """尝试将 JSON 字符串解析为 Python 对象(list/dict)；解析失败则原样返回。"""
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
        """插入一条记录。
        JSON 字段：record、audio、video、ptime 支持直接传 list/dict，将自动序列化为 JSON 字符串。
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
        """按需更新指定 id 记录的字段。
        - 对 JSON 列（audio、video、ptime、record）若传入 list/dict，将自动序列化为 JSON 字符串。
        - 未提供的字段不会更新。
        """
        payload: Dict[str, Optional[object]] = {}
        if audio is not None:
            payload["audio"] = self._ensure_json_string(audio)
        if video is not None:
            payload["video"] = self._ensure_json_string(video)
        if record_text is not None:
            print(self._ensure_json_string(record_text))
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

    def get_user_history_data(self, username: str) -> Dict[str, any]:
        """获取指定用户的历史数据，用于分数展示页面
        
        Args:
            username: 用户名
            
        Returns:
            包含当前数据和历史数据的字典
        """
        try:
            # 查询该用户的所有历史记录
            all_records = self.db.query_all("test")
            user_records = [record for record in all_records if record.get('name') == username]
            
            # 按时间排序，获取最新的记录
            user_records.sort(key=lambda x: x.get('datetime', ''), reverse=True)
            
            # 提取历史数据
            blood_pressure_history = []
            accuracy_history = []
            score_history = []
            
            for record in user_records:
                # 血压脉搏（只取收缩压）
                blood_data = record.get('blood', '')
                if blood_data and '/' in blood_data:
                    try:
                        systolic = int(blood_data.split('/')[0])
                        blood_pressure_history.append(systolic)
                    except (ValueError, IndexError):
                        pass
                
                # 舒尔特准确率
                accuracy = record.get('accuracy')
                if accuracy is not None:
                    accuracy_history.append(float(accuracy))
                
                # 舒尔特综合得分
                score = record.get('score')
                if score is not None:
                    score_history.append(int(score))
            
            # 获取当前最新数据
            current_blood = blood_pressure_history[0] if blood_pressure_history else 0
            current_accuracy = accuracy_history[0] if accuracy_history else 0
            current_score = score_history[0] if score_history else 0
            
            return {
                "血压脉搏": current_blood,
                "舒尔特准确率": current_accuracy,
                "舒尔特综合得分": current_score,
                "历史": {
                    "血压脉搏": blood_pressure_history[::-1] if blood_pressure_history else [0],  # 反转顺序，最新的在后面
                    "舒尔特准确率": accuracy_history[::-1] if accuracy_history else [0],  # 反转顺序，最新的在后面
                    "舒尔特综合得分": score_history[::-1] if score_history else [0],  # 反转顺序，最新的在后面
                }
            }
            
        except Exception as e:
            # 如果查询失败，返回空数据
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
    """构建存储目录，路径格式：root/username/时间戳。"""
    ts = start_time or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    store_dir = os.path.join(root, username, ts)
    os.makedirs(store_dir, exist_ok=True)
    return store_dir
