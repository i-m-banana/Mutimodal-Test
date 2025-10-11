"""

- build_store_dir：构建"root/username/时间戳"存储目录
- transcribe_audio：使用 Faster-Whisper 将音频文件转换为文字
"""

import datetime
import os
import threading
import time
from typing import Optional, List, Dict
from faster_whisper import WhisperModel
from opencc import OpenCC


def build_store_dir(root: str, username: str, start_time: Optional[str] = None) -> str:
    """构建存储目录，路径格式：root/username/时间戳。"""
    ts = start_time or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    store_dir = os.path.join(root, username, ts)
    os.makedirs(store_dir, exist_ok=True)
    return store_dir


def transcribe_audio(audio_path: str, language: str = "zh", local_files_only = False) -> str:
    """使用 Faster-Whisper 将音频文件转换为文字（固定 CPU + int8，基础优化）"""
    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
        local_files_only=local_files_only
    )

    # 转写音频：启用 VAD、束搜索，固定中文，减少重复与幻觉
    segments, info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    # 合并文本并去除相邻重复
    merged = []
    prev = None
    for seg in segments:
        t = (seg.text or "").strip()
        if not t:
            continue
        if t != prev:
            merged.append(t)
            prev = t
    text = "".join(merged)
    # 繁转简（t2s）
    cc = OpenCC('t2s')
    return cc.convert(text)


class AsyncSpeechRecognizer:
    """异步语音识别管理器，避免阻塞主线程。"""

    def __init__(self) -> None:
        self._queue: List[Dict] = []
        self._results: List[Dict] = []
        self._lock = threading.Lock()
        self._running = False
        self._worker: Optional[threading.Thread] = None

    def add(self, *, audio_path: str, question_index: int, question_text: str) -> None:
        with self._lock:
            self._queue.append({
                "audio_path": audio_path,
                "question_index": question_index,
                "question_text": question_text,
                "timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            })
        if not self._running:
            self._start()

    def _start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self) -> None:
        while self._running:
            task: Optional[Dict] = None
            with self._lock:
                if self._queue:
                    task = self._queue.pop(0)
            if task is None:
                time.sleep(0.05)
                continue
            try:
                print(f"🎤 开始识别第 {task['question_index']} 题音频...")
                text = transcribe_audio(task["audio_path"], language="zh", local_files_only=False)
                if text and text.strip():
                    with self._lock:
                        result = {
                            "question_index": task["question_index"],
                            "question_text": task["question_text"],
                            "recognized_text": text.strip(),
                            "audio_path": task["audio_path"],
                            "timestamp": task["timestamp"],
                        }
                        self._results.append(result)
                    
                    # 输出识别成功提示
                    print(f"✅ 语音识别成功！")
                    print(f"   题目: {task['question_text']}")
                    print(f"   识别结果: {text.strip()}")
                    print(f"   音频文件: {task['audio_path']}")
                    print(f"   时间: {task['timestamp']}")
                    print("-" * 50)
                else:
                    with self._lock:
                        result = {
                            "question_index": task["question_index"],
                            "question_text": task["question_text"],
                            "recognized_text": '识别结果为空',
                            "audio_path": task["audio_path"],
                            "timestamp": task["timestamp"],
                        }                    
                    print(f"⚠️  第 {task['question_index'] + 1} 题识别结果为空")
                    print(f"   音频文件: {task['audio_path']}")
                    print("-" * 50)
                if task["question_index"]>5:
                    print('5条语音均已识别完成')
            except Exception as e:
                print(f"❌ 第 {task['question_index'] + 1} 题语音识别失败: {e}")
                print(f"   音频文件: {task['audio_path']}")
                print("-" * 50)

    def get_results(self) -> List[Dict]:
        with self._lock:
            return list(self._results)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._results.clear()


_GLOBAL_RECOGNIZER: Optional[AsyncSpeechRecognizer] = None


def _get_recognizer() -> AsyncSpeechRecognizer:
    global _GLOBAL_RECOGNIZER
    if _GLOBAL_RECOGNIZER is None:
        _GLOBAL_RECOGNIZER = AsyncSpeechRecognizer()
    return _GLOBAL_RECOGNIZER


def add_audio_for_recognition(audio_path: str, question_index: int, question_text: str) -> None:
    _get_recognizer().add(audio_path=audio_path, question_index=question_index, question_text=question_text)


def get_recognition_results() -> List[Dict]:
    return _get_recognizer().get_results()


def clear_recognition_results() -> None:
    _get_recognizer().clear()