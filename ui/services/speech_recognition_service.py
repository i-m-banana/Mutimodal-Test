"""
Speech Recognition Service

- transcribe_audio: Use Faster-Whisper to transcribe audio to text
- AsyncSpeechRecognizer: Background queue for non-blocking recognition
- Public APIs: add_audio_for_recognition, get_recognition_results,
               clear_recognition_results, stop_recognition

Notes:
- On Windows, set KMP_DUPLICATE_LIB_OK to avoid libiomp conflict with torch
- Model load prefers local cache; can force local-only via UI_WHISPER_LOCAL_ONLY=1
"""
from __future__ import annotations

import datetime
import os
import threading
import time
from typing import Optional, List, Dict

# Prevent libiomp duplication when faster-whisper and torch load together on Windows
if os.name == "nt" and not os.getenv("KMP_DUPLICATE_LIB_OK"):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from faster_whisper import WhisperModel
from opencc import OpenCC

from ui.utils_common.thread_process_manager import get_process_manager


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


WHISPER_MODEL_NAME = os.getenv("UI_WHISPER_MODEL", "base")
WHISPER_MODEL_DIR = os.getenv("UI_WHISPER_MODEL_DIR", "").strip()
FORCE_LOCAL_WHISPER = _env_flag("UI_WHISPER_LOCAL_ONLY")
_WHISPER_MODEL: Optional[WhisperModel] = None
_WHISPER_MODEL_LOCAL_ONLY: Optional[bool] = None
_WHISPER_LOCK = threading.Lock()


def _load_whisper_model(local_only: bool) -> WhisperModel:
    global _WHISPER_MODEL, _WHISPER_MODEL_LOCAL_ONLY
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is not None:
            if _WHISPER_MODEL_LOCAL_ONLY or _WHISPER_MODEL_LOCAL_ONLY == local_only:
                return _WHISPER_MODEL

        model_id = WHISPER_MODEL_DIR or WHISPER_MODEL_NAME
        use_local_only = local_only or bool(WHISPER_MODEL_DIR)

        try:
            model = WhisperModel(
                model_id,
                device="cpu",
                compute_type="int8",
                local_files_only=use_local_only,
            )
            _WHISPER_MODEL = model
            _WHISPER_MODEL_LOCAL_ONLY = use_local_only
            return model
        except Exception as primary_error:
            if use_local_only or FORCE_LOCAL_WHISPER or WHISPER_MODEL_DIR:
                raise

            print(f"⚠️  Whisper 模型联网加载失败，将尝试使用本地缓存: {primary_error}")
            try:
                model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=True,
                )
                _WHISPER_MODEL = model
                _WHISPER_MODEL_LOCAL_ONLY = True
                return model
            except Exception as fallback_error:
                raise RuntimeError(
                    "无法加载 Faster-Whisper 模型。请联网运行一次以下载模型，"
                    "或手动将模型缓存到本地，并可设置 UI_WHISPER_LOCAL_ONLY=1 跳过联网加载。"
                ) from fallback_error


def transcribe_audio(audio_path: str, language: str = "zh", local_files_only: bool = False) -> str:
    """使用 Faster-Whisper 将音频文件转换为文字（固定 CPU + int8，支持离线回退）。"""
    prefer_local_only = local_files_only or FORCE_LOCAL_WHISPER
    model = _load_whisper_model(prefer_local_only)

    segments, _info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    merged: List[str] = []
    prev: Optional[str] = None
    for seg in segments:
        t = (seg.text or "").strip()
        if not t:
            continue
        if t != prev:
            merged.append(t)
            prev = t
    text = "".join(merged)
    cc = OpenCC("t2s")
    return cc.convert(text)


class AsyncSpeechRecognizer:
    """异步语音识别管理器，避免阻塞主线程。"""

    def __init__(self) -> None:
        self._queue: List[Dict] = []
        self._results: List[Dict] = []
        self._lock = threading.Lock()
        self._running = False
        self.process_manager = None
        self.task_id: Optional[str] = None
        self._completed_count = 0
        self._max_tasks = 5  # 最大任务数

    def add(self, *, audio_path: str, question_index: int, question_text: str) -> None:
        with self._lock:
            self._queue.append(
                {
                    "audio_path": audio_path,
                    "question_index": question_index,
                    "question_text": question_text,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        if not self._running:
            self._start()

    def _start(self) -> None:
        self._running = True
        self.process_manager = get_process_manager()
        self.task_id = self.process_manager.submit_inference_task(self._loop, task_name="语音识别")
        try:
            info = self.process_manager.get_process_info()
            print(info)
        except Exception as e:
            print("有问题：", e)
        try:
            info = self.process_manager.get_process_pool_status()
            print(info)
        except Exception as e:
            print("有问题：", e)

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
                text = transcribe_audio(task["audio_path"], language="zh")
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
                    print("✅ 语音识别成功！")
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
                            "recognized_text": "识别结果为空",
                            "audio_path": task["audio_path"],
                            "timestamp": task["timestamp"],
                        }
                    print(f"⚠️  第 {task['question_index']} 题识别结果为空")
                    print(f"   音频文件: {task['audio_path']}")
                    print("-" * 50)

                self._completed_count += 1
                if self._completed_count >= self._max_tasks:
                    print("🎉 所有语音识别任务已完成！")
                    self._stop()
                    break

            except Exception as e:
                print(f"❌ 第 {task['question_index'] + 1} 题语音识别失败: {e}")
                print(f"   音频文件: {task['audio_path']}")
                print("-" * 50)
                self._completed_count += 1
                if self._completed_count >= self._max_tasks:
                    print("🎉 所有语音识别任务已完成！")
                    self._stop()
                    break

    def _stop(self) -> None:
        with self._lock:
            self._running = False
        if self.task_id:
            self.process_manager.cancel_task(self.task_id)
            self.task_id = None

    def get_results(self) -> List[Dict]:
        with self._lock:
            return list(self._results)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
            self._results.clear()
            self._completed_count = 0
            self._running = False


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


def stop_recognition() -> None:
    _get_recognizer()._stop()


def clear_recognition_results() -> None:
    _get_recognizer().clear()
