"""Text-to-speech service running inside the backend process."""

from __future__ import annotations

import base64
import concurrent.futures
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

try:  # pragma: no cover - optional dependency
    import pyttsx3  # type: ignore[import]
except ImportError:  # pragma: no cover
    pyttsx3 = None  # type: ignore[assignment]

try:  # pragma: no cover - optional on non-Windows platforms
    import pythoncom  # type: ignore[import]
except ImportError:  # pragma: no cover
    pythoncom = None  # type: ignore[assignment]

from ..core.thread_pool import get_thread_pool


class TTSUnavailable(RuntimeError):
    """Raised when no TTS backend can be executed."""


class TTSService:
    """Backend-hosted TTS engine that executes speech requests sequentially."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("service.tts")
        self._queue: queue.Queue[Tuple[str, Dict[str, Any], concurrent.futures.Future]] = queue.Queue()
        self._thread_pool = get_thread_pool()
        self._thread_name = "tts-worker"
        thread = self._thread_pool.register_managed_thread(
            self._thread_name,
            self._loop,
            daemon=True
        )
        thread.start()

        default_backend = "powershell" if sys.platform.startswith("win") else "pyttsx3"
        env_backend = os.getenv("UI_TTS_BACKEND", "").strip().lower()
        self._default_backend = env_backend or default_backend
        self._default_voice = os.getenv("UI_TTS_VOICE", "").strip()
        self._default_rate = self._safe_float(os.getenv("UI_TTS_RATE"), 150.0)
        self._default_volume = self._safe_float(os.getenv("UI_TTS_VOLUME"), 1.0)

        self._pyttsx3_engine = None
        self._pyttsx3_voices: List[Any] = []
        self._pyttsx3_voices_logged = False
        self._pythoncom = pythoncom
        self._com_initialized = False

        self._powershell_checked = False
        self._powershell_available = False

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_float(value: Optional[str], default: float) -> float:
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    # ------------------------------------------------------------------
    def speak(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = payload.get("text")
        if not text:
            raise ValueError("'text' is required for tts.speak")

        options = {
            "voice": payload.get("voice") or self._default_voice,
            "rate": self._safe_float(payload.get("rate"), self._default_rate),
            "volume": self._safe_float(payload.get("volume"), self._default_volume),
            "backend": (payload.get("backend") or self._default_backend or "pyttsx3").lower(),
        }
        timeout = payload.get("timeout")
        if timeout is None:
            timeout = self._calculate_timeout(text, options["backend"])
        try:
            timeout_float = float(timeout)
        except (TypeError, ValueError):
            timeout_float = self._calculate_timeout(text, options["backend"])
        options["timeout"] = timeout_float

        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((text, options, future))
        try:
            # Allow a small grace period for cleanup.
            result = future.result(timeout=timeout_float + 5.0)
        except Exception:
            raise
        if result is None:
            raise RuntimeError("TTS execution returned empty result")
        return result

    # ------------------------------------------------------------------
    def _calculate_timeout(self, text: str, backend: str) -> float:
        length = max(len(text), 1)
        if backend == "powershell":
            # PowerShell 的朗读脚本启动和释放 COM 组件会额外耗时，
            # 根据实测需要更长的缓冲时间。
            base = 8.0 + length / 2.0
            return max(15.0, min(240.0, base))
        base = 5.0 + length / 3.5
        return max(10.0, min(180.0, base))

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:  # pragma: no cover - not used in normal flow
                self._queue.task_done()
                break
            text, options, future = item
            try:
                backend_used, elapsed = self._perform_speak(text, options)
                future.set_result({
                    "backend": backend_used,
                    "elapsed": elapsed,
                })
            except Exception as exc:  # pragma: no cover - defensive
                future.set_exception(exc)
            finally:
                self._queue.task_done()

    # ------------------------------------------------------------------
    def _perform_speak(self, text: str, options: Dict[str, Any]) -> Tuple[str, float]:
        preferred = options.get("backend", "pyttsx3")
        attempt_order = self._build_backend_attempt_order(preferred)
        errors: Dict[str, Exception] = {}
        for backend in attempt_order:
            if backend == "powershell":
                if not self._powershell_supported():
                    continue
                try:
                    elapsed = self._speak_with_powershell(text, options)
                    return "powershell", elapsed
                except subprocess.TimeoutExpired as exc:
                    self._disable_powershell(exc)
                    errors["powershell"] = exc
                    continue
                except Exception as exc:
                    self.logger.error("PowerShell TTS失败: %s", exc)
                    errors["powershell"] = exc
                    continue
            elif backend == "pyttsx3":
                if pyttsx3 is None:
                    continue
                try:
                    elapsed = self._speak_with_pyttsx3(text, options)
                    return "pyttsx3", elapsed
                except Exception as exc:
                    self.logger.exception("pyttsx3 TTS 失败")
                    errors["pyttsx3"] = exc
                    continue
        if errors:
            raise TTSUnavailable(
                "; ".join(f"{name}: {err}" for name, err in errors.items())
            )
        raise TTSUnavailable("No TTS backend available")

    # ------------------------------------------------------------------
    def _build_backend_attempt_order(self, preferred: str) -> List[str]:
        pref = (preferred or "pyttsx3").lower()
        order: List[str]
        if pref == "powershell":
            order = ["powershell", "pyttsx3"]
        elif pref == "pyttsx3":
            order = ["pyttsx3", "powershell"]
        else:
            order = [pref, "pyttsx3", "powershell"]
        seen: set[str] = set()
        filtered: List[str] = []
        for backend in order:
            if backend == "powershell" and not self._powershell_supported():
                continue
            if backend not in seen:
                filtered.append(backend)
                seen.add(backend)
        return filtered

    # ------------------------------------------------------------------
    def _powershell_supported(self) -> bool:
        if not sys.platform.startswith("win"):
            return False
        if not self._powershell_checked:
            self._powershell_available = shutil.which("powershell") is not None
            self._powershell_checked = True
        return self._powershell_available

    def _disable_powershell(self, exc: Exception) -> None:
        if self._powershell_available:
            self.logger.warning("PowerShell TTS 将被禁用: %s", exc)
        self._powershell_available = False
        self._powershell_checked = True

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "default_backend": self._default_backend,
            "powershell_available": self._powershell_supported(),
            "pyttsx3_available": pyttsx3 is not None,
        }

    # ------------------------------------------------------------------
    def _speak_with_powershell(self, text: str, options: Dict[str, Any]) -> float:
        volume = int(max(0, min(options.get("volume", 1.0), 1.0)) * 100)
        rate = self._convert_powershell_rate(options.get("rate", 150.0))
        voice = options.get("voice") or ""

        script_parts = [
            "Add-Type -AssemblyName System.Speech",
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer",
            f"$synth.Volume = {volume}",
            f"$synth.Rate = {rate}",
        ]
        if voice:
            pattern = f"*{voice}*"
            pattern_json = json.dumps(pattern)
            voice_script = (
                "$voice = $synth.GetInstalledVoices() | Where-Object { "
                f"$_.VoiceInfo.Name -like {pattern_json} -or "
                f"$_.VoiceInfo.Culture.Name -like {pattern_json} -or "
                f"$_.VoiceInfo.Description -like {pattern_json} "
                "} | Select-Object -First 1"
            )
            script_parts.append(voice_script)
            script_parts.append("if ($voice) { $synth.SelectVoice($voice.VoiceInfo.Name) }")
        sanitized_text = json.dumps(text)
        encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
        script_parts.extend([
            f"$bytes = [System.Convert]::FromBase64String('{encoded_text}')",
            "$text = [System.Text.Encoding]::UTF8.GetString($bytes)",
            "$synth.Speak($text)",
        ])
        script = "; ".join(script_parts)
        timeout_opt = options.get("timeout")
        if timeout_opt is None:
            timeout = max(10, min(240, int(len(text) / 2 + 20)))
        else:
            try:
                timeout = max(10, min(240, int(float(timeout_opt))))
            except (TypeError, ValueError):
                timeout = max(10, min(240, int(len(text) / 2 + 20)))
        start = time.perf_counter()
        result = subprocess.run(
            [
                "powershell",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            stdout = result.stdout.strip()
            raise RuntimeError(stderr or stdout or "PowerShell speech failed")
        return time.perf_counter() - start

    # ------------------------------------------------------------------
    def _ensure_pyttsx3_engine(self) -> Any:
        if self._pyttsx3_engine is not None:
            return self._pyttsx3_engine
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not installed")
        if self._pythoncom and not self._com_initialized:
            try:
                self._pythoncom.CoInitialize()  # type: ignore[attr-defined]
                self._com_initialized = True
            except Exception as exc:
                self.logger.warning("COM 初始化失败: %s", exc)
        engine = pyttsx3.init()
        engine.connect('error', lambda name, exc: self.logger.error("TTS 引擎执行失败 [%s]: %s", name, exc))
        self._pyttsx3_engine = engine
        try:
            self._pyttsx3_voices = engine.getProperty('voices') or []
            if self._pyttsx3_voices and not self._pyttsx3_voices_logged:
                desc = "; ".join(
                    f"{idx}:{voice.name} ({voice.id})" for idx, voice in enumerate(self._pyttsx3_voices)
                )
                self.logger.info("TTS 可用语音列表: %s", desc)
                self._pyttsx3_voices_logged = True
        except Exception as exc:
            self.logger.warning("获取 TTS 语音列表失败: %s", exc)
        return engine

    # ------------------------------------------------------------------
    def _speak_with_pyttsx3(self, text: str, options: Dict[str, Any]) -> float:
        engine = self._ensure_pyttsx3_engine()
        voice = options.get("voice") or ""
        selected_voice_id = self._match_voice(engine, voice)
        if selected_voice_id:
            try:
                engine.setProperty('voice', selected_voice_id)
                self.logger.debug("pyttsx3 使用语音: %s", selected_voice_id)
            except Exception as exc:
                self.logger.warning("设置语音失败(%s): %s", selected_voice_id, exc)
        elif voice:
            self.logger.warning("未找到匹配语音 '%s'，继续使用默认语音", voice)
        rate = int(options.get("rate", 150.0))
        volume = max(0.0, min(float(options.get("volume", 1.0)), 1.0))
        try:
            engine.setProperty('rate', rate)
        except Exception as exc:
            self.logger.debug("设置朗读速度失败: %s", exc)
        try:
            engine.setProperty('volume', volume)
        except Exception as exc:
            self.logger.debug("设置朗读音量失败: %s", exc)
        start = time.perf_counter()
        utterance_id = f"tts-{time.time_ns()}"
        engine.say(text, utterance_id)
        engine.runAndWait()
        return time.perf_counter() - start

    # ------------------------------------------------------------------
    def _match_voice(self, engine: Any, voice_hint: str) -> Optional[str]:
        if not voice_hint:
            # 根据默认偏好自动匹配中文语音
            for voice in self._pyttsx3_voices:
                blob = f"{voice.id} {voice.name}".lower()
                if any(keyword in blob for keyword in ["zh", "chs", "chinese", "xiaoyi", "xiaoyan", "huihui", "晓", "语音"]):
                    return voice.id
            return None
        target = voice_hint.lower()
        for voice in self._pyttsx3_voices:
            blob = f"{voice.id} {voice.name}".lower()
            if target in blob:
                return voice.id
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _convert_powershell_rate(rate_value: float) -> int:
        try:
            base = float(rate_value)
        except (TypeError, ValueError):
            base = 150.0
        normalized = int(round((base - 150.0) / 20.0))
        return max(-10, min(10, normalized))

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        try:
            self._queue.put_nowait(None)
        except Exception:  # pragma: no cover - best effort
            pass
        self._thread_pool.unregister_managed_thread(self._thread_name, timeout=2.0)
        if self._pythoncom and self._com_initialized:
            try:
                self._pythoncom.CoUninitialize()  # type: ignore[attr-defined]
            except Exception:
                pass
            self._com_initialized = False
        self._pyttsx3_engine = None


__all__ = ["TTSService", "TTSUnavailable"]
