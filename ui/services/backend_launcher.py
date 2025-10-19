"""Automatic backend orchestrator launcher for UI."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from ..runtime.ui_thread_pool import get_ui_thread_pool
except ImportError:  # pragma: no cover - fallback
    from ui.runtime.ui_thread_pool import get_ui_thread_pool  # type: ignore

_logger = logging.getLogger("ui.backend_launcher")


class BackendLauncher:
    """Manages the backend orchestrator process lifecycle."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parents[2]
        self._process: Optional[subprocess.Popen] = None
        self._thread_pool = get_ui_thread_pool()
        self._monitor_thread_name = "ui-backend-monitor"
        self._should_stop_event = None  # 将在start时创建

    def start(self) -> bool:
        """Start the backend orchestrator in a subprocess."""
        if self._process is not None and self._process.poll() is None:
            _logger.info("后端服务器已在运行中")
            return True

        try:
            # 使用当前Python解释器启动后端
            cmd = [
                sys.executable,
                "-m", "src.main",
                "--root", str(self.project_root)
            ]
            
            _logger.info("启动后端服务器: %s", " ".join(cmd))
            
            # 在Windows上需要CREATE_NO_WINDOW标志来避免弹出控制台窗口
            startupinfo = None
            if sys.platform == "win32":
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
            
            self._process = subprocess.Popen(
                cmd,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                startupinfo=startupinfo,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # 启动监控线程来记录后端输出
            import threading  # 本地导入用于Event
            self._should_stop_event = threading.Event()
            thread = self._thread_pool.register_managed_thread(
                self._monitor_thread_name,
                self._monitor_output,
                daemon=True
            )
            thread.start()
            
            # 等待后端启动（最多10秒，检查 WebSocket 服务器启动）
            _logger.info("等待后端 WebSocket 服务器启动...")
            for i in range(100):
                if self._process.poll() is not None:
                    # 进程退出了，读取错误输出
                    stderr_output = ""
                    if self._process.stderr:
                        try:
                            stderr_output = self._process.stderr.read()
                        except:
                            pass
                    _logger.error("后端服务器启动失败，进程已退出。错误: %s", stderr_output[:500])
                    return False
                time.sleep(0.1)
                # 给后端更多时间启动（2秒）
                if i >= 20:
                    _logger.info("✅ 后端服务器已启动 (PID: %d)", self._process.pid)
                    return True
            
            return True
            
        except Exception as exc:
            _logger.error("启动后端服务器失败: %s", exc)
            return False

    def stop(self) -> None:
        """Stop the backend orchestrator."""
        if self._process is None:
            return

        try:
            _logger.info("正在停止后端服务器...")
            if self._should_stop_event:
                self._should_stop_event.set()
            
            # 尝试优雅关闭
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
                _logger.info("后端服务器已正常停止")
            except subprocess.TimeoutExpired:
                _logger.warning("后端服务器未响应，强制终止")
                self._process.kill()
                self._process.wait()
                
        except Exception as exc:
            _logger.warning("停止后端服务器时出错: %s", exc)
        finally:
            self._process = None
            self._thread_pool.unregister_managed_thread(self._monitor_thread_name, timeout=2.0)

    def is_running(self) -> bool:
        """Check if the backend is running."""
        return self._process is not None and self._process.poll() is None

    def _monitor_output(self) -> None:
        """Monitor and log backend process output."""
        if self._process is None or self._process.stdout is None:
            return

        backend_logger = logging.getLogger("backend")
        
        try:
            # 记录前50行启动日志，帮助诊断
            line_count = 0
            for line in self._process.stdout:
                if self._should_stop_event and self._should_stop_event.is_set():
                    break
                line = line.rstrip()
                if line:
                    # 前50行全部记录
                    if line_count < 50:
                        backend_logger.info("[Backend] %s", line)
                        line_count += 1
                    else:
                        # 之后只记录重要日志
                        if any(keyword in line.lower() for keyword in [
                            "error", "warning", "exception", "failed", 
                            "started", "stopped", "listening", "critical"
                        ]):
                            backend_logger.info("[Backend] %s", line)
        except Exception as exc:
            if not (self._should_stop_event and self._should_stop_event.is_set()):
                _logger.debug("后端输出监控中断: %s", exc)


# 全局单例
_backend_launcher: Optional[BackendLauncher] = None


def get_backend_launcher() -> BackendLauncher:
    """Get or create the global backend launcher instance."""
    global _backend_launcher
    if _backend_launcher is None:
        _backend_launcher = BackendLauncher()
    return _backend_launcher


def start_backend() -> bool:
    """Start the backend orchestrator (convenience function)."""
    return get_backend_launcher().start()


def stop_backend() -> None:
    """Stop the backend orchestrator (convenience function)."""
    get_backend_launcher().stop()


__all__ = ["BackendLauncher", "get_backend_launcher", "start_backend", "stop_backend"]
