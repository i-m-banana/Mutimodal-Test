"""Helpers for loading YAML configuration files with optional auto-reload."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import yaml


@dataclass(frozen=True)
class ConfigSnapshot:
    """Immutable wrapper holding a configuration payload and metadata."""

    content: Dict[str, Any]
    path: Path
    mtime: float

    def to_json(self) -> str:
        return json.dumps(self.content, ensure_ascii=False, indent=2)


class ConfigLoader:
    """Load YAML files and notify subscribers on changes."""

    def __init__(self, path: str | Path, *, auto_reload: bool = False, poll_interval: float = 3.0):
        self._path = Path(path)
        self._auto_reload = auto_reload
        self._poll_interval = poll_interval
        self._subscribers: list[Callable[[ConfigSnapshot], None]] = []
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot = self._load()
        if auto_reload:
            self._start_watcher()

    def _load(self) -> ConfigSnapshot:
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        snapshot = ConfigSnapshot(content=data, path=self._path, mtime=self._path.stat().st_mtime)
        return snapshot

    def get(self) -> ConfigSnapshot:
        with self._lock:
            return self._snapshot

    def subscribe(self, callback: Callable[[ConfigSnapshot], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def _notify(self, snapshot: ConfigSnapshot) -> None:
        for callback in list(self._subscribers):
            try:
                callback(snapshot)
            except Exception:  # pragma: no cover - defensive logging handled by caller
                pass

    def _start_watcher(self) -> None:
        self._thread = threading.Thread(target=self._watch_loop, name=f"ConfigWatcher:{self._path.name}", daemon=True)
        self._thread.start()

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                stat = self._path.stat()
            except FileNotFoundError:
                time.sleep(self._poll_interval)
                continue
            with self._lock:
                if stat.st_mtime > self._snapshot.mtime:
                    self._snapshot = self._load()
                    self._notify(self._snapshot)
            time.sleep(self._poll_interval)

    def reload_now(self) -> ConfigSnapshot:
        with self._lock:
            self._snapshot = self._load()
            self._notify(self._snapshot)
            return self._snapshot

    def close(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._poll_interval * 2)


__all__ = ["ConfigLoader", "ConfigSnapshot"]
