"""Collector that watches a directory and batches newly arrived files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..constants import EventTopic
from ..core.event_bus import Event
from .base_collector import BaseCollector


class FileCollector(BaseCollector):
    def __init__(self, name: str, bus, *, options: Optional[Dict[str, Any]] = None, logger=None) -> None:
        super().__init__(name, bus, options=options, logger=logger)
        watch_dir = self.options.get("watch_dir", "data/raw")
        self.watch_dir = Path(watch_dir)
        self.pattern = self.options.get("pattern", "*.json")
        self._seen: set[Path] = set()

    def on_start(self) -> None:
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self._seen = set(self.watch_dir.glob(self.pattern))

    def run_once(self) -> None:
        new_files = sorted(set(self.watch_dir.glob(self.pattern)) - self._seen)
        if not new_files:
            return
        self._seen.update(new_files)
        payload = {
            "collector": self.name,
            "timestamp": self._current_time(),
            "files": [str(path) for path in new_files],
        }
        self.bus.publish(Event(topic=EventTopic.FILE_BATCH, payload=payload))

    def _current_time(self) -> float:
        import time

        return time.time()


__all__ = ["FileCollector"]
