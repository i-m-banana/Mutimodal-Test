"""OCR detector placeholder reacting to file batches."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..constants import EventTopic
from ..core.event_bus import Event
from .base_detector import BaseDetector


class OCRDetector(BaseDetector):
    def topics(self) -> Iterable[EventTopic]:
        return [EventTopic.FILE_BATCH]

    def handle_event(self, event: Event) -> None:
        files = event.payload.get("files", [])
        summaries = []
        for path in files:
            summaries.append(self._summarise(Path(path)))
        payload = {
            "detector": self.name,
            "files": files,
            "summaries": summaries,
        }
        self.bus.publish(Event(topic=EventTopic.DETECTION_RESULT, payload=payload))

    def _summarise(self, path: Path) -> str:
        if not path.exists():
            return f"missing:{path.name}"
        try:
            with path.open("r", encoding="utf-8") as fh:
                head = fh.read(80).replace("\n", " ")
            return f"{path.name}:{head[:60]}"
        except Exception:
            return f"unreadable:{path.name}"


__all__ = ["OCRDetector"]
