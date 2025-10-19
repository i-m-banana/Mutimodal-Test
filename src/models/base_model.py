"""Base inference model abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional


class BaseModel(ABC):
    def __init__(self, name: str, model_path: Optional[str] = None, **kwargs: Any) -> None:
        self.name = name
        self.model_path = Path(model_path) if model_path else None
        self.kwargs = kwargs

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def unload(self) -> None:
        """Optional cleanup hook."""


__all__ = ["BaseModel"]
