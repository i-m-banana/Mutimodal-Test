"""PyTorch model wrapper with graceful degradation when torch is unavailable."""

from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover
    torch = None

from .base_model import BaseModel


class TorchModel(BaseModel):
    """Wrap a TorchScript model or provide a deterministic stub during testing."""

    def __init__(self, name: str, model_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(name, model_path, **kwargs)
        self._model = None

    def load(self) -> None:
        if torch is None or self.model_path is None:
            return
        self._model = torch.jit.load(str(self.model_path)) if self.model_path.suffix in {".pt", ".pth"} else torch.load(str(self.model_path))
        self._model.eval()

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if torch is None or self._model is None:
            # Provide deterministic mock behaviour for tests
            sorted_items = sorted(payload.items())
            raw = repr(sorted_items)
            score = sum(ord(ch) for ch in raw) % 100 / 100
            return {"model": self.name, "confidence": round(score, 3), "stub": True}
        with torch.no_grad():
            tensor = torch.tensor(payload["data"])  # type: ignore[index]
            result = self._model(tensor)
            confidence = float(result.squeeze().mean().item())
            return {"model": self.name, "confidence": confidence, "stub": False}

    def unload(self) -> None:
        self._model = None


__all__ = ["TorchModel"]
