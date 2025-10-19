"""ONNX model wrapper for inference."""

from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from .base_model import BaseModel


class ONNXModel(BaseModel):
    def __init__(self, name: str, model_path: str | None = None, **kwargs: Any) -> None:
        super().__init__(name, model_path, **kwargs)
        self._session: Any = None

    def load(self) -> None:
        if ort is None or self.model_path is None:
            return
        providers = self.kwargs.get("providers") or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self.model_path), providers=providers)

    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if ort is None or self._session is None:
            checksum = sum(len(str(v)) for v in payload.values()) % 100 / 100
            return {"model": self.name, "confidence": round(checksum, 3), "stub": True}
        inputs = {self._session.get_inputs()[0].name: payload["data"]}
        outputs = self._session.run(None, inputs)
        return {"model": self.name, "confidence": float(outputs[0].mean()), "stub": False}

    def unload(self) -> None:
        self._session = None


__all__ = ["ONNXModel"]
