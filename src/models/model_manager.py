"""Centralised model registry supporting hot loading."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .base_model import BaseModel


@dataclass
class ModelConfig:
    name: str
    class_path: str
    path: Optional[str]
    enabled: bool
    options: Dict[str, Any]


def _import_class(path: str) -> type[BaseModel]:
    if not path.startswith("src."):
        qualified = f"src.{path}"
    else:
        qualified = path
    module_name, _, class_name = qualified.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, BaseModel):
        raise TypeError(f"{qualified} is not a BaseModel subclass")
    return cls


class ModelManager:
    def __init__(self, configs: Iterable[Dict[str, Any]], *, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("model")
        self._configs = [self._parse_config(cfg) for cfg in configs]
        self._models: Dict[str, BaseModel] = {}

    @staticmethod
    def _parse_config(raw: Dict[str, Any]) -> ModelConfig:
        return ModelConfig(
            name=raw["name"],
            class_path=raw["class"],
            path=raw.get("path"),
            enabled=bool(raw.get("enabled", True)),
            options=raw.get("options", {}),
        )

    def load_enabled(self) -> None:
        for config in self._configs:
            if not config.enabled:
                continue
            cls = _import_class(config.class_path)
            model = cls(config.name, config.path, **config.options)
            model.load()
            self._models[config.name] = model
            self.logger.info("Loaded model %s using %s", config.name, cls.__name__)

    def infer(self, model_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        model = self._models.get(model_name)
        if not model:
            raise KeyError(f"Model {model_name} not loaded")
        return model.infer(payload)

    def unload_all(self) -> None:
        for name, model in list(self._models.items()):
            try:
                model.unload()
            finally:
                self.logger.info("Unloaded model %s", name)
        self._models.clear()

    def list_models(self) -> Dict[str, str]:
        return {name: type(model).__name__ for name, model in self._models.items()}


__all__ = ["ModelManager"]
