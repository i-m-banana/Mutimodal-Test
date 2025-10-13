"""Central orchestrator coordinating collectors, models, and detectors."""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

from ..constants import DEFAULT_COMPONENT_TIMEOUT, EventTopic, ModuleState
from ..collectors.base_collector import BaseCollector
from ..detectors.base_detector import BaseDetector
from ..models.model_manager import ModelManager
from ..utils.logger import setup_logging
from ..interfaces.base import BaseInterface
from ..services.ui_command_router import UICommandRouter
from .event_bus import EventBus
from .system_monitor import SystemMonitor


@dataclass
class ComponentConfig:
    name: str
    class_path: str
    enabled: bool
    options: Dict[str, Any]


def _import_component(path: str, base) -> type:
    qualified = path if path.startswith("src.") else f"src.{path}"
    module_name, _, class_name = qualified.rpartition(".")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not issubclass(cls, base):
        raise TypeError(f"{qualified} is not a subclass of {base.__name__}")
    return cls


def _parse_components(configs: Iterable[Dict[str, Any]]) -> list[ComponentConfig]:
    parsed = []
    for cfg in configs:
        parsed.append(ComponentConfig(
            name=cfg["name"],
            class_path=cfg["class"],
            enabled=bool(cfg.get("enabled", True)),
            options=cfg.get("options", {}),
        ))
    return parsed


class Orchestrator:
    def __init__(self, *, system: Dict[str, Any], collectors: Iterable[Dict[str, Any]],
                 detectors: Iterable[Dict[str, Any]], models: Iterable[Dict[str, Any]],
                 interfaces: Iterable[Dict[str, Any]] | None = None,
                 log_root: Optional[str] = None) -> None:
        setup_logging(log_root)
        self.logger = logging.getLogger("orchestrator")
        self.bus = EventBus()
        self.system_config = system
        self.collector_configs = _parse_components(collectors)
        self.detector_configs = _parse_components(detectors)
        self.model_manager = ModelManager(models, logger=logging.getLogger("model"))
        interval = float(system.get("heartbeat_interval", 5.0))
        self.monitor = SystemMonitor(self.bus, interval=interval)
        self.collectors: Dict[str, BaseCollector] = {}
        self.detectors: Dict[str, BaseDetector] = {}
        self.interface_configs = _parse_components(interfaces or [])
        self.interfaces: Dict[str, BaseInterface] = {}
        self.command_router = UICommandRouter(self.bus, logger=logging.getLogger("ui.command"))
        self._running = False

    @classmethod
    def from_config_directory(cls, directory: str | Path) -> "Orchestrator":
        directory = Path(directory)
        system = cls._read_yaml(directory / "config" / "system.yaml")
        collectors = cls._read_yaml(directory / "config" / "collectors.yaml").get("collectors", [])
        detectors = cls._read_yaml(directory / "config" / "detectors.yaml").get("detectors", [])
        models = cls._read_yaml(directory / "config" / "models.yaml").get("models", [])
        interfaces = cls._read_yaml(directory / "config" / "interfaces.yaml").get("interfaces", [])
        return cls(system=system, collectors=collectors, detectors=detectors, models=models,
                   interfaces=interfaces, log_root=directory / "logs")

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _instantiate_collectors(self) -> None:
        for cfg in self.collector_configs:
            if not cfg.enabled:
                self.logger.info("Collector %s disabled via config", cfg.name)
                continue
            cls = _import_component(cfg.class_path, BaseCollector)
            instance = cls(cfg.name, self.bus, options=cfg.options, logger=logging.getLogger("collector"))
            self.collectors[cfg.name] = instance

    def _instantiate_detectors(self) -> None:
        for cfg in self.detector_configs:
            if not cfg.enabled:
                self.logger.info("Detector %s disabled via config", cfg.name)
                continue
            cls = _import_component(cfg.class_path, BaseDetector)
            instance = cls(cfg.name, self.bus, self.model_manager, options=cfg.options,
                           logger=logging.getLogger("detector"))
            self.detectors[cfg.name] = instance

    def _instantiate_interfaces(self) -> None:
        for cfg in self.interface_configs:
            if not cfg.enabled:
                self.logger.info("Interface %s disabled via config", cfg.name)
                continue
            cls = _import_component(cfg.class_path, BaseInterface)
            instance = cls(cfg.name, self.bus, options=cfg.options,
                           logger=logging.getLogger("interface"))
            self.interfaces[cfg.name] = instance

    def start(self) -> None:
        if self._running:
            return
        self.logger.info("Starting orchestrator")
        self.model_manager.load_enabled()
        self._instantiate_collectors()
        self._instantiate_detectors()
        self._instantiate_interfaces()
        self.command_router.start()
        for detector in self.detectors.values():
            detector.start()
        for collector in self.collectors.values():
            collector.start()
        if self.system_config.get("enable_monitor", True):
            self.monitor.start()
        for interface in self.interfaces.values():
            try:
                interface.start()
            except Exception:
                self.logger.exception("Failed to start interface %s", interface.name)
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self.logger.info("Stopping orchestrator")
        for collector in self.collectors.values():
            collector.stop(timeout=self.system_config.get("collector_stop_timeout", DEFAULT_COMPONENT_TIMEOUT))
        for detector in self.detectors.values():
            detector.stop()
        self.monitor.stop()
        for interface in self.interfaces.values():
            try:
                interface.stop()
            except Exception:
                self.logger.exception("Failed to stop interface %s", interface.name)
        try:
            self.command_router.stop()
        except Exception:
            self.logger.exception("Failed to stop UI command router")
        self.model_manager.unload_all()
        self._running = False

    def run_for(self, seconds: float) -> None:
        import time

        self.start()
        try:
            time.sleep(seconds)
        finally:
            self.stop()


__all__ = ["Orchestrator"]
