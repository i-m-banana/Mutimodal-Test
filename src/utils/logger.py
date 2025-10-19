"""Project-wide logging utilities."""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def ensure_log_directories(base_dir: Path) -> None:
    """Ensure that the standard log directories exist."""

    collector_dir = base_dir / "collector"
    model_dir = base_dir / "model"
    detector_dir = base_dir / "detector"
    for directory in (collector_dir, model_dir, detector_dir):
        directory.mkdir(parents=True, exist_ok=True)


def build_default_dict(log_root: Path) -> Dict[str, Any]:
    """Create a dictConfig-compatible logging configuration."""

    ensure_log_directories(log_root)

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": _DEFAULT_FORMAT,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
            },
            "collector_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(log_root / "collector" / "collector.log"),
                "encoding": "utf-8",
            },
            "model_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(log_root / "model" / "model.log"),
                "encoding": "utf-8",
            },
            "detector_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(log_root / "detector" / "detector.log"),
                "encoding": "utf-8",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
        "loggers": {
            "collector": {
                "level": "INFO",
                "handlers": ["collector_file", "console"],
                "propagate": False,
            },
            "model": {
                "level": "INFO",
                "handlers": ["model_file", "console"],
                "propagate": False,
            },
            "detector": {
                "level": "INFO",
                "handlers": ["detector_file", "console"],
                "propagate": False,
            },
        },
    }


def setup_logging(log_root: Optional[str | os.PathLike[str]] = None, *, config: Optional[Dict[str, Any]] = None) -> None:
    """Initialise the logging subsystem.

    Args:
        log_root: Optional override for the root directory. Defaults to ``./logs``.
        config: Optional dictConfig mapping. When omitted, a pragmatic default is used.
    """

    resolved_root = Path(log_root or "logs")
    resolved_root.mkdir(parents=True, exist_ok=True)

    logging_config = config or build_default_dict(resolved_root)
    logging.config.dictConfig(logging_config)


__all__ = ["setup_logging", "ensure_log_directories", "build_default_dict"]
