"""Project-wide logging utilities."""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def build_default_dict(log_root: Path) -> Dict[str, Any]:
    """Create a dictConfig-compatible logging configuration (console only)."""
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
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
        },
    }


def setup_logging(log_root: Optional[str | os.PathLike[str]] = None, *, config: Optional[Dict[str, Any]] = None) -> None:
    """Initialise the logging subsystem (console only)."""
    logging_config = config or build_default_dict(Path(log_root or "logs"))
    logging.config.dictConfig(logging_config)


__all__ = ["setup_logging", "build_default_dict"]
