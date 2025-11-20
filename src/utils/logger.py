"""Project-wide logging utilities."""

from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

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
    # Compute the project-level logs root (expected: <project>/logs)
    logs_root = Path(log_root or "logs")
    try:
        logs_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        logs_root = Path(".")

    # Ensure a session timestamp shared across UI and backend runs
    ts_file = logs_root / "session_timestamp.txt"
    try:
        # Append a new timestamp line for this run and use the last non-empty
        # line as the canonical session id. This preserves historical session
        # timestamps while ensuring each run produces a new timestamp.
        new_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            with ts_file.open("a", encoding="utf-8") as fh:
                fh.write(new_ts + "\n")
        except Exception:
            # Best-effort append; fall back to creating the file
            ts_file.write_text(new_ts + "\n", encoding="utf-8")

        # Read last non-empty line
        try:
            with ts_file.open("r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]
            timestamp = lines[-1] if lines else new_ts
        except Exception:
            timestamp = new_ts
    except Exception:
        # Fallback to immediate timestamp if file ops fail
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Directory for backend logs
    src_dir = logs_root / "src"
    try:
        src_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        src_dir = Path(".")

    logging_config = config or build_default_dict(src_dir)

    # Add a file handler that writes to logs/src/src_log_<timestamp>.txt
    try:
        handlers = logging_config.setdefault("handlers", {})
        if "file" not in handlers:
            handlers["file"] = {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "standard",
                "filename": str(src_dir / f"src_log_{timestamp}.txt"),
                "encoding": "utf-8",
            }

        root_cfg = logging_config.setdefault("root", {})
        root_handlers = list(root_cfg.get("handlers", ["console"]))
        if "file" not in root_handlers:
            root_handlers.append("file")
        root_cfg["handlers"] = root_handlers
    except Exception:
        # If anything goes wrong modifying the dictConfig, fall back to console-only
        pass

    logging.config.dictConfig(logging_config)


__all__ = ["setup_logging", "build_default_dict"]
