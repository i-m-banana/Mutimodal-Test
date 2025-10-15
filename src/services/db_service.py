"""Database command handlers exposed to the UI command router."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

try:  # pragma: no cover - backend service can run without database module in tests
    from .database import TestTableStore
except ImportError:  # pragma: no cover
    TestTableStore = None  # type: ignore


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


class DatabaseUnavailable(RuntimeError):
    """Raised when the database layer is disabled or missing."""


class DatabaseService:
    """Wraps MySQL access so that the UI can talk to it via the backend."""

    _INSERT_FIELDS = {
        "name",
        "dt",
        "score",
        "audio",
        "video",
        "record_text",
        "rgb",
        "depth",
        "tobii",
        "blood",
        "eeg1",
        "eeg2",
        "ptime",
        "accuracy",
        "elapsed",
    }
    _UPDATE_FIELDS = {
        "audio",
        "video",
        "record_text",
        "rgb",
        "depth",
        "tobii",
        "blood",
        "eeg1",
        "eeg2",
        "ptime",
        "accuracy",
        "elapsed",
        "score",
    }

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("service.db")
        self._store: Any | None = None
        self._db_disabled = _env_flag("UI_SKIP_DATABASE")
        self._config = {
            "host": os.getenv("UI_DB_HOST", "localhost"),
            "user": os.getenv("UI_DB_USER", "root"),
            "password": os.getenv("UI_DB_PASSWORD", "123456"),
            "database": os.getenv("UI_DB_NAME", "tired"),
        }
        if TestTableStore is None:
            self.logger.warning("TestTableStore unavailable; database commands will be disabled")
            self._db_disabled = True

    # ------------------------------------------------------------------
    def _ensure_store(self):
        if self._db_disabled:
            raise DatabaseUnavailable("database access disabled (UI_SKIP_DATABASE=1 or module missing)")
        if self._store is None:
            try:
                self._store = TestTableStore(**self._config)  # type: ignore[arg-type]
                self.logger.info(
                    "Database connection ready -> %s/%s",
                    self._config["host"],
                    self._config["database"],
                )
            except Exception as exc:  # pragma: no cover - network/db failures are environmental
                self.logger.error("Failed to connect to database: %s", exc)
                raise DatabaseUnavailable(f"database connection failed: {exc}") from exc
        return self._store

    # ------------------------------------------------------------------
    def insert_test_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        store = self._ensure_store()
        fields = {key: payload.get(key) for key in self._INSERT_FIELDS if key in payload}
        if not fields.get("name"):
            raise ValueError("'name' is required to insert a test record")
        try:
            row_id = store.insert_row(**fields)
            self.logger.debug("Inserted test record for %s with id %s", fields.get("name"), row_id)
            return {"row_id": row_id}
        except Exception:
            self.logger.exception("Insert test record failed")
            raise

    def update_test_record(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        store = self._ensure_store()
        row_id = payload.get("row_id")
        if not row_id:
            raise ValueError("'row_id' is required to update a test record")
        updates = {key: payload.get(key) for key in self._UPDATE_FIELDS if key in payload}
        if not updates:
            self.logger.debug("No update fields supplied for row %s", row_id)
            return {"row_id": row_id, "updated": False}
        try:
            store.update_values(row_id=row_id, **updates)
            self.logger.debug("Updated test record %s with keys %s", row_id, list(updates.keys()))
            return {"row_id": row_id, "updated": True}
        except Exception:
            self.logger.exception("Update test record %s failed", row_id)
            raise

    def get_user_history(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        store = self._ensure_store()
        username = payload.get("name") or payload.get("username")
        if not username:
            raise ValueError("'name' is required to fetch user history")
        try:
            history = store.get_user_history_data(username)
            return {"history": history}
        except Exception:
            self.logger.exception("Fetching history for %s failed", username)
            raise

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "disabled": self._db_disabled,
            "configured_host": self._config.get("host"),
            "configured_database": self._config.get("database"),
        }


__all__ = ["DatabaseService", "DatabaseUnavailable"]
