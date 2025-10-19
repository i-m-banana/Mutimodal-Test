"""CLI entrypoint for the refactored multimodal platform."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from .constants import EventTopic
from .core.event_bus import Event
from .core.orchestrator import Orchestrator

_logger = logging.getLogger("entrypoint")

_KEY_INFO_MODE_DEFAULT = os.getenv("BACKEND_KEY_INFO_MODE", "1").strip().lower() not in {"0", "false", "off"}


def _make_event_logger(key_info_mode: bool):
    """Create an event logger that optionally suppresses noisy updates."""

    last_payload: dict[EventTopic, object] = {}

    def _log(event: Event) -> None:
        if not key_info_mode:
            _logger.info("Event[%s] -> %s", event.topic.value, event.payload)
            return

        if event.topic is EventTopic.SYSTEM_HEARTBEAT:
            # 心跳频率很快，关键模式下默认跳过
            return

        payload = event.payload or {}
        if event.topic is EventTopic.DETECTION_RESULT:
            key = (
                payload.get("detector"),
                payload.get("status"),
                payload.get("label"),
            )
            if last_payload.get(event.topic) == key:
                return
            last_payload[event.topic] = key
            _logger.info(
                "[关键检测] detector=%s status=%s label=%s",
                key[0],
                key[1],
                key[2],
            )
            return

        # 其他事件按payload哈希过滤重复
        key = tuple(sorted(payload.items())) if payload else None
        if last_payload.get(event.topic) == key:
            return
        last_payload[event.topic] = key
        _logger.info("Event[%s] -> %s", event.topic.value, payload)

    return _log


def _install_signal_handler(stop_callback) -> None:
    if threading.current_thread() is not threading.main_thread():  # pragma: no cover - safety guard
        return

    def handler(signum, frame):  # pragma: no cover - signal path
        _logger.info("Received signal %s, shutting down", signum)
        stop_callback()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def run(
    orchestrator: Orchestrator,
    *,
    attach_listeners: bool = True,
    key_info_mode: bool = _KEY_INFO_MODE_DEFAULT,
) -> None:
    stop_event = threading.Event()

    if attach_listeners:
        event_logger = _make_event_logger(key_info_mode)
        orchestrator.bus.subscribe(EventTopic.DETECTION_RESULT, event_logger)
        orchestrator.bus.subscribe(EventTopic.SYSTEM_HEARTBEAT, event_logger)

    def stop() -> None:
        stop_event.set()

    _install_signal_handler(stop)

    orchestrator.start()
    _logger.info("Orchestrator started. Press Ctrl+C to stop.")

    try:
        while not stop_event.is_set():
            stop_event.wait(timeout=1.0)
    finally:
        orchestrator.stop()
        _logger.info("Orchestrator stopped")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal platform orchestrator")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Project root directory containing the config folder")
    parser.add_argument("--no-listeners", dest="attach_listeners", action="store_false",
                        help="Disable default console event listeners")
    parser.add_argument("--full-events", dest="key_info_mode", action="store_false",
                        help="显示全部事件日志（关闭关键交互信息模式）")
    parser.set_defaults(key_info_mode=_KEY_INFO_MODE_DEFAULT)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    orchestrator = Orchestrator.from_config_directory(args.root)
    run(orchestrator, attach_listeners=args.attach_listeners, key_info_mode=args.key_info_mode)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    sys.exit(main())
