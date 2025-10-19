from __future__ import annotations

from pathlib import Path

from src.constants import EventTopic
from src.core.orchestrator import Orchestrator


def test_orchestrator_end_to_end():
    project_root = Path(__file__).resolve().parents[2]
    orchestrator = Orchestrator.from_config_directory(project_root)

    events = []
    orchestrator.bus.subscribe(EventTopic.DETECTION_RESULT, lambda event: events.append(event))

    orchestrator.run_for(0.3)

    assert events, "End-to-end run should produce detection results"
