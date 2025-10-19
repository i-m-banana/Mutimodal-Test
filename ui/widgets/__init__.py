"""UI widgets and reusable components.

Avoid importing heavy submodules at package import time to prevent circular
imports. Import specific widgets from their modules instead, e.g.:

	from ui.widgets.camera_preview import CameraPreviewWidget
"""

from __future__ import annotations

__all__: list[str] = []
