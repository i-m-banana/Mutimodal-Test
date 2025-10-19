"""Entrypoint for the modularised non-contact assessment UI."""

from __future__ import annotations

import sys

from .app.application import create_application, main as run_main

__all__ = ["create_application", "run_main", "main"]


def main() -> int:
    """Launch the UI application and return the Qt exit code."""
    return run_main()


if __name__ == "__main__":
    sys.exit(main())
