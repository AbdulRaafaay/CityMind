"""Command-line entry point: launch the CityMind GUI.

Works two ways:
    python main.py              (run from inside the citymind folder)
    python -m citymind          (run from the parent folder)
"""

from __future__ import annotations

import os
import sys


def _ensure_package_on_path() -> None:
    """Make citymind importable when this file is run directly as a script."""
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def main() -> None:
    _ensure_package_on_path()
    from citymind.ui.app import run_app  # imported here so the path fix runs first
    run_app()


if __name__ == "__main__":
    main()
