"""Ensure local 'src' is importable without setting PYTHONPATH.

Python automatically imports 'sitecustomize' if present on sys.path.
Placing this file at the project root guarantees that running Python
from the repository will append 'src' into sys.path.
"""

from __future__ import annotations

import os
import sys


def _ensure_src_on_path() -> None:
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if os.path.isdir(src_path) and src_path not in sys.path:
        sys.path.insert(0, src_path)


_ensure_src_on_path()


