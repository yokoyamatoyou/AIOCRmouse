import os
import sys
from pathlib import Path


def pytest_sessionstart(session):
    """Ensure `src` is on sys.path for imports like `from core import ...`."""
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))



