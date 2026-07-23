from __future__ import annotations

import sys
from pathlib import Path


LANGSMITH_SOURCE = Path(__file__).resolve().parents[2] / "src" / "langsmith"
sys.path.insert(0, str(LANGSMITH_SOURCE))
