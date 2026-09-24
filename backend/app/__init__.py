"""FastAPI service exposing the toxicity classifier in ml/.

Ensures the repo root is on sys.path so `ml.*` imports resolve regardless of
the working directory the app is launched from (uvicorn, pytest, etc.).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
