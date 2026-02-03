from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Shim for Streamlit Cloud: forward the old entrypoint to the new location.
ROOT = Path(__file__).resolve().parents[2]
LEGACY_APP = ROOT / "legacy" / "capital_programme_optimiser" / "frontend" / "app.py"

if not LEGACY_APP.exists():
    raise FileNotFoundError(
        f"Legacy Streamlit entrypoint not found at {LEGACY_APP}. Update the shim or fix the repository layout."
    )

legacy_root = ROOT / "legacy"
if str(legacy_root) not in sys.path:
    sys.path.insert(0, str(legacy_root))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

runpy.run_path(str(LEGACY_APP), run_name="__main__")
