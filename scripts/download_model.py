"""Optional convenience CLI: pre-fetch the trained model during a
build step (e.g. Render's Build Command) instead of waiting for the
FastAPI app's own startup-time check (app/model_state.py) to do it.

Safe to always include in a build command, even locally: it's a no-op
if the model is already present, and prints (not raises) if it's
missing and ECG_MODEL_URL isn't set, so it never breaks a local dev
build that trains its own model instead of downloading one.

See ecg/model_bootstrap.py for the actual download logic - this is
just a CLI entrypoint over the same function app/model_state.py calls
at startup.

Usage:
    python scripts/download_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg.model_bootstrap import ensure_model_present


def main() -> None:
    path = ensure_model_present()
    if path.exists():
        print(f"Model present at {path} ({path.stat().st_size} bytes)")
    else:
        print(
            f"Model NOT present at {path} and ECG_MODEL_URL is not set. "
            "The app will fail to start until one of those is fixed - "
            "see README 'Deployment' / MODEL_CARD.md."
        )


if __name__ == "__main__":
    main()
