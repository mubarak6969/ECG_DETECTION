"""Fetch the trained RCNN model at deploy time if it isn't already
present on disk, so the ~11.6MB model binary never has to live in git
alongside the training/PTB-XL-dependent code.

Driven entirely by the `ECG_MODEL_URL` environment variable - this
project does not hardcode or invent any hosting URL, bucket, or
credential. To deploy, host your own trained `rcnn_model.h5` somewhere
with a stable, direct-download link (e.g. a GitHub Release asset on
your own repo, or an object-storage URL) and set `ECG_MODEL_URL` to it.

Called once from the FastAPI app's lifespan startup (app/model_state.py),
so it runs identically whether the process is started via `uvicorn
app.main:app`, a Dockerfile CMD, or a plain `python app/main.py` - one
code path for every deployment target. If the model is already present
(the normal case for local development), this is a complete no-op: no
network access is attempted.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

DOWNLOAD_TIMEOUT_SECONDS = 60


def ensure_model_present(model_path: Path | None = None) -> Path:
    """Return a path to the model, downloading it first if missing and
    `ECG_MODEL_URL` is set. Never raises for a missing model when
    `ECG_MODEL_URL` is unset - the caller (app/main.py) already has a
    clear, existing error message for that case."""
    model_path = Path(model_path) if model_path else config.MODEL_PATH
    if model_path.exists():
        return model_path

    url = os.environ.get("ECG_MODEL_URL")
    if not url:
        return model_path

    logger.info("Model not found at %s; downloading from ECG_MODEL_URL", model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            with open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out)
        tmp_path.rename(model_path)  # atomic-ish: never leaves a partial file at the real path
        logger.info("Downloaded model to %s (%d bytes)", model_path, model_path.stat().st_size)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        logger.exception("Failed to download model from ECG_MODEL_URL")
        raise
    return model_path
