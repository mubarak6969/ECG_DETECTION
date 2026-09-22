"""App-level model lifecycle: load-once state and the non-sensitive
summary dict served by GET /model-info and embedded in POST /predict
responses.

This module is HTTP-service wiring, not ML logic - it holds the loaded
Keras model as process-wide state and reads the small, already-computed
JSON reports in artifacts/reports/. All ECG math (preprocessing,
aggregation, calibration, explainability) stays in ecg/, untouched by
this migration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from tensorflow.keras.saving import load_model

from ecg import config
from ecg.model_bootstrap import ensure_model_present

logger = logging.getLogger("ecg.app")

_model = None
_model_name: Optional[str] = None


def load() -> None:
    """Ensure the trained model is on disk (downloading it via
    ECG_MODEL_URL first if it's missing) and load it into memory. Called
    once from the FastAPI app's lifespan startup."""
    global _model, _model_name
    ensure_model_present()

    model_path = config.MODEL_PATH if config.MODEL_PATH.exists() else config.LEGACY_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {config.MODEL_PATH} or {config.LEGACY_MODEL_PATH}, and "
            "ECG_MODEL_URL is not set (or the download failed). "
            "Either run scripts/prepare_data.py and scripts/train_model.py first, "
            "or set ECG_MODEL_URL to a direct-download link for a trained rcnn_model.h5."
        )
    logger.info("Loading model from %s", model_path)
    _model = load_model(str(model_path), compile=False)
    _model_name = "rcnn" if "rcnn" in model_path.name else "cnn"


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded - model_state.load() must run at startup")
    return _model


def get_model_name() -> str:
    if _model_name is None:
        raise RuntimeError("Model not loaded - model_state.load() must run at startup")
    return _model_name


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_model_info() -> dict:
    """Non-sensitive summary for the API/UI - never includes filesystem
    paths. See ModelInfo in app/schemas/prediction.py for the response
    shape this fills."""
    name = get_model_name()
    summary = _load_json(config.REPORTS_DIR / "training_summary.json").get(name, {})
    evaluation = _load_json(config.REPORTS_DIR / "evaluation_report.json").get(name, {})
    record_level = evaluation.get("record_level", {})
    return {
        "architecture": "RCNN (1D CNN + LSTM)" if name == "rcnn" else "CNN (baseline)",
        "input_window_seconds": round(config.WINDOW_SIZE / config.SAMPLING_RATE_HZ, 2),
        "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        "epochs_trained": summary.get("epochs_run"),
        "test_accuracy": record_level.get("accuracy"),
        "test_roc_auc": record_level.get("roc_auc"),
        "test_f1_macro": record_level.get("f1_macro"),
        "test_n_records": record_level.get("n_samples"),
        "test_metric_granularity": "record-level (per uploaded ECG)",
    }
