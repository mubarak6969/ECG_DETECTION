"""Flask application - the single canonical web UI for ECG classification.

Security/reliability fixes vs. the original app.py:
  * filenames are sanitized with secure_filename() and given a random,
    per-request subdirectory, closing a path-traversal / overwrite hole
  * only .dat/.hea extensions are accepted, and upload size is capped
    (MAX_CONTENT_LENGTH) to bound memory/disk use
  * every request cleans up its temp files in a `finally` block, even on
    error, instead of only on the success path
  * raw exception text is logged server-side but never sent to the
    client; the client gets a generic message
  * uses the same ecg.preprocessing pipeline as training, so inference
    and training can never silently diverge
  * record-level aggregation uses ecg.postprocessing with the method and
    calibration temperature selected on the validation fold - fixing a
    real bug where /predict was previously computing the fraction of
    beats individually voted abnormal (mean of booleans) rather than
    the mean beat probability, silently diverging from what
    scripts/evaluate_model.py was reporting as "record-level" metrics
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.saving import load_model
from werkzeug.utils import secure_filename

from ecg import config
from ecg.explainability import beat_saliency, most_influential_beat
from ecg.model_bootstrap import ensure_model_present
from ecg.postprocessing import aggregate, apply_temperature
from ecg.preprocessing import prepare_segments_for_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ecg.app")

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES

# CORS: the Vercel static frontend calls this API cross-origin. Only the
# exact origin in ECG_FRONTEND_ORIGIN is allowed - deliberately not "*",
# so an unset value means no CORS header at all (same-origin requests,
# e.g. hitting this Flask app's own "/" directly, are unaffected either
# way; only cross-origin browser calls need this).
_FRONTEND_ORIGIN = os.environ.get("ECG_FRONTEND_ORIGIN")


@app.after_request
def _apply_cors(response):
    if _FRONTEND_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = _FRONTEND_ORIGIN
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# No-op if the model is already on disk (the normal case for local dev);
# downloads it from ECG_MODEL_URL first if it's missing and that env var
# is set (the deployment case - see ecg/model_bootstrap.py).
ensure_model_present()

_model_path = config.MODEL_PATH if config.MODEL_PATH.exists() else config.LEGACY_MODEL_PATH
if not _model_path.exists():
    raise FileNotFoundError(
        f"No model found at {config.MODEL_PATH} or {config.LEGACY_MODEL_PATH}, and "
        "ECG_MODEL_URL is not set (or the download failed). "
        "Either run scripts/prepare_data.py and scripts/train_model.py first, "
        "or set ECG_MODEL_URL to a direct-download link for a trained rcnn_model.h5."
    )
logger.info("Loading model from %s", _model_path)
model = load_model(str(_model_path), compile=False)
_model_name = "rcnn" if "rcnn" in _model_path.name else "cnn"


def _load_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _model_info() -> dict:
    """Non-sensitive summary for the UI - never includes filesystem paths.

    Test metrics are the RECORD-level numbers from evaluation_report.json
    (mean-pooled beat probabilities per record, thresholded at 0.5) since
    that is the same granularity /predict returns to a user - one
    prediction per uploaded record, not per beat.
    """
    summary = _load_json(config.REPORTS_DIR / "training_summary.json").get(_model_name, {})
    evaluation = _load_json(config.REPORTS_DIR / "evaluation_report.json").get(_model_name, {})
    record_level = evaluation.get("record_level", {})
    return {
        "architecture": "RCNN (1D CNN + LSTM)" if _model_name == "rcnn" else "CNN (baseline)",
        "input_window_seconds": round(config.WINDOW_SIZE / config.SAMPLING_RATE_HZ, 2),
        "sampling_rate_hz": config.SAMPLING_RATE_HZ,
        "epochs_trained": summary.get("epochs_run"),
        "test_accuracy": record_level.get("accuracy"),
        "test_roc_auc": record_level.get("roc_auc"),
        "test_f1_macro": record_level.get("f1_macro"),
        "test_n_records": record_level.get("n_samples"),
        "test_metric_granularity": "record-level (per uploaded ECG)",
    }


def _reject(message: str, status: int):
    logger.info("Rejecting request: %s", message)
    return jsonify({"error": message}), status


@app.errorhandler(413)
def handle_too_large(_exc):
    return _reject("Uploaded file exceeds the maximum allowed size", 413)


@app.errorhandler(Exception)
def handle_unexpected(exc):
    # Belt-and-braces: no code path should ever leak a raw exception or
    # traceback to the client, even one not explicitly wrapped above.
    from werkzeug.exceptions import HTTPException
    if isinstance(exc, HTTPException):
        return exc
    logger.exception("Unhandled exception")
    return _reject("An unexpected error occurred", 500)


@app.get("/")
def index():
    return render_template("index.html", model_info=_model_info())


@app.get("/health")
def health():
    # Deliberately no filesystem paths in a public endpoint.
    return jsonify({"status": "ok", "model_loaded": True, "model_type": _model_name})


@app.get("/api/model-info")
def model_info():
    return jsonify(_model_info())


@app.post("/predict")
def predict():
    t0 = time.perf_counter()

    if "file" not in request.files or request.files["file"].filename == "":
        return _reject("No file uploaded", 400)

    upload = request.files["file"]
    filename = secure_filename(upload.filename)
    if not filename or not filename.lower().endswith(".dat"):
        return _reject("Only .dat files are accepted", 400)
    if len(filename) > config.MAX_FILENAME_LENGTH:
        return _reject("Filename is too long", 400)

    hea_upload = request.files.get("hea_file")
    if hea_upload is not None and hea_upload.filename:
        hea_filename_check = secure_filename(hea_upload.filename)
        if not hea_filename_check.lower().endswith(".hea"):
            return _reject("Header file must be a .hea file", 400)
        if len(hea_filename_check) > config.MAX_FILENAME_LENGTH:
            return _reject("Header filename is too long", 400)

    request_dir = config.UPLOADS_DIR / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=True)
    try:
        dat_path = request_dir / filename
        try:
            upload.save(dat_path)
            if hea_upload is not None and hea_upload.filename:
                # matched to the .dat basename - wfdb requires exact name pairing
                hea_upload.save(request_dir / (filename[:-4] + ".hea"))
        except OSError:
            logger.exception("Failed to save upload %s", filename)
            return _reject("Could not save the uploaded file", 400)

        record_path = str(dat_path)[:-4]
        try:
            segments, preview_signal = prepare_segments_for_record(record_path)
        except Exception:
            logger.exception("Preprocessing failed for upload %s", filename)
            return _reject("Could not read this file as a valid ECG record", 422)

        if segments.shape[0] == 0:
            return _reject(
                "No usable heartbeats could be extracted from this signal "
                "(too short, too noisy, or no detectable R-peaks)",
                422,
            )

        X = segments.reshape((-1, segments.shape[1], 1))
        try:
            beat_probs = model.predict(X, verbose=0).ravel()
        except Exception:
            logger.exception("Inference failed for upload %s", filename)
            return _reject("Model inference failed", 500)

        # Same aggregation method + calibration temperature chosen once
        # on the validation fold by scripts/select_aggregation_and_calibration.py
        # and used identically by scripts/evaluate_model.py, so the
        # reported test metrics describe exactly this code path.
        record_prob = aggregate(beat_probs, method=config.AGGREGATION_METHOD)
        calibrated_prob = float(apply_temperature(np.array([record_prob]), config.CALIBRATION_TEMPERATURE)[0])
        label = 1 if calibrated_prob > 0.5 else 0
        confidence = calibrated_prob if label else 1 - calibrated_prob

        explainability = None
        try:
            idx = most_influential_beat(beat_probs)
            explainability = {
                "beat_window": segments[idx].tolist(),
                "saliency": beat_saliency(model, segments[idx]).tolist(),
                "note": (
                    "Gradient-based model attribution showing which parts of the most "
                    "influential detected beat drove this prediction. This reflects the "
                    "model's internal weighting only - it is not a clinical explanation."
                ),
            }
        except Exception:
            logger.exception("Explainability computation failed for upload %s (non-fatal)", filename)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return jsonify({
            "prediction": "abnormal" if label else "normal",
            "confidence": round(confidence, 4),
            "beats_analyzed": int(X.shape[0]),
            "processing_time_ms": elapsed_ms,
            "explainability": explainability,
            "model": _model_info(),
            "signal": preview_signal.tolist() if preview_signal is not None else [],
        })
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)


if __name__ == "__main__":
    host = os.environ.get("ECG_APP_HOST", "127.0.0.1")
    port = int(os.environ.get("ECG_APP_PORT", 5000))
    app.run(host=host, port=port, debug=False)
