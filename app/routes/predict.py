"""POST /predict - the only route that touches an uploaded file.

All ECG-specific math (signal loading, resampling, R-peak detection,
windowing, aggregation, calibration, explainability) lives in ecg/ and
is called here unchanged from the Flask version this replaces. This
module is pure HTTP plumbing: validation, temp-file handling, response
shaping - no ML logic is implemented here.
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app import model_state
from app.schemas.prediction import PredictionResponse
from app.uploads import read_capped, secure_filename
from ecg import config
from ecg.explainability import beat_saliency, most_influential_beat
from ecg.postprocessing import aggregate, apply_temperature
from ecg.preprocessing import prepare_segments_for_record

logger = logging.getLogger("ecg.app")
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an uploaded ECG as normal or abnormal",
    description=(
        "Multipart upload: `file` (required, WFDB `.dat`) and `hea_file` (optional but "
        "recommended, matching `.hea` header - without it the signal cannot be "
        "gain/baseline-calibrated). Returns a calibrated prediction, confidence, beat "
        "count, the raw waveform for plotting, and gradient-saliency model attribution."
    ),
    responses={
        400: {"description": "No/invalid file, wrong extension, or filename too long"},
        413: {"description": "Upload exceeds the configured size cap"},
        422: {"description": "File parsed but no usable heartbeats could be extracted"},
        500: {"description": "Model inference or an unanticipated server error"},
    },
)
async def predict(
    request: Request,
    file: Optional[UploadFile] = File(None, description="WFDB signal file, .dat extension only."),
    hea_file: Optional[UploadFile] = File(None, description="Matching WFDB header, .hea extension only."),
) -> PredictionResponse:
    t0 = time.perf_counter()

    # Fast-path check on an honest Content-Length header; the per-file
    # read below (read_capped) enforces the same cap against a lying or
    # chunked-transfer client, so this is a speed optimization, not the
    # only guard.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > config.MAX_UPLOAD_BYTES:
                raise HTTPException(413, "Uploaded file exceeds the maximum allowed size")
        except ValueError:
            pass

    if file is None or not file.filename:
        raise HTTPException(400, "No file uploaded")

    filename = secure_filename(file.filename)
    if not filename or not filename.lower().endswith(".dat"):
        raise HTTPException(400, "Only .dat files are accepted")
    if len(filename) > config.MAX_FILENAME_LENGTH:
        raise HTTPException(400, "Filename is too long")

    hea_filename = None
    if hea_file is not None and hea_file.filename:
        hea_filename = secure_filename(hea_file.filename)
        if not hea_filename.lower().endswith(".hea"):
            raise HTTPException(400, "Header file must be a .hea file")
        if len(hea_filename) > config.MAX_FILENAME_LENGTH:
            raise HTTPException(400, "Header filename is too long")

    dat_bytes = await read_capped(file, config.MAX_UPLOAD_BYTES)
    hea_bytes = await read_capped(hea_file, config.MAX_UPLOAD_BYTES) if hea_filename else None

    request_dir = config.UPLOADS_DIR / uuid.uuid4().hex
    request_dir.mkdir(parents=True, exist_ok=True)
    try:
        dat_path = request_dir / filename
        try:
            dat_path.write_bytes(dat_bytes)
            if hea_bytes is not None:
                # matched to the .dat basename - wfdb requires exact name pairing
                (request_dir / (filename[:-4] + ".hea")).write_bytes(hea_bytes)
        except OSError:
            logger.exception("Failed to save upload %s", filename)
            raise HTTPException(400, "Could not save the uploaded file")

        record_path = str(dat_path)[:-4]
        try:
            segments, preview_signal = prepare_segments_for_record(record_path)
        except Exception:
            logger.exception("Preprocessing failed for upload %s", filename)
            raise HTTPException(422, "Could not read this file as a valid ECG record")

        if segments.shape[0] == 0:
            raise HTTPException(
                422,
                "No usable heartbeats could be extracted from this signal "
                "(too short, too noisy, or no detectable R-peaks)",
            )

        X = segments.reshape((-1, segments.shape[1], 1))
        model = model_state.get_model()
        try:
            beat_probs = model.predict(X, verbose=0).ravel()
        except Exception:
            logger.exception("Inference failed for upload %s", filename)
            raise HTTPException(500, "Model inference failed")

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

        return PredictionResponse(
            prediction="abnormal" if label else "normal",
            confidence=round(confidence, 4),
            beats_analyzed=int(X.shape[0]),
            processing_time_ms=elapsed_ms,
            explainability=explainability,
            model=model_state.build_model_info(),
            signal=preview_signal.tolist() if preview_signal is not None else [],
        )
    finally:
        shutil.rmtree(request_dir, ignore_errors=True)
