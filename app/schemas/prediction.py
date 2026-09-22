"""Pydantic response schemas for the ECG Intelligence API.

These describe HTTP response shapes only - they carry no ECG math. The
actual prediction/aggregation/calibration/explainability logic lives in
ecg/ and is unchanged by this schema layer.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Non-sensitive model summary - never includes filesystem paths.

    Test metrics are the RECORD-level numbers from the last run of
    scripts/evaluate_model.py (mean-pooled beat probabilities per
    record, thresholded at 0.5) since that is the same granularity
    /predict returns to a user. They are None until that script has
    been run at least once (e.g. a fresh checkout with no reports yet).
    """

    architecture: str = Field(..., examples=["RCNN (1D CNN + LSTM)"])
    input_window_seconds: float = Field(..., examples=[2.0])
    sampling_rate_hz: int = Field(..., examples=[100])
    epochs_trained: Optional[int] = None
    test_accuracy: Optional[float] = None
    test_roc_auc: Optional[float] = None
    test_f1_macro: Optional[float] = None
    test_n_records: Optional[int] = None
    test_metric_granularity: str = Field(..., examples=["record-level (per uploaded ECG)"])


class ExplainabilityInfo(BaseModel):
    """Gradient-saliency model attribution for the single most-confident
    detected beat. This is model attribution, not a clinical
    explanation - see `note`."""

    beat_window: list[float]
    saliency: list[float] = Field(..., description="Per-timestep importance, normalized to [0, 1].")
    note: str


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool
    model_type: str = Field(..., examples=["rcnn"])


class PredictionResponse(BaseModel):
    prediction: str = Field(..., examples=["normal"], description='"normal" or "abnormal".')
    confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated probability, not a clinical certainty.")
    beats_analyzed: int = Field(..., ge=0)
    processing_time_ms: float
    explainability: Optional[ExplainabilityInfo] = None
    model: ModelInfo
    signal: list[float] = Field(..., description="Up to 1000 raw samples for the waveform chart.")


class ErrorResponse(BaseModel):
    error: str
