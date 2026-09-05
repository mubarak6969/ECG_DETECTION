"""Record-level aggregation and probability calibration.

Both are chosen ONCE via scripts/select_aggregation_and_calibration.py
using the VALIDATION fold only, then frozen into ecg/config.py
(AGGREGATION_METHOD, CALIBRATION_TEMPERATURE). The held-out test fold is
scored exactly once, after selection, in scripts/evaluate_model.py - it
is never used to choose a method or tune a parameter.

This module is imported by both scripts/evaluate_model.py and
app/main.py so a record's prediction is computed identically whether it
comes from the offline evaluation or a live upload.
"""
from __future__ import annotations

import numpy as np

AGGREGATION_METHODS = ("mean", "median", "majority_vote", "confidence_weighted", "top_k_mean")


def aggregate(beat_probs: np.ndarray, method: str = "mean", k: int = 3) -> float:
    """Combine one record's beat-level probabilities into a single
    record-level probability of "abnormal"."""
    beat_probs = np.asarray(beat_probs, dtype=np.float64)
    if beat_probs.size == 0:
        raise ValueError("aggregate() called with zero beats")

    if method == "mean":
        return float(beat_probs.mean())

    if method == "median":
        return float(np.median(beat_probs))

    if method == "majority_vote":
        # fraction of beats individually classified abnormal
        return float((beat_probs > 0.5).mean())

    if method == "confidence_weighted":
        # each beat's vote is weighted by how far it is from the
        # decision boundary, so a handful of very confident beats can
        # outweigh many near-0.5 (uninformative) beats
        weights = np.abs(beat_probs - 0.5)
        if weights.sum() < 1e-12:
            return float(beat_probs.mean())
        return float(np.average(beat_probs, weights=weights))

    if method == "top_k_mean":
        # average of the k most confident (most extreme) beats only
        k_eff = min(k, len(beat_probs))
        idx = np.argsort(np.abs(beat_probs - 0.5))[-k_eff:]
        return float(beat_probs[idx].mean())

    raise ValueError(f"Unknown aggregation method: {method!r}. Choose from {AGGREGATION_METHODS}")


def aggregate_by_record(beat_probs: np.ndarray, record_ids: np.ndarray, method: str = "mean", k: int = 3):
    """Vectorized aggregate() over many records at once.

    Returns (unique_ids, record_probs) with unique_ids sorted, matching
    the order produced by np.unique.
    """
    unique_ids = np.unique(record_ids)
    record_probs = np.array([
        aggregate(beat_probs[record_ids == rid], method=method, k=k) for rid in unique_ids
    ])
    return unique_ids, record_probs


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scale probabilities: convert to logits, divide by T,
    convert back. T=1.0 is a no-op (uncalibrated). T>1 softens
    (less extreme) probabilities, T<1 sharpens them."""
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    logits = np.log(probs / (1 - probs))
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(y_true)) ** 2))
