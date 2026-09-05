"""Choose the record-level aggregation method and calibration
temperature using the VALIDATION fold only, then freeze the choice to
artifacts/reports/aggregation_calibration_selection.json, which
ecg/config.py reads at import time.

The test fold is never touched by this script. scripts/evaluate_model.py
applies whatever this script selects and scores it on test exactly once.

Decision rule (see PRIORITY 2 in the brief - "keep the simplest method
that genuinely improves"): mean-pooling is the default; another
aggregation method is only chosen if it beats mean by more than
MIN_IMPROVEMENT in record-level F1 on validation, to avoid chasing
validation-set noise with a more complex method for no real gain.

Usage:
    python scripts/select_aggregation_and_calibration.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, log_loss
from tensorflow.keras.models import load_model

from ecg import config
from ecg.postprocessing import AGGREGATION_METHODS, aggregate_by_record, apply_temperature, brier_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("select_aggregation_and_calibration")

MIN_IMPROVEMENT = 0.005  # 0.5pp of macro/binary F1 on the abnormal class


def fit_temperature(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Find the temperature T>0 minimizing log-loss (NLL) of the
    temperature-scaled probabilities against validation labels."""
    def nll(t):
        if t <= 0:
            return np.inf
        calibrated = apply_temperature(probs, t)
        return log_loss(y_true, calibrated, labels=[0, 1])

    result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(result.x)


def main() -> None:
    X_val = np.load(config.SPLITS_DIR / "X_val.npy").reshape(-1, config.WINDOW_SIZE, 1)
    y_val_beat = np.load(config.SPLITS_DIR / "y_val.npy")
    ids_val = np.load(config.SPLITS_DIR / "ids_val.npy")

    model = load_model(config.MODELS_DIR / "rcnn_model.h5", compile=False)
    beat_probs = model.predict(X_val, verbose=0).ravel()

    unique_ids = np.unique(ids_val)
    y_val_record = np.array([y_val_beat[ids_val == rid][0] for rid in unique_ids])

    logger.info("Validation: %d beats from %d records", len(beat_probs), len(unique_ids))

    method_scores = {}
    for method in AGGREGATION_METHODS:
        _, record_probs = aggregate_by_record(beat_probs, ids_val, method=method)
        preds = (record_probs > 0.5).astype(int)
        f1 = f1_score(y_val_record, preds, pos_label=1, zero_division=0)
        method_scores[method] = f1
        logger.info("  %-20s abnormal-F1=%.4f", method, f1)

    baseline_f1 = method_scores["mean"]
    best_method = max(method_scores, key=method_scores.get)
    if method_scores[best_method] - baseline_f1 >= MIN_IMPROVEMENT:
        chosen_method = best_method
        logger.info(
            "Chose '%s' (F1 %.4f) over 'mean' (F1 %.4f): improvement %.4f >= threshold %.4f",
            chosen_method, method_scores[best_method], baseline_f1,
            method_scores[best_method] - baseline_f1, MIN_IMPROVEMENT,
        )
    else:
        chosen_method = "mean"
        logger.info(
            "Keeping simplest method 'mean' (F1 %.4f): best alternative '%s' (F1 %.4f) "
            "did not clear the %.4f improvement threshold",
            baseline_f1, best_method, method_scores[best_method], MIN_IMPROVEMENT,
        )

    _, chosen_record_probs = aggregate_by_record(beat_probs, ids_val, method=chosen_method)

    brier_before = brier_score(chosen_record_probs, y_val_record)
    temperature = fit_temperature(chosen_record_probs, y_val_record)
    calibrated_probs = apply_temperature(chosen_record_probs, temperature)
    brier_after = brier_score(calibrated_probs, y_val_record)

    logger.info("Calibration temperature (validation-fit): %.4f", temperature)
    logger.info("Brier score on validation: uncalibrated=%.4f, calibrated=%.4f", brier_before, brier_after)

    # Only apply calibration if it doesn't make things worse (temperature
    # scaling is monotonic and preserves the decision boundary at 0.5, so
    # accuracy/F1 are unaffected either way - this is purely about
    # whether reported probabilities are honest).
    if brier_after > brier_before + 1e-6:
        logger.warning("Calibration did not improve Brier score; falling back to temperature=1.0 (no calibration)")
        temperature = 1.0
        brier_after = brier_before

    selection = {
        "candidate_methods_val_f1": method_scores,
        "chosen_method": chosen_method,
        "chosen_temperature": temperature,
        "validation_n_records": int(len(unique_ids)),
        "validation_brier_uncalibrated": brier_before,
        "validation_brier_calibrated": brier_after,
    }
    out_path = config.REPORTS_DIR / "aggregation_calibration_selection.json"
    with open(out_path, "w") as f:
        json.dump(selection, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
