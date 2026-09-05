"""Evaluate the trained CNN and RCNN models on the held-out test fold
(fold 10 - never seen during training or model selection, natural class
distribution). Produces a JSON report plus plots so results are
reproducible and inspectable, instead of an unverifiable pre-trained
.h5 with no accompanying metrics.

Reports metrics at TWO granularities, because they answer different
questions and are not interchangeable:

* beat-level  - one prediction per extracted R-peak window. This is what
  the network is trained on and what a per-window classification_report
  measures.
* record-level - one prediction per uploaded ECG record, using whatever
  aggregation method and calibration temperature were selected by
  scripts/select_aggregation_and_calibration.py on the VALIDATION fold
  (ecg/config.py: AGGREGATION_METHOD, CALIBRATION_TEMPERATURE). This is
  exactly what app/main.py's /predict returns, so it is the number that
  matters for judging real-world behavior. The test fold is scored with
  this frozen choice exactly once - no method/parameter is tuned here.

Usage:
    python scripts/evaluate_model.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from tensorflow.keras.models import load_model

from ecg import config
from ecg.postprocessing import aggregate_by_record, apply_temperature, brier_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("evaluate_model")


def _score(y_true, y_pred, probs) -> dict:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        auc = float(roc_auc_score(y_true, probs))
        pr_auc = float(average_precision_score(y_true, probs))
    except ValueError:
        auc, pr_auc = None, None  # only one class present in y_true

    # Abnormal (1) is the condition of interest -> normal's (0) recall is
    # the true-negative rate, i.e. specificity.
    specificity = report.get("0", {}).get("recall")

    hist, edges = np.histogram(probs, bins=10, range=(0, 1))

    return {
        "n_samples": int(len(y_true)),
        "class_distribution": {int(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))},
        "accuracy": report["accuracy"],
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "specificity": specificity,
        "per_class": {
            "normal": report.get("0", {}),
            "abnormal": report.get("1", {}),
        },
        "roc_auc": auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),  # rows/cols ordered [normal, abnormal]
        "probability_distribution": {"bin_edges": edges.tolist(), "counts": hist.tolist()},
    }


def _plot_confusion(cm, title, path):
    plt.figure(figsize=(5, 4))
    class_names = ["Normal", "Abnormal"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_roc(y_true, probs, auc, title, path):
    if auc is None:
        return
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _plot_calibration(y_true, uncalibrated, calibrated, title, path):
    plt.figure(figsize=(5, 4))
    for probs, label, style in ((uncalibrated, "uncalibrated", "o-"), (calibrated, "calibrated", "s--")):
        frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=8, strategy="quantile")
        plt.plot(mean_pred, frac_pos, style, label=label)
    plt.plot([0, 1], [0, 1], ":", color="gray", label="perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency of 'abnormal'")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def evaluate(model_type: str, X_test, y_test, record_ids) -> dict | None:
    model_path = config.MODELS_DIR / f"{model_type}_model.h5"
    if not model_path.exists():
        logger.warning("Skipping %s: %s not found", model_type, model_path)
        return None

    model = load_model(model_path, compile=False)
    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs > 0.5).astype(int)

    beat = _score(y_test, preds, probs)
    _plot_confusion(
        np.array(beat["confusion_matrix"]),
        f"{model_type.upper()} Confusion Matrix (beat-level, test fold)",
        config.REPORTS_DIR / f"{model_type}_confusion_matrix.png",
    )
    _plot_roc(
        y_test, probs, beat["roc_auc"],
        f"{model_type.upper()} ROC Curve (beat-level, test fold)",
        config.REPORTS_DIR / f"{model_type}_roc_curve.png",
    )

    # Record-level: aggregate with the method chosen (on validation only)
    # by scripts/select_aggregation_and_calibration.py - identical to
    # what app/main.py's /predict does.
    unique_ids, record_probs = aggregate_by_record(probs, record_ids, method=config.AGGREGATION_METHOD)
    # label is constant across a record's own segments - verified in tests/test_dataset.py
    record_true = np.array([y_test[record_ids == rid][0] for rid in unique_ids])
    record_preds = (record_probs > 0.5).astype(int)
    record = _score(record_true, record_preds, record_probs)
    record["aggregation_method"] = config.AGGREGATION_METHOD
    _plot_confusion(
        np.array(record["confusion_matrix"]),
        f"{model_type.upper()} Confusion Matrix (record-level, test fold)",
        config.REPORTS_DIR / f"{model_type}_confusion_matrix_record_level.png",
    )

    # Calibration: monotonic, so it never changes accuracy/F1/AUC (all
    # rank- or threshold-at-0.5-based) - only whether the *stated*
    # probability matches empirical frequency. Scored on test exactly
    # once, using the temperature already fit on validation.
    calibrated_probs = apply_temperature(record_probs, config.CALIBRATION_TEMPERATURE)
    calibration = {
        "temperature": config.CALIBRATION_TEMPERATURE,
        "brier_uncalibrated": brier_score(record_probs, record_true),
        "brier_calibrated": brier_score(calibrated_probs, record_true),
    }
    _plot_calibration(
        record_true, record_probs, calibrated_probs,
        f"{model_type.upper()} Calibration Curve (record-level, test fold)",
        config.REPORTS_DIR / f"{model_type}_calibration_curve.png",
    )

    logger.info(
        "%s BEAT-level: accuracy=%.4f macro-F1=%.4f AUC=%s | RECORD-level (%s): accuracy=%.4f macro-F1=%.4f "
        "AUC=%s | Brier uncal=%.4f cal=%.4f",
        model_type, beat["accuracy"], beat["f1_macro"], beat["roc_auc"], config.AGGREGATION_METHOD,
        record["accuracy"], record["f1_macro"], record["roc_auc"],
        calibration["brier_uncalibrated"], calibration["brier_calibrated"],
    )

    return {"beat_level": beat, "record_level": record, "calibration": calibration}


def main() -> None:
    X_test = np.load(config.SPLITS_DIR / "X_test.npy")
    y_test = np.load(config.SPLITS_DIR / "y_test.npy")
    record_ids = np.load(config.SPLITS_DIR / "ids_test.npy")
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    logger.info(
        "Test set: %d beat-level segments from %d records, class distribution %s",
        X_test.shape[0], len(np.unique(record_ids)), dict(zip(*np.unique(y_test, return_counts=True))),
    )
    logger.info("Aggregation method (frozen from validation): %s", config.AGGREGATION_METHOD)
    logger.info("Calibration temperature (frozen from validation): %.4f", config.CALIBRATION_TEMPERATURE)

    results = {}
    for model_type in ("cnn", "rcnn"):
        result = evaluate(model_type, X_test, y_test, record_ids)
        if result:
            results[model_type] = result

    if "cnn" in results and "rcnn" in results:
        fig, ax = plt.subplots(figsize=(7, 5))
        metrics = ["accuracy", "f1_macro"]
        x = np.arange(len(metrics))
        width = 0.35
        cnn_vals = [results["cnn"]["record_level"][m] for m in metrics]
        rcnn_vals = [results["rcnn"]["record_level"][m] for m in metrics]
        ax.bar(x - width / 2, cnn_vals, width, label="CNN", color="skyblue")
        ax.bar(x + width / 2, rcnn_vals, width, label="RCNN", color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(["Accuracy", "Macro F1"])
        ax.set_title("CNN vs RCNN - record-level, held-out test fold")
        ax.legend()
        plt.tight_layout()
        plt.savefig(config.REPORTS_DIR / "model_comparison.png")
        plt.close()

    with open(config.REPORTS_DIR / "evaluation_report.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", config.REPORTS_DIR / "evaluation_report.json")


if __name__ == "__main__":
    main()
