"""EXPERIMENTAL: train the multi-label diagnostic-superclass RCNN
(NORM/MI/STTC/CD/HYP). Requires scripts/prepare_multilabel_data.py to
have been run first.

This is infrastructure + a proof-of-concept run, not a production
model: by default it trains on a small stratified-cap-free but
record-capped subset (see prepare_multilabel_data.py) for a handful of
epochs, purely to prove dataset -> model -> per-class evaluation works
correctly end-to-end without leakage. Do not present its numbers as
comparable to the primary, fully-evaluated binary RCNN. For a real
multi-label model, rerun prepare_multilabel_data.py with
--max-train-records 0 and this script with more --epochs on a GPU.

Usage:
    python scripts/train_multilabel_model.py
    python scripts/train_multilabel_model.py --epochs 30   # after a full (uncapped) data prep, on a GPU
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ecg import config
from ecg import model as model_lib
from ecg.multilabel import SUPERCLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_multilabel_model")


def load_split(name: str):
    X = np.load(config.SPLITS_DIR / f"multilabel_X_{name}.npy").reshape(-1, config.WINDOW_SIZE, 1)
    Y = np.load(config.SPLITS_DIR / f"multilabel_Y_{name}.npy")
    return X, Y


def per_class_report(y_true: np.ndarray, probs: np.ndarray) -> dict:
    preds = (probs > 0.5).astype(int)
    report = {}
    for i, sc in enumerate(SUPERCLASSES):
        support = int(y_true[:, i].sum())
        entry = {
            "support": support,
            "precision": float(precision_score(y_true[:, i], preds[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], preds[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], preds[:, i], zero_division=0)),
        }
        try:
            entry["roc_auc"] = float(roc_auc_score(y_true[:, i], probs[:, i]))
            entry["pr_auc"] = float(average_precision_score(y_true[:, i], probs[:, i]))
        except ValueError:
            entry["roc_auc"], entry["pr_auc"] = None, None
        report[sc] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5, help="small default: this is a proof-of-concept run")
    args = parser.parse_args()

    for name in ("train", "val", "test"):
        if not (config.SPLITS_DIR / f"multilabel_X_{name}.npy").exists():
            raise FileNotFoundError(
                f"multilabel_X_{name}.npy not found. Run scripts/prepare_multilabel_data.py first."
            )

    model_lib.set_global_seed()
    X_train, Y_train = load_split("train")
    X_val, Y_val = load_split("val")
    X_test, Y_test = load_split("test")
    logger.info("Train: %s, Val: %s, Test: %s", X_train.shape, X_val.shape, X_test.shape)
    logger.info("PROOF-OF-CONCEPT run (%d epochs, %d train segments) - see module docstring", args.epochs, len(X_train))

    model = model_lib.build_multilabel_rcnn(window_size=X_train.shape[1], n_classes=len(SUPERCLASSES))

    checkpoint_path = config.MODELS_DIR / "multilabel_rcnn_model.h5"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        X_train, Y_train,
        epochs=args.epochs,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=2,
    )
    model.save(checkpoint_path)

    test_probs = model.predict(X_test, verbose=0)
    test_report = per_class_report(Y_test, test_probs)
    for sc, m in test_report.items():
        logger.info(
            "  %-5s support=%-5d precision=%.3f recall=%.3f f1=%.3f AUC=%s",
            sc, m["support"], m["precision"], m["recall"], m["f1"], m["roc_auc"],
        )

    result = {
        "status": "PROOF_OF_CONCEPT - not a production multi-label model, see script docstring",
        "epochs_run": len(history.history["loss"]),
        "n_train_segments": int(len(X_train)),
        "n_test_segments": int(len(X_test)),
        "test_per_class": test_report,
    }
    out_path = config.REPORTS_DIR / "multilabel_evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
