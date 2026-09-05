"""Train the CNN baseline and the primary RCNN model on the prepared
PTB-XL segment splits. Requires scripts/prepare_data.py to have been run
first.

Usage:
    python scripts/train_model.py --model rcnn
    python scripts/train_model.py --model both --epochs 30
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from ecg import config
from ecg import model as model_lib

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_model")


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    X = np.load(config.SPLITS_DIR / f"X_{name}.npy")
    y = np.load(config.SPLITS_DIR / f"y_{name}.npy")
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def train_one(model_type: str, epochs: int, X_train, y_train, X_val, y_val) -> dict:
    model_lib.set_global_seed()
    logger.info("Building %s model", model_type)
    model = model_lib.MODEL_BUILDERS[model_type](window_size=X_train.shape[1])

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), class_weights)}
    logger.info("Class weights: %s", class_weight_dict)

    checkpoint_path = config.MODELS_DIR / f"{model_type}_model.h5"
    plateau_patience = max(3, config.EARLY_STOPPING_PATIENCE // 2)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=plateau_patience, min_lr=1e-6),
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True),
    ]

    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - t0
    epochs_run = len(history.history["loss"])
    logger.info("%s trained in %.1fs over %d epochs (stopped early if <max)", model_type, elapsed, epochs_run)

    model.save(checkpoint_path)  # ensure final weights are on disk even if checkpoint dir changed
    history_path = config.REPORTS_DIR / f"{model_type}_training_history.json"
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    return {
        "model_path": str(checkpoint_path),
        "history_path": str(history_path),
        "epochs_run": epochs_run,
        "seconds": round(elapsed, 1),
        "class_weight": class_weight_dict,
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "best_val_loss": float(min(history.history["val_loss"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "rcnn", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=config.MAX_EPOCHS)
    args = parser.parse_args()

    for name in ("X_train", "y_train", "X_val", "y_val"):
        if not (config.SPLITS_DIR / f"{name}.npy").exists():
            raise FileNotFoundError(f"{name}.npy not found in {config.SPLITS_DIR}. Run scripts/prepare_data.py first.")

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    logger.info("Train segments: %s, Val segments: %s", X_train.shape, X_val.shape)

    models_to_run = ["cnn", "rcnn"] if args.model == "both" else [args.model]

    summary_path = config.REPORTS_DIR / "training_summary.json"
    try:
        with open(summary_path) as f:
            summary = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        summary = {}

    for model_type in models_to_run:
        logger.info("=== Training %s ===", model_type.upper())
        summary[model_type] = train_one(model_type, args.epochs, X_train, y_train, X_val, y_val)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Training summary written to %s", summary_path)


if __name__ == "__main__":
    main()
