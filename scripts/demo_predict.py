"""Minimal CLI demo: run the trained model against the bundled sample
WFDB record using the same preprocessing pipeline as the web app.

Usage:
    python scripts/demo_predict.py
    python scripts/demo_predict.py --record path/to/record_without_extension
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tensorflow.keras.saving import load_model

from ecg import config
from ecg.preprocessing import prepare_segments_for_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=str, default=str(config.DEMO_RECORD_PATH))
    args = parser.parse_args()

    model_path = config.MODEL_PATH if config.MODEL_PATH.exists() else config.LEGACY_MODEL_PATH
    print(f"Loading model from {model_path}")
    model = load_model(str(model_path), compile=False)

    print(f"Reading record {args.record}")
    segments, _ = prepare_segments_for_record(args.record)
    if segments.shape[0] == 0:
        print("No usable beats extracted from this record.")
        return

    X = segments.reshape((-1, segments.shape[1], 1))
    predictions = model.predict(X, verbose=0)
    avg_pred = float(predictions.mean())
    label = "ABNORMAL" if avg_pred > 0.5 else "NORMAL"
    confidence = avg_pred if avg_pred > 0.5 else 1 - avg_pred

    print(f"Beats analyzed: {X.shape[0]}")
    print(f"Prediction: {label} (confidence {confidence:.2%})")


if __name__ == "__main__":
    main()
