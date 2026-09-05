"""Build train/val/test segment arrays from PTB-XL using a leakage-safe,
fold-based split. See ecg/dataset.py for the methodology.

Usage:
    python scripts/prepare_data.py
    python scripts/prepare_data.py --max-records 300   # fast smoke test
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

from ecg import config, dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", type=int, default=None, help="cap records per split, for fast smoke tests")
    args = parser.parse_args()

    logger.info("Dataset root: %s", config.DATASET_ROOT)
    df = dataset.load_metadata()
    logger.info("Loaded metadata: %d records, %d patients", len(df), df["patient_id"].nunique())
    logger.info("Full-dataset label distribution: %s", df["label"].value_counts().to_dict())

    train_df, val_df, test_df = dataset.fold_split(df)
    logger.info(
        "Fold split -> train folds %s (%d rows), val fold %s (%d rows), test fold %s (%d rows)",
        config.TRAIN_FOLDS, len(train_df), config.VAL_FOLD, len(val_df), config.TEST_FOLD, len(test_df),
    )

    train_df = dataset.balance_train(train_df)
    logger.info("Balanced training records: %s", train_df["label"].value_counts().to_dict())

    if args.max_records:
        train_df = train_df.head(args.max_records)
        val_df = val_df.head(args.max_records)
        test_df = test_df.head(args.max_records)
        logger.warning("--max-records=%d applied: this is a smoke test, not a full run", args.max_records)

    report = {"dataset_root": str(config.DATASET_ROOT), "record_rate_hz": config.SAMPLING_RATE_HZ}

    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        t0 = time.time()
        logger.info("Building %s segments from %d records...", name, len(split_df))
        seg = dataset.build_segments(split_df)
        elapsed = time.time() - t0
        seg_dist = dataset.class_distribution(seg.y)
        logger.info("%s: %d segments in %.1fs, class distribution %s", name, len(seg.X), elapsed, seg_dist)

        np.save(config.SPLITS_DIR / f"X_{name}.npy", seg.X)
        np.save(config.SPLITS_DIR / f"y_{name}.npy", seg.y)
        np.save(config.SPLITS_DIR / f"ids_{name}.npy", seg.record_ids)
        split_df.to_csv(config.SPLITS_DIR / f"{name}_records.csv", index=False)

        report[name] = {
            "n_records": int(len(split_df)),
            "n_segments": int(len(seg.X)),
            "segment_class_distribution": dataset.class_distribution(seg.y),
            "seconds": round(elapsed, 1),
        }

    # Leakage check: no ecg_id should appear in more than one split's record list.
    train_ids = set(train_df["ecg_id"])
    val_ids = set(val_df["ecg_id"])
    test_ids = set(test_df["ecg_id"])
    overlap = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    report["record_overlap_across_splits"] = len(overlap)
    if overlap:
        logger.error("LEAKAGE DETECTED: %d records appear in multiple splits!", len(overlap))
    else:
        logger.info("Leakage check passed: 0 records shared across train/val/test")

    # Patient-level leakage check as well (belt and braces on top of strat_fold).
    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])
    test_patients = set(test_df["patient_id"])
    patient_overlap = (
        (train_patients & val_patients) | (train_patients & test_patients) | (val_patients & test_patients)
    )
    report["patient_overlap_across_splits"] = len(patient_overlap)
    if patient_overlap:
        logger.error("PATIENT LEAKAGE DETECTED: %d patients appear in multiple splits!", len(patient_overlap))
    else:
        logger.info("Patient leakage check passed: 0 patients shared across train/val/test")

    with open(config.REPORTS_DIR / "data_preparation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", config.REPORTS_DIR / "data_preparation_report.json")


if __name__ == "__main__":
    main()
