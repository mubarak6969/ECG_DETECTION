"""EXPERIMENTAL: build train/val/test segment arrays for the multi-label
diagnostic superclass task (NORM/MI/STTC/CD/HYP). Separate from, and
does not touch, the primary binary pipeline's splits.

Uses the exact same leakage-safe fold assignment (strat_fold 1-8/9/10)
as the binary task, so the two tasks' test sets are drawn from the same
untouched patients/records - just re-derives the label from
scp_statements.csv instead of the NORM-vs-rest rule.

Usage:
    python scripts/prepare_multilabel_data.py                  # small proof-of-concept cap
    python scripts/prepare_multilabel_data.py --max-train-records 0   # full training fold (~17k records, hours on CPU)
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

from ecg import config, dataset, multilabel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_multilabel_data")

# Deliberately small: this pass only proves the pipeline is correct
# end-to-end, not a production multi-label model. See README/model card.
DEFAULT_POC_TRAIN_RECORDS = 800


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train-records", type=int, default=DEFAULT_POC_TRAIN_RECORDS,
                         help="0 = use the full training fold (no cap)")
    args = parser.parse_args()

    df = multilabel.load_multilabel_metadata()
    logger.info("Multi-label metadata: %d records, %d patients", len(df), df["patient_id"].nunique())
    logger.info("Per-superclass record counts (full dataset): %s", multilabel.class_distribution(df))
    logger.info(
        "Multi-label records (>1 superclass): %d (%.1f%%)",
        df["multilabel"].sum(), 100 * df["multilabel"].mean(),
    )

    train_df, val_df, test_df = dataset.fold_split(df)
    logger.info("Fold split -> train %d / val %d / test %d records", len(train_df), len(val_df), len(test_df))

    if args.max_train_records and len(train_df) > args.max_train_records:
        train_df = train_df.sample(n=args.max_train_records, random_state=config.RANDOM_SEED)
        logger.warning(
            "PROOF-OF-CONCEPT CAP: using %d/%d available training records (--max-train-records=0 for the full fold)",
            args.max_train_records, len(dataset.fold_split(df)[0]),
        )

    report = {"dataset_root": str(config.DATASET_ROOT), "superclasses": list(multilabel.SUPERCLASSES)}
    for name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        t0 = time.time()
        logger.info("Building multilabel %s segments from %d records...", name, len(split_df))
        X, Y, ids = multilabel.build_multilabel_segments(split_df)
        elapsed = time.time() - t0
        per_class_segment_counts = {sc: int(Y[:, i].sum()) for i, sc in enumerate(multilabel.SUPERCLASSES)}
        logger.info("%s: %d segments in %.1fs, per-class counts %s", name, len(X), elapsed, per_class_segment_counts)

        np.save(config.SPLITS_DIR / f"multilabel_X_{name}.npy", X)
        np.save(config.SPLITS_DIR / f"multilabel_Y_{name}.npy", Y)
        np.save(config.SPLITS_DIR / f"multilabel_ids_{name}.npy", ids)

        report[name] = {
            "n_records": int(len(split_df)),
            "n_segments": int(len(X)),
            "per_class_segment_counts": per_class_segment_counts,
            "seconds": round(elapsed, 1),
        }

    with open(config.REPORTS_DIR / "multilabel_data_report.json", "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", config.REPORTS_DIR / "multilabel_data_report.json")


if __name__ == "__main__":
    main()
