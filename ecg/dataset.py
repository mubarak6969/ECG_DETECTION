"""PTB-XL dataset loading, labeling and leakage-safe splitting.

Methodology (see README for full rationale):

* Label: binary normal (0) vs abnormal (1). A record is "normal" if the
  only SCP-ECG statement present at nonzero diagnostic likelihood is
  ``NORM``. PTB-XL's ``scp_codes`` values are likelihood percentages
  (0-100), not boolean flags: e.g. ``{'NORM': 100.0, 'SR': 0.0}`` means
  NORM at 100% confidence and SR (sinus rhythm) explicitly *ruled out*
  at 0%, i.e. this record is normal. The original project (and an
  earlier version of this module) treated the raw key set instead of
  filtering by likelihood, which misclassified 8,786 of 21,799 records
  (40%) as abnormal purely because of a co-listed 0%-likelihood code -
  this is fixed here.
* Split: PTB-XL's official ``strat_fold`` column (1-10) is used instead
  of a random row-level split. ``strat_fold`` is stratified at the
  *patient* level (verified: 0 patients span more than one fold), so
  folds 1-8 / 9 / 10 as train/val/test guarantees no patient's ECGs
  leak across the split - this is the split protocol recommended by the
  PTB-XL paper (Wagner et al., 2020).
* Balancing: only the *training* fold is class-balanced (all normals +
  a fixed multiple of randomly sampled abnormals). Validation and test
  keep the natural, heavily imbalanced class ratio so reported metrics
  reflect real-world performance, not an artificially balanced sample.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .preprocessing import prepare_segments_for_record

logger = logging.getLogger(__name__)


def classify_label(scp_codes: str) -> int:
    try:
        codes = ast.literal_eval(scp_codes)
        if not isinstance(codes, dict):
            return 1
        present = {code for code, likelihood in codes.items() if likelihood > 0}
        if present == {"NORM"}:
            return 0
        return 1
    except (ValueError, SyntaxError):
        return 1


def load_metadata() -> pd.DataFrame:
    if not config.PTBXL_DATABASE_CSV.exists():
        raise FileNotFoundError(
            f"ptbxl_database.csv not found at {config.PTBXL_DATABASE_CSV}. "
            "Set ECG_DATASET_ROOT to the folder containing the extracted "
            "PTB-XL release (see .env.example)."
        )
    df = pd.read_csv(config.PTBXL_DATABASE_CSV)
    df["label"] = df["scp_codes"].apply(classify_label)
    return df


def fold_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["strat_fold"].isin(config.TRAIN_FOLDS)].copy()
    val_df = df[df["strat_fold"] == config.VAL_FOLD].copy()
    test_df = df[df["strat_fold"] == config.TEST_FOLD].copy()
    return train_df, val_df, test_df


def balance_train(df: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    normal = df[df["label"] == 0]
    abnormal_pool = df[df["label"] == 1]
    n_abnormal = min(len(abnormal_pool), len(normal) * config.TRAIN_ABNORMAL_PER_NORMAL)
    abnormal = abnormal_pool.sample(n=n_abnormal, random_state=seed)
    balanced = pd.concat([normal, abnormal]).sample(frac=1, random_state=seed)

    cap = config.TRAIN_MAX_RECORDS
    if cap and len(balanced) > cap:
        frac = cap / len(balanced)
        balanced = (
            balanced.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=max(1, round(len(g) * frac)), random_state=seed))
        )

    return balanced


@dataclass
class SegmentSet:
    X: np.ndarray          # (n_segments, window_size)
    y: np.ndarray          # (n_segments,)
    record_ids: np.ndarray  # ecg_id per segment, for provenance/debugging


def build_segments(df: pd.DataFrame, log_every: int = 500) -> SegmentSet:
    all_segments, all_labels, all_ids = [], [], []
    n_records = len(df)
    n_ok, n_empty = 0, 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        record_path = config.DATASET_ROOT / row[config.FILENAME_COLUMN]
        try:
            segments, _ = prepare_segments_for_record(str(record_path))
        except Exception as exc:  # corrupt/missing file - skip, don't crash the whole run
            logger.warning("Skipping record %s (%s)", row["ecg_id"], exc)
            segments = np.empty((0, config.WINDOW_SIZE))
        if segments.shape[0] == 0:
            n_empty += 1
        else:
            n_ok += 1
            all_segments.append(segments)
            all_labels.extend([row["label"]] * len(segments))
            all_ids.extend([row["ecg_id"]] * len(segments))
        if log_every and i % log_every == 0:
            logger.info("Processed %d/%d records (%d ok, %d empty)", i, n_records, n_ok, n_empty)

    if not all_segments:
        return SegmentSet(np.empty((0, config.WINDOW_SIZE)), np.empty((0,)), np.empty((0,)))

    X = np.concatenate(all_segments, axis=0)
    y = np.array(all_labels)
    ids = np.array(all_ids)
    logger.info("Built %d segments from %d/%d records (%d had no usable beats)", len(X), n_ok, n_records, n_empty)
    return SegmentSet(X=X, y=y, record_ids=ids)


def class_distribution(y: np.ndarray) -> dict:
    values, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}
