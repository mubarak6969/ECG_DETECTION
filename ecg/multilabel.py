"""Experimental multi-label diagnostic pipeline (PTB-XL superclasses).

Separate from, and does not modify, the primary binary normal/abnormal
pipeline in ecg/dataset.py - this module is additive infrastructure for
PRIORITY 3 of the hardening pass: a clean path to eventually classify
*which kind* of abnormality is present, not just that one exists.

Labels are PTB-XL's own five diagnostic superclasses (`diagnostic_class`
in scp_statements.csv), the standard grouping used in the PTB-XL paper
(Wagner et al., 2020) and clinically meaningful on their own:

    NORM  - normal ECG
    MI    - myocardial infarction (ischemic/infarction-related)
    STTC  - ST/T wave changes
    CD    - conduction disturbance
    HYP   - hypertrophy

This was chosen over the ~70 individual SCP-ECG statement codes because
most individual codes have too few samples to train on reliably; the
five superclasses each have thousands of records (see
class_distribution() output / artifacts/reports/multilabel_data_report.json).

A record can map to more than one superclass (true multi-label task,
~23% of records in this dataset carry more than one) or to zero (no
mapped superclass at all, ~2% of records) - those are excluded rather
than guessed at.
"""
from __future__ import annotations

import ast
import logging

import numpy as np
import pandas as pd

from . import config
from .preprocessing import prepare_segments_for_record

logger = logging.getLogger(__name__)

SUPERCLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")


def load_code_to_class() -> dict[str, str]:
    scp = pd.read_csv(config.SCP_STATEMENTS_CSV, index_col=0)
    return scp["diagnostic_class"].dropna().to_dict()


def _target_vector(scp_codes_str: str, code_to_class: dict[str, str]) -> np.ndarray | None:
    try:
        codes = ast.literal_eval(scp_codes_str)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(codes, dict):
        return None
    present = {c for c, likelihood in codes.items() if likelihood > 0}
    classes_present = {code_to_class[c] for c in present if c in code_to_class}
    if not classes_present:
        return None  # no mappable diagnostic superclass for this record
    return np.array([1.0 if sc in classes_present else 0.0 for sc in SUPERCLASSES])


def load_multilabel_metadata() -> pd.DataFrame:
    """Same PTB-XL metadata as the binary pipeline, plus a `target`
    column (one multi-hot np.ndarray of length len(SUPERCLASSES) per
    row) and a `multilabel` column flagging records with >1 superclass.
    Records with zero mappable superclasses are dropped.
    """
    if not config.PTBXL_DATABASE_CSV.exists():
        raise FileNotFoundError(
            f"ptbxl_database.csv not found at {config.PTBXL_DATABASE_CSV}. "
            "Set ECG_DATASET_ROOT (see .env.example)."
        )
    code_to_class = load_code_to_class()
    df = pd.read_csv(config.PTBXL_DATABASE_CSV)
    df["target"] = df["scp_codes"].apply(lambda s: _target_vector(s, code_to_class))
    n_before = len(df)
    df = df[df["target"].notna()].reset_index(drop=True)
    logger.info("Dropped %d/%d records with no mappable diagnostic superclass", n_before - len(df), n_before)
    df["multilabel"] = df["target"].apply(lambda t: int(t.sum()) > 1)
    return df


def class_distribution(df: pd.DataFrame) -> dict[str, int]:
    counts = {sc: 0 for sc in SUPERCLASSES}
    for t in df["target"]:
        for i, sc in enumerate(SUPERCLASSES):
            counts[sc] += int(t[i])
    return counts


def build_multilabel_segments(df: pd.DataFrame, log_every: int = 500):
    """Same beat extraction as the binary pipeline (ecg.dataset.build_segments)
    but carrying a multi-hot target vector per segment instead of a
    scalar label."""
    all_segments, all_targets, all_ids = [], [], []
    n_records = len(df)
    n_ok, n_empty = 0, 0
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        record_path = config.DATASET_ROOT / row[config.FILENAME_COLUMN]
        try:
            segments, _ = prepare_segments_for_record(str(record_path))
        except Exception as exc:
            logger.warning("Skipping record %s (%s)", row["ecg_id"], exc)
            segments = np.empty((0, config.WINDOW_SIZE))
        if segments.shape[0] == 0:
            n_empty += 1
        else:
            n_ok += 1
            all_segments.append(segments)
            all_targets.extend([row["target"]] * len(segments))
            all_ids.extend([row["ecg_id"]] * len(segments))
        if log_every and i % log_every == 0:
            logger.info("Processed %d/%d records (%d ok, %d empty)", i, n_records, n_ok, n_empty)

    if not all_segments:
        return np.empty((0, config.WINDOW_SIZE)), np.empty((0, len(SUPERCLASSES))), np.empty((0,))

    X = np.concatenate(all_segments, axis=0)
    Y = np.stack(all_targets)
    ids = np.array(all_ids)
    logger.info("Built %d segments from %d/%d records (%d had no usable beats)", len(X), n_ok, n_records, n_empty)
    return X, Y, ids
