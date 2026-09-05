"""Central configuration: paths, seeds and hyperparameters.

Every value here can be overridden with an environment variable of the same
name (see .env.example). Nothing in this project should hardcode a dataset
path or window size outside of this module.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Dataset location -------------------------------------------------
# The full PTB-XL dataset (several GB) is never stored inside the repo.
# Point ECG_DATASET_ROOT at a local extraction of the PhysioNet PTB-XL
# release (the folder that directly contains ptbxl_database.csv,
# records100/ and records500/).
DEFAULT_DATASET_ROOT = (
    Path.home() / "Desktop" / "ECG-Files" / "dataset"
    / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
)
DATASET_ROOT = Path(os.environ.get("ECG_DATASET_ROOT", DEFAULT_DATASET_ROOT))

PTBXL_DATABASE_CSV = DATASET_ROOT / "ptbxl_database.csv"
SCP_STATEMENTS_CSV = DATASET_ROOT / "scp_statements.csv"

# 100Hz records are used: a 2-second window at 100Hz captures a full
# QRS-T complex around each R-peak, whereas the same 200-sample window at
# 500Hz only spans 0.4s. See README "Methodology notes".
RECORD_RATE = "lr"  # 'lr' -> records100 (100Hz), 'hr' -> records500 (500Hz)
SAMPLING_RATE_HZ = 100 if RECORD_RATE == "lr" else 500
FILENAME_COLUMN = "filename_lr" if RECORD_RATE == "lr" else "filename_hr"

# --- Artifacts (generated, gitignored) ---------------------------------
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SPLITS_DIR = ARTIFACTS_DIR / "splits"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for _d in (ARTIFACTS_DIR, SPLITS_DIR, MODELS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reproducibility ----------------------------------------------------
RANDOM_SEED = int(os.environ.get("ECG_RANDOM_SEED", 42))

# --- Preprocessing --------------------------------------------------
WINDOW_SIZE = int(os.environ.get("ECG_WINDOW_SIZE", 200))
LEAD_INDEX = int(os.environ.get("ECG_LEAD_INDEX", 1))  # Lead II
PEAK_HEIGHT_RATIO = float(os.environ.get("ECG_PEAK_HEIGHT_RATIO", 0.35))
MIN_RR_SECONDS = float(os.environ.get("ECG_MIN_RR_SECONDS", 0.3))  # caps HR at 200bpm

# --- PTB-XL fold split (paper-recommended, patient-safe) ----------------
# strat_fold is stratified at the patient level: no patient appears in more
# than one fold, so splitting on it (rather than a random row-level split)
# prevents patient/record leakage between train, validation and test.
TRAIN_FOLDS = list(range(1, 9))   # folds 1-8
VAL_FOLD = 9
TEST_FOLD = 10

# --- Class balancing (train split only; val/test stay at natural ratio) -
TRAIN_ABNORMAL_PER_NORMAL = int(os.environ.get("ECG_TRAIN_ABNORMAL_RATIO", 3))

# Optional stratified cap on the number of *training* records, applied
# after balancing. With the corrected label rule (see dataset.py) folds
# 1-8 hold ~17k records; training the CNN+LSTM RCNN on all of them on
# CPU takes hours per epoch. This cap keeps a from-scratch CPU run in
# this repo tractable while still being ~7x more training data than the
# original project used. Val/test are never capped - real evaluation
# always runs on the full, natural-distribution held-out fold. Set to 0
# (or unset the env var and pass None) to disable and train on the full
# balanced set given enough compute time.
TRAIN_MAX_RECORDS = int(os.environ.get("ECG_TRAIN_MAX_RECORDS", 4000))

# --- Training ------------------------------------------------------
MAX_EPOCHS = int(os.environ.get("ECG_MAX_EPOCHS", 60))
BATCH_SIZE = int(os.environ.get("ECG_BATCH_SIZE", 64))
LEARNING_RATE = float(os.environ.get("ECG_LEARNING_RATE", 1e-4))
EARLY_STOPPING_PATIENCE = int(os.environ.get("ECG_EARLY_STOPPING_PATIENCE", 8))

# --- Inference / API -------------------------------------------------
UPLOADS_DIR = PROJECT_ROOT / "uploads"
ALLOWED_UPLOAD_EXTENSIONS = {".dat", ".hea"}
MAX_UPLOAD_BYTES = int(os.environ.get("ECG_MAX_UPLOAD_BYTES", 5 * 1024 * 1024))  # 5 MB
# NTFS/most filesystems cap a single path component at 255 bytes; stay
# well under that so uploads/<uuid>/<filename> never risks an OSError.
MAX_FILENAME_LENGTH = int(os.environ.get("ECG_MAX_FILENAME_LENGTH", 100))
MODEL_PATH = Path(os.environ.get("ECG_MODEL_PATH", MODELS_DIR / "rcnn_model.h5"))
LEGACY_MODEL_PATH = PROJECT_ROOT / "rcnn_model.h5"

DEMO_RECORD_PATH = PROJECT_ROOT / "data" / "demo" / "00001_hr"

# --- Record-level aggregation + calibration -----------------------------
# Chosen ONCE by scripts/select_aggregation_and_calibration.py using the
# VALIDATION fold only (never the test fold), then frozen here by reading
# back its output. Falls back to an uncalibrated plain mean if that
# script has not been run yet (e.g. a fresh checkout).
_AGG_CAL_PATH = ARTIFACTS_DIR / "reports" / "aggregation_calibration_selection.json"


def _load_aggregation_calibration_choice() -> tuple[str, float]:
    try:
        import json
        with open(_AGG_CAL_PATH) as f:
            selection = json.load(f)
        return selection["chosen_method"], float(selection["chosen_temperature"])
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return "mean", 1.0


AGGREGATION_METHOD, CALIBRATION_TEMPERATURE = _load_aggregation_calibration_choice()
# Explicit env overrides still win, for local experimentation.
AGGREGATION_METHOD = os.environ.get("ECG_AGGREGATION_METHOD", AGGREGATION_METHOD)
CALIBRATION_TEMPERATURE = float(os.environ.get("ECG_CALIBRATION_TEMPERATURE", CALIBRATION_TEMPERATURE))
