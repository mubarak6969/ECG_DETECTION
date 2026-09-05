# Model Card: ECG Intelligence RCNN

**This is a research/portfolio prototype and NOT a medical diagnostic device.** It has not been clinically validated. Do not use it to diagnose, triage, or make any real healthcare decision about any person.

## Model architecture

Primary model: **RCNN** (1D CNN + LSTM), `ecg/model.py: build_rcnn`.

```
Conv1D(64, k=5) -> MaxPool -> BatchNorm
Conv1D(128, k=5) -> MaxPool -> BatchNorm
Conv1D(256, k=5) -> MaxPool
LSTM(256, return_sequences=True)
LSTM(128)
Dense(256, relu) -> Dropout(0.4)
Dense(1, sigmoid)
```

Input: one 200-sample (2.0s @ 100Hz) beat window, z-score normalized. Output: probability the beat is "abnormal." A CNN-only baseline (`build_cnn`, same first six layers, no LSTM) is trained alongside it for comparison, not as a fallback.

## Dataset

[PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) - 21,799 clinical 12-lead ECGs from 18,869 patients. Loaded from a local extraction, path configured via `ECG_DATASET_ROOT`; never bundled with this repository.

## Split methodology

PTB-XL's official `strat_fold` column (1-10), patient-stratified: **folds 1-8 train / fold 9 validation / fold 10 test.** Verified programmatically (`scripts/prepare_data.py`, `tests/test_dataset.py`) that zero patients and zero records appear in more than one split. This is the split protocol from the PTB-XL paper (Wagner et al., 2020), not an arbitrary choice.

## Label definition

Binary: a record is **normal** (0) if `NORM` is the only SCP-ECG statement present at nonzero diagnostic likelihood in `scp_codes`; otherwise **abnormal** (1). `scp_codes` values are confidence percentages, not booleans - a naive "raw key set" implementation (matching the project's original, pre-hardening version) misclassified 8,786/21,799 records (40%) as abnormal purely from a co-listed 0%-likelihood code; this was found and fixed. Corrected distribution: 8,976 normal / 12,823 abnormal records.

## Preprocessing

One shared implementation (`ecg/preprocessing.py`) used identically by training, evaluation, and live inference: calibrated signal load (`wfdb`, `physical=True`) → resample to 100Hz (so a window always spans the same physical duration regardless of a record's native rate) → R-peak detection (amplitude threshold + physiological minimum RR interval) → 200-sample window per beat, z-score normalized. Malformed input degrades to a clean rejection rather than a silent bad prediction.

## Training configuration

- Optimizer: Adam, lr 1e-4. Loss: binary cross-entropy.
- `class_weight="balanced"`, `EarlyStopping(patience=8, restore_best_weights=True)`, `ReduceLROnPlateau`, `ModelCheckpoint(save_best_only=True)`.
- Seed fixed (42) across numpy/TensorFlow/Python hash seed.
- Training fold: balanced (all normals + up to 3x sampled abnormals - close to a no-op given the corrected ~1:1.4 ratio), then **stratified-capped at 4,000 records** (`ECG_TRAIN_MAX_RECORDS`) purely for CPU training-time tractability in this environment. The full corrected training fold has **17,418 records** (23% of it was used) - see "How to train at full scale" below. Validation and test are never balanced or capped.
- Actual trained artifact: `artifacts/models/rcnn_model.h5`, 10 epochs (`training_summary.json`), batch size 64.

## Record-level aggregation and calibration

The application returns one prediction per uploaded ECG, not per beat. The aggregation method (mean/median/majority-vote/confidence-weighted/top-k-mean) and a calibration temperature were chosen **once, using the validation fold only**, by `scripts/select_aggregation_and_calibration.py`, then frozen into `ecg/config.py` and applied identically by both the evaluator and the live app (`ecg/postprocessing.py`). Result: plain **mean-pooling** was kept (no alternative method cleared a 0.5pp F1 improvement threshold on validation - the added complexity wasn't justified), with a calibration temperature of ~1.16 (a small, real Brier-score improvement, verified once on the untouched test fold after selection).

**A real bug was found and fixed during this**: the live endpoint had been computing the fraction of beats *individually* voted abnormal (`mean(prob > 0.5)`), not the mean probability - a materially different, uncalibratable quantity that could report exactly 100% confidence whenever all beats agreed in direction regardless of how weak the signal actually was. Confirmed on the bundled demo record: the old code reported 100.0% confidence; the corrected pipeline reports 76.6%.

## Evaluation methodology

`scripts/evaluate_model.py` scores the frozen model **exactly once** against the untouched test fold (fold 10), at two granularities:
- **beat-level**: one prediction per beat window (what the network is directly trained on).
- **record-level**: mean-pooled + calibrated, per uploaded ECG - what the application actually returns.

No metric, method, or hyperparameter was tuned against the test fold at any point.

## Metrics (test fold 10: 2,194 records / 22,212 beats)

| Granularity | Model | Accuracy | F1 (macro) | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Beat-level | CNN | 79.1% | 0.787 | 0.857 | 0.883 | 0.929 |
| Beat-level | RCNN | 79.4% | 0.790 | 0.859 | 0.883 | - |
| Record-level | CNN | 80.4% | 0.803 | 0.882 | 0.890 | - |
| Record-level | RCNN | 80.1% | 0.800 | 0.886 | 0.889 | 0.931 |

Calibration (record-level, RCNN): Brier score 0.1377 (uncalibrated) → 0.1369 (calibrated), temperature 1.16 - a small, genuine improvement, not a large one. Full numbers, confusion matrices, ROC curves, and calibration curves: `artifacts/reports/`.

**Honest comparison**: RCNN's recurrent layers give a small, real edge at beat-level; at record-level - the deployed granularity - RCNN and the CNN baseline are statistically indistinguishable (CNN nominally 0.3pp ahead in F1). RCNN is retained as the primary model per project scope; this is disclosed, not hidden.

## Explainability

Vanilla gradient saliency (`ecg/explainability.py`): for the single most-confident detected beat, the gradient of the model's output with respect to each input timestep, normalized to [0,1]. Returned by `/predict` as `explainability.saliency` alongside `explainability.beat_window`. **This is model attribution, not a clinical explanation** - it shows which parts of the input most moved *this model's* output, not what is medically wrong with the heartbeat. Verified against a closed-form gradient on a synthetic linear model (`tests/test_explainability.py`) to confirm the implementation is mathematically correct, not just "runs without crashing."

## Multi-label diagnostic pipeline (experimental, proof-of-concept only)

`ecg/multilabel.py` + `scripts/prepare_multilabel_data.py` + `scripts/train_multilabel_model.py` implement a second, additive task: PTB-XL's five diagnostic superclasses (NORM, MI, STTC, CD, HYP - the standard grouping from the PTB-XL paper), which are multi-label (23.7% of records carry more than one) and clinically meaningful on their own. Per-superclass sample counts across the full dataset: NORM 9,514 / MI 5,469 / STTC 5,108 / CD 4,898 / HYP 2,649 (424 records with no mappable superclass are excluded).

**Status: infrastructure verified end-to-end; the trained artifact is a small proof-of-concept only** (800 of 17,073 available training records, 5 epochs, CPU) - it exists to prove the dataset → model → per-class-evaluation path is correct and leakage-free, not as a claim of a production multi-label model. See `artifacts/reports/multilabel_evaluation_report.json` for its actual (modest, POC-scale) per-class metrics, and the header of `scripts/train_multilabel_model.py` for the exact command to train at full scale (`--max-train-records 0` in data prep, more epochs, ideally on a GPU).

## How to train at full scale

```bash
# Binary RCNN, full corrected training fold (17,418 records) instead of the 4,000-record cap:
ECG_TRAIN_MAX_RECORDS=0 python scripts/prepare_data.py
python scripts/train_model.py

# Multi-label RCNN, full training fold instead of the 800-record proof-of-concept:
python scripts/prepare_multilabel_data.py --max-train-records 0
python scripts/train_multilabel_model.py --epochs 30
```

Both will take substantially longer per epoch on CPU (hours, not minutes, for the binary case at 17k records with LSTM layers) - a GPU is recommended. Validation and test fold sizes are unaffected either way.

## Limitations and known failure modes

- **Record-level labels applied per-beat introduce label noise**: PTB-XL's diagnosis is one label per 10-second record, but training happens on individual beat windows carrying that record's label. Many beats inside an "abnormal" record can look morphologically normal. This caps achievable beat-level separation and is a property of the task formulation, not a fixable bug.
- **`confidence` is a calibrated probability mass, not a validated clinical certainty.** Calibration was fit and checked on Brier score only; no clinical ground-truth-agreement study has been done.
- **RCNN vs. CNN is close at record-level** (see Metrics) - the recurrent layers' benefit is real but small once beat probabilities are aggregated.
- R-peak detection is a fixed-threshold `find_peaks` call, not a dedicated QRS detector; noisy or low-amplitude leads can under/over-detect beats.
- Headerless `.dat`-only uploads cannot be gain/baseline-calibrated and are inherently less reliable.
- Training used a CPU-tractability cap (4,000/17,418 available training records for binary; 800/17,073 for the multi-label proof-of-concept) - both are documented, reproducible, and not hidden.
- The multi-label path is infrastructure + a small proof-of-concept, not a validated model - do not deploy it as-is.

## Intended use

Educational and portfolio demonstration of an end-to-end, leakage-safe ECG deep-learning pipeline: dataset methodology, preprocessing, model training/evaluation, calibration, explainability, and a served application. Suitable for technical review, learning, and demoing engineering practice.

## Prohibited use

Any real clinical, diagnostic, triage, insurance, or employment decision about a real person. Any use implying regulatory clearance or clinical validation, which this project does not have.

## Reproducibility

```bash
pip install -r requirements-dev.txt
cp .env.example .env   # set ECG_DATASET_ROOT
python scripts/prepare_data.py
python scripts/train_model.py
python scripts/select_aggregation_and_calibration.py
python scripts/evaluate_model.py
python -m pytest
```

Fixed seed (42) is used throughout; exact epoch counts, class weights, and per-run timing are written to `artifacts/reports/training_summary.json` on every training run.
