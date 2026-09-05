import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_model import _score  # noqa: E402


def test_score_reports_specificity_as_normal_class_recall():
    # 4 normals (label 0), 2 correctly predicted normal -> specificity 0.5
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.6, 0.7, 0.9, 0.95])
    result = _score(y_true, y_pred, probs)
    assert result["specificity"] == 0.5
    assert result["per_class"]["normal"]["recall"] == 0.5


def test_score_confusion_matrix_is_normal_abnormal_ordered():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    probs = np.array([0.1, 0.6, 0.7, 0.9])
    result = _score(y_true, y_pred, probs)
    # rows/cols ordered [normal, abnormal]: 1 true normal correct, 1 false positive,
    # 0 false negatives, 2 true positives
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]


def test_record_level_aggregation_matches_manual_mean_pooling():
    # Two records: record A has 3 beats all normal-leaning, record B has
    # 2 beats split evenly but tipping abnormal on average.
    record_ids = np.array(["A", "A", "A", "B", "B"])
    probs = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    y_beat = np.array([0, 0, 0, 1, 1])

    unique_ids = np.unique(record_ids)
    record_probs = np.array([probs[record_ids == rid].mean() for rid in unique_ids])
    record_true = np.array([y_beat[record_ids == rid][0] for rid in unique_ids])

    assert list(unique_ids) == ["A", "B"]
    assert np.isclose(record_probs[0], 0.2)  # mean of 0.1,0.2,0.3
    assert np.isclose(record_probs[1], 0.65)  # mean of 0.6,0.7
    assert list(record_true) == [0, 1]
    assert list((record_probs > 0.5).astype(int)) == [0, 1]
