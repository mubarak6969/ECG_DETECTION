import numpy as np
import pytest

from ecg.postprocessing import (
    aggregate,
    aggregate_by_record,
    apply_temperature,
    brier_score,
)


def test_aggregate_mean():
    assert aggregate(np.array([0.2, 0.4, 0.9]), method="mean") == pytest.approx(0.5)


def test_aggregate_median():
    assert aggregate(np.array([0.1, 0.2, 0.9]), method="median") == pytest.approx(0.2)


def test_aggregate_majority_vote():
    # 2 of 3 beats individually vote abnormal (>0.5)
    assert aggregate(np.array([0.2, 0.6, 0.7]), method="majority_vote") == pytest.approx(2 / 3)


def test_aggregate_confidence_weighted_favors_extreme_beats():
    # one very confident abnormal beat should outweigh two near-boundary beats
    probs = np.array([0.51, 0.49, 0.99])
    result = aggregate(probs, method="confidence_weighted")
    assert result > probs.mean()  # weighting shifts it above the plain mean


def test_aggregate_top_k_mean_uses_only_most_confident():
    probs = np.array([0.5, 0.51, 0.49, 0.95, 0.9])
    result = aggregate(probs, method="top_k_mean", k=2)
    assert result == pytest.approx((0.95 + 0.9) / 2)


def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError):
        aggregate(np.array([]), method="mean")


def test_aggregate_rejects_unknown_method():
    with pytest.raises(ValueError):
        aggregate(np.array([0.5]), method="not_a_real_method")


def test_aggregate_by_record_groups_correctly():
    probs = np.array([0.1, 0.2, 0.8, 0.9])
    ids = np.array(["a", "a", "b", "b"])
    unique_ids, record_probs = aggregate_by_record(probs, ids, method="mean")
    assert list(unique_ids) == ["a", "b"]
    assert record_probs[0] == pytest.approx(0.15)
    assert record_probs[1] == pytest.approx(0.85)


def test_apply_temperature_is_identity_at_one():
    probs = np.array([0.1, 0.5, 0.9])
    result = apply_temperature(probs, temperature=1.0)
    assert np.allclose(result, probs, atol=1e-6)


def test_apply_temperature_preserves_decision_boundary():
    # monotonic transform: whatever is >0.5 stays >0.5 regardless of T
    probs = np.array([0.3, 0.5, 0.7, 0.95])
    for t in (0.3, 0.7, 1.5, 3.0):
        calibrated = apply_temperature(probs, temperature=t)
        assert np.array_equal(probs > 0.5, calibrated > 0.5)


def test_apply_temperature_softens_with_larger_t():
    probs = np.array([0.9])
    softened = apply_temperature(probs, temperature=3.0)
    assert softened[0] < probs[0]  # T>1 pulls extreme probabilities toward 0.5


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score(np.array([0.0, 1.0]), np.array([0, 1])) == pytest.approx(0.0)


def test_brier_score_worst_predictions_is_one():
    assert brier_score(np.array([1.0, 0.0]), np.array([0, 1])) == pytest.approx(1.0)
