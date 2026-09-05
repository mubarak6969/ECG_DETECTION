import numpy as np
import pytest
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

from ecg.explainability import beat_saliency, most_influential_beat


def test_most_influential_beat_picks_furthest_from_boundary():
    probs = np.array([0.5, 0.51, 0.9, 0.49])
    assert most_influential_beat(probs) == 2


@pytest.fixture
def linear_model():
    # A model whose gradient is known exactly: sigmoid(w . flatten(x)),
    # so d(output)/dx is a positive scalar multiple of w everywhere -
    # lets us assert saliency reflects the *weights*, not noise.
    model = Sequential([
        Flatten(input_shape=(10, 1)),
        Dense(1, activation="sigmoid", use_bias=False),
    ])
    known_weights = np.linspace(-1, 1, 10).reshape(10, 1)
    model.set_weights([known_weights])
    return model, known_weights.ravel()


def test_beat_saliency_matches_known_linear_gradient(linear_model):
    model, weights = linear_model
    x = np.random.default_rng(0).normal(size=10)
    saliency = beat_saliency(model, x)

    assert saliency.shape == (10,)
    assert saliency.min() >= 0.0
    assert saliency.max() == pytest.approx(1.0)

    # For sigmoid(w . x), d(output)/dx = output*(1-output)*w, which is a
    # positive scalar multiple of w - so |gradient| ranks positions in
    # exactly the same order as |w|, and normalizing by the max recovers
    # |w| / max(|w|) exactly.
    expected = np.abs(weights) / np.abs(weights).max()
    assert np.allclose(saliency, expected, atol=1e-4)


def test_beat_saliency_handles_flat_input_without_crashing():
    from tensorflow.keras.layers import Flatten
    model = Sequential([Flatten(input_shape=(5, 1)), Dense(1, activation="sigmoid")])
    result = beat_saliency(model, np.zeros(5))
    assert result.shape == (5,)
    assert np.all(np.isfinite(result))
