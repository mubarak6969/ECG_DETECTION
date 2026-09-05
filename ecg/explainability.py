"""Lightweight, research-grade model attribution.

Uses vanilla gradient saliency (Simonyan et al., 2013): the gradient of
the model's output probability with respect to each input timestep,
which requires no architecture change and works for any differentiable
Keras model - safe to apply to the existing Conv1D+LSTM RCNN as-is.

This is model attribution, not clinical reasoning: it shows which parts
of a beat window most influenced *this model's* output, not a medical
explanation of what is wrong with the heartbeat. Callers must present it
with that distinction intact - see app/main.py and the UI copy.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf


def beat_saliency(model, beat_window: np.ndarray) -> np.ndarray:
    """Gradient-based per-timestep importance for one beat window.

    ``beat_window`` is a single window, shape (window_size,) or
    (window_size, 1). Returns an array of shape (window_size,) with
    values normalized to [0, 1] (0 = no influence on the prediction,
    1 = most influential timestep in this window).
    """
    x = tf.convert_to_tensor(beat_window.reshape(1, -1, 1), dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x, training=False)
    gradient = tape.gradient(prediction, x)
    if gradient is None:
        return np.zeros(beat_window.shape[0])

    importance = np.abs(gradient.numpy().reshape(-1))
    peak = importance.max()
    if peak < 1e-12:
        return np.zeros_like(importance)
    return importance / peak


def most_influential_beat(beat_probs: np.ndarray) -> int:
    """Index of the beat furthest from the decision boundary - the one
    the model was most confident about, and therefore the most
    representative single beat to explain."""
    return int(np.argmax(np.abs(beat_probs - 0.5)))
