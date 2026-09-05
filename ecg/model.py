"""Model architectures. The RCNN (CNN + LSTM) stays the primary model,
unchanged in structure from the original project - only training
mechanics (seeding, callbacks, checkpointing) are improved elsewhere.
"""
from __future__ import annotations

import os
import random

import numpy as np
from tensorflow.keras.layers import (
    LSTM,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
)
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from . import config


def set_global_seed(seed: int = config.RANDOM_SEED) -> None:
    """Best-effort determinism across numpy/tensorflow/random/python hash seed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def build_cnn(window_size: int = config.WINDOW_SIZE) -> Sequential:
    model = Sequential([
        Conv1D(64, kernel_size=5, activation="relu", input_shape=(window_size, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        Conv1D(256, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_rcnn(window_size: int = config.WINDOW_SIZE) -> Sequential:
    model = Sequential([
        Conv1D(64, kernel_size=5, activation="relu", input_shape=(window_size, 1)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(128, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(256, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        LSTM(256, return_sequences=True),
        LSTM(128, return_sequences=False),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    return model


MODEL_BUILDERS = {"cnn": build_cnn, "rcnn": build_rcnn}


def build_multilabel_rcnn(window_size: int = config.WINDOW_SIZE, n_classes: int = 5) -> Sequential:
    """Same RCNN backbone as build_rcnn, with an n_classes-way sigmoid
    head for independent per-superclass probabilities (multi-label,
    not softmax/multi-class) - see ecg/multilabel.py."""
    model = Sequential([
        Conv1D(64, kernel_size=5, activation="relu", input_shape=(window_size, 1)),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(128, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        BatchNormalization(),
        Conv1D(256, kernel_size=5, activation="relu"),
        MaxPooling1D(pool_size=2),
        LSTM(256, return_sequences=True),
        LSTM(128, return_sequences=False),
        Dense(256, activation="relu"),
        Dropout(0.4),
        Dense(n_classes, activation="sigmoid"),
    ])
    # NOTE: the generic "accuracy" string metric hits a Windows-specific
    # TF/Keras bug for multi-output sigmoid heads (OverflowError in
    # keras.src.backend.tensorflow.numpy.signbit: 0x80000000 doesn't fit
    # a 32-bit C long on Windows). BinaryAccuracy avoids that code path.
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[BinaryAccuracy(name="binary_accuracy")],
    )
    return model
