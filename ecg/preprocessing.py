"""Single source of truth for ECG signal loading, R-peak detection and
beat-windowing. Used identically by the offline data-prep script and the
online inference path in the web app, so training and serving can never
drift apart.

Two real bugs in the original scripts are fixed here:

1. Records were read with ``physical=False`` (raw, uncalibrated ADC
   counts). Different PTB-XL records can use different amplifier
   gain/baseline, so raw digital counts are not comparable across
   records. We now read ``physical=True`` (``p_signal``), which is in
   calibrated millivolts and is safe to compare/normalize across the
   whole dataset.
2. Windows were never amplitude-normalized before being fed to the
   network. Each extracted beat is now z-score normalized
   (zero mean, unit variance), a standard step for ECG deep learning
   that makes the model robust to inter-patient amplitude differences.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb
from scipy.signal import find_peaks, resample_poly

from . import config

logger = logging.getLogger(__name__)


@dataclass
class LoadedSignal:
    signal: np.ndarray  # 1D lead signal
    fs: float           # sampling rate in Hz
    calibrated: bool    # True if physical units (mV), False if raw fallback


def load_lead_signal(record_path: str, lead_index: int = config.LEAD_INDEX) -> LoadedSignal:
    """Load a single lead from a WFDB record.

    ``record_path`` is a path *without* extension, matching wfdb's
    convention (e.g. ``.../00001_lr``). Falls back to a best-effort raw
    ``.dat`` read (assuming 12 interleaved int16 channels) when no
    ``.hea`` header is available, matching what the original app
    supported for headerless uploads - but this fallback cannot apply
    gain/baseline calibration, so results are less reliable.
    """
    hea_path = Path(str(record_path) + ".hea")
    if hea_path.exists():
        record = wfdb.rdrecord(str(record_path), physical=True)
        signal = record.p_signal[:, lead_index]
        fs = float(record.fs)
        return LoadedSignal(signal=signal, fs=fs, calibrated=True)

    dat_path = Path(str(record_path) + ".dat")
    logger.warning("No .hea header found for %s; falling back to uncalibrated raw read", record_path)
    raw = np.fromfile(dat_path, dtype=np.int16)
    num_channels = 12
    usable = (len(raw) // num_channels) * num_channels
    signal_data = raw[:usable].reshape(-1, num_channels)
    signal = signal_data[:, lead_index].astype(np.float64)
    return LoadedSignal(signal=signal, fs=float(config.SAMPLING_RATE_HZ), calibrated=False)


def resample_to_target_rate(
    signal: np.ndarray, fs: float, target_fs: float = config.SAMPLING_RATE_HZ,
) -> tuple[np.ndarray, float]:
    """Resample so a fixed-sample-count window always spans the same
    physical duration regardless of a record's native rate. Training
    uses 100Hz records; without this, a 500Hz upload's 200-sample window
    would cover 0.4s instead of the 2s window the model was trained on.
    """
    if abs(fs - target_fs) < 1e-6:
        return signal, fs
    ratio = Fraction(target_fs).limit_denominator(1000) / Fraction(fs).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    resampled = resample_poly(signal, up, down)
    return resampled, target_fs


def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks with an amplitude threshold and a physiological
    minimum RR interval (derived from fs so it behaves consistently at
    100Hz or 500Hz, unlike the original hardcoded ``distance=30``)."""
    if signal.size == 0:
        return np.array([], dtype=int)
    min_distance = max(1, int(round(config.MIN_RR_SECONDS * fs)))
    height = float(np.max(signal)) * config.PEAK_HEIGHT_RATIO
    peaks, _ = find_peaks(signal, height=height, distance=min_distance)
    return peaks


def extract_windows(signal: np.ndarray, peaks: np.ndarray, window_size: int = config.WINDOW_SIZE) -> np.ndarray:
    """Slice fixed-size windows centered on each R-peak, normalizing each
    window to zero mean / unit variance."""
    half = window_size // 2
    segments = []
    for peak in peaks:
        start = peak - half
        end = start + window_size
        if start < 0 or end > len(signal):
            continue
        segment = signal[start:end].astype(np.float64)
        std = segment.std()
        if std < 1e-8:
            continue  # flat/dead segment, not a usable beat
        segment = (segment - segment.mean()) / std
        segments.append(segment)
    if not segments:
        return np.empty((0, window_size), dtype=np.float64)
    return np.stack(segments)


def prepare_segments_for_record(
    record_path: str,
    lead_index: int = config.LEAD_INDEX,
    window_size: int = config.WINDOW_SIZE,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Full pipeline for one record: load -> detect peaks -> window.

    Returns ``(segments, preview_signal)`` where ``preview_signal`` is a
    short unnormalized snippet suitable for plotting in the UI.
    """
    loaded = load_lead_signal(record_path, lead_index=lead_index)
    signal, fs = resample_to_target_rate(loaded.signal, loaded.fs)
    peaks = detect_r_peaks(signal, fs)
    segments = extract_windows(signal, peaks, window_size=window_size)
    preview = signal[: min(1000, len(signal))]
    return segments, preview
