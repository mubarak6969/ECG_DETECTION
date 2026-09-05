from pathlib import Path

import numpy as np
import pytest

from ecg import config
from ecg.preprocessing import (
    detect_r_peaks,
    extract_windows,
    load_lead_signal,
    prepare_segments_for_record,
    resample_to_target_rate,
)


def make_synthetic_ecg(n_beats=10, fs=100, beat_period_s=1.0):
    """A crude synthetic signal: sharp spikes on a flat baseline, spaced
    one per second, so we know exactly where the peaks should land."""
    period_samples = int(beat_period_s * fs)
    length = n_beats * period_samples + period_samples
    signal = np.zeros(length)
    expected_peaks = []
    for i in range(n_beats):
        idx = period_samples // 2 + i * period_samples
        signal[idx] = 1.0
        expected_peaks.append(idx)
    return signal, np.array(expected_peaks)


def test_detect_r_peaks_finds_all_synthetic_beats():
    signal, expected_peaks = make_synthetic_ecg(n_beats=8, fs=100)
    peaks = detect_r_peaks(signal, fs=100)
    assert len(peaks) == len(expected_peaks)
    assert np.array_equal(peaks, expected_peaks)


def test_detect_r_peaks_empty_signal():
    assert detect_r_peaks(np.array([]), fs=100).size == 0


def test_detect_r_peaks_respects_min_rr_distance():
    # Two spikes 5 samples apart at fs=100 (0.05s) are physiologically
    # impossible (>1000bpm) and should be collapsed to one detection.
    signal = np.zeros(200)
    signal[100] = 1.0
    signal[105] = 0.99
    peaks = detect_r_peaks(signal, fs=100)
    assert len(peaks) == 1


def test_extract_windows_shape_and_normalization():
    signal, peaks = make_synthetic_ecg(n_beats=5, fs=100)
    # give beats some shape so std > 0
    rng = np.random.default_rng(0)
    signal = signal + rng.normal(0, 0.01, size=signal.shape)
    windows = extract_windows(signal, peaks, window_size=50)
    assert windows.shape[1] == 50
    assert windows.shape[0] <= len(peaks)
    for w in windows:
        assert abs(w.mean()) < 1e-6
        assert abs(w.std() - 1.0) < 1e-6


def test_extract_windows_drops_out_of_bounds_peaks():
    signal = np.random.default_rng(1).normal(0, 1, size=100)
    peaks = np.array([0, 5, 50, 99])  # first/last too close to the edges for window_size=40
    windows = extract_windows(signal, peaks, window_size=40)
    assert windows.shape[0] == 1  # only peak at index 50 has room on both sides


def test_extract_windows_drops_flat_segments():
    flat_signal = np.zeros(100)
    windows = extract_windows(flat_signal, np.array([50]), window_size=10)
    assert windows.shape[0] == 0


def test_resample_no_op_when_already_at_target_rate():
    signal = np.arange(100.0)
    resampled, fs = resample_to_target_rate(signal, fs=config.SAMPLING_RATE_HZ)
    assert fs == config.SAMPLING_RATE_HZ
    assert np.array_equal(resampled, signal)


def test_resample_preserves_physical_duration():
    # 5 seconds of signal at 500Hz -> resampled to 100Hz should still be
    # ~5 seconds (500 samples -> ~100 samples), so a fixed-length window
    # covers the same physical duration regardless of source rate.
    fs_native = 500
    duration_s = 5.0
    signal = np.sin(np.linspace(0, 2 * np.pi * 5, int(fs_native * duration_s)))
    resampled, fs_out = resample_to_target_rate(signal, fs=fs_native, target_fs=100)
    assert fs_out == 100
    assert abs(len(resampled) / fs_out - duration_s) < 0.05


def test_load_lead_signal_missing_record_raises():
    with pytest.raises(FileNotFoundError):
        load_lead_signal(str(config.PROJECT_ROOT / "data" / "demo" / "does_not_exist"))


def test_prepare_segments_handles_out_of_range_lead_index():
    # A malformed/unexpected lead_index (e.g. a record with fewer leads
    # than expected) must surface as a clean exception the caller can
    # catch, not a silent wrong-channel read.
    demo = config.DEMO_RECORD_PATH
    if not Path(str(demo) + ".hea").exists():
        pytest.skip("bundled demo record not present")
    with pytest.raises(IndexError):
        prepare_segments_for_record(str(demo), lead_index=99)
