"""R-peak detection and beat-relative waveform regions (numpy/scipy only).

Used by xai_population.py to map Integrated-Gradients attributions onto
P / PR / QRS / ST / T regions. The region windows are FIXED offsets from the
R peak – a deliberate, documented approximation (no wave delineation); at
100 Hz one sample is 10 ms.

    region   window relative to the R peak
    P        [-250, -120) ms
    PR       [-120,  -50) ms
    QRS      [ -50,  +50] ms
    ST       ( +50, +150] ms
    T        (+150, +400] ms
    other    everything else (mostly the T-P interval / diastole)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

REGIONS = ["P", "PR", "QRS", "ST", "T", "other"]
REGION_MS = {"P": (-250, -120), "PR": (-120, -50), "QRS": (-50, 50), "ST": (50, 150), "T": (150, 400)}


def _bandpass(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 15.0) -> np.ndarray:
    nyq = fs / 2.0
    hi = min(hi, 0.9 * nyq)
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, x, axis=-1)


def detect_r_peaks(x_mv: np.ndarray, fs: float, refractory_s: float = 0.27) -> np.ndarray:
    """Multi-lead Pan–Tompkins-style detector. x_mv: [leads, T] → sample indices."""
    x = np.asarray(x_mv, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    x = x - np.median(x, axis=1, keepdims=True)
    bp = _bandpass(x, fs)
    energy = (np.diff(bp, axis=1, prepend=bp[:, :1]) ** 2).sum(axis=0)
    win = max(int(round(0.12 * fs)), 1)
    env = np.convolve(energy, np.ones(win) / win, mode="same")
    if not np.any(env > 0):
        return np.array([], dtype=int)
    height = 0.3 * np.percentile(env, 99)
    cand, _ = find_peaks(env, height=height, distance=max(int(refractory_s * fs), 1))
    # refine to the QRS apex: max of summed |band-passed| signal within ±80 ms
    mag = np.abs(bp).sum(axis=0)
    half = max(int(round(0.08 * fs)), 1)
    peaks = []
    for c in cand:
        lo, hi = max(c - half, 0), min(c + half + 1, x.shape[1])
        peaks.append(lo + int(np.argmax(mag[lo:hi])))
    peaks = np.unique(np.asarray(peaks, dtype=int))
    # merge peaks closer than the refractory period after refinement
    if len(peaks) > 1:
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= refractory_s * fs:
                keep.append(p)
            elif mag[p] > mag[keep[-1]]:
                keep[-1] = p
        peaks = np.asarray(keep, dtype=int)
    return peaks


def region_labels(n_samples: int, r_peaks: np.ndarray, fs: float) -> np.ndarray:
    """Label every sample with a REGIONS index (len(REGIONS)-1 = 'other').

    Priority when windows of neighbouring beats overlap (fast heart rate):
    QRS > ST > PR > T > P > other.
    """
    other = len(REGIONS) - 1
    labels = np.full(n_samples, other, dtype=int)
    if len(r_peaks) == 0:
        return labels
    r = np.sort(np.asarray(r_peaks))
    t = np.arange(n_samples)
    pos = np.searchsorted(r, t, side="right")          # index of next R (strictly after t)
    prev_r = np.where(pos > 0, r[np.clip(pos - 1, 0, len(r) - 1)], -10 ** 9)
    next_r = np.where(pos < len(r), r[np.clip(pos, 0, len(r) - 1)], 10 ** 9)
    after = (t - prev_r) * 1000.0 / fs                  # ms since previous R (>=0)
    before = (next_r - t) * 1000.0 / fs                 # ms until next R (>0)
    idx = {name: REGIONS.index(name) for name in REGIONS}
    # assign from lowest to highest priority so later writes win
    labels[(before > 120) & (before <= 250)] = idx["P"]
    labels[(after > 150) & (after <= 400)] = idx["T"]
    labels[(before > 50) & (before <= 120)] = idx["PR"]
    labels[(after > 50) & (after <= 150)] = idx["ST"]
    labels[(after <= 50) | (before <= 50)] = idx["QRS"]
    return labels


def beat_templates(x: np.ndarray, r_peaks: np.ndarray, fs: float, pre_s: float = 0.25, post_s: float = 0.45):
    """Mean beat-aligned window of x [leads, T]; returns (template [leads, W], n_beats, offsets_ms)."""
    pre, post = int(round(pre_s * fs)), int(round(post_s * fs))
    beats = [x[:, r - pre:r + post] for r in r_peaks if r - pre >= 0 and r + post <= x.shape[1]]
    offsets = (np.arange(-pre, post) * 1000.0 / fs)
    if not beats:
        return None, 0, offsets
    return np.mean(beats, axis=0), len(beats), offsets


def heart_rate_bpm(r_peaks: np.ndarray, fs: float) -> float:
    if len(r_peaks) < 2:
        return float("nan")
    return float(60.0 * fs / np.median(np.diff(r_peaks)))
