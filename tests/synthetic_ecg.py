"""Simulated 12-lead ECG for tests only (Gaussian P/QRS/T waves)."""
from __future__ import annotations

import numpy as np


def synth_ecg(fs=100, dur=10, hr=70, seed=0, noise=0.02, amp=1.0):
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    t = np.arange(n) / fs
    r_times = np.arange(0.4, dur - 0.2, 60 / hr)
    x = np.zeros((12, n))
    gains = rng.uniform(0.5, 1.5, 12) * np.where(rng.random(12) < 0.2, -1, 1)
    for rt in r_times:
        for off, w, a in [(-0.17, 0.025, 0.15), (-0.02, 0.008, -0.1), (0, 0.012, amp),
                          (0.025, 0.01, -0.25), (0.28, 0.05, 0.3)]:
            x += gains[:, None] * a * np.exp(-0.5 * ((t - rt - off) / w) ** 2)[None, :]
    x += 0.1 * np.sin(2 * np.pi * 0.2 * t)[None, :] + rng.normal(0, noise, x.shape)
    return x.astype(np.float32), np.round(r_times * fs).astype(int)
