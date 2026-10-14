"""R-peak detection / region labelling on simulated ECG (numpy + scipy only)."""
from __future__ import annotations

import numpy as np
import pytest

from ecg_beats import REGIONS, detect_r_peaks, heart_rate_bpm, region_labels
from synthetic_ecg import synth_ecg


@pytest.mark.parametrize("fs,hr", [(100, 50), (100, 120), (500, 75)])
def test_r_peak_detection_on_synthetic_ecg(fs, hr):
    x, true = synth_ecg(fs=fs, hr=hr, seed=3)
    found = detect_r_peaks(x, fs)
    assert len(found) == len(true)
    assert np.max(np.abs(found - true)) * 1000 / fs <= 30
    assert abs(heart_rate_bpm(found, fs) - hr) < 3


def test_region_labels_layout():
    labels = region_labels(300, np.array([100, 200]), 100)
    names = [REGIONS[i] for i in labels]
    assert names[100] == "QRS" and names[110] == "ST" and names[125] == "T"
    assert names[90] == "PR" and names[80] == "P" and names[190] == "PR" and names[130] == "T"
    assert names[250] == "other" and names[0] == "other"
