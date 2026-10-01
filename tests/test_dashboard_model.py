"""Synthetic test records and random test-only weights; never research outputs."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
wfdb = pytest.importorskip("wfdb")

from dashboard.model_service import LEADS, inspect_dataset, inspect_record, load_model, predict_record, read_record_signal
from ecg_age_common import ECGAgeResNet
from predict_ecg_age import evaluate_checkpoint, select_subset


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("synthetic-model") / "test_only_random_weights.pt"
    torch.manual_seed(42)
    torch.save({"model_state": ECGAgeResNet().state_dict(), "sampling_rate": 100, "epoch": 1}, path)
    return path


@pytest.fixture
def dataset(tmp_path):
    records = tmp_path / "records100"
    records.mkdir()
    rows = []
    # Reverse lead order to exercise real WFDB reorder handling.
    names = LEADS[::-1]
    for ecg_id, (age, fold) in enumerate([(45, 1), (51, 8), (22, 9), (38, 9), (57, 9), (80, 9),
                                          (24, 10), (44, 10), (62, 10), (85, 10), (16, 10)], 1):
        signal = np.tile(np.arange(1, 13, dtype=float) / 2, (64, 1))
        # A small time variation keeps WFDB quantization and IG non-degenerate.
        signal += np.sin(np.arange(64)[:, None] / 5) * 0.02
        name = f"record_{ecg_id}"
        wfdb.wrsamp(name, fs=100, units=["mV"] * 12, sig_name=names, p_signal=signal,
                    fmt=["16"] * 12, write_dir=str(records))
        rows.append({"ecg_id": ecg_id, "age": age, "strat_fold": fold, "patient_id": ecg_id,
                     "filename_lr": f"records100/{name}", "sex": ecg_id % 2})
    pd.DataFrame(rows).to_csv(tmp_path / "ptbxl_database.csv", index=False)
    return tmp_path


def update_metadata(dataset, change):
    path = dataset / "ptbxl_database.csv"
    frame = pd.read_csv(path)
    change(frame)
    frame.to_csv(path, index=False)


def test_dataset_filters_training_age_range_and_checks_split(dataset):
    info = inspect_dataset(dataset)
    assert (info["n_train"], info["n_validation"], info["n_test"], info["total"]) == (2, 4, 4, 10)
    assert info["n_excluded_age"] == 1
    assert info["record_ids"] == list(range(1, 11))
    assert info["leakage"]["passed"]


def test_patient_leakage_rejected(dataset):
    update_metadata(dataset, lambda frame: frame.loc.__setitem__((2, "patient_id"), 1))
    with pytest.raises(ValueError, match="adatszivárgás"):
        inspect_dataset(dataset)


@pytest.mark.parametrize("column,value,error", [("strat_fold", 0, "strat_fold"),
                                                 ("strat_fold", 11, "strat_fold"),
                                                 ("age", float("inf"), "életkor"),
                                                 ("ecg_id", 2, "Ismétlődő")])
def test_invalid_metadata_rejected(dataset, column, value, error):
    def corrupt(frame):
        frame[column] = frame[column].astype(object)
        frame.loc[0, column] = value
    update_metadata(dataset, corrupt)
    with pytest.raises(ValueError, match=error):
        inspect_dataset(dataset)


def test_single_prediction_reorders_and_clips_real_wfdb(dataset, checkpoint):
    prediction = predict_record(dataset, checkpoint, 3)
    assert prediction["ecg_id"] == 3
    assert prediction["split"] == "validation"
    assert prediction["sampling_rate"] == 100
    assert prediction["signal"].shape == (12, 64)
    assert prediction["leads"] == LEADS
    # I is the last channel on disk (6 mV), V6 is the first (0.5 mV).
    assert prediction["signal"][0, 0] == pytest.approx(1.0)
    assert prediction["signal"][-1, 0] == pytest.approx(0.1, abs=0.001)
    assert np.isfinite(prediction["predicted_age"])
    assert prediction["age_gap_raw"] == pytest.approx(prediction["predicted_age"] - 22)


def test_inspect_record_without_checkpoint_or_torch(dataset, monkeypatch):
    def no_torch():
        raise AssertionError("Waveform inspection must not use Torch")
    monkeypatch.setattr("dashboard.model_service._torch", no_torch)
    result = inspect_record(dataset, 3)
    assert result["ecg_id"] == 3
    assert result["age"] == 22
    assert result["sampling_rate"] == 100
    assert result["split"] == "validation"
    assert result["sex"] == 1
    assert result["leads"] == LEADS
    assert result["signal"].shape == (12, 64)
    assert result["signal"][0, 0] == pytest.approx(1.0)
    assert result["signal"][-1, 0] == pytest.approx(0.1, abs=0.001)
    assert not {"predicted_age", "age_gap_raw", "checkpoint", "attribution"}.intersection(result)


@pytest.mark.parametrize("rate", [0, 250, True, "100"])
def test_inspect_record_rejects_invalid_sampling_rate(dataset, rate):
    with pytest.raises(ValueError, match="mintavétel"):
        inspect_record(dataset, 3, sampling_rate=rate)


@pytest.mark.parametrize("failure,error", [("rate", "mintavétel"), ("leads", "elvezetés"),
                                          ("nonfinite", "nem véges"), ("units", "mértékegysége")])
def test_signal_validation_rejects_incompatible_input(dataset, monkeypatch, failure, error):
    signal = np.zeros((64, 12), dtype=np.float32)
    fields = {"sig_name": LEADS, "fs": 100, "units": ["mV"] * 12}
    if failure == "rate":
        fields["fs"] = 500
    elif failure == "leads":
        fields["sig_name"] = ["I"] * 12
    elif failure == "nonfinite":
        signal[0, 0] = np.nan
    elif failure == "units":
        fields["units"] = ["V"] * 12
    monkeypatch.setattr(wfdb, "rdsamp", lambda path: (signal, fields))
    with pytest.raises(ValueError, match=error):
        read_record_signal(dataset, pd.Series({"filename_lr": "records100/record_1"}), 100)


def test_missing_and_incompatible_checkpoints_never_predict(tmp_path, checkpoint):
    with pytest.raises(ValueError, match="Nem található"):
        load_model(tmp_path / "missing.pt")
    bad = tmp_path / "bad.pt"
    torch.save({"model_state": {}, "sampling_rate": 100}, bad)
    with pytest.raises(ValueError, match="nem kompatibilis"):
        load_model(bad)
    state = torch.load(checkpoint, weights_only=True)
    state.pop("sampling_rate")
    torch.save(state, bad)
    with pytest.raises(ValueError, match="sampling_rate"):
        load_model(bad)


def test_explanation_returns_finite_aligned_attributions(dataset, checkpoint):
    pytest.importorskip("captum")
    result = predict_record(dataset, checkpoint, 3, explain=True, steps=8)
    assert result["attribution"].shape == result["signal"].shape
    assert result["lead_importance"].shape == (12,)
    assert np.isfinite(result["attribution"]).all()
    assert np.isfinite(result["convergence_delta"])


def test_age_stratified_limit_is_reproducible():
    frame = pd.DataFrame({"age": [22] * 60 + [45] * 30 + [82] * 10}, index=range(100))
    selected = select_subset(frame, 10)
    assert len(selected) == 10
    assert selected.groupby("age").size().to_dict() == {22: 6, 45: 3, 82: 1}
    assert selected.equals(select_subset(frame, 10))


def test_cli_produces_both_splits_and_subset_provenance(dataset, checkpoint, tmp_path):
    output = tmp_path / "evaluation"
    script = Path(__file__).resolve().parents[1] / "predict_ecg_age.py"
    result = subprocess.run([sys.executable, str(script), "--data-dir", str(dataset),
                             "--checkpoint", str(checkpoint), "--output-dir", str(output),
                             "--batch-size", "2", "--device", "cpu", "--limit", "2"],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    predictions = pd.read_csv(output / "predictions.csv")
    assert predictions.groupby("split").size().to_dict() == {"test": 2, "validation": 2}
    assert not set(predictions.ecg_id).intersection({1, 2, 11})
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "subset"
    assert metrics["subset"] is True
    assert metrics["provenance"]["training_performed"] is False
    assert metrics["provenance"]["available_test"] == 4
    assert len(metrics["provenance"]["checkpoint_sha256"]) == 64
    assert metrics["validation"]["mae"] == pytest.approx(
        np.abs(predictions.query("split == 'validation'").predicted_age - predictions.query("split == 'validation'").age).mean())


def test_full_evaluation_matches_single_record_prediction(dataset, checkpoint, tmp_path):
    output = tmp_path / "full"
    metrics = evaluate_checkpoint(dataset, checkpoint, output, batch_size=4)
    assert metrics["status"] == "full"
    assert metrics["subset"] is False
    assert (metrics["n_validation"], metrics["n_test"]) == (4, 4)
    predictions = pd.read_csv(output / "predictions.csv")
    single = predict_record(dataset, checkpoint, 3)
    assert predictions.set_index("ecg_id").loc[3, "predicted_age"] == pytest.approx(single["predicted_age"], abs=1e-6)


def test_evaluation_requires_both_held_out_splits(dataset, checkpoint, tmp_path):
    update_metadata(dataset, lambda frame: frame.loc.__setitem__((frame.strat_fold.eq(9), "strat_fold"), 8))
    with pytest.raises(ValueError, match="Validációs"):
        evaluate_checkpoint(dataset, checkpoint, tmp_path / "missing-validation")
    assert not (tmp_path / "missing-validation").exists()
