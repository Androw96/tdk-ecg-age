"""End-to-end smoke test of the research scripts on SYNTHETIC PTB-XL-like data.

Nothing here is a research result: signals are simulated, models are tiny and
trained for two epochs. The test only proves that every script runs and that
old checkpoints stay compatible.

    .venv/bin/python -m pytest tests/test_research_pipeline.py -q
"""
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

from synthetic_ecg import synth_ecg  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]


def test_holm_and_corrections():
    from evaluate_agegap import Correction, holm

    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(0)
    age = rng.uniform(18, 89, 3000)
    pred = 0.7 * age + 0.3 * 55 + rng.normal(0, 5, len(age))
    frame = pd.DataFrame({"age": age, "age_gap_raw": pred - age})
    for method in ("linear", "quadratic", "two_step"):
        gap = Correction(method).fit(frame).transform(frame)
        assert abs(np.corrcoef(age, gap)[0, 1]) < 0.05




def test_default_model_is_the_original_baseline():
    from ecg_age_common import ECGAgeResNet, count_parameters

    model = ECGAgeResNet()
    assert count_parameters(model) == 7_345_281
    keys = set(model.state_dict())
    for key in ("stem.0.weight", "features.0.conv1.weight", "features.2.skip.0.weight",
                "features.7.bn2.running_var", "head.2.weight", "head.2.bias"):
        assert key in keys
    assert len([k for k in keys if k.startswith("features.")]) > 0
    assert max(int(k.split(".")[1]) for k in keys if k.startswith("features.")) == 7


def test_old_checkpoint_still_loads(tmp_path):
    from dashboard.model_service import load_model
    from ecg_age_common import ECGAgeResNet, load_checkpoint

    path = tmp_path / "old.pt"
    torch.save({"model_state": ECGAgeResNet().state_dict(), "sampling_rate": 100, "epoch": 1}, path)
    model, ckpt = load_checkpoint(path)
    assert ckpt["preprocessing"]["mode"] == "clip5"
    model2, info = load_model(path, "cpu")
    assert info["preprocessing"]["mode"] == "clip5"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_ptbxl")
    (root / "records100").mkdir()
    pd.DataFrame({"diagnostic": [1, 1, 1, 1, 1, np.nan],
                  "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", np.nan]},
                 index=pd.Index(["NORM", "IMI", "NDT", "LAFB", "LVH", "SR"], name="")).to_csv(root / "scp_statements.csv")
    rng = np.random.default_rng(7)
    rows = []
    for ecg_id in range(1, 181):
        age = int(rng.integers(18, 90))
        fold = (ecg_id % 10) + 1
        hr = float(rng.uniform(50, 100))
        x, _ = synth_ecg(hr=hr, seed=ecg_id, amp=1.4 - age / 150)
        codes = {"SR": 0}
        if rng.random() < 0.55:
            codes["NORM"] = 100
        else:
            codes[rng.choice(["IMI", "NDT", "LAFB", "LVH"])] = 100
        name = f"{ecg_id:05d}_lr"
        wfdb.wrsamp(name, fs=100, units=["mV"] * 12, sig_name=["I", "II", "III", "aVR", "aVL", "aVF",
                                                               "V1", "V2", "V3", "V4", "V5", "V6"],
                    p_signal=x.T.astype(float), fmt=["16"] * 12, write_dir=str(root / "records100"))
        rows.append({"ecg_id": ecg_id, "patient_id": float(ecg_id), "age": age, "sex": ecg_id % 2,
                     "strat_fold": fold, "scp_codes": str(codes), "filename_lr": f"records100/{name}"})
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    return root


def run(*args):
    result = subprocess.run([sys.executable, *map(str, args)], cwd=PROJECT, capture_output=True, text=True,
                            env={**__import__("os").environ, "MPLBACKEND": "Agg"})
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    return result.stdout


@pytest.fixture(scope="module")
def trained_run(dataset, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    run("train_ecg_age.py", "--data-dir", dataset, "--output-dir", out, "--epochs", 2, "--batch-size", 8,
        "--widths", "8,16", "--blocks", "1,1", "--train-cohort", "norm", "--augment", "--crop-seconds", 8,
        "--preprocessing", "lead_standardize", "--scheduler", "cosine", "--cache-dir", out / "cache")
    return out


def test_training_outputs(trained_run):
    for name in ("best_model.pt", "metrics.json", "predictions.csv", "training_history.csv", "config.json"):
        assert (trained_run / name).is_file(), name
    metrics = json.loads((trained_run / "metrics.json").read_text())
    assert metrics["train_cohort"] == "norm"
    assert "train_mean" in metrics["constant_baseline_test"]
    pred = pd.read_csv(trained_run / "predictions.csv")
    assert {"in_training_cohort", "strict_norm", "NORM"} <= set(pred.columns)
    assert set(pred.split) == {"validation", "test"}
    assert (trained_run / "cache" / "ptbxl_100hz.npy").is_file()


def test_dashboard_evaluation_of_new_checkpoint(dataset, trained_run, tmp_path):
    run("predict_ecg_age.py", "--data-dir", dataset, "--checkpoint", trained_run / "best_model.pt",
        "--output-dir", tmp_path, "--batch-size", 4, "--device", "cpu", "--limit", 5)
    assert (tmp_path / "metrics.json").is_file()


def test_analysis_xai_and_ensemble(dataset, trained_run, tmp_path):
    analysis = trained_run / "analysis"
    run("evaluate_agegap.py", "--data-dir", dataset, "--predictions", trained_run / "predictions.csv",
        "--output-dir", analysis, "--n-boot", 50)
    for name in ("results_summary.md", "prediction_metrics.csv", "population_table.csv",
                 "test_agegap_with_labels.csv", "logistic_regression.json"):
        assert (analysis / name).is_file(), name
    summary = json.loads((analysis / "agegap_summary.json").read_text())
    assert summary["bias_correction"]["fitted_on"].startswith("validation (training")

    xai = tmp_path / "xai"
    run("xai_population.py", "--data-dir", dataset, "--checkpoint", trained_run / "best_model.pt",
        "--agegap-csv", analysis / "test_agegap_with_labels.csv", "--output-dir", xai,
        "--n-per-group", 2, "--steps", 4, "--internal-batch", 4, "--device", "cpu", "--sanity-random",
        "--cache-dir", trained_run / "cache")
    per = pd.read_csv(xai / "per_record_attribution.csv")
    assert len(per) > 0 and (per.n_beats > 0).all()
    lead_cols = [c for c in per.columns if c.startswith("lead_")]
    assert np.allclose(per[lead_cols].sum(axis=1), 1.0, atol=1e-5)

    run("explain_integrated_gradients.py", "--data-dir", dataset, "--checkpoint", trained_run / "best_model.pt",
        "--ecg-id", int(per.ecg_id.iloc[0]), "--output-dir", tmp_path / "xai_one", "--steps", 4)

    ens = tmp_path / "ens"
    run("ensemble_runs.py", "--runs", trained_run, trained_run, "--output-dir", ens)
    assert (ens / "predictions.csv").is_file()
