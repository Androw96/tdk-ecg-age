"""The hosted UI profiles (browser / cloud) on SYNTHETIC bundles – never research outputs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.analytics import demo_predictions

APP = Path(__file__).resolve().parents[1] / "dashboard_app.py"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def sidebar_pages(view):
    return list(view.radio(key="page").options)


@pytest.mark.parametrize("profile", ["browser", "cloud"])
def test_hosted_profiles_without_bundle_offer_demo_only(profile, monkeypatch, tmp_path):
    monkeypatch.setenv("ECG_AGE_PROFILE", profile)
    monkeypatch.setenv("ECG_AGE_DEMO_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("ECG_AGE_WEB_DIR", str(tmp_path / "none"))
    view = app()
    assert sidebar_pages(view) == ["EKG-vizsgálat", "Kimutatások"]
    assert list(view.radio(key="data_mode").options) == ["Bemutató"]
    assert not any(t.label == "PTB-XL adatbázis mappája" for t in view.text_input)
    for page in ["EKG-vizsgálat", "Kimutatások"]:
        view.radio(key="page").set_value(page).run()
        assert not view.exception, [e.message for e in view.exception]


def test_local_profile_keeps_all_pages(monkeypatch):
    monkeypatch.delenv("ECG_AGE_PROFILE", raising=False)
    view = app()
    assert sidebar_pages(view) == ["EKG-vizsgálat", "Kimutatások", "Modelltesztek", "Kísérletek"]


@pytest.fixture
def web_bundle(tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    rng = np.random.default_rng(0)
    ids = [101, 102, 103]
    signals = rng.normal(0, 0.2, (3, 12, 1000)).astype(np.float16)
    np.savez_compressed(web / "examples.npz", signals=signals, attributions=(signals * 0.01).astype(np.float16))
    records = [{"ecg_id": i, "age": 40.0 + k * 10, "predicted_age": 45.0 + k * 9, "split": "test", "sex": k % 2,
                "ig_convergence_delta": 0.001, "NORM": 1, "MI": 0, "STTC": 0, "CD": 0, "HYP": 0}
               for k, i in enumerate(ids)]
    (web / "examples.json").write_text(json.dumps({"sampling_rate": 100, "best_epoch": 12, "ig_steps": 32,
                                                   "records": records}), encoding="utf-8")
    preds = demo_predictions().drop(columns=["synthetic"], errors="ignore")
    preds.to_csv(web / "predictions.csv", index=False)
    return web


def test_browser_profile_with_precomputed_examples(web_bundle, monkeypatch):
    monkeypatch.setenv("ECG_AGE_PROFILE", "browser")
    monkeypatch.setenv("ECG_AGE_WEB_DIR", str(web_bundle))
    view = app()
    view.radio(key="data_mode").set_value("Valós példák").run()
    assert not view.exception, [e.message for e in view.exception]
    assert view.metric[0].value == "40 év"
    assert view.metric[1].value == "45.00 év"
    next(c for c in view.checkbox if c.label == "Integrated Gradients").check().run()
    assert not view.exception, [e.message for e in view.exception]
    assert any("Mire figyelt" in h.value for h in view.subheader)
    view.radio(key="page").set_value("Kimutatások").run()
    assert not view.exception, [e.message for e in view.exception]
    assert len(view.metric) == 5


def test_cloud_profile_runs_live_model_on_bundle(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    wfdb = pytest.importorskip("wfdb")
    from ecg_age_common import ECGAgeResNet

    demo = tmp_path / "demo_data"
    ptb = demo / "ptbxl"
    (ptb / "records100").mkdir(parents=True)
    rows = []
    for ecg_id, (age, fold) in enumerate([(45, 1), (52, 9), (61, 10), (70, 10)], 1):
        t = np.arange(1000) / 100
        signal = np.stack([np.sin(2 * np.pi * 1.2 * t + i) * 0.5 for i in range(12)], axis=1)
        wfdb.wrsamp(f"r{ecg_id}", fs=100, units=["mV"] * 12, sig_name=LEADS, p_signal=signal,
                    fmt=["16"] * 12, write_dir=str(ptb / "records100"))
        rows.append({"ecg_id": ecg_id, "patient_id": ecg_id, "age": age, "sex": ecg_id % 2, "strat_fold": fold,
                     "scp_codes": "{'NORM': 100}", "filename_lr": f"records100/r{ecg_id}"})
    pd.DataFrame(rows).to_csv(ptb / "ptbxl_database.csv", index=False)
    pd.DataFrame({"diagnostic": [1], "diagnostic_class": ["NORM"]}, index=pd.Index(["NORM"])).to_csv(
        ptb / "scp_statements.csv")
    torch.manual_seed(0)
    torch.save({"model_state": ECGAgeResNet().state_dict(), "sampling_rate": 100, "epoch": 1}, demo / "best_model.pt")
    pd.DataFrame({"ecg_id": [3, 4], "age": [61, 70], "predicted_age": [60.0, 66.0]}).to_csv(
        demo / "examples.csv", index=False)
    pd.DataFrame({"ecg_id": [2, 3, 4], "age": [52, 61, 70], "predicted_age": [50.0, 60.0, 66.0],
                  "split": ["validation", "test", "test"]}).to_csv(demo / "predictions.csv", index=False)

    monkeypatch.setenv("ECG_AGE_PROFILE", "cloud")
    monkeypatch.setenv("ECG_AGE_DEMO_DIR", str(demo))
    view = app()
    view.radio(key="data_mode").set_value("Valós példák").run()
    next(b for b in view.button if b.label == "EKG vizsgálata").click().run()
    assert not view.exception, [e.message for e in view.exception]
    assert not view.error, [e.value for e in view.error]
    assert view.metric[0].value == "61 év"
    view.radio(key="page").set_value("Kimutatások").run()
    assert not view.exception, [e.message for e in view.exception]
