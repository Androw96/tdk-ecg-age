"""Exercise rendered user flows without needing a private dataset."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboard.analytics import demo_predictions, metrics

APP = Path(__file__).resolve().parents[1] / "dashboard_app.py"


def app():
    return AppTest.from_file(str(APP), default_timeout=30).run()


@pytest.mark.parametrize("mode", ["Bemutató", "Saját adatok"])
def test_all_pages_and_empty_states(mode):
    view = app()
    view.radio(key="data_mode").set_value(mode).run()
    for page in ["EKG-vizsgálat", "Kimutatások", "Modelltesztek", "Kísérletek"]:
        view.radio(key="page").set_value(page).run()
        assert not view.exception, [e.message for e in view.exception]
        assert not view.error


def test_demo_filters_recompute_metrics_and_handle_empty_group():
    view = app()
    view.radio(key="page").set_value("Kimutatások").run()
    expected = metrics(demo_predictions().query("split == 'test'"))
    assert view.metric[0].value == str(expected["n"])
    assert view.metric[1].value == f"{expected['mae']:.2f} év"
    next(s for s in view.slider if s.label == "Életkor (év)").set_value((30, 39)).run()
    expected = metrics(demo_predictions().query("split == 'test' and 30 <= age <= 39"))
    assert view.metric[0].value == str(expected["n"])
    assert not view.exception
    next(s for s in view.selectbox if s.label == "Nem (PTB-XL kód)").set_value("Hiányzó").run()
    assert any("nincs megjeleníthető" in w.value for w in view.warning)
    assert not view.exception


def test_real_record_missing_configuration_is_actionable():
    view = app()
    view.radio(key="data_mode").set_value("Saját adatok").run()
    next(b for b in view.button if b.label == "EKG vizsgálata").click().run()
    assert not view.exception
    assert any("Add meg az adatbázis" in e.value for e in view.error)
    assert len(view.metric) == 0


def test_local_csv_demo_provenance_survives_import(tmp_path):
    path = tmp_path / "predictions.csv"
    demo_predictions().to_csv(path, index=False)
    view = app()
    view.radio(key="data_mode").set_value("Saját adatok").run()
    view.radio(key="page").set_value("Kimutatások").run()
    view.text_input(key="predictions_path").set_value(str(path)).run()
    assert not view.exception
    assert any("szintetikus bemutatóadatokat" in w.value for w in view.warning)
    assert len(view.metric) == 5
    path.write_text("age,predicted_age\nnot-an-age,55\n")
    view.run()
    assert not view.exception
    assert any("nem dolgozható fel" in e.value for e in view.error)
    assert len(view.metric) == 0


def test_demo_cannot_start_real_jobs():
    view = app()
    for page, labels in [("Modelltesztek", ["Adatok és modell ellenőrzése", "Modellteszt indítása"]),
                         ("Kísérletek", ["Tanítás indítása", "Elemzés indítása"])]:
        view.radio(key="page").set_value(page).run()
        for label in labels:
            assert next(b for b in view.button if b.label == label).disabled


def test_record_view_handles_no_leads_selected():
    view = app()
    view.multiselect[0].set_value([]).run()
    assert not view.exception
    assert any("legalább egy elvezetést" in info.value for info in view.info)


def test_metadata_enrichment_checks_labels_patient_and_split(tmp_path):
    metadata = pd.DataFrame({"ecg_id": [1, 2, 3, 4], "patient_id": [101, 102, 103, 104],
                             "age": [30, 40, 50, 60], "strat_fold": [9, 9, 9, 10], "sex": [0, 1, 0, 1]})
    metadata.to_csv(tmp_path / "ptbxl_database.csv", index=False)
    predictions = pd.DataFrame({"ecg_id": [1, 2, 3, 4], "age": [30, 40, 50, 60],
                                "predicted_age": [35, 43, 51, 58], "split": ["validation"]*3 + ["test"]})
    for label in ["NORM", "MI", "STTC", "CD", "HYP"]:
        predictions[label] = 1 if label == "NORM" else 0
    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    view = app()
    view.radio(key="data_mode").set_value("Saját adatok").run()
    view.text_input(key="data_dir").set_value(str(tmp_path)).run()
    view.radio(key="page").set_value("Kimutatások").run()
    view.text_input(key="predictions_path").set_value(str(path)).run()
    assert not view.exception
    assert not view.error
    assert "patient_id" in view.dataframe[0].value.columns
    assert view.metric[0].value == "1"
    predictions.loc[3, "split"] = "validation"
    predictions.to_csv(path, index=False)
    view.run()
    assert not view.exception
    assert any("hivatalos felosztásától" in e.value for e in view.error)
