"""Dashboard analysis checks; run with python -m unittest discover -s tests."""

import io
import json
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from dashboard.analytics import (
    demo_predictions,
    export_report,
    load_predictions,
    metrics,
    prepare_predictions,
    subgroup_metrics,
)


class PredictionValidationTests(unittest.TestCase):
    def test_minimal_csv_has_explicit_unspecified_split(self):
        result = load_predictions(io.StringIO("age,predicted_age\n18,23.5\n89,90\n"))
        self.assertEqual(result["split"].tolist(), ["unspecified", "unspecified"])
        self.assertEqual(result["predicted_age"].tolist(), [23.5, 90.0])

    def test_required_numeric_ages_and_split_validation(self):
        invalid_frames = [
            pd.DataFrame({"age": [40]}),
            pd.DataFrame({"age": [], "predicted_age": []}),
            pd.DataFrame({"age": [17], "predicted_age": [30]}),
            pd.DataFrame({"age": [90], "predicted_age": [30]}),
            pd.DataFrame({"age": [40], "predicted_age": [np.nan]}),
            pd.DataFrame({"age": [40], "predicted_age": [np.inf]}),
            pd.DataFrame({"age": [40], "predicted_age": ["bad"]}),
            pd.DataFrame({"age": [40], "predicted_age": [True]}),
            pd.DataFrame({"age": [None], "predicted_age": [30]}),
            pd.DataFrame({"age": [40], "predicted_age": [30], "split": [None]}),
            pd.DataFrame({"age": [40], "predicted_age": [30], "split": ["val"]}),
        ]
        for frame in invalid_frames:
            with self.subTest(frame=frame.to_dict()), self.assertRaises(ValueError):
                load_predictions(frame)

    def test_preserves_labels_and_ids_and_canonicalizes_split(self):
        result = load_predictions(io.StringIO("ecg_id,patient_id,age,predicted_age,split,NORM,sex\n001,p01,30,34, TEST ,1,0\n"))
        self.assertEqual(result.loc[0, "ecg_id"], "001")
        self.assertEqual(result.loc[0, "patient_id"], "p01")
        self.assertEqual(result.loc[0, "split"], "test")
        self.assertEqual(result.loc[0, "NORM"], 1)
        self.assertEqual(result.loc[0, "sex"], 0)

    def test_rejects_record_and_patient_leakage(self):
        base = pd.DataFrame({"age": [40, 41], "predicted_age": [43, 42], "split": ["validation", "test"]})
        with self.assertRaisesRegex(ValueError, "ecg_id"):
            load_predictions(base.assign(ecg_id=["same", "same"]))
        with self.assertRaisesRegex(ValueError, "patient_id"):
            load_predictions(base.assign(patient_id=["same", "same"]))
        # Repeated measurements from a single patient are legitimate in one split.
        valid = load_predictions(base.assign(split="test", patient_id="same", ecg_id=[1, 2]))
        self.assertEqual(len(valid), 2)
        for column in ["ecg_id", "patient_id"]:
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, column):
                load_predictions(base.assign(**{column: [None, "ok"]}))

    def test_synthetic_column_is_validated_and_normalized(self):
        frame = pd.DataFrame({"age": [40, 50], "predicted_age": [42, 53]})
        for values, expected in [([True, False], [True, False]), ([" FALSE ", "true"], [False, True]), ([0, 1], [False, True])]:
            with self.subTest(values=values):
                result = load_predictions(frame.assign(synthetic=values))
                self.assertEqual(result["synthetic"].tolist(), expected)
                self.assertTrue(result.attrs["synthetic"])
        real = load_predictions(frame.assign(synthetic=False))
        self.assertFalse(real.attrs["synthetic"])
        for values in [["yes", "no"], [None, False], [2, 0], ["", False]]:
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "synthetic"):
                load_predictions(frame.assign(synthetic=values))
        marked = frame.assign(synthetic=False)
        marked.attrs["synthetic"] = True
        self.assertTrue(load_predictions(marked).attrs["synthetic"])


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "age": [20, 40, 60, 70, 80],
            "predicted_age": [32, 54, 76, 100, 90],
            "split": ["validation"] * 3 + ["test"] * 2,
        })

    def test_corrects_validation_bias_and_keeps_raw_errors(self):
        result, calibration = prepare_predictions(self.frame)
        self.assertAlmostEqual(calibration["intercept"], 10)
        self.assertAlmostEqual(calibration["slope"], 0.1)
        self.assertEqual(calibration["n_validation"], 3)
        np.testing.assert_allclose(result["age_gap_corrected"], [0, 0, 0, 13, -8], atol=1e-12)
        np.testing.assert_allclose(result["abs_error"], [12, 14, 16, 30, 10])
        self.assertNotIn("age_gap_raw", self.frame)

    def test_test_outcomes_never_change_calibration(self):
        _, original = prepare_predictions(self.frame)
        changed = self.frame.copy()
        changed.loc[changed["split"].eq("test"), "predicted_age"] = [-1000, 1000]
        _, calibration = prepare_predictions(changed)
        self.assertEqual(original, calibration)
        prepared, _ = prepare_predictions(self.frame)
        filtered = prepared.loc[prepared["age"].ge(60)]
        self.assertAlmostEqual(filtered.loc[3, "age_gap_corrected"], 13)

    def test_missing_small_or_constant_validation_disables_correction(self):
        cases = [self.frame.assign(split="test"), self.frame.iloc[:2], self.frame.assign(age=40)]
        for frame in cases:
            with self.subTest(frame=frame.to_dict()):
                prepared, calibration = prepare_predictions(frame)
                self.assertIsNone(calibration)
                self.assertTrue(prepared["age_gap_corrected"].isna().all())


class MetricTests(unittest.TestCase):
    def test_known_errors(self):
        result = metrics(pd.DataFrame({"age": [20, 30, 40], "predicted_age": [25, 30, 30]}))
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mae"], 5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(125 / 3))
        self.assertAlmostEqual(result["r2"], 0.375)
        self.assertAlmostEqual(result["mean_gap"], -5 / 3)
        self.assertAlmostEqual(result["within_5"], 2 / 3)
        self.assertAlmostEqual(result["pearson_r"], np.sqrt(3) / 2)

    def test_undefined_statistics_remain_undefined(self):
        single = metrics(pd.DataFrame({"age": [40], "predicted_age": [43]}))
        self.assertEqual(single["mae"], 3)
        self.assertIsNone(single["r2"])
        self.assertIsNone(single["pearson_r"])
        constant = metrics(pd.DataFrame({"age": [40, 40], "predicted_age": [43, 41]}))
        self.assertIsNone(constant["r2"])
        self.assertIsNone(constant["pearson_r"])
        constant_prediction = metrics(pd.DataFrame({"age": [40, 50], "predicted_age": [43, 43]}))
        self.assertIsNotNone(constant_prediction["r2"])
        self.assertIsNone(constant_prediction["pearson_r"])
        empty = metrics(pd.DataFrame(columns=["age", "predicted_age"]))
        self.assertEqual(empty["n"], 0)
        self.assertTrue(all(value is None for key, value in empty.items() if key != "n"))

    def test_subgroups_include_missing_labels(self):
        frame = pd.DataFrame({"age": [40, 50, 60], "predicted_age": [42, 48, 67], "sex": [0, 0, None]})
        result = subgroup_metrics(frame, "sex")
        self.assertEqual(result["n"].sum(), 3)
        self.assertEqual(result["mae"].tolist(), [2, 7])
        self.assertTrue(result["mean_gap_corrected"].isna().all())


class ReportTests(unittest.TestCase):
    def test_demo_is_reproducible_and_has_independent_splits(self):
        first = demo_predictions()
        pd.testing.assert_frame_equal(first, demo_predictions())
        self.assertFalse(first.equals(demo_predictions(seed=1)))
        self.assertTrue(first.attrs["synthetic"])
        self.assertEqual(set(first["split"]), {"validation", "test"})
        self.assertTrue(first["ecg_id"].is_unique)
        self.assertTrue(first["patient_id"].is_unique)

    def test_demo_csv_roundtrip_retains_synthetic_provenance(self):
        original, _ = prepare_predictions(demo_predictions())
        with ZipFile(io.BytesIO(export_report(original, {"source": "demo"}))) as archive:
            reloaded = load_predictions(io.BytesIO(archive.read("predictions.csv")))
        self.assertTrue(reloaded.attrs["synthetic"])
        self.assertTrue(reloaded["synthetic"].all())
        prepared, _ = prepare_predictions(reloaded)
        with ZipFile(io.BytesIO(export_report(prepared.head(2), {"synthetic": False, "source": "upload"}))) as archive:
            report = json.loads(archive.read("metrics.json"))
            self.assertTrue(report["metadata"]["synthetic"])
            self.assertIn("SZINTETIKUS DEMÓADATOK", archive.read("summary.html").decode("utf-8"))

    def test_export_filters_provenance_and_escaped_html(self):
        prepared, calibration = prepare_predictions(demo_predictions())
        selected = prepared.loc[prepared["split"].eq("test")].head(1)
        metadata = {"synthetic": False, "source": '<script>alert("x")</script>', "filters": {"split": "test", "top_n": 1}, "calibration": calibration, "unavailable": np.nan}
        data = export_report(selected, metadata)
        with ZipFile(io.BytesIO(data)) as archive:
            self.assertEqual(set(archive.namelist()), {"predictions.csv", "metrics.json", "summary.html"})
            exported = pd.read_csv(archive.open("predictions.csv"))
            self.assertEqual(len(exported), 1)
            self.assertEqual(exported.loc[0, "ecg_id"], selected.iloc[0]["ecg_id"])
            report = json.loads(archive.read("metrics.json"), parse_constant=lambda value: self.fail(f"Invalid JSON constant: {value}"))
            self.assertEqual(report["metrics"]["n"], 1)
            self.assertIsNone(report["metrics"]["r2"])
            self.assertTrue(report["metadata"]["synthetic"])
            self.assertEqual(report["metadata"]["filters"], metadata["filters"])
            self.assertEqual(report["metadata"]["calibration"], calibration)
            self.assertIsNone(report["metadata"]["unavailable"])
            page = archive.read("summary.html").decode("utf-8")
            self.assertIn("SZINTETIKUS DEMÓADATOK", page)
            self.assertNotIn("<script>", page)
            self.assertIn("&lt;script&gt;", page)
        self.assertFalse(metadata["synthetic"], "Caller metadata must not be mutated")

    def test_empty_filtered_export_uses_null_metrics(self):
        empty = pd.DataFrame(columns=["age", "predicted_age"])
        with ZipFile(io.BytesIO(export_report(empty, {"synthetic": False}))) as archive:
            report = json.loads(archive.read("metrics.json"))
            self.assertEqual(report["metrics"]["n"], 0)
            self.assertIsNone(report["metrics"]["mae"])


if __name__ == "__main__":
    unittest.main()
