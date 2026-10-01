"""Pure analysis helpers for the ECG age dashboard.

The training pipeline is deliberately not imported here: uploaded predictions can
be explored without PyTorch or access to ECG recordings. Correction follows the
project's ``evaluate_agegap.py``: regress the raw gap on age using validation only.
"""

from __future__ import annotations

import html
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test", "unspecified")
SUPERCLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
METRIC_COLUMNS = ("n", "mae", "rmse", "r2", "pearson_r", "mean_gap", "within_5")


def _validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return an independent, normalized frame or raise an actionable ValueError."""
    if not frame.columns.is_unique:
        raise ValueError("A CSV azonos nevű oszlopokat tartalmaz.")
    missing = {"age", "predicted_age"}.difference(frame.columns)
    if missing:
        raise ValueError("Hiányzó kötelező oszlop: " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("A predikciós táblázat nem tartalmaz adatot.")
    result = frame.copy(deep=True)
    for column in ("age", "predicted_age"):
        if pd.api.types.is_bool_dtype(result[column]):
            raise ValueError(f"A(z) {column} oszlop numerikus életkorokat vár, nem logikai értékeket.")
        values = pd.to_numeric(result[column], errors="coerce").astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"A(z) {column} oszlop hiányzó, nem numerikus vagy végtelen értéket tartalmaz.")
        result[column] = values
    if not result["age"].between(18, 89, inclusive="both").all():
        raise ValueError("Az age oszlopban csak 18–89 év közötti életkor szerepelhet.")

    if "split" not in result:
        result["split"] = "unspecified"
    else:
        split = result["split"].astype("string").str.strip().str.lower()
        if split.isna().any() or not split.isin(SPLITS).all():
            raise ValueError("A split oszlop megengedett értékei: train, validation, test, unspecified; hiányzó érték nem használható.")
        result["split"] = split.astype(str)

    if "synthetic" in result:
        # Persist provenance in CSV as well as attrs: attrs alone disappear on
        # export, allowing a re-uploaded demo to look like measured output.
        synthetic = result["synthetic"].astype("string").str.strip().str.lower()
        allowed = {"true": True, "false": False, "1": True, "0": False}
        if synthetic.isna().any() or not synthetic.isin(allowed).all():
            raise ValueError("A synthetic oszlop csak true/false vagy 1/0 értéket tartalmazhat, hiányzó érték nélkül.")
        result["synthetic"] = synthetic.map(allowed).astype(bool)
        result.attrs["synthetic"] = bool(result["synthetic"].any() or result.attrs.get("synthetic", False))

    for column in ("ecg_id", "patient_id"):
        if column not in result:
            continue
        identifiers = result[column].astype("string").str.strip()
        if identifiers.isna().any() or identifiers.eq("").any():
            raise ValueError(f"A(z) {column} oszlop hiányzó azonosítót tartalmaz.")
        result[column] = identifiers
    if "ecg_id" in result and result["ecg_id"].duplicated().any():
        raise ValueError("Ismétlődő ecg_id: egy EKG csak egyszer szerepelhet, külön splitben sem ismételhető.")
    if "patient_id" in result:
        split_counts = result.groupby("patient_id", observed=True)["split"].nunique()
        if (split_counts > 1).any():
            raise ValueError("Betegszintű adatszivárgás: ugyanaz a patient_id több splitben szerepel.")
    return result


def load_predictions(source: str | Path | BinaryIO | TextIO | pd.DataFrame) -> pd.DataFrame:
    """Read UTF-8 comma-separated model output and validate independent splits.

    ``age`` and ``predicted_age`` are required; optional identifiers, diagnostic
    labels and other columns are preserved. Missing split *column* is allowed,
    but an existing split column cannot contain missing or unknown values.
    DataFrames are accepted for generated data and programmatic callers.
    """
    if isinstance(source, pd.DataFrame):
        return _validate_predictions(source)
    try:
        frame = pd.read_csv(source, dtype={"ecg_id": "string", "patient_id": "string"})
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("A fájl nem olvasható UTF-8, vesszővel elválasztott CSV-ként.") from exc
    return _validate_predictions(frame)


def prepare_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Compute raw errors and fit one correction on the complete validation set.

    Call this before applying user filters. Test/train records never participate
    in fitting. A correction requires three validation records and two ages.
    Imported gap columns are recomputed, so they cannot silently supply a
    correction fitted on an unknown population.
    """
    result = _validate_predictions(frame)
    result["age_gap_raw"] = result["predicted_age"] - result["age"]
    result["abs_error"] = result["age_gap_raw"].abs()
    result["age_gap_corrected"] = np.nan
    validation = result.loc[result["split"].eq("validation")]
    calibration = None
    if len(validation) >= 3 and validation["age"].nunique() >= 2:
        ages = validation["age"].to_numpy(dtype=float)
        gaps = validation["age_gap_raw"].to_numpy(dtype=float)
        centered = ages - ages.mean()
        slope = float(np.dot(centered, gaps - gaps.mean()) / np.dot(centered, centered))
        intercept = float(gaps.mean() - slope * ages.mean())
        if np.isfinite([intercept, slope]).all():
            calibration = {"intercept": intercept, "slope": slope, "n_validation": int(len(validation))}
            result["age_gap_corrected"] = result["age_gap_raw"] - (intercept + slope * result["age"])
    return result, calibration


def _finite_or_none(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def metrics(frame: pd.DataFrame) -> dict[str, int | float | None]:
    """Raw age-prediction metrics; within_5 is a fraction between zero and one.

    Empty groups and undefined R²/Pearson correlation return None, never NaN.
    Corrected AgeGap is not an independently predicted age and is intentionally
    excluded from prediction-accuracy metrics.
    """
    n = len(frame)
    result: dict[str, int | float | None] = {key: None for key in METRIC_COLUMNS}
    result["n"] = n
    if not n:
        return result
    ages = frame["age"].to_numpy(dtype=float)
    predictions = frame["predicted_age"].to_numpy(dtype=float)
    if not np.isfinite(ages).all() or not np.isfinite(predictions).all():
        raise ValueError("A metrikák számításához véges age és predicted_age értékek szükségesek.")
    gaps = predictions - ages
    absolute = np.abs(gaps)
    # Scaling keeps RMSE finite even for valid but extreme numeric predictions.
    scale = float(absolute.max())
    rmse = float(scale * np.sqrt(np.mean((gaps / scale) ** 2))) if scale else 0.0
    mae = float(scale * np.mean(absolute / scale)) if scale else 0.0
    mean_gap = float(scale * np.mean(gaps / scale)) if scale else 0.0
    result.update(mae=mae, rmse=rmse, mean_gap=mean_gap, within_5=float(np.mean(absolute <= 5)))
    if n >= 2 and np.ptp(ages) > 0:
        centered_ages = ages - ages.mean()
        variance = float(np.mean(centered_ages**2))
        with np.errstate(over="ignore", invalid="ignore"):
            result["r2"] = _finite_or_none(1.0 - (np.float64(rmse) ** 2) / variance)
        if np.ptp(predictions) > 0:
            # Normalize before centering, avoiding overflow in covariance.
            pred_scale = float(np.max(np.abs(predictions)))
            scaled_predictions = predictions / pred_scale
            centered_predictions = scaled_predictions - scaled_predictions.mean()
            denominator = np.sqrt(np.dot(centered_ages, centered_ages)) * np.sqrt(np.dot(centered_predictions, centered_predictions))
            if denominator > 0:
                result["pearson_r"] = _finite_or_none(np.clip(np.dot(centered_ages, centered_predictions) / denominator, -1, 1))
    return result


def subgroup_metrics(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarize a categorical column, retaining an explicit missing-data group."""
    if group_col not in frame:
        raise ValueError(f"Hiányzó csoportosító oszlop: {group_col}")
    columns = [group_col, *METRIC_COLUMNS, "mean_gap_corrected"]
    rows = []
    for name, group in frame.groupby(group_col, dropna=False, observed=True, sort=False):
        row = {group_col: name, **metrics(group), "mean_gap_corrected": None}
        if "age_gap_corrected" in group:
            corrected = group["age_gap_corrected"].dropna()
            if len(corrected):
                row["mean_gap_corrected"] = _finite_or_none(corrected.mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def demo_predictions(seed: int = 42) -> pd.DataFrame:
    """Create reproducible synthetic data, never representing measured performance."""
    rng = np.random.default_rng(seed)
    n = 720
    ages = rng.integers(18, 90, size=n)
    sex = rng.integers(0, 2, size=n)
    labels = {
        "MI": rng.binomial(1, 0.17, n),
        "STTC": rng.binomial(1, 0.22, n),
        "CD": rng.binomial(1, 0.16, n),
        "HYP": rng.binomial(1, 0.08, n),
    }
    labels["NORM"] = (sum(labels.values()) == 0).astype(int)
    prediction = ages + (54 - ages) * 0.22 + 2.8 * labels["MI"] + rng.normal(0, 6.5, n)
    frame = pd.DataFrame({
        "ecg_id": [f"demo-ecg-{i:04d}" for i in range(n)],
        "patient_id": [f"demo-patient-{i:04d}" for i in range(n)],
        "age": ages,
        "predicted_age": np.round(prediction, 3),
        "split": np.where(np.arange(n) < 240, "validation", "test"),
        "sex": sex,
        "synthetic": True,
        **{label: labels[label] for label in SUPERCLASSES},
    })
    frame.attrs["synthetic"] = True
    frame.attrs["source"] = f"Deterministic synthetic demonstration (seed={seed})"
    return _validate_predictions(frame)


def _json_safe(value: Any) -> Any:
    """Produce standards-compliant JSON, including provenance from numpy/pandas."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, np.ndarray, pd.Index)):
        return [_json_safe(item) for item in value]
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (float, np.floating)):
        return _finite_or_none(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (str,)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def export_report(frame: pd.DataFrame, metadata: dict[str, Any]) -> bytes:
    """Export the *already filtered* rows and their raw metrics as a ZIP report.

    Pass source, filters and calibration in metadata. Synthetic provenance from
    demo DataFrame attributes cannot be overridden by caller metadata.
    """
    provenance = dict(metadata)
    synthetic = provenance.get("synthetic", frame.attrs.get("synthetic", False))
    if not isinstance(synthetic, (bool, np.bool_)):
        raise ValueError("A synthetic metaadat értéke true vagy false lehet.")
    provenance["synthetic"] = bool(synthetic or frame.attrs.get("synthetic", False))
    provenance.setdefault("source", frame.attrs.get("source", "unspecified"))
    provenance.setdefault("filters", {})
    provenance.setdefault("created_at_utc", datetime.now(timezone.utc).isoformat())
    provenance["exported_rows"] = len(frame)
    provenance["metrics_basis"] = "Raw predicted_age versus chronological age; within_5 is a fraction."
    report = _json_safe({"metrics": metrics(frame), "metadata": provenance})
    report_json = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    synthetic_banner = (
        '<p class="notice"><strong>SZINTETIKUS DEMÓADATOK.</strong> Ez a riport nem a modell mért teljesítményét mutatja.</p>'
        if provenance["synthetic"] else ""
    )
    metric_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape('Nem értelmezhető' if value is None else str(value))}</td></tr>"
        for key, value in report["metrics"].items()
    )
    metadata_json = json.dumps(report["metadata"], ensure_ascii=False, indent=2, allow_nan=False)
    summary = f"""<!doctype html>
<html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TDK · EKG-életkor riport</title>
<style>body{{font:16px system-ui,sans-serif;max-width:960px;margin:48px auto;padding:0 24px;color:#172b3a}}h1{{font-size:28px}}table{{border-collapse:collapse;width:100%}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #dde5e9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:20px}}.notice{{background:#fff3cd;padding:18px;border-left:4px solid #b78200}}</style>
</head><body><h1>TDK · EKG-életkor riport</h1>{synthetic_banner}
<p>A kimutatás a kiválasztott, szűrt rekordokra vonatkozik. A pontossági metrikák az eredeti becslést használják; az AgeGap-korrekció nem önálló életkorbecslés.</p>
<table><thead><tr><th>Metrika</th><th>Érték</th></tr></thead><tbody>{metric_rows}</tbody></table>
<p>MAE, RMSE és mean_gap mértékegysége: év. within_5: az ±5 éven belüli becslések aránya (0–1). A mean_gap előjele: becsült életkor − valós életkor.</p>
<h2>Forrás, szűrők és reprodukálhatóság</h2><pre>{html.escape(metadata_json)}</pre>
</body></html>"""
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("predictions.csv", frame.to_csv(index=False))
        archive.writestr("metrics.json", report_json)
        archive.writestr("summary.html", summary)
    return buffer.getvalue()
