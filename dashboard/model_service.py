"""Validated, checkpoint-backed inference shared by the UI and evaluation CLI.

Torch and WFDB are imported only when inference is requested, so reports and
metadata inspection remain usable without the optional model runtime.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _metadata(data_dir: str | Path) -> tuple[pd.DataFrame, dict]:
    root = Path(data_dir).expanduser().resolve()
    path = root / "ptbxl_database.csv"
    if not path.is_file():
        raise ValueError(f"Nem található a PTB-XL metaadatfájl: {path}")
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"A PTB-XL metaadatfájl nem olvasható: {exc}") from exc
    required = {"ecg_id", "age", "strat_fold", "patient_id"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Hiányzó metaadatoszlopok: {', '.join(sorted(missing))}.")
    filename_cols = [col for col in ("filename_lr", "filename_hr") if col in raw.columns]
    if not filename_cols:
        raise ValueError("Hiányzik a filename_lr vagy filename_hr EKG-fájlhivatkozás.")
    if raw.empty:
        raise ValueError("A PTB-XL metaadattábla üres.")
    for name, lo, hi in (("ecg_id", 1, None), ("strat_fold", 1, 10)):
        numeric = pd.to_numeric(raw[name], errors="coerce")
        valid = numeric.notna() & np.isfinite(numeric) & numeric.eq(numeric.round()) & numeric.ge(lo)
        if hi is not None:
            valid &= numeric.le(hi)
        if not valid.all():
            raise ValueError(f"Érvénytelen {name}: {int((~valid).sum())} sor; pozitív egész "
                             + ("1–10 fold szükséges." if name == "strat_fold" else "EKG-azonosító szükséges."))
        raw[name] = numeric.astype(int)
    if raw.ecg_id.duplicated().any():
        raise ValueError("Ismétlődő ecg_id található a metaadatokban.")
    if raw.patient_id.isna().any() or raw.patient_id.astype(str).str.strip().eq("").any():
        raise ValueError("Hiányzó patient_id: a betegszintű adatszivárgás nem ellenőrizhető.")
    ages = pd.to_numeric(raw.age, errors="coerce")
    malformed = raw.age.notna() & (ages.isna() | ~np.isfinite(ages))
    if malformed.any():
        raise ValueError(f"Nem numerikus vagy nem véges életkor: {int(malformed.sum())} sor.")
    raw["age"] = ages
    raw["split"] = np.where(raw.strat_fold.le(8), "train", np.where(raw.strat_fold.eq(9), "validation", "test"))
    patient_splits = raw.groupby("patient_id")["split"].nunique()
    leaked = patient_splits[patient_splits.gt(1)]
    # Follow the age range used by train_ecg_age.py exactly; excluded rows are
    # counted explicitly instead of silently making an apparently complete set.
    db = raw[raw.age.between(18, 89, inclusive="both")].copy().set_index("ecg_id")
    if db.empty:
        raise ValueError("Nincs értékelhető EKG a modell 18–89 éves korcsoportjában.")
    info = {
        "data_dir": str(root),
        "total": len(db),
        "n_train": int(db.split.eq("train").sum()),
        "n_validation": int(db.split.eq("validation").sum()),
        "n_test": int(db.split.eq("test").sum()),
        "record_ids": [int(v) for v in db.index],
        "n_source_records": len(raw),
        "n_excluded_age": len(raw) - len(db),
        "age_min": float(db.age.min()),
        "age_max": float(db.age.max()),
        "available_sampling_rates": [rate for col, rate in (("filename_lr", 100), ("filename_hr", 500)) if col in raw],
        "leakage": {"checked": True, "n_overlapping_patients": len(leaked), "passed": len(leaked) == 0},
        "patient_leakage": len(leaked) > 0,
    }
    if len(leaked):
        raise ValueError(f"Betegszintű adatszivárgás: {len(leaked)} beteg több tanítási/validációs/teszt "
                         "részhalmazban szerepel. Javítsd a strat_fold felosztást.")
    return db, info


def inspect_dataset(data_dir: str | Path) -> dict:
    """Inspect metadata, age eligibility and patient-disjoint fold allocation."""
    return _metadata(data_dir)[1]


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ValueError("A modell futtatásához telepítsd a projekt függőségeit (torch).") from exc
    return torch


def _device(device_name: str):
    torch = _torch()
    if device_name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Ismeretlen eszköz; cpu, auto, cuda vagy mps használható.")
    cuda = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if device_name == "auto":
        device_name = "cuda" if cuda else "mps" if mps else "cpu"
    if device_name == "cuda" and not cuda:
        raise ValueError("CUDA nem érhető el ezen a gépen; válaszd a cpu vagy auto módot.")
    if device_name == "mps" and not mps:
        raise ValueError("MPS nem érhető el ezen a gépen; válaszd a cpu vagy auto módot.")
    return torch.device(device_name)


def load_model(checkpoint: str | Path, device_name: str = "cpu") -> tuple[Any, dict]:
    """Load only a complete compatible state dict; never fall back to random weights."""
    path = Path(checkpoint).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Nem található a betanított modell checkpointja: {path}")
    torch = _torch()
    from ecg_age_common import DEFAULT_PREPROCESSING, PREPROCESSING_MODES, build_model

    device = _device(device_name)
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"A checkpoint biztonságos betöltése sikertelen: {exc}") from exc
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("model_state"), dict):
        raise ValueError("A checkpointból hiányzik a model_state állapotszótár.")
    rate = ckpt.get("sampling_rate")
    if isinstance(rate, bool) or rate not in (100, 500):
        raise ValueError("A checkpoint sampling_rate értéke kötelező, és csak 100 vagy 500 Hz lehet.")
    for key, value in ckpt["model_state"].items():
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Érvénytelen modellparaméter: {key} nem tenzor.")
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nem véges modellparaméter: {key}.")
    model_config = ckpt.get("model_config")
    if model_config is not None and not isinstance(model_config, dict):
        raise ValueError("A checkpoint model_config mezője érvénytelen.")
    preprocessing = ckpt.get("preprocessing") or dict(DEFAULT_PREPROCESSING)
    if not isinstance(preprocessing, dict) or preprocessing.get("mode") not in PREPROCESSING_MODES:
        raise ValueError("A checkpoint preprocessing mezője érvénytelen.")
    if float(preprocessing.get("clip_mv", 5.0)) != 5.0:
        raise ValueError("A felület csak ±5 mV-os klippeléssel tanított modellt támogat.")
    try:
        model = build_model(model_config)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model = model.to(device).eval()
    except Exception as exc:
        raise ValueError(f"A checkpoint nem kompatibilis az ECGAgeResNet modellel: {exc}") from exc
    epoch = ckpt.get("epoch")
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1):
        raise ValueError("A checkpoint epoch értéke pozitív egész kell legyen.")
    return model, {"sampling_rate": int(rate), "best_epoch": epoch, "device": str(device), "checkpoint": str(path),
                   "preprocessing": preprocessing, "train_cohort": ckpt.get("train_cohort", "all")}


def model_input(signal: np.ndarray, info: dict) -> np.ndarray:
    """Convert a read_record_signal() array (±5 mV clipped, /5) to the checkpoint's input space."""
    preprocessing = info.get("preprocessing") or {"mode": "clip5"}
    if preprocessing.get("mode", "clip5") == "clip5":
        return signal
    from ecg_age_common import preprocess

    return preprocess(np.asarray(signal, dtype=np.float32) * 5.0, preprocessing)


def read_record_signal(data_dir: str | Path, row: pd.Series, sampling_rate: int) -> np.ndarray:
    """Read physical mV samples, enforce the 12 leads, reorder and normalize."""
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if column not in row or pd.isna(row[column]) or not str(row[column]).strip():
        raise ValueError(f"Az EKG {sampling_rate} Hz-es fájlhivatkozása ({column}) hiányzik.")
    root = Path(data_dir).expanduser().resolve()
    record = (root / str(row[column])).resolve()
    if not record.is_relative_to(root):
        raise ValueError("Az EKG fájlhivatkozása az adatkönyvtáron kívülre mutat.")
    try:
        import wfdb
    except ImportError as exc:
        raise ValueError("Az EKG beolvasásához telepítsd a wfdb csomagot.") from exc
    try:
        signal, fields = wfdb.rdsamp(str(record))
    except Exception as exc:
        raise ValueError(f"Az EKG nem olvasható ({record.name}): {exc}") from exc
    names = list(fields.get("sig_name") or [])
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(LEADS):
        raise ValueError(f"Pontosan a 12 szabványos elvezetés szükséges. Talált elvezetések: {names}.")
    actual_rate = fields.get("fs")
    if actual_rate is None or not np.isclose(float(actual_rate), sampling_rate, rtol=0, atol=1e-6):
        raise ValueError(f"Eltérő mintavétel: a jel {actual_rate} Hz, a checkpoint {sampling_rate} Hz.")
    units = fields.get("units")
    if units and any(str(unit).lower() != "mv" for unit in units):
        raise ValueError("A modell mV egységű jelet vár; az EKG fizikai mértékegysége eltér.")
    signal = np.asarray(signal)
    if signal.ndim != 2 or signal.shape[1] != 12 or signal.shape[0] < 16:
        raise ValueError(f"Érvénytelen EKG-jelalak: {signal.shape}; legalább 16 × 12 minta szükséges.")
    if not np.isfinite(signal).all():
        raise ValueError("Az EKG hiányzó vagy nem véges mintát tartalmaz.")
    x = signal[:, [names.index(lead) for lead in LEADS]].astype(np.float32).T
    return np.ascontiguousarray(np.clip(x, -5.0, 5.0) / 5.0)


def inspect_record(data_dir: str | Path, ecg_id: int, sampling_rate: int = 100) -> dict:
    """Read one eligible ECG without loading Torch or producing a model estimate."""
    if isinstance(sampling_rate, bool) or sampling_rate not in (100, 500):
        raise ValueError("A kért mintavétel csak 100 vagy 500 Hz lehet.")
    db, _ = _metadata(data_dir)
    if isinstance(ecg_id, bool) or int(ecg_id) != ecg_id or int(ecg_id) not in db.index:
        raise ValueError(f"Az EKG-azonosító nem található a 18–89 éves mintában: {ecg_id}.")
    row = db.loc[int(ecg_id)]
    signal = read_record_signal(data_dir, row, int(sampling_rate))
    result = {"ecg_id": int(ecg_id), "age": float(row.age), "sampling_rate": int(sampling_rate),
              "signal": signal, "leads": LEADS.copy(), "split": str(row["split"])}
    for key in ("sex", "scp_codes", "report"):
        if key in row and pd.notna(row[key]):
            value = row[key]
            result[key] = value.item() if isinstance(value, np.generic) else value
    return result


def predict_record(data_dir: str | Path, checkpoint: str | Path, ecg_id: int,
                   device_name: str = "cpu", explain: bool = False, steps: int = 32) -> dict:
    """Predict one eligible PTB-XL record, optionally using zero-baseline IG."""
    model, info = load_model(checkpoint, device_name)
    result = inspect_record(data_dir, ecg_id, info["sampling_rate"])
    torch = _torch()
    x = torch.from_numpy(model_input(result["signal"], info)).unsqueeze(0).to(info["device"])
    with torch.inference_mode():
        predicted_age = float(model(x).item())
    if not np.isfinite(predicted_age):
        raise ValueError("A modell nem véges becslést adott; ellenőrizd a checkpointot és a bemenetet.")
    result.update(predicted_age=predicted_age, age_gap_raw=predicted_age - result["age"],
                  device=info["device"], checkpoint=info["checkpoint"], best_epoch=info["best_epoch"])
    if explain:
        if not isinstance(steps, int) or isinstance(steps, bool) or not 8 <= steps <= 256:
            raise ValueError("Az Integrated Gradients lépésszáma 8–256 közötti egész lehet.")
        try:
            from captum.attr import IntegratedGradients
        except ImportError as exc:
            raise ValueError("A magyarázathoz telepítsd a captum csomagot.") from exc
        attr, delta = IntegratedGradients(model).attribute(
            x, baselines=torch.zeros_like(x), n_steps=steps,
            internal_batch_size=min(4, steps), return_convergence_delta=True,
        )
        attribution = attr.squeeze(0).detach().cpu().numpy()
        convergence_delta = float(delta.detach().cpu().item())
        if not np.isfinite(attribution).all() or not np.isfinite(convergence_delta):
            raise ValueError("Az Integrated Gradients nem véges attribúciót adott.")
        result.update(attribution=attribution, convergence_delta=convergence_delta,
                      lead_importance=np.mean(np.abs(attribution), axis=1))
    return result
