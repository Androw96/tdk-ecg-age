"""Evaluate an existing checkpoint on both held-out PTB-XL splits; no training."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard.model_service import _metadata, _torch, load_model, model_input, read_record_signal


def select_subset(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Deterministic proportional age-decade sampling within one held-out split."""
    if limit <= 0 or len(frame) <= limit:
        return frame.sort_index()
    groups = [(key, group.sort_index().sample(frac=1, random_state=42))
              for key, group in frame.groupby((frame.age // 10).astype(int), sort=True)]
    quotas = np.array([len(group) * limit / len(frame) for _, group in groups])
    allocated = np.floor(quotas).astype(int)
    remainder = limit - int(allocated.sum())
    for index in np.argsort(-(quotas - allocated), kind="stable")[:remainder]:
        allocated[index] += 1
    return pd.concat([group.iloc[:int(count)] for (_, group), count in zip(groups, allocated)]).sort_index()


def regression_metrics(frame: pd.DataFrame) -> dict:
    age = frame.age.to_numpy(dtype=float)
    prediction = frame.predicted_age.to_numpy(dtype=float)
    error = prediction - age
    denominator = float(np.sum((age - age.mean()) ** 2))
    correlation = None
    if len(age) >= 3 and np.std(age) > 0 and np.std(prediction) > 0:
        correlation = float(np.corrcoef(age, prediction)[0, 1])
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(1 - np.sum(error ** 2) / denominator) if len(age) >= 2 and denominator > 0 else None,
            "pearson_r": correlation, "mean_error": float(error.mean()), "n": len(frame)}


def evaluate_checkpoint(data_dir: str | Path, checkpoint: str | Path, output_dir: str | Path,
                        batch_size: int = 16, device_name: str = "cpu", limit: int = 0) -> dict:
    if batch_size < 1 or batch_size > 512:
        raise ValueError("A batch-size értéke 1–512 közé essen.")
    if limit < 0:
        raise ValueError("A limit nem lehet negatív; 0 jelenti a teljes kiértékelést.")
    db, dataset = _metadata(data_dir)
    if not dataset["n_validation"] or not dataset["n_test"]:
        raise ValueError("Validációs (fold 9) és teszt (fold 10) EKG is szükséges a kiértékeléshez.")
    model, info = load_model(checkpoint, device_name)
    torch = _torch()
    print(f"Eszköz: {info['device']}; mintavétel: {info['sampling_rate']} Hz", flush=True)
    frames = []
    split_metrics = {}
    for split in ("validation", "test"):
        selected = select_subset(db[db.split.eq(split)], limit)
        rows = []
        for start in range(0, len(selected), batch_size):
            batch = selected.iloc[start:start + batch_size]
            signals = [model_input(read_record_signal(data_dir, row, info["sampling_rate"]), info)
                       for _, row in batch.iterrows()]
            if len({signal.shape for signal in signals}) != 1:
                raise ValueError("A köteg EKG-felvételei eltérő hosszúságúak. Használj batch-size=1 értéket.")
            x = torch.from_numpy(np.stack(signals)).to(info["device"])
            with torch.inference_mode():
                predictions = model(x).detach().cpu().numpy().reshape(-1)
            if not np.isfinite(predictions).all():
                raise ValueError(f"A modell nem véges becslést adott a(z) {split} részhalmazon.")
            rows.extend({"ecg_id": int(ecg_id), "age": float(row.age), "predicted_age": float(prediction),
                         "age_gap_raw": float(prediction - row.age), "split": split}
                        for (ecg_id, row), prediction in zip(batch.iterrows(), predictions))
            print(f"{split}: {min(start + len(batch), len(selected))}/{len(selected)} EKG kész", flush=True)
        frame = pd.DataFrame(rows)
        frames.append(frame)
        split_metrics[split] = regression_metrics(frame)
    predictions = pd.concat(frames, ignore_index=True)
    full = len(predictions) == dataset["n_validation"] + dataset["n_test"]
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metrics = {**info, "status": "full" if full else "subset", "subset": not full,
               "n_train": dataset["n_train"], "n_validation": split_metrics["validation"]["n"],
               "n_test": split_metrics["test"]["n"], **split_metrics,
               "provenance": {"created_at": datetime.now(timezone.utc).isoformat(),
                              "data_dir": dataset["data_dir"], "checkpoint_sha256": digest.hexdigest(),
                              "limit_per_split": limit, "sampling": "age_decade_proportional_seed_42" if not full else "all",
                              "available_validation": dataset["n_validation"], "available_test": dataset["n_test"],
                              "age_filter": [18, 89], "patient_leakage_checked": True,
                              "training_performed": False}}
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"Kiértékelés kész ({metrics['status']}): {out.resolve()}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "auto", "cuda", "mps"], default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Legfeljebb ennyi EKG részhalmazonként; 0: összes.")
    args = parser.parse_args()
    try:
        evaluate_checkpoint(args.data_dir, args.checkpoint, args.output_dir, args.batch_size, args.device, args.limit)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Kiértékelési hiba: {exc}\n")


if __name__ == "__main__":
    main()
