"""Combine several training runs (e.g. different seeds) into one ensemble.

    python ensemble_runs.py --runs runs/seed42 runs/seed43 runs/seed44 \
        --output-dir runs/ensemble_3seeds

Writes predictions.csv (mean predicted age per ECG; evaluate_agegap.py can use
it directly), config.json (copied from the first run, with the member list) and
seed_variability.csv: per-run test metrics plus mean ± SD across runs – report
this in the thesis to show how much results depend on the random seed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(frame: pd.DataFrame) -> dict:
    e = frame.predicted_age - frame.age
    ss = ((frame.age - frame.age.mean()) ** 2).sum()
    return {"n": len(frame), "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan,
            "pearson_r": float(np.corrcoef(frame.age, frame.predicted_age)[0, 1])}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    runs = [Path(r) for r in args.runs]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames, rows, cohorts = [], [], set()
    for run in runs:
        pred = pd.read_csv(run / "predictions.csv")
        cfg = json.loads((run / "config.json").read_text()) if (run / "config.json").is_file() else {}
        cohorts.add(cfg.get("train_cohort", "all"))
        frames.append(pred)
        for split in ("validation", "test"):
            rows.append({"run": str(run), "seed": cfg.get("seed"), "split": split,
                         **metrics(pred[pred.split == split])})
    if len(cohorts) > 1:
        raise SystemExit(f"Runs use different training cohorts: {cohorts}")
    keys = ["ecg_id", "split"]
    base = frames[0].drop(columns=[c for c in ("predicted_age", "age_gap_raw") if c in frames[0]])
    stacked = pd.concat([f[keys + ["predicted_age"]] for f in frames])
    counts = stacked.groupby(keys).size()
    if (counts != len(frames)).any():
        raise SystemExit("Runs do not contain the same validation/test ECGs.")
    mean_pred = stacked.groupby(keys, as_index=False)["predicted_age"].mean()
    sd_pred = stacked.groupby(keys)["predicted_age"].std().rename("predicted_age_sd_across_runs").reset_index()
    ens = base.merge(mean_pred, on=keys).merge(sd_pred, on=keys)
    ens["age_gap_raw"] = ens["predicted_age"] - ens["age"]
    ens.to_csv(out / "predictions.csv", index=False)

    for split in ("validation", "test"):
        rows.append({"run": "ENSEMBLE", "seed": None, "split": split, **metrics(ens[ens.split == split])})
    table = pd.DataFrame(rows)
    single = table[(table.run != "ENSEMBLE")]
    agg = single.groupby("split")[["mae", "rmse", "r2", "pearson_r"]].agg(["mean", "std"])
    table.to_csv(out / "seed_variability.csv", index=False)
    agg.to_csv(out / "seed_variability_summary.csv")

    first_cfg = runs[0] / "config.json"
    cfg = json.loads(first_cfg.read_text()) if first_cfg.is_file() else {}
    cfg.update(ensemble_members=[str(r) for r in runs], seed=None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    hist = runs[0] / "training_history.csv"
    print(table.to_string(index=False))
    print("\nMean ± SD over single runs:\n", agg.round(3).to_string())
    if hist.is_file():
        print(f"(training_history.csv not merged; see individual runs, e.g. {hist})")


if __name__ == "__main__":
    main()
