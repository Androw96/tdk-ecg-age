"""Build the example bundles for the hosted UI variants from one finished run.

Run this on your own machine (it needs PyTorch, WFDB and the local PTB-XL copy):

    .venv/bin/python scripts/build_demo_bundle.py \
        --data-dir /UTVONAL/ptb-xl-1.0.3 --run runs/E1_all_seed42 --n 40

It writes two folders that ARE meant to be committed to git:

  demo_data/      for Streamlit Community Cloud (live model on the server):
                  ptbxl/ptbxl_database.csv, ptbxl/scp_statements.csv, the WFDB files of
                  the selected TEST records, best_model.pt, predictions.csv, metrics.json,
                  examples.csv, README.md (licence + provenance)
  web/examples/   for the GitHub Pages / stlite variant (no model in the browser):
                  examples.npz (signals in mV + Integrated Gradients, float16),
                  examples.json (ages, predictions, labels), predictions.csv, metrics.json

Only test-fold (fold 10) records are bundled. PTB-XL is CC BY 4.0: keep the
attribution in demo_data/README.md and on the web page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.model_service import _metadata, load_model, predict_record  # noqa: E402

LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
ATTRIBUTION = (
    "Az EKG-felvételek a PTB-XL adatbázisból származnak (Wagner P, Strodthoff N, Bousseljot R-D, et al. "
    "PTB-XL, a large publicly available electrocardiography dataset. Scientific Data 2020;7:154; "
    "PhysioNet, v1.0.3, doi:10.13026/kfzx-aw45), Creative Commons Attribution 4.0 (CC BY 4.0) licenc alatt."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_records(test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Spread the examples over age decades (equal share per decade where possible)."""
    test = test.copy()
    test["decade"] = (test.age // 10).astype(int)
    groups = [g.sample(frac=1, random_state=seed) for _, g in test.groupby("decade", sort=True)]
    per = int(np.ceil(n / max(len(groups), 1)))
    picked = pd.concat([g.head(per) for g in groups])
    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed)
    return picked.sort_values("ecg_id")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="PTB-XL root (ptbxl_database.csv, records100/)")
    p.add_argument("--run", required=True, help="run folder with best_model.pt and predictions.csv")
    p.add_argument("--n", type=int, default=40, help="number of example test ECGs (default 40)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ig-steps", type=int, default=32)
    p.add_argument("--no-ig", action="store_true", help="skip Integrated Gradients for the web bundle")
    p.add_argument("--demo-out", default=str(ROOT / "demo_data"))
    p.add_argument("--web-out", default=str(ROOT / "web" / "examples"))
    p.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "mps"])
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    run = Path(args.run).expanduser().resolve()
    checkpoint = run / "best_model.pt"
    predictions_path = run / "predictions.csv"
    for path in (checkpoint, predictions_path, data_dir / "ptbxl_database.csv", data_dir / "scp_statements.csv"):
        if not path.is_file():
            sys.exit(f"Hiányzó fájl: {path}")

    db, _ = _metadata(data_dir)
    _, info = load_model(checkpoint, "cpu")
    rate = info["sampling_rate"]
    column = "filename_lr" if rate == 100 else "filename_hr"

    predictions = pd.read_csv(predictions_path)
    test = predictions[predictions.split == "test"]
    if test.empty:
        sys.exit("A predictions.csv nem tartalmaz teszt (fold 10) sorokat.")
    chosen = choose_records(test, args.n, args.seed)
    print(f"{len(chosen)} teszt-EKG kiválasztva ({rate} Hz).", flush=True)

    # ---------------- demo_data/ (Streamlit Community Cloud) ---------------------------
    demo = Path(args.demo_out).resolve()
    ptb = demo / "ptbxl"
    ptb.mkdir(parents=True, exist_ok=True)
    shutil.copy2(data_dir / "ptbxl_database.csv", ptb / "ptbxl_database.csv")
    shutil.copy2(data_dir / "scp_statements.csv", ptb / "scp_statements.csv")
    for ecg_id in chosen.ecg_id:
        rel = Path(str(db.loc[int(ecg_id), column]))
        for suffix in (".hea", ".dat"):
            src = data_dir / rel.with_suffix(suffix)
            dst = ptb / rel.with_suffix(suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    shutil.copy2(checkpoint, demo / "best_model.pt")
    shutil.copy2(predictions_path, demo / "predictions.csv")
    for name in ("metrics.json", "config.json"):
        if (run / name).is_file():
            shutil.copy2(run / name, demo / name)
    chosen[["ecg_id", "age", "predicted_age"]].to_csv(demo / "examples.csv", index=False)
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run": run.name,
        "checkpoint_sha256": sha256(checkpoint),
        "sampling_rate": rate,
        "n_examples": int(len(chosen)),
        "selection": f"fold 10 (test), age-decade spread, seed {args.seed}",
    }
    (demo / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (demo / "README.md").write_text(
        "# Példacsomag az online felülethez\n\n"
        "Ezt a mappát a `scripts/build_demo_bundle.py` készítette; a Streamlit Community Cloud változat használja.\n\n"
        f"- Forrás futás: `{run.name}`\n- Modell SHA-256: `{provenance['checkpoint_sha256']}`\n"
        f"- Példák: {len(chosen)} teszt-EKG (fold 10), {rate} Hz\n\n"
        f"## Licenc\n\n{ATTRIBUTION}\n\n"
        "A modell becslései kutatási célúak; az EKG-életkor és az AgeGap nem diagnózis.\n",
        encoding="utf-8")
    print(f"Kész: {demo}", flush=True)

    # ---------------- web/examples/ (GitHub Pages, stlite) -----------------------------
    web = Path(args.web_out).resolve()
    web.mkdir(parents=True, exist_ok=True)
    signals, attributions, records = [], [], []
    for i, row in enumerate(chosen.itertuples(index=False), 1):
        result = predict_record(data_dir, checkpoint, int(row.ecg_id), device_name=args.device,
                                explain=not args.no_ig, steps=args.ig_steps)
        signals.append((np.asarray(result["signal"]) * 5.0).astype(np.float16))
        if not args.no_ig:
            attributions.append(np.asarray(result["attribution"]).astype(np.float16))
        if abs(result["predicted_age"] - float(row.predicted_age)) > 0.1:
            print(f"  figyelem: EKG #{row.ecg_id} becslése eltér a predictions.csv-től "
                  f"({result['predicted_age']:.2f} vs {row.predicted_age:.2f})", flush=True)
        meta = db.loc[int(row.ecg_id)]
        record = {"ecg_id": int(row.ecg_id), "age": float(result["age"]),
                  "predicted_age": round(float(result["predicted_age"]), 3), "split": "test",
                  "sex": None if pd.isna(meta.get("sex")) else int(meta["sex"])}
        if not args.no_ig:
            record["ig_convergence_delta"] = float(result["convergence_delta"])
        for label in LABELS:
            if label in predictions.columns:
                value = predictions.loc[predictions.ecg_id == row.ecg_id, label]
                record[label] = None if value.empty or pd.isna(value.iloc[0]) else int(value.iloc[0])
        records.append(record)
        print(f"  {i}/{len(chosen)} EKG kész", flush=True)
    arrays = {"signals": np.stack(signals)}
    if attributions:
        arrays["attributions"] = np.stack(attributions)
    np.savez_compressed(web / "examples.npz", **arrays)
    metrics = json.loads((run / "metrics.json").read_text()) if (run / "metrics.json").is_file() else {}
    (web / "examples.json").write_text(json.dumps({
        "sampling_rate": rate, "best_epoch": metrics.get("best_epoch", "—"), "ig_steps": args.ig_steps,
        "units": "mV, ±5 mV-ra vágva", "attribution": ATTRIBUTION, **provenance, "records": records,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(predictions_path, web / "predictions.csv")
    if (run / "metrics.json").is_file():
        shutil.copy2(run / "metrics.json", web / "metrics.json")
    size = sum(f.stat().st_size for f in web.iterdir()) / 1e6
    print(f"Kész: {web} ({size:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
