"""Train the 1D ResNet ECG-age regressor on PTB-XL (folds 1-8 / 9 / 10).

Baseline call (identical defaults to the first version, plus the signal cache):
    python train_ecg_age.py --data-dir PTBXL --output-dir runs/baseline

Improved / experimental options (see README, "Kísérleti kapcsolók"):
    --train-cohort norm      train + early-stop only on strictly normal ECGs (E7)
    --augment                mild noise / baseline-wander / scale augmentation
    --preprocessing ...      clip5 | record_zscore | lead_standardize (E6)
    --sampling-rate 500      500 Hz ablation (E5)
    --scheduler cosine       cosine LR with warm-up instead of ReduceLROnPlateau
"""
from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from ecg_age_common import (
    DEFAULT_PREPROCESSING,
    PREPROCESSING_MODES,
    SUPERCLASSES,
    TRAIN_COHORTS,
    ECGAugment,
    PTBXLAgeDataset,
    SignalCache,
    add_diagnostic_superclasses,
    build_model,
    compute_lead_stats,
    count_parameters,
    load_ptbxl_metadata,
    pick_device,
    save_json,
    seed_everything,
    strict_normal_mask,
    training_cohort_mask,
)


def regression_metrics(age: np.ndarray, pred: np.ndarray) -> dict:
    age = np.asarray(age, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(age) == 0:
        return {"n": 0}
    out = {
        "n": int(len(age)),
        "mae": float(mean_absolute_error(age, pred)),
        "rmse": float(math.sqrt(mean_squared_error(age, pred))),
        "mean_error": float(np.mean(pred - age)),
    }
    out["r2"] = float(r2_score(age, pred)) if len(age) > 1 else float("nan")
    out["pearson_r"] = float(pearsonr(age, pred).statistic) if len(age) > 2 and np.std(pred) > 0 else float("nan")
    return out


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, age, ecg_id in loader:
            pred = model(x.to(device)).detach().cpu().numpy().reshape(-1)
            age_np = age.numpy()
            for i in range(len(pred)):
                rows.append((int(ecg_id[i]), float(age_np[i]), float(pred[i])))
    df = pd.DataFrame(rows, columns=["ecg_id", "age", "predicted_age"])
    m = regression_metrics(df.age, df.predicted_age)
    return df, {k: m[k] for k in ("mae", "rmse", "r2", "pearson_r")}


def parse_int_list(text: str) -> list[int]:
    return [int(v) for v in str(text).split(",") if v.strip()]


def make_loss(name: str, beta: float):
    if name == "huber":
        return torch.nn.SmoothL1Loss(beta=beta)
    if name == "l1":
        return torch.nn.L1Loss()
    if name == "mse":
        return torch.nn.MSELoss()
    raise ValueError(name)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", default="runs/baseline")
    p.add_argument("--sampling-rate", type=int, default=100, choices=[100, 500])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=7)
    # --- new options (defaults reproduce the original baseline) ---
    p.add_argument("--train-cohort", choices=TRAIN_COHORTS, default="all",
                   help="all: every 18-89 y record; norm: only strictly normal ECGs (healthy reference model)")
    p.add_argument("--preprocessing", choices=PREPROCESSING_MODES, default="clip5")
    p.add_argument("--augment", action="store_true", help="train-time noise / baseline wander / scale")
    p.add_argument("--crop-seconds", type=float, default=None,
                   help="random crop length during training (e.g. 8); evaluation uses the full 10 s")
    p.add_argument("--widths", default="64,128,256,512")
    p.add_argument("--blocks", default="2,2,2,2", help="residual blocks per stage")
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--loss", choices=["huber", "l1", "mse"], default="huber")
    p.add_argument("--huber-beta", type=float, default=5.0)
    p.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")
    p.add_argument("--warmup-epochs", type=int, default=1, help="only for --scheduler cosine")
    p.add_argument("--no-bias-init", action="store_true",
                   help="do not initialise the output bias at the training mean age")
    p.add_argument("--cache-dir", default="cache", help="memory-mapped signal cache directory")
    p.add_argument("--no-cache", action="store_true", help="read WFDB files every epoch (slow)")
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    db = load_ptbxl_metadata(args.data_dir, min_age=18, max_age=89)
    has_scp = (Path(args.data_dir) / "scp_statements.csv").is_file()
    if has_scp:
        db = add_diagnostic_superclasses(db, args.data_dir)
    elif args.train_cohort != "all":
        sys.exit("--train-cohort norm requires scp_statements.csv in the data directory")
    db["split"] = np.where(db.strat_fold <= 8, "train", np.where(db.strat_fold == 9, "validation", "test"))
    db["in_training_cohort"] = training_cohort_mask(db, args.train_cohort).astype(int)

    train_df = db[(db.split == "train") & (db.in_training_cohort == 1)].copy()
    val_sel_df = db[(db.split == "validation") & (db.in_training_cohort == 1)].copy()  # early stopping
    val_df = db[db.split == "validation"].copy()   # all validation records (for AgeGap correction)
    test_df = db[db.split == "test"].copy()

    print(f"Device: {device}")
    print(f"Training cohort: {args.train_cohort}")
    print(f"Train/val(selection)/val(all)/test: {len(train_df)}/{len(val_sel_df)}/{len(val_df)}/{len(test_df)}")
    if len(train_df) == 0 or len(val_sel_df) == 0:
        sys.exit("Empty training or validation set after filtering.")

    cache = None
    if not args.no_cache:
        cache = SignalCache.build_or_load(db, args.data_dir, args.sampling_rate, args.cache_dir)

    preprocessing = dict(DEFAULT_PREPROCESSING, mode=args.preprocessing)
    if args.preprocessing == "lead_standardize":
        tmp = PTBXLAgeDataset(train_df, args.data_dir, args.sampling_rate, cache=cache)
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(tmp), size=min(2000, len(tmp)), replace=False)
        preprocessing.update(compute_lead_stats((tmp.raw_signal(int(i)) for i in idx),
                                                clip_mv=preprocessing["clip_mv"]))
        print("Lead statistics (train):", preprocessing)

    augment = None
    if args.augment or args.crop_seconds:
        augment = ECGAugment(args.sampling_rate, crop_seconds=args.crop_seconds,
                             **({} if args.augment else {"scale": 0, "noise_mv": 0, "wander_mv": 0}))

    def dataset(frame, aug=None):
        return PTBXLAgeDataset(frame, args.data_dir, args.sampling_rate, preprocessing["clip_mv"],
                               preprocessing=preprocessing, cache=cache, augment=aug)

    pin = device.type == "cuda"
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(dataset(train_df, augment), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, generator=gen,
                              drop_last=len(train_df) > args.batch_size)
    eval_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    val_sel_loader = DataLoader(dataset(val_sel_df), **eval_kw)

    model_config = {"dropout": args.dropout, "widths": parse_int_list(args.widths),
                    "blocks_per_stage": parse_int_list(args.blocks)}
    model = build_model(model_config)
    train_mean_age = float(train_df.age.mean())
    if not args.no_bias_init:
        model.init_output_bias(train_mean_age)
    model = model.to(device)
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    criterion = make_loss(args.loss, args.huber_beta)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    else:
        steps_per_epoch = max(len(train_loader), 1)
        total = args.epochs * steps_per_epoch
        warm = max(args.warmup_epochs * steps_per_epoch, 1)

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(total - warm, 1))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    config = {
        **vars(args),
        "device": str(device),
        "model_config": model_config,
        "preprocessing": preprocessing,
        "augmentation": augment.describe() if augment else None,
        "n_parameters": n_params,
        "train_mean_age": train_mean_age,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "platform": platform.platform(),
    }
    save_json(config, out / "config.json")

    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n = 0
        for x, age, _ in train_loader:
            x = x.to(device)
            age = age.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, age)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if args.scheduler == "cosine":
                scheduler.step()
            running_loss += float(loss.detach().cpu()) * len(age)
            n += len(age)

        train_loss = running_loss / max(n, 1)
        val_pred, val_metrics = evaluate(model, val_sel_loader, device)
        if args.scheduler == "plateau":
            scheduler.step(val_metrics["mae"])
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": train_loss, "lr": lr, **val_metrics,
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(row, flush=True)
        pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

        if val_metrics["mae"] < best_mae - 1e-4:
            best_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "sampling_rate": args.sampling_rate,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "model_config": model_config,
                "preprocessing": preprocessing,
                "train_cohort": args.train_cohort,
                "train_mean_age": train_mean_age,
            }, out / "best_model.pt")
            val_pred.to_csv(out / "validation_predictions.csv", index=False)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    val_pred, _ = evaluate(model, DataLoader(dataset(val_df), **eval_kw), device)
    test_pred, _ = evaluate(model, DataLoader(dataset(test_df), **eval_kw), device)
    val_pred["split"] = "validation"
    test_pred["split"] = "test"
    predictions = pd.concat([val_pred, test_pred], ignore_index=True)
    extra = ["patient_id", "sex", "in_training_cohort"] + ([*SUPERCLASSES, "n_superclasses"] if has_scp else [])
    meta = db[[c for c in extra if c in db.columns]].copy()
    if has_scp:
        meta["strict_norm"] = strict_normal_mask(db).astype(int)
    predictions = predictions.merge(meta, left_on="ecg_id", right_index=True, how="left")
    predictions["age_gap_raw"] = predictions["predicted_age"] - predictions["age"]
    predictions.to_csv(out / "predictions.csv", index=False)

    def block(frame):
        res = {"all": regression_metrics(frame.age, frame.predicted_age)}
        if "strict_norm" in frame:
            for name, mask in (("strict_norm", frame.strict_norm == 1), ("non_norm", frame.strict_norm == 0)):
                res[name] = regression_metrics(frame.age[mask], frame.predicted_age[mask])
        return res

    test_block = block(test_pred.merge(meta, left_on="ecg_id", right_index=True, how="left"))
    val_block = block(val_pred.merge(meta, left_on="ecg_id", right_index=True, how="left"))
    const_mean = np.full(len(test_pred), train_mean_age)
    const_median = np.full(len(test_pred), float(train_df.age.median()))

    metrics = {
        "device": str(device),
        "train_cohort": args.train_cohort,
        "n_train": len(train_df),
        "n_validation_selection": len(val_sel_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        # 'validation'/'test' keep the original flat schema (all records) for the dashboard
        "validation": {k: val_block["all"][k] for k in ("mae", "rmse", "r2", "pearson_r")},
        "test": {k: test_block["all"][k] for k in ("mae", "rmse", "r2", "pearson_r")},
        "validation_by_subgroup": val_block,
        "test_by_subgroup": test_block,
        "constant_baseline_test": {
            "train_mean": regression_metrics(test_pred.age, const_mean),
            "train_median": regression_metrics(test_pred.age, const_median),
        },
        "best_epoch": int(ckpt["epoch"]),
        "n_parameters": n_params,
        "minutes": round((time.time() - started) / 60, 1),
    }
    save_json(metrics, out / "metrics.json")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
