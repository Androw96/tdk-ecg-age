from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from ecg_age_common import (
    ECGAgeResNet,
    PTBXLAgeDataset,
    load_ptbxl_metadata,
    pick_device,
    save_json,
    seed_everything,
)


def evaluate(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, age, ecg_id in loader:
            x = x.to(device)
            pred = model(x).detach().cpu().numpy()
            age_np = age.numpy()
            for i in range(len(pred)):
                rows.append((int(ecg_id[i]), float(age_np[i]), float(pred[i])))
    df = pd.DataFrame(rows, columns=["ecg_id", "age", "predicted_age"])
    mae = mean_absolute_error(df.age, df.predicted_age)
    rmse = math.sqrt(mean_squared_error(df.age, df.predicted_age))
    r2 = r2_score(df.age, df.predicted_age)
    r = pearsonr(df.age, df.predicted_age).statistic if len(df) > 2 else np.nan
    return df, {"mae": float(mae), "rmse": float(rmse), "r2": float(r2), "pearson_r": float(r)}


def main():
    p = argparse.ArgumentParser()
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
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = load_ptbxl_metadata(args.data_dir, min_age=18, max_age=89)
    train_df = db[db.strat_fold <= 8].copy()
    val_df = db[db.strat_fold == 9].copy()
    test_df = db[db.strat_fold == 10].copy()

    print(f"Device: {device}")
    print(f"Train/val/test: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    train_ds = PTBXLAgeDataset(train_df, args.data_dir, args.sampling_rate)
    val_ds = PTBXLAgeDataset(val_df, args.data_dir, args.sampling_rate)
    test_ds = PTBXLAgeDataset(test_df, args.data_dir, args.sampling_rate)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")

    model = ECGAgeResNet().to(device)
    criterion = torch.nn.SmoothL1Loss(beta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
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
            running_loss += float(loss.detach().cpu()) * len(age)
            n += len(age)

        train_loss = running_loss / max(n, 1)
        val_pred, val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mae"])
        lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss": train_loss, "lr": lr, **val_metrics}
        history.append(row)
        print(row)

        if val_metrics["mae"] < best_mae - 1e-4:
            best_mae = val_metrics["mae"]
            epochs_without_improvement = 0
            torch.save({
                "model_state": model.state_dict(),
                "sampling_rate": args.sampling_rate,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, out / "best_model.pt")
            val_pred.to_csv(out / "validation_predictions.csv", index=False)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping.")
                break

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)

    ckpt = torch.load(out / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_pred, val_metrics = evaluate(model, val_loader, device)
    test_pred, test_metrics = evaluate(model, test_loader, device)
    val_pred["split"] = "validation"
    test_pred["split"] = "test"
    predictions = pd.concat([val_pred, test_pred], ignore_index=True)
    predictions.to_csv(out / "predictions.csv", index=False)

    metrics = {
        "device": str(device),
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "validation": val_metrics,
        "test": test_metrics,
        "best_epoch": int(ckpt["epoch"]),
    }
    save_json(metrics, out / "metrics.json")
    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()
