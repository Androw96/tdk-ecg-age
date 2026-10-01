from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients

from ecg_age_common import ECGAgeResNet, LEADS, PTBXLAgeDataset, load_ptbxl_metadata, pick_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--ecg-id", required=True, type=int)
    p.add_argument("--output-dir", default="runs/baseline/xai")
    p.add_argument("--steps", type=int, default=64)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    sampling_rate = int(ckpt.get("sampling_rate", 100))
    model = ECGAgeResNet().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    db = load_ptbxl_metadata(args.data_dir, 18, 89)
    if args.ecg_id not in db.index:
        raise KeyError(f"ECG ID {args.ecg_id} not present after age filtering")
    one = db.loc[[args.ecg_id]].copy()
    ds = PTBXLAgeDataset(one, args.data_dir, sampling_rate=sampling_rate)
    x, age, ecg_id = ds[0]
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_age = float(model(x).item())

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)
    attr, delta = ig.attribute(x, baselines=baseline, n_steps=args.steps, return_convergence_delta=True)
    attr = attr.squeeze(0).detach().cpu().numpy()
    signal = x.squeeze(0).detach().cpu().numpy()

    lead_importance = np.mean(np.abs(attr), axis=1)
    pd.DataFrame({"lead": LEADS, "mean_abs_attribution": lead_importance}).to_csv(
        out / f"ecg_{ecg_id}_lead_importance.csv", index=False
    )

    t = np.arange(signal.shape[1]) / sampling_rate
    # Plot the three most important leads for a compact first-pass visualization.
    top = np.argsort(lead_importance)[::-1][:3]
    for idx in top:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t, signal[idx], linewidth=0.9)
        ax2 = ax.twinx()
        ax2.fill_between(t, 0, np.abs(attr[idx]), alpha=0.25)
        ax.set_xlabel("Idő (s)")
        ax.set_ylabel("Skálázott EKG")
        ax2.set_ylabel("|Integrated Gradients|")
        ax.set_title(f"{LEADS[idx]} – valós életkor {float(age):.0f}, becsült {predicted_age:.1f} év")
        fig.tight_layout()
        fig.savefig(out / f"ecg_{ecg_id}_{LEADS[idx]}_integrated_gradients.png", dpi=180)
        plt.close(fig)

    print({
        "ecg_id": int(ecg_id),
        "chronological_age": float(age),
        "predicted_age": predicted_age,
        "age_gap_raw": predicted_age - float(age),
        "convergence_delta": float(delta.detach().cpu().item()),
        "top_leads": [LEADS[i] for i in top],
    })


if __name__ == "__main__":
    main()
