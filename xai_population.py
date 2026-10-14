"""Population-level Integrated Gradients: leads, waveform regions, AgeGap groups.

    python xai_population.py --data-dir PTBXL \
        --checkpoint runs/baseline/best_model.pt \
        --agegap-csv runs/baseline/analysis/test_agegap_with_labels.csv \
        --output-dir runs/baseline/xai_population

Selects TEST records from the three corrected-AgeGap groups
(negative = lowest 20 %, neutral = middle 20 %, positive = highest 20 %),
computes IG (zero baseline) per record and summarises

  * lead importance      share of total |IG| per lead
  * region enrichment    share of |IG| in P / PR / QRS / ST / T divided by the
                         share of time those regions occupy (1 = no concentration)
  * signed contribution  sum of IG per region in YEARS (IG completeness:
                         sum(IG) = f(x) - f(baseline))
  * beat templates       R-aligned average signal and |IG| per group

and compares positive vs negative AgeGap groups (Mann–Whitney, Holm).
--sanity-random repeats the lead analysis with a randomly initialised network
(Adebayo et al. 2018 model-randomisation test): trained-model maps that look
like random-model maps would not be trustworthy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from ecg_age_common import (  # noqa: E402
    LEADS, PTBXLAgeDataset, SignalCache, build_model, load_checkpoint, load_ptbxl_metadata,
    pick_device, preprocess, save_json, seed_everything,
)
from ecg_beats import REGIONS, beat_templates, detect_r_peaks, heart_rate_bpm, region_labels  # noqa: E402
from evaluate_agegap import holm, style_axes  # noqa: E402

GROUP_COLORS = {"negatív": "#2a78d6", "semleges": "#8a8984", "pozitív": "#eb6834"}
INK = "#2b2b2a"


def select_records(agegap: pd.DataFrame, n_per_group: int, seed: int) -> pd.DataFrame:
    t = agegap.dropna(subset=["age_gap_corrected"]).copy()
    q20, q40, q60, q80 = t.age_gap_corrected.quantile([0.2, 0.4, 0.6, 0.8])
    t["group"] = np.select(
        [t.age_gap_corrected <= q20, t.age_gap_corrected.between(q40, q60), t.age_gap_corrected >= q80],
        ["negatív", "semleges", "pozitív"], default="")
    t = t[t.group != ""]
    parts = [g.sample(n=min(n_per_group, len(g)), random_state=seed) for _, g in t.groupby("group")]
    return pd.concat(parts).sort_values("ecg_id")


def integrated_gradients(model, x: torch.Tensor, steps: int, internal_batch: int):
    from captum.attr import IntegratedGradients

    attr, delta = IntegratedGradients(model).attribute(
        x, baselines=torch.zeros_like(x), n_steps=steps,
        internal_batch_size=internal_batch, return_convergence_delta=True)
    return attr.detach().cpu().numpy(), delta.detach().cpu().numpy().reshape(-1)


def record_summary(attr: np.ndarray, x_mv: np.ndarray, fs: float) -> tuple[dict, dict]:
    """attr / x_mv: [12, T]. Returns (flat row of scalars, arrays for templates)."""
    a = np.abs(attr)
    total = a.sum()
    row = {f"lead_{lead}": float(a[i].sum() / total) for i, lead in enumerate(LEADS)}
    peaks = detect_r_peaks(x_mv, fs)
    row["n_beats"] = int(len(peaks))
    row["heart_rate_bpm"] = heart_rate_bpm(peaks, fs)
    labels = region_labels(attr.shape[1], peaks, fs)
    region_lead = np.zeros((12, len(REGIONS)))
    for k, name in enumerate(REGIONS):
        m = labels == k
        time_share = m.mean()
        share = a[:, m].sum() / total
        row[f"time_share_{name}"] = float(time_share)
        row[f"share_{name}"] = float(share)
        row[f"enrichment_{name}"] = float(share / time_share) if time_share > 0 else np.nan
        row[f"signed_years_{name}"] = float(attr[:, m].sum())
        region_lead[:, k] = a[:, m].sum(axis=1) / total
    sig_tpl, n_tpl, offsets = beat_templates(x_mv, peaks, fs)
    att_tpl, _, _ = beat_templates(a, peaks, fs)
    return row, {"region_lead": region_lead, "signal_template": sig_tpl, "attr_template": att_tpl,
                 "offsets_ms": offsets}


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--agegap-csv", required=True, help="analysis/test_agegap_with_labels.csv")
    p.add_argument("--output-dir", default="runs/baseline/xai_population")
    p.add_argument("--n-per-group", type=int, default=100)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--internal-batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--sanity-random", action="store_true", help="model-randomisation sanity check")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(args.checkpoint, device)
    fs = int(ckpt["sampling_rate"])
    prep = ckpt["preprocessing"]
    db = load_ptbxl_metadata(args.data_dir, 18, 89)
    cache = None if args.no_cache else SignalCache.build_or_load(db, args.data_dir, fs, args.cache_dir)
    chosen = select_records(pd.read_csv(args.agegap_csv), args.n_per_group, args.seed)
    chosen["ecg_id"] = chosen["ecg_id"].astype(int)
    chosen = chosen[chosen.ecg_id.isin(db.index)]
    ds = PTBXLAgeDataset(db.loc[chosen.ecg_id], args.data_dir, fs, cache=cache)
    print(f"Device: {device}; records: {len(chosen)} ({chosen.group.value_counts().to_dict()})", flush=True)

    random_model = None
    if args.sanity_random:
        torch.manual_seed(args.seed + 1)
        random_model = build_model(ckpt.get("model_config")).to(device).eval()

    rows, region_lead, sig_tpls, att_tpls, offsets = [], [], {}, {}, None
    for i, (_, rec) in enumerate(chosen.iterrows()):
        x_mv = ds.raw_signal(i)
        x = torch.from_numpy(preprocess(x_mv, prep)).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = float(model(x).item())
            base = float(model(torch.zeros_like(x)).item())
        attr, delta = integrated_gradients(model, x, args.steps, args.internal_batch)
        summ, arrays = record_summary(attr[0], x_mv, fs)
        row = {"ecg_id": int(rec.ecg_id), "group": rec.group, "age": float(rec.age),
               "predicted_age": pred, "baseline_prediction": base,
               "age_gap_corrected": float(rec.age_gap_corrected), "ig_sum": float(attr.sum()),
               "ig_convergence_delta": float(delta[0]), **summ}
        if random_model is not None:
            r_attr, _ = integrated_gradients(random_model, x, args.steps, args.internal_batch)
            r_share = np.abs(r_attr[0]).sum(axis=1) / np.abs(r_attr[0]).sum()
            t_share = np.array([summ[f"lead_{lead}"] for lead in LEADS])
            row["sanity_spearman_lead_vs_random"] = float(spearmanr(t_share, r_share).statistic)
        rows.append(row)
        region_lead.append(arrays["region_lead"])
        if arrays["signal_template"] is not None:
            sig_tpls.setdefault(rec.group, []).append(arrays["signal_template"])
            att_tpls.setdefault(rec.group, []).append(arrays["attr_template"])
            offsets = arrays["offsets_ms"]
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(chosen)}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(out / "per_record_attribution.csv", index=False)
    np.savez_compressed(out / "attribution_arrays.npz", region_lead=np.array(region_lead),
                        ecg_id=per.ecg_id.to_numpy(), group=per.group.to_numpy(),
                        offsets_ms=offsets if offsets is not None else np.array([]),
                        **{f"signal_template_{g}": np.mean(v, axis=0) for g, v in sig_tpls.items()},
                        **{f"attr_template_{g}": np.mean(v, axis=0) for g, v in att_tpls.items()})

    # ---- group summaries + positive vs negative tests ---------------------------------
    features = ([f"lead_{lead}" for lead in LEADS] + [f"enrichment_{r}" for r in REGIONS]
                + [f"share_{r}" for r in REGIONS] + [f"signed_years_{r}" for r in REGIONS] + ["heart_rate_bpm"])
    summary_rows = []
    for feat in features:
        row = {"feature": feat}
        for g in ("negatív", "semleges", "pozitív"):
            v = per.loc[per.group == g, feat].to_numpy(float)
            lo, hi = bootstrap_mean_ci(v, seed=args.seed)
            row.update({f"{g}_mean": float(np.nanmean(v)) if len(v) else np.nan, f"{g}_ci_low": lo, f"{g}_ci_high": hi})
        a = per.loc[per.group == "pozitív", feat].dropna()
        b = per.loc[per.group == "negatív", feat].dropna()
        row["mw_p_pos_vs_neg"] = float(mannwhitneyu(a, b).pvalue) if len(a) > 1 and len(b) > 1 else np.nan
        summary_rows.append(row)
    summ = pd.DataFrame(summary_rows)
    for family in ("lead_", "enrichment_", "share_", "signed_years_"):
        m = summ.feature.str.startswith(family)
        summ.loc[m, "p_holm_within_family"] = holm(summ.loc[m, "mw_p_pos_vs_neg"])
    summ.to_csv(out / "group_attribution_summary.csv", index=False)

    overall = {
        "n_records": int(len(per)),
        "median_ig_convergence_delta": float(per.ig_convergence_delta.abs().median()),
        "median_relative_delta": float((per.ig_convergence_delta.abs()
                                        / (per.predicted_age - per.baseline_prediction).abs().clip(lower=1e-6)).median()),
        "baseline_prediction_zero_signal": float(per.baseline_prediction.median()),
        "records_without_detected_beats": int((per.n_beats == 0).sum()),
        "top_leads_overall": per[[f"lead_{lead}" for lead in LEADS]].mean().sort_values(ascending=False)
        .index.str.replace("lead_", "").tolist()[:5],
    }
    if "sanity_spearman_lead_vs_random" in per:
        overall["sanity_mean_spearman_trained_vs_random"] = float(per.sanity_spearman_lead_vs_random.mean())
    save_json(overall, out / "xai_overview.json")
    print(json.dumps(overall, indent=2, ensure_ascii=False))

    # ---- figures ---------------------------------------------------------------------------
    groups = [g for g in ("negatív", "semleges", "pozitív") if g in set(per.group)]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    width = 0.8 / max(len(groups), 1)
    xs = np.arange(12)
    for k, g in enumerate(groups):
        means = per.loc[per.group == g, [f"lead_{lead}" for lead in LEADS]].mean().to_numpy() * 100
        ax.bar(xs + (k - (len(groups) - 1) / 2) * width, means, width=width * 0.92,
               color=GROUP_COLORS[g], label=f"{g} AgeGap")
    ax.set_xticks(xs, LEADS)
    ax.set_ylabel("|IG| részesedés (%)")
    ax.set_title("Elvezetések hozzájárulása AgeGap-csoportonként", fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lead_importance_by_group.png", dpi=200)
    plt.close(fig)

    rl = np.array(region_lead).mean(axis=0)  # [12, regions] share of total
    time_share = per[[f"time_share_{r}" for r in REGIONS]].mean().to_numpy()
    enr = rl / (time_share[None, :] / 12.0)
    fig, ax = plt.subplots(figsize=(6.4, 5))
    im = ax.imshow(enr, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(REGIONS)), REGIONS)
    ax.set_yticks(range(12), LEADS)
    for (i, j), v in np.ndenumerate(enr):
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7,
                color="white" if v > enr.max() * 0.6 else INK)
    ax.set_title("|IG| dúsulás elvezetés × hullámrégió szerint\n(1 = az időarányos várható érték)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out / "lead_region_enrichment_heatmap.png", dpi=200)
    plt.close(fig)

    if offsets is not None and sig_tpls:
        top = [LEADS.index(lead) for lead in overall["top_leads_overall"][:3]]
        fig, axes = plt.subplots(len(top), 1, figsize=(7, 2.4 * len(top)), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, li in zip(axes, top):
            ax2 = ax.twinx()
            for g in groups:
                if g in sig_tpls and g != "semleges":
                    ax.plot(offsets, np.mean(sig_tpls[g], axis=0)[li], color=GROUP_COLORS[g], linewidth=2,
                            label=f"jel – {g}")
                    ax2.fill_between(offsets, 0, np.mean(att_tpls[g], axis=0)[li], color=GROUP_COLORS[g],
                                     alpha=0.18)
            for name, (lo, hi) in (("P", (-250, -120)), ("QRS", (-50, 50)), ("T", (150, 400))):
                ax.axvspan(lo, hi, color="#e4e3df", alpha=0.35, zorder=0)
                ax.text((lo + hi) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=8, color=INK)
            ax.set_ylabel(f"{LEADS[li]} (mV)")
            ax2.set_ylabel("|IG|", color="#8a8984")
            style_axes(ax)
        axes[0].legend(frameon=False, fontsize=8, loc="upper right")
        axes[-1].set_xlabel("Idő az R-csúcshoz képest (ms)")
        fig.suptitle("R-csúcsra szinkronizált átlagos ütés és |IG| (pozitív vs. negatív AgeGap)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "beat_synchronous_attribution.png", dpi=200)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.8 / max(len(groups), 1)
    xs = np.arange(len(REGIONS))
    for k, g in enumerate(groups):
        means = per.loc[per.group == g, [f"signed_years_{r}" for r in REGIONS]].mean().to_numpy()
        ax.bar(xs + (k - (len(groups) - 1) / 2) * width, means, width=width * 0.92, color=GROUP_COLORS[g], label=g)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(xs, REGIONS)
    ax.set_ylabel("Előjeles IG (év)")
    ax.set_title("Régiók előjeles hozzájárulása a becsült életkorhoz", fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "region_signed_contribution.png", dpi=200)
    plt.close(fig)
    print(f"Done: {out.resolve()}")


if __name__ == "__main__":
    main()
