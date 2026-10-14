"""AgeGap analysis for a finished run: metrics with CIs, bias correction, diagnostics.

    python evaluate_agegap.py --data-dir PTBXL \
        --predictions runs/baseline/predictions.csv \
        --output-dir runs/baseline/analysis

Outputs (all in --output-dir):
    results_summary.md            Hungarian summary tables, ready for TDK ch. 6
    population_table.csv          6.1  records / patients / age / sex / labels per split
    prediction_metrics.csv        6.2  MAE, RMSE, R2, r with patient-cluster bootstrap 95% CI,
                                       constant baselines and the paired MAE difference
    error_by_age_decade.csv       age-group error analysis
    agegap_summary.json           6.3  bias-correction parameters and age correlations
    correction_methods.csv        linear / quadratic / two-step (Beheshti) comparison
    test_agegap_with_labels.csv   per-record corrected AgeGap + labels
    diagnostic_group_agegap.csv   6.4  group medians, Mann-Whitney, Cliff's delta, Holm
    logistic_regression.json      6.4  OR per +5 y AgeGap (age + sex adjusted, cluster-robust SE)
    logistic_sensitivity.csv      the same ORs for every correction method
    agegap_quartiles.csv          label prevalence by corrected-AgeGap quartile (cf. Park et al.)
    *.png                         figures
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
from scipy.stats import chi2_contingency, mannwhitneyu, pearsonr, spearmanr  # noqa: E402

from ecg_age_common import SUPERCLASSES, add_diagnostic_superclasses, load_ptbxl_metadata, save_json  # noqa: E402

BLUE, ORANGE, AQUA, GREY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#2b2b2a"
DECADES = [(18, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 79), (80, 89)]
CORRECTIONS = ("linear", "quadratic", "two_step")


# --------------------------------------------------------------------------- helpers
def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        out[idx] = running
    return out


def weighted_metrics(y, pred, w=None) -> dict:
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    sw = w.sum()
    err = pred - y
    mae = float((w * np.abs(err)).sum() / sw)
    rmse = float(np.sqrt((w * err ** 2).sum() / sw))
    ym = (w * y).sum() / sw
    ss_tot = (w * (y - ym) ** 2).sum()
    r2 = float(1 - (w * err ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan
    pm = (w * pred).sum() / sw
    cov = (w * (y - ym) * (pred - pm)).sum()
    vp = (w * (pred - pm) ** 2).sum()
    r = float(cov / np.sqrt(ss_tot * vp)) if ss_tot > 0 and vp > 0 else np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "pearson_r": r, "mean_error": float((w * err).sum() / sw)}


def cluster_bootstrap(frame: pd.DataFrame, stat_fn, n_boot: int, seed: int) -> np.ndarray:
    """Patient-level (cluster) bootstrap via multinomial group weights.

    Repeated ECGs of one patient are resampled together, so CIs are not
    artificially narrow. stat_fn(weights) -> dict of floats.
    """
    groups = frame["patient_id"] if "patient_id" in frame and frame["patient_id"].notna().all() else frame["ecg_id"]
    codes, _ = pd.factorize(groups)
    n_groups = codes.max() + 1
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        gw = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
        rows.append(stat_fn(gw[codes].astype(float)))
    return pd.DataFrame(rows)


def ci(series: pd.Series) -> tuple[float, float]:
    s = pd.Series(series).dropna()
    if s.empty:
        return (np.nan, np.nan)
    return float(s.quantile(0.025)), float(s.quantile(0.975))


def fmt(v, d=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- bias correction
def fit_bias_correction(df_val: pd.DataFrame):
    """Linear correction (kept for backward compatibility): returns (intercept, slope)."""
    slope, intercept = np.polyfit(df_val["age"].to_numpy(float), df_val["age_gap_raw"].to_numpy(float), 1)
    return float(intercept), float(slope)


class Correction:
    """AgeGap correction fitted on validation data only, applied unchanged to test.

    linear    : gap - (b0 + b1*age)                          (de Lange & Cole 2020)
    quadratic : gap - (b0 + b1*age + b2*age^2)
    two_step  : linear, then subtract the mean residual per integer age
                (Beheshti et al. 2019; used by Barthels et al. 2026). Ages with
                <5 validation records borrow the mean of their 5-year bin.
    """

    def __init__(self, method: str):
        self.method = method

    def fit(self, val: pd.DataFrame):
        age = val["age"].to_numpy(float)
        gap = val["age_gap_raw"].to_numpy(float)
        deg = 2 if self.method == "quadratic" else 1
        self.coef = np.polyfit(age, gap, deg)
        if self.method == "two_step":
            resid = gap - np.polyval(self.coef, age)
            d = pd.DataFrame({"age": np.round(age).astype(int), "r": resid})
            per_age = d.groupby("age")["r"].agg(["mean", "size"])
            d["bin"] = (d["age"] // 5) * 5
            per_bin = d.groupby("bin")["r"].mean()
            self.age_offset = {}
            for a in range(18, 90):
                if a in per_age.index and per_age.loc[a, "size"] >= 5:
                    self.age_offset[a] = float(per_age.loc[a, "mean"])
                else:
                    self.age_offset[a] = float(per_bin.get((a // 5) * 5, 0.0))
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        age = frame["age"].to_numpy(float)
        out = frame["age_gap_raw"].to_numpy(float) - np.polyval(self.coef, age)
        if self.method == "two_step":
            out = out - np.array([self.age_offset.get(int(round(a)), 0.0) for a in age])
        return out

    def params(self) -> dict:
        p = {"method": self.method, "poly_coef_highest_first": [float(c) for c in self.coef]}
        if self.method == "linear":
            p.update(intercept=float(self.coef[1]), slope=float(self.coef[0]))
        return p


# --------------------------------------------------------------------------- statistics
def logistic_models(df: pd.DataFrame, gap_col: str = "age_gap_corrected"):
    import statsmodels.api as sm

    results = {}
    cluster = "patient_id" in df.columns and df["patient_id"].notna().all()
    for label in SUPERCLASSES:
        cols = [gap_col, "age"] + (["sex"] if "sex" in df.columns else [])
        d = df.dropna(subset=[label, *cols]).copy()
        if d[label].nunique() < 2:
            continue
        X = sm.add_constant(d[cols].astype(float), has_constant="add")
        y = d[label].astype(int)
        try:
            if cluster:
                groups = pd.factorize(d["patient_id"])[0]
                fit = sm.Logit(y, X).fit(disp=False, cov_type="cluster", cov_kwds={"groups": groups})
            else:
                fit = sm.Logit(y, X).fit(disp=False)
            beta = float(fit.params[gap_col])
            se = float(fit.bse[gap_col])
            results[label] = {
                "n": int(len(d)),
                "n_positive": int(y.sum()),
                "odds_ratio_per_5y_agegap": float(np.exp(beta * 5)),
                "ci95_low": float(np.exp((beta - 1.96 * se) * 5)),
                "ci95_high": float(np.exp((beta + 1.96 * se) * 5)),
                "p_value": float(fit.pvalues[gap_col]),
                "covariates": cols[1:],
                "se_type": "cluster(patient_id)" if cluster else "model",
            }
        except Exception as e:  # perfect separation, singular matrix, ...
            results[label] = {"error": str(e)}
    labels = [k for k, v in results.items() if "p_value" in v]
    for k, ph in zip(labels, holm([results[k]["p_value"] for k in labels])):
        results[k]["p_holm"] = float(ph)
    return results


def group_statistics(test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in SUPERCLASSES:
        pos = test.loc[test[label] == 1, "age_gap_corrected"].dropna()
        neg = test.loc[test[label] == 0, "age_gap_corrected"].dropna()
        if len(pos) and len(neg):
            u = mannwhitneyu(pos, neg, alternative="two-sided")
            rows.append({
                "label": label,
                "n_positive": len(pos), "mean_positive": float(pos.mean()), "median_positive": float(pos.median()),
                "q1_positive": float(pos.quantile(0.25)), "q3_positive": float(pos.quantile(0.75)),
                "n_negative": len(neg), "mean_negative": float(neg.mean()), "median_negative": float(neg.median()),
                "mean_difference": float(pos.mean() - neg.mean()),
                "cliffs_delta": float(2 * u.statistic / (len(pos) * len(neg)) - 1),
                "mannwhitney_p": float(u.pvalue),
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["mannwhitney_p_holm"] = holm(out["mannwhitney_p"])
    return out


def quartile_table(test: pd.DataFrame) -> pd.DataFrame:
    t = test.dropna(subset=["age_gap_corrected"]).copy()
    t["quartile"] = pd.qcut(t["age_gap_corrected"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    rows = []
    for q, g in t.groupby("quartile", observed=True):
        row = {"quartile": str(q), "n": len(g), "agegap_min": float(g.age_gap_corrected.min()),
               "agegap_max": float(g.age_gap_corrected.max()), "mean_age": float(g.age.mean())}
        for label in SUPERCLASSES:
            row[f"{label}_pct"] = float(100 * g[label].mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    q1, q4 = t[t.quartile == "Q1"], t[t.quartile == "Q4"]
    for label in SUPERCLASSES:
        table = [[int(q4[label].sum()), int((1 - q4[label]).sum())], [int(q1[label].sum()), int((1 - q1[label]).sum())]]
        try:
            p = chi2_contingency(table)[1]
        except ValueError:
            p = np.nan
        out.loc[out.quartile == "Q4", f"{label}_p_Q4_vs_Q1"] = p
    return out


def population_table(db: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in ("train", "validation", "test", "all"):
        d = db if split == "all" else db[db.split == split]
        row = {"split": split, "records": len(d), "patients": int(d.patient_id.nunique()),
               "age_mean": float(d.age.mean()), "age_sd": float(d.age.std()),
               "age_median": float(d.age.median()),
               "female_n": int((d.sex == 1).sum()), "female_pct": float(100 * (d.sex == 1).mean())}
        for label in SUPERCLASSES + ["strict_norm"]:
            if label in d:
                row[f"{label}_n"] = int(d[label].sum())
                row[f"{label}_pct"] = float(100 * d[label].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def figure_scatter(test, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(test.age, test.predicted_age, s=6, alpha=0.35, color=BLUE, linewidths=0, label="Teszt EKG")
    lims = [15, 95]
    ax.plot(lims, lims, color=INK, linewidth=1.2, linestyle="--", label="Identitásvonal (y = x)")
    b1, b0 = np.polyfit(test.age, test.predicted_age, 1)
    xs = np.array(lims)
    ax.plot(xs, b0 + b1 * xs, color=ORANGE, linewidth=2, label=f"Illesztett egyenes (meredekség {b1:.2f})")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Kronológiai életkor (év)")
    ax.set_ylabel("EKG-alapú becsült életkor (év)")
    ax.set_title("Becsült vs. kronológiai életkor – teszthalmaz", fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "predicted_vs_chronological_age.png", dpi=200)
    plt.close(fig)


def figure_decades(dec, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(dec))
    ax.bar(x - 0.2, dec["mae"], width=0.38, color=BLUE, label="MAE")
    ax.bar(x + 0.2, dec["mean_error"], width=0.38, color=ORANGE, label="Átlagos előjeles hiba")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, dec["age_group"])
    ax.set_xlabel("Korcsoport (év)")
    ax.set_ylabel("Év")
    ax.set_title("Hiba korcsoportonként – teszthalmaz", fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "error_by_age_decade.png", dpi=200)
    plt.close(fig)


def figure_gap_vs_age(test, out):
    # Two separate legacy figures (kept) + a combined two-panel figure.
    for col, name, title in (("age_gap_raw", "agegap_raw_vs_age.png", "Nyers AgeGap életkorfüggése"),
                             ("age_gap_corrected", "agegap_corrected_vs_age.png", "Bias-korrigált AgeGap")):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(test["age"], test[col], s=8, alpha=0.35, color=BLUE, linewidths=0)
        ax.axhline(0, linewidth=1, color=INK)
        ax.set_xlabel("Kronológiai életkor (év)")
        ax.set_ylabel(("Nyers" if col == "age_gap_raw" else "Korrigált") + " ECG AgeGap (év)")
        ax.set_title(title)
        style_axes(ax)
        fig.tight_layout()
        fig.savefig(out / name, dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, col, title in zip(axes, ("age_gap_raw", "age_gap_corrected"), ("Nyers AgeGap", "Korrigált AgeGap")):
        rho = spearmanr(test.age, test[col]).statistic
        ax.scatter(test.age, test[col], s=6, alpha=0.35, color=BLUE, linewidths=0)
        ax.axhline(0, color=INK, linewidth=1)
        ax.set_title(f"{title} (Spearman ρ = {rho:.2f})", fontsize=10)
        ax.set_xlabel("Kronológiai életkor (év)")
        style_axes(ax)
    axes[0].set_ylabel("AgeGap (év)")
    fig.tight_layout()
    fig.savefig(out / "agegap_raw_vs_corrected_panel.png", dpi=200)
    plt.close(fig)


def figure_groups(test, out):
    labels, pos, neg = [], [], []
    for label in SUPERCLASSES:
        a = test.loc[test[label] == 1, "age_gap_corrected"].dropna().values
        b = test.loc[test[label] == 0, "age_gap_corrected"].dropna().values
        if len(a) and len(b):
            labels.append(label)
            pos.append(a)
            neg.append(b)
    if not labels:
        return
    # legacy single-box figure
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(pos, tick_labels=labels, showfliers=False)
    ax.axhline(0, linewidth=1, color=INK)
    ax.set_ylabel("Korrigált ECG AgeGap (év)")
    ax.set_title("AgeGap diagnosztikai szuperosztályok szerint")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "agegap_by_diagnostic_superclass.png", dpi=180)
    plt.close(fig)
    # paired present / absent figure
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bp1 = ax.boxplot(pos, positions=x - 0.18, widths=0.32, showfliers=False, patch_artist=True)
    bp2 = ax.boxplot(neg, positions=x + 0.18, widths=0.32, showfliers=False, patch_artist=True)
    for bp, color in ((bp1, ORANGE), (bp2, BLUE)):
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.55, edgecolor=color)
        for item in bp["medians"]:
            item.set(color=INK, linewidth=1.5)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Korrigált ECG AgeGap (év)")
    ax.set_title("Korrigált AgeGap: címke jelen vs. hiányzik (teszthalmaz)", fontsize=11)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["Címke jelen", "Címke hiányzik"], frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / "agegap_by_superclass_present_absent.png", dpi=200)
    plt.close(fig)


def figure_quartiles(q, out):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    colors = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4"]
    for label, color in zip(SUPERCLASSES, colors):
        ax.plot(q.quartile, q[f"{label}_pct"], marker="o", markersize=6, linewidth=2, color=color, label=label)
    ax.set_xlabel("Korrigált AgeGap kvartilis (Q1 = legfiatalabbnak becsült)")
    ax.set_ylabel("Prevalencia (%)")
    ax.set_title("Diagnosztikai címkék gyakorisága AgeGap-kvartilisenként", fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(out / "agegap_quartile_prevalence.png", dpi=200)
    plt.close(fig)


def figure_history(hist_path: Path, out):
    h = pd.read_csv(hist_path)
    if "mae" not in h:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h.epoch, h.mae, marker="o", markersize=4, linewidth=2, color=BLUE, label="Validációs MAE")
    best = h.loc[h.mae.idxmin()]
    ax.scatter([best.epoch], [best.mae], s=60, color=ORANGE, zorder=3, label=f"Legjobb: {best.mae:.2f} év")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAE (év)")
    ax.set_title("Tanítási görbe", fontsize=11)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", default="runs/baseline/analysis")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--correction", choices=CORRECTIONS, default="linear", help="primary correction method")
    p.add_argument("--correction-cohort", choices=["auto", "all", "training"], default="auto",
                   help="validation rows used to fit the correction; auto = training cohort if the run used one")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions)
    run_dir = pred_path.parent
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").is_file() else {}
    train_cohort = config.get("train_cohort", "all")

    # ---- metadata -------------------------------------------------------------------------
    db = load_ptbxl_metadata(args.data_dir, 18, 89)
    db = add_diagnostic_superclasses(db, args.data_dir)
    others = [c for c in SUPERCLASSES if c != "NORM"]
    db["strict_norm"] = ((db.NORM == 1) & (db[others].sum(axis=1) == 0)).astype(int)
    db["split"] = np.where(db.strat_fold <= 8, "train", np.where(db.strat_fold == 9, "validation", "test"))
    pop = population_table(db)
    pop.to_csv(out / "population_table.csv", index=False)

    # ---- predictions ----------------------------------------------------------------------
    pred = pd.read_csv(pred_path)
    keep = [c for c in pred.columns if c in ("ecg_id", "age", "predicted_age", "split", "in_training_cohort")]
    pred = pred[keep].copy()
    pred["age_gap_raw"] = pred["predicted_age"] - pred["age"]
    meta_cols = ["patient_id", "sex", *SUPERCLASSES, "strict_norm"]
    pred = pred.merge(db[meta_cols], left_on="ecg_id", right_index=True, how="left")
    if "in_training_cohort" not in pred:
        pred["in_training_cohort"] = 1
    val = pred[pred["split"] == "validation"].copy()
    test = pred[pred["split"] == "test"].copy()
    if len(val) < 3 or len(test) < 3:
        raise SystemExit("predictions.csv needs validation and test rows")

    # ---- 6.2 prediction metrics with cluster bootstrap --------------------------------------
    train_frame = db[db.split == "train"]
    if train_cohort == "norm":
        train_frame = train_frame[train_frame.strict_norm == 1]
    const = {"Konstans (tanító átlag)": float(train_frame.age.mean()),
             "Konstans (tanító medián)": float(train_frame.age.median())}
    metric_rows = []
    subsets = [("test", "all", test), ("test", "strict_norm", test[test.strict_norm == 1]),
               ("test", "non_norm", test[test.strict_norm == 0]), ("validation", "all", val)]
    for split, subgroup, frame in subsets:
        if len(frame) < 3:
            continue
        y, yhat = frame.age.to_numpy(float), frame.predicted_age.to_numpy(float)
        point = weighted_metrics(y, yhat)
        boot = cluster_bootstrap(frame, lambda w: weighted_metrics(y, yhat, w), args.n_boot, args.seed)
        row = {"model": "1D ResNet", "split": split, "subgroup": subgroup, "n": len(frame)}
        for k in ("mae", "rmse", "r2", "pearson_r", "mean_error"):
            lo, hi = ci(boot[k])
            row.update({k: point[k], f"{k}_ci_low": lo, f"{k}_ci_high": hi})
        metric_rows.append(row)
        if split == "test" and subgroup == "all":
            for name, value in const.items():
                c = np.full_like(y, value)
                cp = weighted_metrics(y, c)
                cb = cluster_bootstrap(frame, lambda w: weighted_metrics(y, c, w), args.n_boot, args.seed)
                crow = {"model": name, "split": split, "subgroup": subgroup, "n": len(frame)}
                for k in ("mae", "rmse", "r2", "mean_error"):
                    lo, hi = ci(cb[k])
                    crow.update({k: cp[k], f"{k}_ci_low": lo, f"{k}_ci_high": hi})
                crow["pearson_r"] = np.nan
                metric_rows.append(crow)
            # paired difference: constant-mean MAE minus model MAE
            c = np.full_like(y, const["Konstans (tanító átlag)"])
            diff = cluster_bootstrap(frame, lambda w: {"d": weighted_metrics(y, c, w)["mae"]
                                                      - weighted_metrics(y, yhat, w)["mae"]},
                                     args.n_boot, args.seed)["d"]
            mae_gain = {"point": weighted_metrics(y, c)["mae"] - point["mae"], "ci": ci(diff)}
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out / "prediction_metrics.csv", index=False)

    # ---- error by decade -------------------------------------------------------------------
    dec_rows = []
    for lo, hi in DECADES:
        g = test[test.age.between(lo, hi)]
        if len(g):
            e = g.predicted_age - g.age
            dec_rows.append({"age_group": f"{lo}–{hi}", "n": len(g), "mae": float(e.abs().mean()),
                             "mean_error": float(e.mean()), "sd_error": float(e.std())})
    dec = pd.DataFrame(dec_rows)
    dec.to_csv(out / "error_by_age_decade.csv", index=False)

    # ---- 6.3 bias correction (validation only) ----------------------------------------------
    cohort = args.correction_cohort
    if cohort == "auto":
        cohort = "training" if train_cohort != "all" else "all"
    fit_rows = val[val.in_training_cohort == 1] if cohort == "training" else val
    corr_rows, corrected = [], {}
    for method in CORRECTIONS:
        c = Correction(method).fit(fit_rows)
        corrected[method] = c
        test_gap = c.transform(test)
        corr_rows.append({"method": method,
                          "test_spearman_rho_vs_age": float(spearmanr(test.age, test_gap).statistic),
                          "test_pearson_r_vs_age": float(pearsonr(test.age, test_gap).statistic),
                          "test_mean": float(np.mean(test_gap)), "test_sd": float(np.std(test_gap, ddof=1))})
    pd.DataFrame(corr_rows).to_csv(out / "correction_methods.csv", index=False)
    primary = corrected[args.correction]
    for d in (val, test):
        d["age_gap_corrected"] = primary.transform(d)
        for method, c in corrected.items():
            d[f"age_gap_{method}"] = c.transform(d)
    test.to_csv(out / "test_agegap_with_labels.csv", index=False)

    rho_raw, p_raw = spearmanr(test["age"], test["age_gap_raw"])
    rho_corr, p_corr = spearmanr(test["age"], test["age_gap_corrected"])
    lin = corrected["linear"].params()
    summary = {
        "bias_correction": {"intercept": lin["intercept"], "slope": lin["slope"],
                            "primary_method": args.correction, "primary": primary.params(),
                            "fitted_on": f"validation ({cohort} rows, n={len(fit_rows)})"},
        "train_cohort": train_cohort,
        "test_age_vs_raw_agegap_spearman": {"rho": float(rho_raw), "p": float(p_raw)},
        "test_age_vs_corrected_agegap_spearman": {"rho": float(rho_corr), "p": float(p_corr)},
        "test_raw_agegap_mean_sd": [float(test.age_gap_raw.mean()), float(test.age_gap_raw.std())],
        "test_corrected_agegap_mean_sd": [float(test.age_gap_corrected.mean()), float(test.age_gap_corrected.std())],
        "mae_improvement_vs_constant_mean": {"years": float(mae_gain["point"]), "ci95": list(mae_gain["ci"])},
    }
    save_json(summary, out / "agegap_summary.json")

    # ---- 6.4 diagnostics -------------------------------------------------------------------
    groups = group_statistics(test)
    groups.to_csv(out / "diagnostic_group_agegap.csv", index=False)
    logit = logistic_models(test)
    save_json(logit, out / "logistic_regression.json")
    sens_rows = []
    for method in CORRECTIONS:
        res = logistic_models(test, f"age_gap_{method}")
        for label, r in res.items():
            sens_rows.append({"method": method, "label": label, **{k: v for k, v in r.items() if k != "covariates"}})
    pd.DataFrame(sens_rows).to_csv(out / "logistic_sensitivity.csv", index=False)
    quart = quartile_table(test)
    quart.to_csv(out / "agegap_quartiles.csv", index=False)

    # ---- figures ---------------------------------------------------------------------------
    figure_scatter(test, out)
    figure_decades(dec, out)
    figure_gap_vs_age(test, out)
    figure_groups(test, out)
    figure_quartiles(quart, out)
    if (run_dir / "training_history.csv").is_file():
        figure_history(run_dir / "training_history.csv", out)

    # ---- Hungarian markdown summary ---------------------------------------------------------
    write_summary(out, pop, metrics_df, summary, pd.DataFrame(corr_rows), groups, logit, quart, dec, config)
    print(json.dumps({**summary, "logistic_models": logit}, indent=2, ensure_ascii=False))


def write_summary(out, pop, metrics_df, summary, corr, groups, logit, quart, dec, config):
    name = {"train": "Tanító", "validation": "Validációs", "test": "Teszt", "all": "Összesen"}
    t61 = pd.DataFrame({
        "Halmaz": [name[s] for s in pop.split],
        "Rekord": pop.records, "Beteg": pop.patients,
        "Életkor (átlag ± SD)": [f"{m:.1f} ± {s:.1f}" for m, s in zip(pop.age_mean, pop.age_sd)],
        "Nő, n (%)": [f"{n} ({p:.1f})" for n, p in zip(pop.female_n, pop.female_pct)],
        "Kizárólag NORM, n (%)": [f"{n} ({p:.1f})" for n, p in zip(pop.strict_norm_n, pop.strict_norm_pct)],
    })
    m = metrics_df[metrics_df.split == "test"]

    def cell(r, k, d=2):
        lo, hi = r.get(f"{k}_ci_low"), r.get(f"{k}_ci_high")
        if not np.isfinite(r.get(k, np.nan)):
            return "–"
        return f"{r[k]:.{d}f} ({fmt(lo, d)}–{fmt(hi, d)})" if lo is not None and np.isfinite(lo) else f"{r[k]:.{d}f}"

    t62 = pd.DataFrame([{"Modell": r.model + ("" if r.subgroup == "all" else f" – {r.subgroup}"), "N": r.n,
                         "MAE (95% CI)": cell(r, "mae"), "RMSE (95% CI)": cell(r, "rmse"),
                         "R² (95% CI)": cell(r, "r2", 3), "Pearson-r (95% CI)": cell(r, "pearson_r", 3)}
                        for _, r in m.iterrows()])
    t63 = pd.DataFrame([{"Módszer": r.method, "Spearman ρ az életkorral": f"{r.test_spearman_rho_vs_age:.3f}",
                         "Átlag ± SD": f"{r.test_mean:.2f} ± {r.test_sd:.2f}"} for _, r in corr.iterrows()])
    rows64 = []
    for _, g in groups.iterrows():
        lr = logit.get(g.label, {})
        rows64.append({"Csoport": g.label, "N (+)": g.n_positive,
                       "Medián AgeGap (+ / −)": f"{g.median_positive:.2f} / {g.median_negative:.2f}",
                       "Cliff δ": f"{g.cliffs_delta:.3f}", "MW p (Holm)": fmt(g.mannwhitney_p_holm, 4),
                       "OR / +5 év (95% CI)": (f"{lr['odds_ratio_per_5y_agegap']:.2f} "
                                               f"({lr['ci95_low']:.2f}–{lr['ci95_high']:.2f})") if "ci95_low" in lr else "–",
                       "p (Holm)": fmt(lr.get("p_holm"), 4)})
    t64 = pd.DataFrame(rows64)
    gain = summary["mae_improvement_vs_constant_mean"]
    text = f"""# Eredményösszefoglaló – automatikusan generálva

Tanító kohorsz: **{summary['train_cohort']}** · Korrekció: **{summary['bias_correction']['primary_method']}**,
illesztve: {summary['bias_correction']['fitted_on']} · Modell: {config.get('model_config', 'alap 1D ResNet')}

## 6.1 A vizsgált populáció (18–89 év)

{md_table(t61)}

## 6.2 Életkorbecslési teljesítmény (teszthalmaz, betegszintű bootstrap 95% CI)

{md_table(t62)}

A modell MAE-je {gain['years']:.2f} évvel (95% CI {fmt(gain['ci95'][0])}–{fmt(gain['ci95'][1])}) kisebb a
konstans (tanítóhalmaz-átlag) becslésénél.

### Hiba korcsoportonként

{md_table(dec.round(2))}

## 6.3 AgeGap-korrekció (teszthalmaz)

Nyers AgeGap – életkor Spearman ρ: {summary['test_age_vs_raw_agegap_spearman']['rho']:.3f};
korrigált: {summary['test_age_vs_corrected_agegap_spearman']['rho']:.3f}.

{md_table(t63)}

## 6.4 Korrigált AgeGap és diagnosztikai szuperosztályok

Logisztikus regresszió: címke ~ korrigált AgeGap + életkor + nem; betegszintű klaszter-robusztus SE;
p-értékek Holm-korrekcióval (5 összehasonlítás).

{md_table(t64)}

### Prevalencia AgeGap-kvartilisenként (%)

{md_table(quart[['quartile', 'n'] + [f'{s}_pct' for s in SUPERCLASSES]].round(1))}
"""
    (out / "results_summary.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
