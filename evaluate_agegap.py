from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import LinearRegression

from ecg_age_common import SUPERCLASSES, add_diagnostic_superclasses, load_ptbxl_metadata, save_json


def fit_bias_correction(df_val: pd.DataFrame):
    x = df_val[["age"]].values
    y = df_val["age_gap_raw"].values
    reg = LinearRegression().fit(x, y)
    return float(reg.intercept_), float(reg.coef_[0])


def logistic_models(df: pd.DataFrame):
    results = {}
    sex_col = "sex" if "sex" in df.columns else None
    for label in SUPERCLASSES:
        d = df.dropna(subset=[label, "age_gap_corrected", "age"]).copy()
        if d[label].nunique() < 2:
            continue
        cols = ["age_gap_corrected", "age"]
        if sex_col is not None:
            cols.append(sex_col)
        X = sm.add_constant(d[cols].astype(float), has_constant="add")
        y = d[label].astype(int)
        try:
            fit = sm.Logit(y, X).fit(disp=False)
            beta = float(fit.params["age_gap_corrected"])
            se = float(fit.bse["age_gap_corrected"])
            # Report per +5 years of corrected AgeGap.
            or5 = float(np.exp(beta * 5))
            lo5 = float(np.exp((beta - 1.96 * se) * 5))
            hi5 = float(np.exp((beta + 1.96 * se) * 5))
            results[label] = {
                "n": int(len(d)),
                "odds_ratio_per_5y_agegap": or5,
                "ci95_low": lo5,
                "ci95_high": hi5,
                "p_value": float(fit.pvalues["age_gap_corrected"]),
            }
        except Exception as e:
            results[label] = {"error": str(e)}
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--output-dir", default="runs/baseline/analysis")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pred = pd.read_csv(args.predictions)
    pred["age_gap_raw"] = pred["predicted_age"] - pred["age"]
    val = pred[pred["split"] == "validation"].copy()
    test = pred[pred["split"] == "test"].copy()

    intercept, slope = fit_bias_correction(val)
    for d in (val, test):
        d["age_gap_corrected"] = d["age_gap_raw"] - (intercept + slope * d["age"])

    db = load_ptbxl_metadata(args.data_dir, 18, 89)
    db = add_diagnostic_superclasses(db, args.data_dir).reset_index()
    test = test.merge(db[["ecg_id", "patient_id", "sex", "NORM", "MI", "STTC", "CD", "HYP"]],
                      on="ecg_id", how="left")
    test.to_csv(out / "test_agegap_with_labels.csv", index=False)

    rho_raw, p_raw = spearmanr(test["age"], test["age_gap_raw"])
    rho_corr, p_corr = spearmanr(test["age"], test["age_gap_corrected"])

    group_stats = []
    for label in SUPERCLASSES:
        pos = test.loc[test[label] == 1, "age_gap_corrected"].dropna()
        neg = test.loc[test[label] == 0, "age_gap_corrected"].dropna()
        if len(pos) and len(neg):
            u = mannwhitneyu(pos, neg, alternative="two-sided")
            group_stats.append({
                "label": label,
                "n_positive": len(pos),
                "mean_positive": float(pos.mean()),
                "median_positive": float(pos.median()),
                "n_negative": len(neg),
                "mean_negative": float(neg.mean()),
                "median_negative": float(neg.median()),
                "mannwhitney_p": float(u.pvalue),
            })
    pd.DataFrame(group_stats).to_csv(out / "diagnostic_group_agegap.csv", index=False)

    logit = logistic_models(test)
    save_json(logit, out / "logistic_regression.json")

    summary = {
        "bias_correction": {"intercept": intercept, "slope": slope},
        "test_age_vs_raw_agegap_spearman": {"rho": float(rho_raw), "p": float(p_raw)},
        "test_age_vs_corrected_agegap_spearman": {"rho": float(rho_corr), "p": float(p_corr)},
    }
    save_json(summary, out / "agegap_summary.json")

    plt.figure(figsize=(7, 5))
    plt.scatter(test["age"], test["age_gap_raw"], s=8, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Kronológiai életkor (év)")
    plt.ylabel("Nyers ECG AgeGap (év)")
    plt.title("Nyers AgeGap életkorfüggése")
    plt.tight_layout()
    plt.savefig(out / "agegap_raw_vs_age.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(test["age"], test["age_gap_corrected"], s=8, alpha=0.35)
    plt.axhline(0, linewidth=1)
    plt.xlabel("Kronológiai életkor (év)")
    plt.ylabel("Korrigált ECG AgeGap (év)")
    plt.title("Bias-korrigált AgeGap")
    plt.tight_layout()
    plt.savefig(out / "agegap_corrected_vs_age.png", dpi=180)
    plt.close()

    labels = []
    values = []
    for label in SUPERCLASSES:
        arr = test.loc[test[label] == 1, "age_gap_corrected"].dropna().values
        if len(arr):
            labels.append(label)
            values.append(arr)
    if values:
        plt.figure(figsize=(8, 5))
        plt.boxplot(values, tick_labels=labels, showfliers=False)
        plt.axhline(0, linewidth=1)
        plt.ylabel("Korrigált ECG AgeGap (év)")
        plt.title("AgeGap diagnosztikai szuperosztályok szerint")
        plt.tight_layout()
        plt.savefig(out / "agegap_by_diagnostic_superclass.png", dpi=180)
        plt.close()

    print(json.dumps({**summary, "logistic_models": logit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
