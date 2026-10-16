"""Copy a finished run's figures and summary into web/results/ for the GitHub Pages site.

    python web/collect_results.py --run runs/E1_all_ens [--xai runs/E1_all_seed42/xai_population]

Copies analysis/results_summary.md, the analysis PNG figures, the key CSV tables
and (optionally) the population-XAI figures, and writes web/results/manifest.json
with Hungarian captions. Commit web/results/ afterwards; the Pages workflow
publishes it on the start page.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = {
    "predicted_vs_chronological_age.png": "Becsült vs. kronológiai életkor (teszthalmaz)",
    "error_by_age_decade.png": "Becslési hiba korcsoportonként",
    "agegap_raw_vs_corrected_panel.png": "Nyers és korrigált AgeGap az életkor függvényében",
    "agegap_by_superclass_present_absent.png": "Korrigált AgeGap diagnosztikai címkék szerint",
    "agegap_quartile_prevalence.png": "Diagnózisok gyakorisága AgeGap-kvartilisenként",
    "training_curve.png": "Tanítási görbe (validációs MAE)",
    "lead_importance_by_group.png": "Elvezetések hozzájárulása AgeGap-csoportonként (IG)",
    "lead_region_enrichment_heatmap.png": "|IG| dúsulás elvezetés × hullámrégió szerint",
    "beat_synchronous_attribution.png": "R-csúcsra szinkronizált átlagos ütés és |IG|",
    "region_signed_contribution.png": "Hullámrégiók előjeles hozzájárulása (év)",
}
TABLES = ["prediction_metrics.csv", "population_table.csv", "diagnostic_group_agegap.csv",
          "correction_methods.csv", "agegap_quartiles.csv", "error_by_age_decade.csv"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run folder containing analysis/")
    p.add_argument("--xai", help="xai_population output folder (optional)")
    p.add_argument("--out", default=str(ROOT / "web" / "results"))
    args = p.parse_args()

    run = Path(args.run).expanduser().resolve()
    analysis = run / "analysis"
    if not (analysis / "results_summary.md").is_file():
        raise SystemExit(f"Nincs results_summary.md: {analysis} – előbb futtasd az evaluate_agegap.py-t.")
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    figures = []
    sources = [analysis] + ([Path(args.xai).expanduser().resolve()] if args.xai else [])
    for folder in sources:
        for name, caption in CAPTIONS.items():
            if (folder / name).is_file():
                shutil.copy2(folder / name, out / name)
                figures.append({"file": name, "caption": caption})
    shutil.copy2(analysis / "results_summary.md", out / "results_summary.md")
    tables = []
    for name in TABLES:
        if (analysis / name).is_file():
            shutil.copy2(analysis / name, out / name)
            tables.append(name)
    manifest = {"run": run.name, "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "figures": figures, "tables": tables}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(figures)} ábra és {len(tables)} tábla → {out}")


if __name__ == "__main__":
    main()
