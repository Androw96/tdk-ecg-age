#!/usr/bin/env bash
# TDK kísérleti terv (E1–E7) – futtasd a projekt mappájából:
#   DATA=/UTVONAL/ptb-xl-...-1.0.3 ./run_experiments.sh E1
# Egyszerre egy kísérletet indíts; mindegyik külön runs/<név> mappába ír.
# Apple Silicon (MPS) gépen egy 100 Hz-es tanítás nagyságrendileg 20–60 perc
# (az első futás a jelcache-t is felépíti, ~1 GB a cache/ mappában).
set -euo pipefail
: "${DATA:?Állítsd be a DATA változót a PTB-XL gyökérkönyvtárára}"
PY=${PY:-.venv/bin/python}
train() { local name=$1; shift; "$PY" train_ecg_age.py --data-dir "$DATA" --output-dir "runs/$name" "$@"; }
analyse() { "$PY" evaluate_agegap.py --data-dir "$DATA" --predictions "runs/$1/predictions.csv" --output-dir "runs/$1/analysis"; }
xai() { "$PY" xai_population.py --data-dir "$DATA" --checkpoint "runs/$1/best_model.pt" \
          --agegap-csv "runs/$1/analysis/test_agegap_with_labels.csv" --output-dir "runs/$1/xai_population" --sanity-random; }

case "${1:-}" in
  E1)  # baseline, 3 seed + ensemble (seed-variabilitás a dolgozatba)
       for s in 42 43 44; do train "E1_all_seed$s" --seed "$s"; done
       "$PY" ensemble_runs.py --runs runs/E1_all_seed42 runs/E1_all_seed43 runs/E1_all_seed44 --output-dir runs/E1_all_ens
       analyse E1_all_ens ;;
  E1aug)  # baseline + augmentáció + cosine LR (modellfejlesztési lépés)
       for s in 42 43 44; do train "E1aug_seed$s" --seed "$s" --augment --scheduler cosine --epochs 40 --patience 10; done
       "$PY" ensemble_runs.py --runs runs/E1aug_seed42 runs/E1aug_seed43 runs/E1aug_seed44 --output-dir runs/E1aug_ens
       analyse E1aug_ens ;;
  E2E3) analyse "${2:?add meg a futás nevét, pl. E1_all_seed42}" ;;  # bias-korrekció + diagnosztikai elemzés
  E4)  xai "${2:?add meg a futás nevét}" ;;                            # populációs XAI (egyedi modell kell, nem ensemble)
  E5)  train E5_500hz --sampling-rate 500 --batch-size 32 && analyse E5_500hz ;;
  E6)  for m in record_zscore lead_standardize; do train "E6_$m" --preprocessing "$m" && analyse "E6_$m"; done ;;
  E7)  # egészséges referencia-modell: csak kizárólag NORM EKG-kon tanítva
       for s in 42 43 44; do train "E7_norm_seed$s" --seed "$s" --train-cohort norm; done
       "$PY" ensemble_runs.py --runs runs/E7_norm_seed42 runs/E7_norm_seed43 runs/E7_norm_seed44 --output-dir runs/E7_norm_ens
       analyse E7_norm_ens ;;
  *) echo "Használat: DATA=... $0 {E1|E1aug|E2E3 <run>|E4 <run>|E5|E6|E7}"; exit 1 ;;
esac
