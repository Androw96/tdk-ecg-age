# Architektúra – fejlesztői dokumentáció

Ez a dokumentum a kód felépítését, az adatfolyamot és a fájlformátumokat írja le.
Felhasználói leírás: [README.md](../README.md) (kutatási pipeline),
[DASHBOARD.md](../DASHBOARD.md) (ECG Age Lab felület) és [UI_DOKUMENTACIO.md](UI_DOKUMENTACIO.md) (a felület változatai). Kísérleti protokoll:
[EXPERIMENTS.md](EXPERIMENTS.md).

## Modultérkép

| Fájl | Szerep | Függőségek |
|---|---|---|
| `ecg_age_common.py` | Metaadat-betöltés, szuperosztályok, tanító kohorsz, WFDB-olvasás, előfeldolgozás, jelcache, augmentáció, `PTBXLAgeDataset`, `ECGAgeResNet`, checkpoint-betöltés | torch, wfdb |
| `train_ecg_age.py` | Tanítás (fold 1–8), early stopping (fold 9), végső predikció (fold 9–10) | common |
| `evaluate_agegap.py` | Metrikák bootstrap CI-vel, bias-korrekció, diagnosztikai statisztika, ábrák, `results_summary.md` | common, statsmodels |
| `ecg_beats.py` | R-csúcs-detektálás, ütésrégiók (P/PR/QRS/ST/T), ütéssablonok | csak numpy/scipy |
| `xai_population.py` | Populációs Integrated Gradients, régiódúsulás, sanity check | common, captum, ecg_beats |
| `explain_integrated_gradients.py` | Egyetlen EKG IG-ábrái | common, captum |
| `ensemble_runs.py` | Több seed összevonása, seed-variabilitás | pandas |
| `predict_ecg_age.py` | Meglévő checkpoint újraértékelése (a felület „Modelltesztek” oldala hívja) | dashboard.model_service |
| `dashboard_app.py` | Streamlit felület (ECG Age Lab) | streamlit, plotly |
| `dashboard/model_service.py` | Biztonságos checkpoint-betöltés, egyedi EKG-olvasás és -becslés | common (lustán importált torch) |
| `dashboard/jobs.py` | Háttérfolyamatok (tanítás, teszt, elemzés) indítása / leállítása, shell nélkül | – |
| `dashboard/analytics.py` | Feltöltött predikciós CSV-k elemzése, export (PyTorch nélkül) | pandas |
| `run_experiments.sh` | Az E1–E7 kísérletek parancsai | – |
| `deploy/streamlit_cloud/streamlit_app.py` | Community Cloud belépési pont (`cloud` profil) | dashboard_app |
| `scripts/build_demo_bundle.py` | Példacsomag a hosztolt felületekhez (`demo_data/`, `web/examples/`) | model_service |
| `web/build_site.py`, `web/collect_results.py` | GitHub Pages oldal és eredménygyűjtés | markdown |

## Adatfolyam

```
PTB-XL (ptbxl_database.csv, scp_statements.csv, records100/ | records500/)
   │  load_ptbxl_metadata: 18–89 év, érvényes strat_fold
   │  add_diagnostic_superclasses: NORM/MI/STTC/CD/HYP (+ strict_norm)
   ▼
SignalCache (cache/ptbxl_<fs>hz.npy, float32 mV, kanonikus elvezetés-sorrend)
   ▼
PTBXLAgeDataset ─► [ECGAugment, csak train] ─► preprocess() ─► ECGAgeResNet
   ▼
train_ecg_age.py ─► runs/<név>/ best_model.pt, predictions.csv, metrics.json, config.json, training_history.csv
   ├─► ensemble_runs.py ─► runs/<ens>/predictions.csv, seed_variability*.csv
   ├─► evaluate_agegap.py ─► runs/<név>/analysis/ (táblák, ábrák, results_summary.md)
   └─► xai_population.py ─► runs/<név>/xai_population/
```

Felosztás: fold 1–8 tanítás, fold 9 validáció (early stopping + AgeGap-korrekció),
fold 10 teszt (csak végső értékelés). A PTB-XL foldjai betegszinten diszjunktak;
a `dashboard/model_service.py` ezt minden betöltéskor ellenőrzi.

## Előfeldolgozás (`preprocess`)

| Mód | Leírás |
|---|---|
| `clip5` (alap) | ±5 mV klippelés, osztás 5-tel – az eredeti baseline |
| `record_zscore` | klippelés után rekordonkénti, elvezetésenkénti z-score |
| `lead_standardize` | klippelés után a tanítóhalmazon becsült elvezetésenkénti átlag/szórás |

A mód és a statisztikák a checkpointba kerülnek, így a felület és minden szkript
ugyanazt a transzformációt alkalmazza.

## Checkpoint-formátum (`best_model.pt`)

```python
{
  "model_state": state_dict,          # kötelező
  "sampling_rate": 100 | 500,         # kötelező
  "epoch": int,
  "val_metrics": {...},
  "model_config": {"dropout", "widths", "blocks_per_stage"},  # v2; hiányában az eredeti architektúra
  "preprocessing": {"mode", "clip_mv", ["lead_mean", "lead_std"]},  # v2; hiányában clip5
  "train_cohort": "all" | "norm",     # v2
  "train_mean_age": float,            # v2
}
```

Csak primitív típusokat tartalmaz, ezért `torch.load(..., weights_only=True)`-val
biztonságosan betölthető. A v1 checkpointok változtatás nélkül működnek.

## Modell

`ECGAgeResNet` alapértelmezésben az eredeti baseline: stem (Conv1d 12→64, k=15, s=2),
4 szakasz × 2 reziduális blokk (64-128-256-512 csatorna, szakaszonként stride 2),
globális átlag-pooling, dropout, lineáris kimenet – 7 345 281 paraméter
(teszt ellenőrzi). A kimeneti bias a tanítóhalmaz átlagéletkorára inicializálódik.

## `predictions.csv` oszlopai (v2)

`ecg_id, age, predicted_age, split, patient_id, sex, in_training_cohort, NORM, MI, STTC, CD, HYP, n_superclasses, strict_norm, age_gap_raw`

## Tesztek

| Fájl | Mit ellenőriz | Kell hozzá |
|---|---|---|
| `tests/test_ecg_beats.py` | R-csúcs-detektálás, régiócímkék szimulált EKG-n | numpy, scipy |
| `tests/test_research_pipeline.py` | Teljes pipeline szintetikus WFDB-adaton: tanítás (norm kohorsz, augmentáció, lead_standardize, cosine), felületi kiértékelés, AgeGap-elemzés, populációs XAI, egyedi IG, ensemble; régi checkpoint kompatibilitása | torch, wfdb, captum, statsmodels |
| `tests/test_dashboard_*.py` | A felület modellszolgáltatása, háttérfeladatai, analitikája és UI-ja | streamlit |

Futtatás: `.venv/bin/python -m pytest -q` vagy macOS-en dupla kattintás a
`run_tests.command` fájlra (eredmény: `tests_output.log`).

A szintetikus tesztek eredményei **nem kutatási eredmények**.
