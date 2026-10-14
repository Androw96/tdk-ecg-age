# Változásnapló

## v2 – 2026. október

### Modell és tanítás
- Memóriába képezett jelcache (`cache/`), a WFDB-fájlok epochonkénti újraolvasása nélkül.
- Egészséges referencia-modell: `--train-cohort norm`.
- Augmentáció (`--augment`, `--crop-seconds`), normalizálási módok (`--preprocessing`).
- Konfigurálható architektúra (`--widths`, `--blocks`, `--dropout`), veszteség (`--loss`),
  ütemező (`--scheduler cosine`); kimeneti bias a tanító átlagéletkorra inicializálva.
- `config.json` minden futáshoz; bővített `predictions.csv` és `metrics.json`
  (konstans baseline, alcsoport-metrikák).
- `ensemble_runs.py`: több seed összevonása, seed-variabilitás.

### Kiértékelés
- `evaluate_agegap.py`: betegszintű bootstrap CI, populációs tábla, korcsoportos hiba,
  három korrekciós módszer, Cliff-δ, Holm, klaszter-robusztus logisztikus regresszió,
  kvartilis-elemzés, `results_summary.md`.

### XAI
- `ecg_beats.py`, `xai_population.py`: populációs IG, R-csúcs-szinkron régiók,
  előjeles hozzájárulás években, modell-randomizációs sanity check.

### Felület és kompatibilitás
- `dashboard/model_service.py` és `predict_ecg_age.py` kezeli a v2 checkpointok
  `model_config` / `preprocessing` mezőit; a v1 checkpointok változatlanul működnek.

### Dokumentáció és tesztek
- `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, `run_experiments.sh`, `run_tests.command`.
- Új tesztek: `tests/test_ecg_beats.py`, `tests/test_research_pipeline.py`.

## v1 – 2026. szeptember
- Baseline pipeline (1D ResNet, AgeGap, lineáris korrekció, IG) és az ECG Age Lab felület.
