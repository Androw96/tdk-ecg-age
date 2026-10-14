# TDK – ECG Age / AgeGap baseline (PTB-XL)

Munkacím: **Biológiai életkor és egészségi állapot becslése 12 elvezetéses EKG-jelek mélytanulási elemzésével**

Ez a projekt egy reprodukálható kiindulási pipeline a TDK saját kísérleti részéhez.

> **Adatok és modellek nincsenek a repóban.** A PTB-XL-t a PhysioNetről kell letölteni
> (lásd lent); a betanított modellek (`runs/`) és a jelcache (`cache/`) helyben készülnek.

## Dokumentáció

| Dokumentum | Tartalom |
|---|---|
| [README.md](README.md) | Telepítés, kutatási pipeline, parancsok |
| [DASHBOARD.md](DASHBOARD.md) | Az ECG Age Lab helyi felület használata |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Kísérleti protokoll (E1–E7), kimenetek → dolgozat fejezetei |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Modulok, adatfolyam, checkpoint-formátum, tesztek |
| [CHANGELOG.md](CHANGELOG.md) | Verziók és változások |

## Repó felépítése

```
ecg_age_common.py          adat, előfeldolgozás, cache, augmentáció, modell
train_ecg_age.py           tanítás
evaluate_agegap.py         AgeGap-elemzés, statisztika, ábrák
xai_population.py          populációs Integrated Gradients
ecg_beats.py               R-csúcs-detektálás, ütésrégiók
explain_integrated_gradients.py   egyedi EKG magyarázata
ensemble_runs.py           több seed összevonása
predict_ecg_age.py         checkpoint újraértékelése
run_experiments.sh         E1–E7 kísérletek
dashboard_app.py, dashboard/      ECG Age Lab (Streamlit)
tests/                     pytest tesztek (szintetikus adat)
docs/                      fejlesztői és kísérleti dokumentáció
```

## Magyar kezelőfelület

A **ECG Age Lab** helyi felületen egyedi EKG-vizsgálat, Integrated Gradients,
validációs és tesztkiértékelés, szűrhető kimutatások és CSV/JSON/HTML riportok
érhetők el. Tanítások is indíthatók, követhetők és leállíthatók.

macOS-en indítsd a `launch_dashboard.command` fájlt dupla kattintással,
vagy futtasd a projekt mappájából:

```bash
.venv/bin/python -m streamlit run dashboard_app.py
```

A cím: **http://127.0.0.1:8501**. A bemutató módhoz nem kell adatbázis;
valós becsléshez PTB-XL és kompatibilis betanított modell szükséges.
Telepítés, használat, kimenetek: [DASHBOARD.md](DASHBOARD.md).

## Kutatási pipeline

1. PTB-XL v1.0.3 metaadatok és 100 Hz-es, 12 elvezetéses EKG-k betöltése.
2. Felnőtt (18–89 év) minták kiválasztása.
3. Betegszintű hivatalos `strat_fold` felosztás: 1–8 train, 9 validation, 10 test.
4. 1D ResNet regresszió a kronológiai életkor becslésére.
5. Nyers AgeGap: `predicted_age - chronological_age`.
6. Validation seten illesztett lineáris bias-korrekció.
7. Korrigált AgeGap összehasonlítása NORM / MI / STTC / CD / HYP diagnosztikai szuperosztályokkal.
8. Integrated Gradients interpretálhatósági elemzés.

## PTB-XL letöltés

Töltsd le a PTB-XL legfrissebb publikus verzióját a PhysioNetről, majd a projektet így futtasd:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 1. Tanítás

```bash
python train_ecg_age.py \
  --data-dir /UTVONAL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
  --epochs 30 \
  --batch-size 64 \
  --output-dir runs/baseline
```

Apple Silicon gépen a program automatikusan megpróbálja az MPS gyorsítást használni; NVIDIA GPU esetén CUDA-t.

### 2. AgeGap + diagnosztikai elemzés

```bash
python evaluate_agegap.py \
  --data-dir /UTVONAL/PTBXL \
  --predictions runs/baseline/predictions.csv \
  --output-dir runs/baseline/analysis
```

### 3. Integrated Gradients példa

```bash
python explain_integrated_gradients.py \
  --data-dir /UTVONAL/PTBXL \
  --checkpoint runs/baseline/best_model.pt \
  --ecg-id 1 \
  --output-dir runs/baseline/xai
```

## 2026. októberi bővítések (v2)

A régi parancsok és checkpointok változatlanul működnek; az alapértelmezések az
eredeti baseline-t adják. Új elemek:

| Fájl / kapcsoló | Mire jó |
|---|---|
| `cache/` (automatikus) | Az első futás az összes jelet egy memóriába képezett `.npy` fájlba írja; utána egy epoch a WFDB-fájlok újraolvasása nélkül fut (lényegesen gyorsabb). `--no-cache` kikapcsolja. |
| `--train-cohort norm` | **Egészséges referencia-modell (E7):** tanítás és early stopping csak a kizárólag NORM címkéjű EKG-kon. Így a betegségek nem „tanulódnak bele” az életkorbecslésbe, és az AgeGap–diagnózis kapcsolat nem körkörös. |
| `--augment`, `--crop-seconds 8` | Enyhe zaj, alapvonal-vándorlás és globális amplitúdóskálázás (±5%), opcionális véletlen kivágás. |
| `--preprocessing record_zscore / lead_standardize` | Normalizálási ablation (E6). |
| `--scheduler cosine`, `--widths`, `--blocks`, `--dropout`, `--loss` | Modell- és tanítási változatok; a checkpoint a `model_config`-ot is menti. |
| kimeneti bias inicializálás | A kimenet a tanítóhalmaz átlagéletkoráról indul (gyorsabb, stabilabb konvergencia). `--no-bias-init` kikapcsolja. |
| `config.json` | Minden futás összes beállítása, verziók, eszköz – reprodukálhatóság. |
| `predictions.csv` | Már tartalmazza: `patient_id`, `sex`, szuperosztályok, `strict_norm`, `in_training_cohort`. |
| `metrics.json` | Konstans baseline, alcsoport-metrikák (kizárólag NORM vs. kóros). |
| `ensemble_runs.py` | Több seed összevonása, seed-variabilitás (átlag ± SD). |
| `evaluate_agegap.py` | Betegszintű bootstrap 95% CI, konstans baseline + MAE-különbség CI, populációs tábla (6.1), korcsoportos hiba, 3 korrekciós módszer (lineáris, kvadratikus, Beheshti-féle kétlépéses), Cliff-δ, Holm-korrekció, klaszter-robusztus logisztikus regresszió, kvartilis-elemzés, `results_summary.md` magyar táblákkal. |
| `xai_population.py`, `ecg_beats.py` | Populációs IG: elvezetés-fontosság, R-csúcsra szinkronizált P/PR/QRS/ST/T régió-dúsulás, előjeles hozzájárulás években, pozitív vs. negatív AgeGap összehasonlítás, modell-randomizációs sanity check. |
| `run_experiments.sh` | Az E1–E7 kísérletek parancsai egy helyen. |

### Tesztek

```bash
.venv/bin/python -m pytest -q          # összes teszt
.venv/bin/python -m pytest tests/test_research_pipeline.py -q   # új kutatási pipeline, szintetikus adaton
```

### Ajánlott sorrend

```bash
export DATA=/UTVONAL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
./run_experiments.sh E1      # baseline, 3 seed + ensemble + elemzés
./run_experiments.sh E7      # egészséges referencia-modell
./run_experiments.sh E4 E1_all_seed42   # populációs XAI
```

Megjegyzés: a felület (Kimutatások oldal) a korrekciót az összes validációs soron
illeszti; `--train-cohort norm` futásnál a dolgozatba az `evaluate_agegap.py`
eredményeit vedd át (ez a tanító kohorsz validációs sorain illeszt).

## Elsődleges végpontok

- MAE (év)
- RMSE (év)
- R²
- Pearson-r
- `AgeGap_raw`
- `AgeGap_corrected`
- korrigált AgeGap eloszlása diagnosztikai csoportonként
- logisztikus regresszió: diagnosztikai eltérés ~ korrigált AgeGap + kronológiai életkor + nem

## Fontos módszertani megjegyzés

A PTB-XL-ben a 89 év feletti életkorokat anonimizálási okból irreális, 300 év körüli értékek reprezentálhatják, ezért a baseline elemzés 18–89 évre korlátozódik.

A PTB-XL keresztmetszeti adatbázis: ezért a TDK-ban az AgeGap aktuális EKG-fenotípusokkal való kapcsolatát vizsgáljuk. Prospektív jövőbeli kardiovaszkuláris kockázat bizonyításához longitudinális/outcome adatbázis szükséges.
