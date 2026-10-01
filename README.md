# TDK – ECG Age / AgeGap baseline (PTB-XL)

Munkacím: **Biológiai életkor és egészségi állapot becslése 12 elvezetéses EKG-jelek mélytanulási elemzésével**

Ez a projekt egy reprodukálható kiindulási pipeline a TDK saját kísérleti részéhez.

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
