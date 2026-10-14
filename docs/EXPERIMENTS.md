# Kísérleti protokoll (E1–E7)

Minden parancs a projekt mappájából fut; előtte:
`export DATA=/út/a/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`

| Kísérlet | Kérdés / hipotézis | Parancs | Dolgozat |
|---|---|---|---|
| E1 – Baseline, 3 seed + ensemble | K1 / H1 | `./run_experiments.sh E1` | 6.1, 6.2 |
| E1-aug – Augmentáció + cosine LR | modellfejlesztés | `./run_experiments.sh E1aug` | 6.2 |
| E2–E3 – Bias-korrekció, diagnosztika | K2–K3 / H2–H3 | `./run_experiments.sh E2E3 <futás>` | 6.3, 6.4 |
| E4 – Populációs XAI | K4 / H4 | `./run_experiments.sh E4 <egyedi futás>` | 6.5 |
| E5 – 500 Hz | ablation | `./run_experiments.sh E5` | 6.2 |
| E6 – Normalizálás | ablation | `./run_experiments.sh E6` | 6.2 |
| E7 – Egészséges referencia-modell | K5 / H5 | `./run_experiments.sh E7` | 6.2, 6.4 |

Minimális csomag: E1, E2–E3, E4 és E7. Az E4-hez egyedi modell kell
(pl. `E1_all_seed42`), ensemble-hez nincs `best_model.pt`.

## Módszertani rögzítések

- Felosztás: hivatalos `strat_fold` – 1–8 tanítás, 9 validáció, 10 teszt; a teszthalmaz
  hiperparaméter-hangolásra nem használható.
- Kohorsz: 18–89 év; E7-ben a tanítás és az early stopping csak a kizárólag NORM EKG-kon.
- Bias-korrekció: kizárólag a validációs foldon illesztve (E7-ben annak NORM sorain);
  elsődleges a lineáris, érzékenységi vizsgálat a kvadratikus és a kétlépéses (Beheshti).
- Bizonytalanság: betegszintű klaszter-bootstrap (2000), klaszter-robusztus logit SE,
  Holm-korrekció az öt szuperosztályra.
- Referencia: PTB-XL benchmark (Strodthoff és mtsai.), 100 Hz ensemble MAE ≈ 7,1 év
  (összes) / 6,8 év (egészséges).

## Kimenetek → dolgozat

| Fájl | Hová |
|---|---|
| `analysis/results_summary.md` | 6.1–6.4. táblák (értékek 95% CI-vel) |
| `analysis/predicted_vs_chronological_age.png` | 6.2. ábra |
| `analysis/error_by_age_decade.png` | 6.2. korcsoportos hiba |
| `analysis/agegap_raw_vs_corrected_panel.png` | 6.3. ábra |
| `analysis/agegap_by_superclass_present_absent.png`, `agegap_quartile_prevalence.png` | 6.4. ábrák |
| `analysis/logistic_sensitivity.csv`, `correction_methods.csv` | 6.3–6.4 érzékenységi vizsgálat (függelék) |
| `seed_variability_summary.csv` | 6.2. seed-variabilitás |
| `xai_population/*.png`, `group_attribution_summary.csv`, `xai_overview.json` | 6.5 |

## Futási idő (tájékoztató)

Apple Silicon (MPS), 100 Hz: egy tanítás nagyságrendileg 20–60 perc; az első futás
a jelcache-t is felépíti (~1 GB a `cache/` mappában). 500 Hz-en kb. ötszörös
adatmennyiség és lassabb tanítás várható.
