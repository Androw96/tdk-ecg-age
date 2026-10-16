# ECG Age Lab – UI dokumentáció

Ez a dokumentum a projekt felhasználói felületének (ECG Age Lab) mindhárom változatát
írja le: mit tudnak, hogyan épülnek fel, hogyan kell őket telepíteni és frissíteni.
A helyi felület részletes kezelési útmutatója a [DASHBOARD.md](../DASHBOARD.md);
a kód felépítése az [ARCHITECTURE.md](ARCHITECTURE.md).

## Tartalom

1. [Áttekintés](#1-áttekintés)
2. [Funkciók változatonként](#2-funkciók-változatonként)
3. [Hogyan működik: profilok](#3-hogyan-működik-profilok)
4. [Példacsomag készítése valós eredményekből](#4-példacsomag-készítése-valós-eredményekből)
5. [GitHub Pages: kezdőoldal, eredmények, böngészős alkalmazás](#5-github-pages-kezdőoldal-eredmények-böngészős-alkalmazás)
6. [Streamlit Community Cloud: online alkalmazás élő modellel](#6-streamlit-community-cloud-online-alkalmazás-élő-modellel)
7. [Helyi munkafelület](#7-helyi-munkafelület)
8. [Frissítés új eredmények után](#8-frissítés-új-eredmények-után)
9. [Láthatóság, adatvédelem, licenc](#9-láthatóság-adatvédelem-licenc)
10. [Hibakeresés](#10-hibakeresés)
11. [Fájlok jegyzéke](#11-fájlok-jegyzéke)

---

## 1. Áttekintés

Ugyanaz a `dashboard_app.py` három környezetben fut. A környezetet a **profil** határozza meg:

| Változat | Hol fut | Profil | Kinek |
|---|---|---|---|
| **Böngészős alkalmazás** | GitHub Pages, a látogató böngészőjében (stlite + Pyodide, szerver nélkül) | `browser` | Bárkinek, linkkel; bemutatásra |
| **Online alkalmazás élő modellel** | Streamlit Community Cloud szerverén | `cloud` | Témavezető, zsűri, meghívott nézők |
| **Helyi munkafelület** | A saját Macen | `local` | Kutatómunka: tanítás, saját adatok |

Mellette a GitHub Pages egy **statikus kezdőoldalt** is közzétesz a projekt leírásával, az
eredményábrákkal és táblákkal (ha már vannak), valamint ezzel a dokumentációval.

```
                ┌──────────── GitHub repó (main) ────────────┐
                │ dashboard_app.py  dashboard/  web/  docs/   │
                │ demo_data/ (modell + példa-EKG-k)           │
                └──────┬──────────────────────┬──────────────┘
      push → GitHub Actions                   │ Community Cloud figyeli a repót
                       ▼                      ▼
   GitHub Pages (statikus)            Streamlit Community Cloud (szerver)
   ├─ index.html  kezdőlap+eredmények  └─ deploy/streamlit_cloud/streamlit_app.py
   ├─ docs/*.html dokumentáció            profil: cloud, PyTorch CPU, élő becslés + IG
   └─ app/        stlite → dashboard_app.py
                  profil: browser, előre kiszámított példák
```

## 2. Funkciók változatonként

| Funkció | Böngészős (Pages) | Online (Cloud) | Helyi |
|---|:---:|:---:|:---:|
| Bemutató mód (szintetikus adatok) | ✓ | ✓ | ✓ |
| Valós PTB-XL példa-EKG-k megjelenítése | ✓ (előre számolt) | ✓ | ✓ (bármely EKG) |
| Életkorbecslés | előre kiszámított | **élő**, a szerveren | élő |
| Integrated Gradients | előre kiszámított (32 lépés) | **élő**, választható lépésszám | élő |
| Csak jel megjelenítése, 100/500 Hz | – | ✓ | ✓ |
| Kimutatások a példamodell eredményeiből | ✓ | ✓ | ✓ (bármely futás) |
| Saját `predictions.csv` feltöltése, szűrés, export (CSV/ZIP) | ✓ | ✓ | ✓ |
| Két eredményfájl összehasonlítása | ✓ | ✓ | ✓ |
| Helyi elérési utak, saját PTB-XL | – | – | ✓ |
| Modelltesztek, tanítás, háttérfolyamatok | – | – | ✓ |
| Kell hozzá telepítés | nem | nem | igen |
| Első betöltés | 20–60 s (Python letöltése a böngészőbe) | gyors (alvó app ébresztése lassabb) | gyors |

A feltöltött CSV-k a böngészős változatban **el sem hagyják a látogató gépét**; a Cloud
változatban a szerver memóriájában dolgozódnak fel, nem mentődnek.

## 3. Hogyan működik: profilok

A `dashboard_app.py` az indításkor kiolvassa a profilt:

| Beállítás | Jelentés | Alapérték |
|---|---|---|
| `ECG_AGE_PROFILE` | `local`, `cloud` vagy `browser` | `local`; Pyodide alatt (böngésző) automatikusan `browser` |
| `ECG_AGE_DEMO_DIR` | a Cloud példacsomag mappája | `demo_data/` |
| `ECG_AGE_WEB_DIR` | a böngészős példacsomag mappája | `web/examples/` |

Mit változtat a profil:

- **`local`**: minden marad a régiben – négy oldal, saját mappák és modellfájlok, háttérfolyamatok.
- **`cloud`**: csak az *EKG-vizsgálat* és a *Kimutatások* oldal; az adatforrás *Bemutató* vagy
  *Valós példák*. A valós példák a `demo_data/` mappából jönnek (PTB-XL részlet + `best_model.pt`),
  a becslés és az IG élőben, CPU-n fut. Elérési utat nem lehet megadni, tanítás nem indítható.
- **`browser`**: ugyanez a két oldal, de PyTorch nélkül: a *Valós példák* a `web/examples/`
  mappa előre kiszámított jeleit, becsléseit és IG-térképeit mutatják.

Ha egy hosztolt változatban a példacsomag hiányzik, a felület automatikusan csak a
*Bemutató* módot kínálja – így a Pages-oldal valódi eredmények nélkül is működik.

## 4. Példacsomag készítése valós eredményekből

A két hosztolt változat egy **betanított modell** eredményeiből dolgozik. Ha egy futás
elkészült (pl. `./run_experiments.sh E1`), a Macen, a projekt mappájából:

```bash
.venv/bin/python scripts/build_demo_bundle.py \
    --data-dir /UTVONAL/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3 \
    --run runs/E1_all_seed42 \
    --n 40
```

| Kapcsoló | Jelentés |
|---|---|
| `--run` | egyedi futás mappája (`best_model.pt` + `predictions.csv`; ensemble-hez nincs modell) |
| `--n` | példa-EKG-k száma (alap 40; csak teszt, fold 10, életkor-évtizedenként elosztva) |
| `--ig-steps` | IG-lépések a böngészős változathoz (alap 32) |
| `--no-ig` | IG nélkül (kisebb, gyorsabb) |

Eredmény:

- `demo_data/` – a Cloud változathoz: `ptbxl/ptbxl_database.csv`, `ptbxl/scp_statements.csv`,
  a kiválasztott EKG-k WFDB-fájljai, `best_model.pt` (~30 MB), `predictions.csv`, `metrics.json`,
  `examples.csv`, `provenance.json` (forrás futás, modell SHA-256) és `README.md` (licenc).
- `web/examples/` – a böngészős változathoz: `examples.npz` (jelek mV-ban és IG, float16,
  40 EKG-ra ~2 MB), `examples.json`, `predictions.csv`, `metrics.json`.

A szkript ellenőrzi, hogy az élő becslés egyezik-e a `predictions.csv` értékével (eltérésnél figyelmeztet).
Mindkét mappát commitold (`.gitignore` a `demo_data/best_model.pt`-t szándékosan engedi):

```bash
git add demo_data web/examples && git commit -m "UI: példacsomag (E1_all_seed42)" && git push
```

## 5. GitHub Pages: kezdőoldal, eredmények, böngészős alkalmazás

### 5.1 Egyszeri beállítás

1. **Láthatóság.** Ingyenes GitHub-fiókkal a Pages csak **publikus** repóból működik; privát
   repóból GitHub Pro kell (diákként a GitHub Education csomaggal elérhető). A közzétett oldal
   **mindkét esetben nyilvános**. Lásd a [9. fejezetet](#9-láthatóság-adatvédelem-licenc).
2. **A workflow bemásolása** (egyszer). A GitHub csak a `.github/workflows/` mappából futtat munkafolyamatot;
   a fájl forrása a `deploy/github_pages/pages.yml`:
   ```bash
   mkdir -p .github/workflows && cp deploy/github_pages/pages.yml .github/workflows/pages.yml
   git add .github/workflows/pages.yml && git commit -m "Pages workflow"
   ```
3. GitHub → a repó → **Settings → Pages → Build and deployment → Source: GitHub Actions**.
4. Pushold a `main` ágat (vagy Actions fül → *GitHub Pages* → *Run workflow*).
5. Pár perc múlva az oldal címe: `https://<felhasználónév>.github.io/<repó>/`
   (ennél a repónál: `https://androw96.github.io/tdk-ecg-age/`). A pontos címet az Actions-futás
   *deploy* lépése és a Settings → Pages oldal is mutatja.

### 5.2 Mi történik pusholáskor

A `.github/workflows/pages.yml` (forrása: `deploy/github_pages/pages.yml`) minden `main`-re pusholáskor lefut:

1. telepíti a `markdown` csomagot,
2. `python web/build_site.py --out _site` – felépíti az oldalt,
3. feltölti és közzéteszi a `_site` mappát.

| Útvonal | Tartalom |
|---|---|
| `/` | kezdőlap: projektleírás, a három változat, eredmények, dokumentáció |
| `/app/` | a böngészős ECG Age Lab |
| `/docs/*.html` | README, DASHBOARD, UI dokumentáció, kísérleti protokoll, architektúra, változásnapló |
| `/results/` | ábrák és CSV-táblák (ha a `web/results/` létezik) |

### 5.3 A böngészős alkalmazás technikai részletei

- A `app/index.html` a [stlite](https://github.com/whitphx/stlite) `@stlite/browser` csomagját tölti be
  a jsDelivr CDN-ről (verzió: `STLITE_VERSION` a `web/build_site.py` elején), az pedig Pyodide-ot,
  vagyis WebAssemblyre fordított Pythont.
- A felület fájljai (`dashboard_app.py`, `dashboard/analytics.py`) és a `web/examples/` tartalma a
  böngésző virtuális fájlrendszerébe kerül; a `plotly` a PyPI-ról töltődik (`BROWSER_REQUIREMENTS`).
- A téma a `.streamlit/config.toml`-ból jön, így megegyezik a helyi felületével.
- PyTorch nem fut a böngészőben, ezért itt nincs élő becslés – ezt a Cloud változat adja.

Helyi előnézet (a böngészős app ehhez is internetet igényel a CDN miatt):

```bash
python3 -m pip install markdown
python3 web/build_site.py --out _site
python3 -m http.server -d _site 8000     # majd: http://127.0.0.1:8000
```

### 5.4 Eredmények a kezdőlapon

Egy lefutott elemzés után:

```bash
python3 web/collect_results.py --run runs/E1_all_ens --xai runs/E1_all_seed42/xai_population
git add web/results && git commit -m "Pages: eredmények" && git push
```

A szkript a `results_summary.md`-t, az elemzés és a populációs XAI ábráit, valamint a fő CSV-táblákat
másolja a `web/results/` mappába, magyar ábraaláírásokkal (`manifest.json`). Amíg ez nincs meg, a
kezdőlap jelzi, hogy eredmény még nincs közzétéve.

### 5.5 A Cloud-alkalmazás linkje a kezdőlapon

Ha a Cloud változat elkészült: repó → **Settings → Secrets and variables → Actions → Variables →
New repository variable**: név `CLOUD_APP_URL`, érték az app címe (pl. `https://tdk-ecg-age.streamlit.app`).
A következő pusholás után a kezdőlapon megjelenik az „Online alkalmazás élő modellel” kártya.

## 6. Streamlit Community Cloud: online alkalmazás élő modellel

### 6.1 Előfeltétel

A `demo_data/` mappa legyen a repóban (4. fejezet). Nélküle az app csak bemutató módban fut.

### 6.2 Telepítés

1. [share.streamlit.io](https://share.streamlit.io) → bejelentkezés GitHubbal. Privát repóhoz a
   Streamlit külön engedélyt kér (`repo` hozzáférés) és egy csak olvasható deploy kulcsot hoz létre.
2. **Create app** → *Deploy a public app from GitHub* / repó kiválasztása:
   - Repository: `Androw96/tdk-ecg-age`, Branch: `main`
   - Main file path: **`deploy/streamlit_cloud/streamlit_app.py`**
   - App URL: pl. `tdk-ecg-age`
3. **Advanced settings** → Python version: 3.12 (vagy a legújabb támogatott). Titok nem kell.
4. **Deploy.** Az első telepítés több percig tarthat (PyTorch CPU-csomag).

Miért ez a belépési pont: a `deploy/streamlit_cloud/streamlit_app.py` beállítja a `cloud` profilt és
elindítja a szokásos `dashboard_app.py`-t. Mellette a `deploy/streamlit_cloud/requirements.txt` csak a
felülethez szükséges csomagokat és a **CPU-s PyTorch**-ot tartalmazza; a Community Cloud a belépési pont
mappájában keresi először a függőségi fájlt, ezért a gyökér `requirements.txt` (Jupyter, statsmodels stb.)
nem települ. A `.streamlit/config.toml` a repó gyökerében marad, ahogy a Community Cloud elvárja.

### 6.3 Láthatóság

Privát repóból telepített app alapból **privát**: csak a meghívottak látják (App settings → Sharing,
vagy a Share gomb; meghívás e-mail címmel). A Community Cloud egyszerre **egy privát appot** enged.
Publikus repóból telepített app alapból nyilvános.

### 6.4 Működés és korlátok

- A szerver ingyenes, korlátozott erőforrású; a modell CPU-n fut, egy becslés 1 mp alatti, az IG
  a lépésszámtól függően néhány másodperc.
- Hosszabb inaktivitás után az app alvó állapotba kerülhet; ilyenkor az oldalon egy gombbal felébreszthető.
- A repó minden `main`-re pusholása után az app automatikusan frissül.

### 6.5 Alternatíva: Hugging Face Spaces

Ugyanez a belépési pont Hugging Face Spaces-en is használható (Streamlit vagy Docker Space), ha a
Community Cloud korlátai szűkek. Ehhez a Space-be a repó tartalmát kell feltölteni, és a
`deploy/streamlit_cloud/requirements.txt`-t kell `requirements.txt`-ként használni.

## 7. Helyi munkafelület

Változatlan, részletek: [DASHBOARD.md](../DASHBOARD.md). Indítás: dupla kattintás a
`launch_dashboard.command` fájlra, vagy:

```bash
.venv/bin/python -m streamlit run dashboard_app.py --server.address=127.0.0.1
```

A `--server.address=127.0.0.1` miatt csak a saját gépről érhető el. (A beállítás a
`.streamlit/config.toml`-ból átkerült az indítóba, hogy a felhős telepítés is működjön.)
A hosztolt profilok helyben is kipróbálhatók:

```bash
ECG_AGE_PROFILE=cloud   .venv/bin/python -m streamlit run dashboard_app.py --server.address=127.0.0.1
ECG_AGE_PROFILE=browser .venv/bin/python -m streamlit run dashboard_app.py --server.address=127.0.0.1
```

## 8. Frissítés új eredmények után

| Mi változott | Teendő |
|---|---|
| Új / jobb modell | `scripts/build_demo_bundle.py --run runs/<új>` → commit `demo_data`, `web/examples` → push |
| Új elemzés | `web/collect_results.py --run …` → commit `web/results` → push |
| Felület kódja | push – a Pages és a Cloud is automatikusan frissül |
| Dokumentáció | push – a Pages újraépül |

## 9. Láthatóság, adatvédelem, licenc

- **A GitHub Pages oldal nyilvános**, akkor is, ha a repó privát (Pro csomaggal). A közzétett tartalom:
  a felület kódja, a dokumentáció, a példacsomag és – ha összegyűjtötted – az eredményábrák. Leadás előtti,
  még nem publikus eredményeknél gondold át, mit teszel ki (a `web/results/` kihagyható).
- **Betegadat:** a PTB-XL anonimizált, nyilvános adatbázis; a példacsomag csak ennek néhány felvételét
  tartalmazza. Saját, nem nyilvános EKG-t ne tegyél a `demo_data/` vagy `web/examples/` mappába.
- **Licenc:** a PTB-XL CC BY 4.0 licencű; a forrásmegjelölés a kezdőlap láblécében és a
  `demo_data/README.md`-ben szerepel – ezeket ne töröld.
- **Felhasználói feltöltések:** a böngészős változatban a feltöltött CSV a látogató gépén marad; a Cloud
  változatban csak a futó munkamenet memóriájában van.
- **Orvosi felelősség:** minden felület jelzi, hogy kutatási célú becslésről van szó, és az AgeGap nem diagnózis.

## 10. Hibakeresés

| Jelenség | Ok és teendő |
|---|---|
| A Pages-oldal 404 | Settings → Pages forrása legyen „GitHub Actions”; nézd meg az Actions fül futását. |
| Nincs „GitHub Pages” futás az Actions fülön | A workflow nincs a `.github/workflows/` mappában (5.1/2. lépés). |
| Az Actions-futás „Pages not enabled” hibával áll le | A 5.1 lépés hiányzik, vagy privát repó ingyenes fiókkal. |
| A böngészős app sokáig tölt | Első indításkor 20–60 s normális; vállalati hálózat tilthatja a jsDelivr CDN-t vagy a PyPI-t. |
| A böngészős appban csak „Bemutató” mód van | Nincs `web/examples/` a repóban (4. fejezet). |
| Cloud: „A checkpoint nem kompatibilis” | A `demo_data/best_model.pt` más kódverzióval készült; építsd újra a csomagot. |
| Cloud: telepítési hiba a PyTorch-nál | Ellenőrizd, hogy a Main file path `deploy/streamlit_cloud/streamlit_app.py`, így a CPU-s requirements települ. |
| Cloud: csak „Bemutató” mód | Hiányzik a `demo_data/best_model.pt`, `demo_data/examples.csv` vagy `demo_data/ptbxl/ptbxl_database.csv`. |
| Helyben a felület más gépről is elérhető | Indíts a `launch_dashboard.command`-dal vagy `--server.address=127.0.0.1` kapcsolóval. |

## 11. Fájlok jegyzéke

| Fájl | Szerep |
|---|---|
| `dashboard_app.py` | a felület; profilkezelés: `PROFILE`, `hosted_sidebar`, `web_record`, `hosted_result_source` |
| `dashboard/analytics.py` | PyTorch nélküli elemzés és export (a böngészős változat is ezt használja) |
| `dashboard/model_service.py` | élő becslés és IG (helyi és Cloud) |
| `deploy/streamlit_cloud/streamlit_app.py` | Community Cloud belépési pont (`cloud` profil) |
| `deploy/streamlit_cloud/requirements.txt` | Cloud függőségek, CPU-s PyTorch |
| `scripts/build_demo_bundle.py` | példacsomag: `demo_data/` és `web/examples/` |
| `web/build_site.py` | a Pages-oldal építése (kezdőlap, dokumentáció, stlite app) |
| `web/collect_results.py` | eredményábrák és -táblák összegyűjtése a kezdőlaphoz |
| `deploy/github_pages/pages.yml` | GitHub Actions workflow a Pages-közzétételhez (bemásolandó a `.github/workflows/` mappába) |
| `tests/test_dashboard_profiles.py` | a `browser` és `cloud` profil tesztjei szintetikus csomagokon |
| `.streamlit/config.toml` | közös téma; a szerver címe az indítóban van |
