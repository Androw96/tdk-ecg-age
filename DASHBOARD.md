# ECG Age Lab – TDK kezelőfelület

Helyben futó, magyar nyelvű felület az EKG-életkor modellhez. Egyedi EKG-kat lehet megjeleníteni és kiértékelni, tanított modellt ellenőrizni, eredményeket szűrni és exportálni, illetve új tanítási futást indítani.

## Indítás

macOS-en kattints duplán a `launch_dashboard.command` fájlra. A felület címe: **http://127.0.0.1:8501**. A program a projekt `.venv` környezetét használja, és csak a saját gépen figyel hálózati kapcsolatra.

Ha még nincsenek telepítve a csomagok, Python 3.11 vagy újabb környezetben, a projekt mappájából futtasd:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dashboard.txt
./launch_dashboard.command
```

Az indító ellenőrzi a függőségeket, de nem telepít automatikusan. A felülethez Streamlit 1.50 vagy újabb, 2.0-nál korábbi verzió szükséges. A PyTorch és a kutatási csomagok telepítése hosszabb ideig tarthat. Terminálból közvetlenül is indítható:

```bash
.venv/bin/python -m streamlit run dashboard_app.py \
  --server.address=127.0.0.1 --server.port=8501 \
  --browser.gatherUsageStats=false
```

Leállítás előtt a felületen állítsd le a futó modellfeladatot, majd az indító terminálban nyomj **Ctrl+C**-t. A böngészőlap bezárása önmagában nem állítja le a kiszolgálót vagy a modellfeladatot.

## Demó és valódi adatok

A demó szintetikus adatokkal mutatja meg a felület működését. A demóértékek nem egy tanított modell teljesítménymutatói, és nem használhatók a TDK kísérleti eredményeként.

Valódi modellvizsgálathoz a következők szükségesek:

- A helyi PTB-XL adatkönyvtár, benne a `ptbxl_database.csv` metaadatokkal és a megfelelő `records100` vagy `records500` jelfájlokkal. Diagnosztikai AgeGap-elemzéshez a `scp_statements.csv` is szükséges.
- A projekt architektúrájával tanított modell: általában `best_model.pt`.
- Korábbi kimutatások betöltéséhez a futás `predictions.csv` fájlja. Saját CSV helyi elérési úttal vagy a feltöltővel is megadható.

A projektcsomag nem tartalmazza a PTB-XL teljes adatállományát vagy előtanított modelljét. Modellfájl nélkül a demó és a már meglévő predikciós CSV-k kimutatásai használhatók. Valódi egyedi EKG-vizsgálathoz tanított modellfájl kell. Egy véletlenül inicializált modell kimenete nem jelenik meg valódi eredményként.

## Egyedi EKG vizsgálata

Az **EKG-vizsgálat** oldalon válassz rekordot és jelenítsd meg a 12 elvezetés jelét. Valódi adatmódban add meg az adatkönyvtárat és a megfelelő modellfájlt. A modellbecslés mellett az ismert kronológiai életkor és a nyers AgeGap is ellenőrizhető.

A **Csak jel megjelenítése (modell nélkül)** kapcsolóval betanított modell nélkül is vizsgálhatsz PTB-XL rekordot; válassz 100 vagy 500 Hz-es mintavételt. Ebben a módban nem készül életkorbecslés. A megjelenített, ±5 mV-re korlátozott jel időbélyeges CSV-ként is letölthető. Modellbecslésnél az **Integrated Gradients** kapcsoló elvezetésenkénti hozzájárulást és időbeli hőtérképet kapcsol be; a JSON-export az összesített hozzájárulásokat is tartalmazza.

A rekordválasztás a PTB-XL `ecg_id` azonosítóira és a metaadatban hivatkozott WFDB-jelfájlokra épül. A modellfájlban tárolt mintavételi frekvencia választja ki a 100 vagy 500 Hz-es jelet. A program ellenőrzi a 12 elvezetést, azok sorrendjét és a mintavételi frekvenciát; a tanítással azonos, ±5 mV-ra korlátozott és 5-tel osztott jelet adja a modellnek. Tetszőleges kép vagy PDF-EKG közvetlen kiértékelése nem része ennek a felületnek.

A nyers AgeGap számítása: `becsült EKG-életkor − kronológiai életkor`. Pozitív érték esetén a modell az ismert életkornál idősebb életkort becsült. A korrigált AgeGap külön, a validációs mintán illesztett korrekciót igényel. A felület kutatási modellkimeneteket mutat; az EKG-életkor nem igazolt biológiai életkor vagy egyéni diagnózis.

Az **Integrated Gradients** kapcsolóval elvezetésenkénti hozzájárulás és időbeli hőtérkép kérhető. A **Vizsgálati eredmény letöltése · JSON** gomb a rekord azonosítóját, az életkorbecslést, az AgeGap-et és a vizsgálat metaadatait menti. A JSON a jelgörbe és a teljes attribúciós tömb nélkül készül.

## Modelltesztelés és tanítás

A **Modelltesztek** oldalon egy meglévő modellfájlt lehet a PTB-XL validációs és teszthalmazán újra kiértékelni. A rekordkorlát gyors működésellenőrzésre szolgál: a korlát külön-külön vonatkozik a validációs és teszthalmazra. A `0` érték a teljes halmazt jelenti. Részmintás eredményeket ne hasonlíts közvetlenül teljes teszthalmazon mért eredményekhez.

A **Kísérletek** oldalon indítható tanítás. A modell a hivatalos PTB-XL felosztást használja: az 1–8. fold tanítás, a 9. validáció, a 10. teszt. A baseline 18–89 éves rekordokat tart meg. A tanítás végén a legjobb validációs eredményhez tartozó modell kerül tesztelésre.

Egyszerre egy háttérfeladat futhat. A feladat állapota és naplójának vége megtekinthető; a teljes napló a futás könyvtárában található. A leállítás csak a felületről indított aktuális folyamatot állítja meg. A félbeszakadt futás könyvtára megmarad, de a részleges fájlok nem jelentenek sikeresen befejezett kísérletet.

A kiszolgáló újraindítása után a korábbi futások fájljai megmaradnak, az aktív folyamatkezelés azonban az aktuális kiszolgáló munkamenetéhez tartozik. A futástörténeti `job.json` az utolsó ellenőrzött állapotot őrzi; váratlan kiszolgáló-leállás után ezt a naplóval és a fájlokkal együtt kell értelmezni.

## Kimutatások és AgeGap

A **Kimutatások** oldalon válassz egy korábbi futásból származó CSV-t, adj meg helyi CSV-elérési utat, vagy válaszd a **CSV feltöltése** lehetőséget. A predikciós táblázat két kötelező oszlopa:

```text
age,predicted_age
```

A CSV vesszővel elválasztott, UTF-8 kódolású legyen; az `age` értékei 18–89 év közé essenek. Ajánlott további oszlopok: `ecg_id`, `patient_id`, `split`, `sex`, illetve a `NORM`, `MI`, `STTC`, `CD`, `HYP` diagnosztikai címkék.

A `split` megengedett értékei: `train`, `validation`, `test`, `unspecified`. Ha ez az oszlop hiányzik, a program `unspecified` halmazként jeleníti meg az adatokat. A validációs sorok a korrekció illesztésére szolgálnak, a tesztsorok a független ellenőrzésre. Korrigált AgeGap-hez legalább három validációs rekord és legalább két különböző életkor szükséges; a szűrés csak az illesztés után történik. A korrekciót nem szabad a teszthalmazra illeszteni.

Az `ecg_id` szükséges a rekordok azonosításához, a PTB-XL címkék hozzárendeléséhez és két modell azonos EKG-kon történő összehasonlításához. A `patient_id` és `split` alapján ellenőrizhető a betegszintű átfedés; azonosítók nélkül a feltöltött CSV-ből ez nem állapítható meg. A **Kísérletek** oldal teljes diagnosztikai AgeGap-elemzéséhez `ecg_id` és `split` is kell, valamint validációs és tesztsorok egyaránt szükségesek.

Az AgeGap-elemzés a `evaluate_agegap.py` kutatási programot futtatja a kiválasztott predikciós CSV-vel és adatkönyvtárral. A diagnosztikai csoporteredmények a PTB-XL címkékkel való kapcsolatra vonatkoznak; prospektív kardiovaszkuláris kockázatot ezek az adatok önmagukban nem igazolnak.

A **Szűrt eredmények · CSV** gomb a kiválasztott rekordokat menti. A **Teljes kimutatás · ZIP** gomb ugyanezekhez a szűrt rekordokhoz három fájlt csomagol: `predictions.csv`, `metrics.json` és böngészőben megnyitható `summary.html`. A JSON és HTML a metrikák mellett a forrást, az alkalmazott szűrőket, a korrekciót és a demójelölést is tartalmazza. A ZIP nem a teljes eredeti adatállományt, hanem a jelenlegi szűrés eredményét exportálja.

## Mentett fájlok

Minden új feladat egyedi könyvtárat kap a projekt `runs` mappájában. Meglévő eredménykönyvtár nem kerül felülírásra.

| Feladat | Fő kimenetek |
| --- | --- |
| Minden háttérfeladat | `process.log`, `job.json` |
| Tanítás | `best_model.pt`, `training_history.csv`, `predictions.csv`, `metrics.json` |
| Modelltesztelés | `predictions.csv`, `metrics.json` |
| AgeGap-elemzés | `test_agegap_with_labels.csv`, `diagnostic_group_agegap.csv`, `logistic_regression.json`, `agegap_summary.json`, PNG-ábrák |

A teljes folyamat sorrendje: **tanítás → modellteszt vagy a tanítás predikciói → AgeGap-elemzés → kimutatások és export**. Egyedi EKG-becslés bármely kompatibilis tanított modellfájllal külön is elvégezhető.

## Hibakeresés és ellenőrzés

- **Hiányzó adatok:** az adatkönyvtár közvetlenül tartalmazza a `ptbxl_database.csv` fájlt. A metaadat-CSV önmagában nem elegendő jelbetöltéshez.
- **Hiányzó modell:** először taníts modellt, vagy válassz egy korábbi `best_model.pt` fájlt.
- **Memóriahiba:** csökkentsd a batch méretét; modelltesztnél kisebb rekordkorláttal is ellenőrizheted a működést.
- **Gyorsítóhiba:** modelltesztnél válaszd a CPU-t; csak a ténylegesen elérhető gyorsítót használd.
- **Sikertelen háttérfeladat:** a pontos kivétel a felületen megjelenített naplóban és a `process.log` fájlban olvasható.
- **Foglalt 8501-es port:** állítsd le az előző kiszolgálót, vagy a terminálos indításnál válassz másik `--server.port` értéket.

A helyi ellenőrzések futtatása a projekt könyvtárából, a fejlesztői tesztfuttató egyszeri telepítésével:

```bash
.venv/bin/python -m pip install pytest
.venv/bin/python -m pytest tests -q
```

A háttérfeladat-kezelő tesztjei apró ideiglenes programokat indítanak. Nem tanítanak modellt, és nem igényelnek PTB-XL adathalmazt.
