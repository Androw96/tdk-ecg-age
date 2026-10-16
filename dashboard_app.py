from __future__ import annotations

import html
import io
import json
import os
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard.analytics import (
    demo_predictions, export_report, load_predictions, metrics,
    prepare_predictions, subgroup_metrics,
)

PROJECT = Path(__file__).resolve().parent
# Deployment profile:
#   local   – full research workstation (default when run on your own machine)
#   cloud   – hosted server (Streamlit Community Cloud): bundled example ECGs + live model, no jobs
#   browser – GitHub Pages via stlite/Pyodide: precomputed examples, no PyTorch, no jobs
PROFILE = os.environ.get("ECG_AGE_PROFILE", "").strip().lower() or (
    "browser" if sys.platform == "emscripten" else "local")
if PROFILE not in {"local", "cloud", "browser"}:
    PROFILE = "local"
LOCAL = PROFILE == "local"
DEMO_DIR = Path(os.environ.get("ECG_AGE_DEMO_DIR", PROJECT / "demo_data")).expanduser()
WEB_DIR = Path(os.environ.get("ECG_AGE_WEB_DIR", PROJECT / "web" / "examples")).expanduser()
PAGES = (["EKG-vizsgálat", "Kimutatások", "Modelltesztek", "Kísérletek"] if LOCAL
         else ["EKG-vizsgálat", "Kimutatások"])
REAL_MODE = "Saját adatok" if LOCAL else "Valós példák"
FOOTER = {
    "local": "ECG Age Lab / TDK · Helyi kutatói munkafelület · A bemutatóadatok nem kutatási eredmények.",
    "cloud": "ECG Age Lab / TDK · Online bemutató (szerver) · PTB-XL példák, CC BY 4.0 · Kutatási célú becslés, nem diagnózis.",
    "browser": "ECG Age Lab / TDK · Böngészős változat (stlite) · Előre kiszámított eredmények · Kutatási célú becslés, nem diagnózis.",
}[PROFILE]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABELS = ["NORM", "MI", "STTC", "CD", "HYP"]
TEAL, BLUE, AMBER = "#087f83", "#4f70ac", "#d39b38"
st.set_page_config(page_title="ECG Age Lab · TDK", page_icon="🫀", layout="wide")


def style():
    st.markdown("""<style>
    .block-container {max-width:1500px;padding-top:4rem;padding-bottom:3rem}
    [data-testid="stSidebar"] {border-right:1px solid #dde5ec}
    h1,h2,h3 {letter-spacing:-.035em;color:#162d3d}
    h1 {font-size:2.35rem!important} h3 {font-size:1.2rem!important}
    [data-testid="stMetric"] {background:white;border:1px solid #e0e7ed;border-radius:12px;padding:16px 20px}
    [data-testid="stMetricLabel"] {color:#637788;font-size:.83rem}
    [data-testid="stMetricValue"] {font-size:1.85rem;font-weight:650}
    .eyebrow {font-size:.72rem;font-weight:750;letter-spacing:.18em;color:#087f83;margin-bottom:.65rem}
    .subtle {color:#6b7f8c;font-size:.9rem;line-height:1.6}
    .brand {font-size:1.5rem;font-weight:750;letter-spacing:-.06em;color:#162d3d}
    .brand span {color:#087f83}
    .badge {display:inline-block;border:1px solid #b8dcda;background:#e4f2ef;color:#16706c;
      padding:4px 10px;border-radius:20px;font-size:.72rem;font-weight:650;margin:5px 0 14px}
    div[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:12px}
    button[kind="primary"] {border-radius:8px}
    </style>""", unsafe_allow_html=True)


@st.cache_resource
def job_manager():
    from dashboard.jobs import JobManager

    return JobManager(PROJECT)


def real_examples_available() -> bool:
    """Whether the hosted profiles have their example bundle (see docs/UI_DOKUMENTACIO.md)."""
    if PROFILE == "cloud":
        return ((DEMO_DIR / "best_model.pt").is_file() and (DEMO_DIR / "ptbxl" / "ptbxl_database.csv").is_file()
                and (DEMO_DIR / "examples.csv").is_file())
    if PROFILE == "browser":
        return (WEB_DIR / "examples.npz").is_file() and (WEB_DIR / "examples.json").is_file()
    return True


@st.cache_data
def web_examples(folder=None):
    folder = Path(folder or WEB_DIR)
    meta = json.loads((folder / "examples.json").read_text(encoding="utf-8"))
    with np.load(folder / "examples.npz") as arrays:
        return meta, {key: arrays[key] for key in arrays.files}


@st.cache_data
def bundle_ids(folder=None):
    return pd.read_csv(Path(folder or DEMO_DIR) / "examples.csv").ecg_id.astype(int).tolist()


def example_ids():
    if PROFILE == "browser":
        return [int(r["ecg_id"]) for r in web_examples(str(WEB_DIR))[0]["records"]]
    return bundle_ids(str(DEMO_DIR))


def web_record(ecg_id, explain):
    """A precomputed real example (browser profile): signal, prediction and optional IG."""
    meta, arrays = web_examples(str(WEB_DIR))
    index = next(i for i, r in enumerate(meta["records"]) if int(r["ecg_id"]) == int(ecg_id))
    rec = meta["records"][index]
    result = {"ecg_id": str(rec["ecg_id"]), "age": float(rec["age"]), "predicted_age": float(rec["predicted_age"]),
              "age_gap_raw": float(rec["predicted_age"]) - float(rec["age"]),
              "sampling_rate": int(meta["sampling_rate"]), "leads": LEADS, "split": rec.get("split", "test"),
              "signal": arrays["signals"][index].astype(np.float32) / 5.0,
              "device": "előre kiszámított becslés", "best_epoch": meta.get("best_epoch", "—"), "precomputed": True}
    if explain and "attributions" in arrays:
        attribution = arrays["attributions"][index].astype(np.float32)
        result.update(attribution=attribution, lead_importance=np.abs(attribution).mean(axis=1),
                      convergence_delta=float(rec.get("ig_convergence_delta", float("nan"))),
                      ig_steps=int(meta.get("ig_steps", 32)))
    return result


def bundled_predictions_path():
    path = (DEMO_DIR if PROFILE == "cloud" else WEB_DIR) / "predictions.csv"
    return path if path.is_file() else None


def chart(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=30, b=25),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
                      font=dict(family="Arial, sans-serif", color="#355165", size=12),
                      colorway=[TEAL, BLUE, AMBER], legend_title_text="",
                      legend=dict(orientation="h", yanchor="top", y=-.25, x=0))
    fig.update_xaxes(gridcolor="#edf1f5", zerolinecolor="#d9e1e7")
    fig.update_yaxes(gridcolor="#edf1f5", zerolinecolor="#d9e1e7")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def number(value, suffix=""):
    return "—" if value is None or not np.isfinite(value) else f"{value:.2f}{suffix}"


def file_version(path):
    try:
        p = Path(path).expanduser()
        s = p.stat()
        return str(p.resolve()), s.st_mtime_ns, s.st_size
    except OSError:
        return path, None, None


def sidebar():
    with st.sidebar:
        st.markdown('<div class="brand">ECG<span> Age Lab</span></div>', unsafe_allow_html=True)
        st.caption("TDK · Kutatói munkafelület")
        st.divider()
        page = st.radio("Munkaterület", PAGES, label_visibility="collapsed", key="page")
        st.divider()
        modes = ["Bemutató", REAL_MODE] if real_examples_available() else ["Bemutató"]
        mode = st.radio("Adatforrás", modes, horizontal=True, key="data_mode")
        if mode == "Bemutató":
            st.caption("Szintetikus példák a felület kipróbálásához.")
        if not LOCAL:
            return page, mode == "Bemutató", *hosted_sidebar(mode)
        st.text_input("PTB-XL adatbázis mappája", key="data_dir",
                      placeholder="/…/ptb-xl-1.0.3", help="A ptbxl_database.csv és records100/records500 mappák helye.")
        checkpoints = sorted((PROJECT / "runs").glob("**/best_model.pt")) if (PROJECT / "runs").exists() else []
        if checkpoints:
            selection = st.selectbox("Megtalált modellek", ["Egyedi elérési út"] + [str(p) for p in checkpoints],
                                     format_func=lambda p: p if p == "Egyedi elérési út" else str(Path(p).relative_to(PROJECT)))
            if selection != "Egyedi elérési út":
                st.session_state.checkpoint = selection
        st.text_input("Modellfájl (.pt)", key="checkpoint", placeholder="/…/best_model.pt")
        device = st.selectbox("Számítás", ["cpu", "auto"],
                              format_func=lambda v: {"cpu": "CPU · biztos kompatibilitás", "auto": "Automatikus · GPU / MPS / CPU"}[v])
        st.divider()
        st.caption("12 elvezetés · 1D ResNet · 18–89 év")
        st.caption("Kutatási célú életkorbecslés. Az AgeGap önmagában nem diagnózis.")
    return page, mode == "Bemutató", st.session_state.data_dir.strip(), st.session_state.checkpoint.strip(), device


def hosted_sidebar(mode):
    """Sidebar tail for the cloud / browser profiles: no local paths, CPU only."""
    if PROFILE == "cloud":
        data_dir, checkpoint = str(DEMO_DIR / "ptbxl"), str(DEMO_DIR / "best_model.pt")
        if mode != "Bemutató":
            st.caption("Válogatott PTB-XL teszt-EKG-k; a becslést a betanított modell a szerveren számolja.")
    else:
        data_dir, checkpoint = "", ""
        if mode != "Bemutató":
            st.caption("Válogatott PTB-XL teszt-EKG-k előre kiszámított becsléssel; a böngészőben nem fut modell.")
    if not real_examples_available():
        st.caption("A valós példacsomag nincs telepítve, ezért csak a bemutató mód érhető el.")
    st.divider()
    st.caption("12 elvezetés · 1D ResNet · 18–89 év")
    st.caption("Kutatási célú életkorbecslés. Az AgeGap önmagában nem diagnózis.")
    st.caption("A teljes munkafelület (tanítás, saját adatok) a helyi telepítésben érhető el.")
    return data_dir, checkpoint, "cpu"


def header(title, subtitle, demo):
    st.markdown('<div class="eyebrow">TDK / EKG-ALAPÚ ÉLETKORBECSLÉS</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<div class="subtle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)
    st.write("")
    if demo:
        st.info("BEMUTATÓ MÓD · A jelgörbék és eredmények szintetikusak. Itt nem fut betanított modell.")


def demo_record(ecg_id):
    df = demo_predictions()
    row = df.loc[df.ecg_id == ecg_id].iloc[0]
    rng = np.random.default_rng(zlib.crc32(str(ecg_id).encode("utf-8")))
    t = np.arange(1000) / 100
    wave = np.zeros_like(t)
    for beat in np.arange(.6, 10, .83):
        for offset, amplitude, width in [(-.18, .12, .028), (-.045, -.12, .012),
                                         (0, 1.1, .012), (.035, -.24, .014), (.26, .28, .065)]:
            wave += amplitude * np.exp(-.5 * ((t - beat - offset) / width) ** 2)
    signal = np.stack([(1 + .18 * np.sin(i)) * wave * (-1 if i == 3 else 1)
                       + .025 * np.sin(2 * np.pi * .25 * t + i) + rng.normal(0, .008, len(t))
                       for i in range(12)]) / 5
    return {"ecg_id": str(ecg_id), "age": float(row.age), "predicted_age": float(row.predicted_age),
            "age_gap_raw": float(row.predicted_age-row.age), "sampling_rate": 100,
            "signal": signal, "leads": LEADS, "device": "szintetikus példa", "synthetic": True}


def record_download(result, checkpoint, data_dir):
    summary = {k: v for k, v in result.items() if k not in {"signal", "attribution", "lead_importance"}}
    summary.update({"checkpoint": None if result.get("synthetic") else checkpoint,
                    "data_dir": None if result.get("synthetic") else data_dir,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "signal_units": "mV after clipping to [-5, 5]", "synthetic": bool(result.get("synthetic", False))})
    if "lead_importance" in result:
        summary["lead_importance"] = dict(zip(result["leads"], np.asarray(result["lead_importance"]).tolist()))
    def serialize(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        return str(v)
    st.download_button("Vizsgálati eredmény letöltése · JSON", json.dumps(summary, ensure_ascii=False, indent=2, default=serialize),
                       f"ecg_{result['ecg_id']}{'_DEMO' if result.get('synthetic') else ''}.json", "application/json")
    signals = pd.DataFrame(np.asarray(result["signal"]).T * 5, columns=result["leads"])
    signals.insert(0, "time_seconds", np.arange(len(signals)) / result["sampling_rate"])
    signals["synthetic"] = bool(result.get("synthetic", False))
    st.download_button("Megjelenített jel · CSV (mV)", signals.to_csv(index=False).encode("utf-8-sig"),
                       f"ecg_{result['ecg_id']}_signal{'_DEMO' if result.get('synthetic') else ''}.csv", "text/csv")


def ecg_page(demo, data_dir, checkpoint, device):
    header("Egy jel. Tizenkét nézőpont.", "Vizsgáld meg az EKG-t, futtass életkorbecslést, és nézd meg, mire figyel a modell.", demo)
    with st.container(border=True):
        st.subheader("Vizsgálat beállítása")
        view_only = st.checkbox("Csak jel megjelenítése (modell nélkül)", value=False,
                                disabled=demo or PROFILE == "browser")
        c1, c2, c3 = st.columns([1, 1, 1.4])
        with c1:
            if demo:
                ids = demo_predictions().ecg_id.tolist()
                ecg_id = st.selectbox("EKG-azonosító", ids[:60], format_func=lambda x: f"EKG #{x}")
            elif not LOCAL:
                ecg_id = st.selectbox("EKG-azonosító (PTB-XL teszt)", example_ids(), format_func=lambda x: f"EKG #{x}")
            else:
                ecg_id = st.number_input("EKG-azonosító (ecg_id)", min_value=1, value=1, step=1)
        with c2:
            explain = st.checkbox("Integrated Gradients", value=False, disabled=demo or view_only,
                                  help="Elvezetésenként megmutatja a bemenet becsléshez való hozzájárulását. Több időt igényel.")
            steps = st.selectbox("IG lépések", [16, 32, 64, 128], index=1,
                                 disabled=not explain or view_only or PROFILE == "browser")
            sampling_rate = st.selectbox("Jel mintavétele (Hz)", [100, 500]) if view_only else 100
        with c3:
            st.caption("A modell a tanításkor használt mintavétellel és előfeldolgozással kapja meg a jelet.")
            run = st.button("Példa megjelenítése" if demo or PROFILE == "browser" else "EKG vizsgálata",
                            type="primary", width="stretch")
    signature = (demo, str(ecg_id), file_version(data_dir), file_version(checkpoint), device, explain, steps, view_only, sampling_rate)
    if demo:
        result = demo_record(ecg_id)
    elif PROFILE == "browser":
        result = web_record(ecg_id, explain)
        steps = result.get("ig_steps", steps)
    else:
        if run:
            st.session_state.pop("record_result", None)
            try:
                if not data_dir:
                    raise ValueError("Add meg az adatbázis mappáját a bal oldalon.")
                if not view_only and not checkpoint:
                    raise ValueError("Add meg az adatbázis mappáját és a modellfájlt a bal oldalon.")
                from dashboard.model_service import inspect_record, predict_record
                with st.spinner("EKG betöltése…" if view_only else "EKG betöltése és modell futtatása…"):
                    if view_only:
                        result = inspect_record(Path(data_dir).expanduser(), int(ecg_id), sampling_rate)
                    else:
                        result = predict_record(Path(data_dir).expanduser(), Path(checkpoint).expanduser(), int(ecg_id),
                                                device_name=device, explain=explain, steps=int(steps))
                        if explain:
                            result["ig_steps"] = int(steps)
                st.session_state.record_result = (signature, result)
            except Exception as exc:
                st.error(f"A vizsgálat nem futott le: {exc}")
        cached = st.session_state.get("record_result")
        result = cached[1] if cached and cached[0] == signature else None
    if result is None:
        st.info("Add meg a PTB-XL adatbázist és az EKG-azonosítót, majd indítsd el a vizsgálatot." if view_only
                else "Válassz PTB-XL adatbázist és egy betanított best_model.pt fájlt, majd indítsd el a vizsgálatot.")
        st.caption("Még nincs modelled? A Kísérletek oldalon indíthatsz tanítást. A Bemutató mód adatbázis nélkül is kipróbálható."
                   if LOCAL else "Válassz egy EKG-t, majd kattints az EKG vizsgálata gombra.")
        return
    if not demo:
        st.caption(f"EKG #{result['ecg_id']} · {result.get('split', 'ismeretlen halmaz')} · "
                   f"{result.get('device', 'csak jelmegjelenítés')} · legjobb epoch: {result.get('best_epoch', '—')}")
    cols = st.columns(4)
    for col, label, val in zip(cols, ["Kronológiai életkor", "Becsült EKG-életkor", "Nyers AgeGap", "Mintavétel"],
                               [f"{result['age']:.0f} év", number(result.get('predicted_age'), " év"),
                                number(result.get('age_gap_raw'), " év"), f"{result['sampling_rate']} Hz"]):
        col.metric(label, val)
    if "predicted_age" not in result:
        st.info("Csak az EKG-jel jelenik meg; modellbecslés nem történt.")
    if result.get("precomputed"):
        st.caption("A becslés és az Integrated Gradients előre, a betanított modellel készült; a böngészőben nem fut modell.")
    st.write("")
    with st.container(border=True):
        a, b = st.columns([2, 1])
        with a:
            st.subheader("EKG-jelgörbe")
            selected = st.multiselect("Elvezetések", LEADS, default=["I", "II", "V2", "V5"])
        with b:
            duration = result["signal"].shape[1] / result["sampling_rate"]
            window = st.slider("Időtartomány (s)", 0.0, float(duration), (0.0, float(min(5, duration))), step=.1)
        signal = np.asarray(result["signal"]) * 5
        t = np.arange(signal.shape[1]) / result["sampling_rate"]
        keep = (t >= window[0]) & (t <= window[1])
        if selected:
            fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True, vertical_spacing=.04)
            for row, lead in enumerate(selected, 1):
                fig.add_trace(go.Scatter(x=t[keep], y=signal[LEADS.index(lead), keep], mode="lines",
                                        line=dict(color=TEAL, width=1.3), name=lead,
                                        hovertemplate="%{x:.2f} s · %{y:.3f} mV<extra>"+lead+"</extra>"), row=row, col=1)
                fig.update_yaxes(title_text=lead, row=row, col=1)
            fig.update_layout(showlegend=False)
            fig.update_xaxes(title_text="Idő (s)", row=len(selected), col=1)
            chart(fig, max(240, len(selected)*115))
        else:
            st.info("Válassz legalább egy elvezetést.")
        st.caption("Amplitúdó: mV, a modell ±5 mV-re vágott bemenete. A görbén húzással nagyíthatsz, dupla kattintással visszaállíthatod a nézetet.")
    if "attribution" in result:
        st.subheader("Mire figyelt a modell?")
        left, right = st.columns([1, 2])
        with left:
            importance = pd.DataFrame({"Elvezetés": LEADS, "Hozzájárulás": result["lead_importance"]})
            chart(px.bar(importance.sort_values("Hozzájárulás"), x="Hozzájárulás", y="Elvezetés", orientation="h"), 380)
        with right:
            attr = np.asarray(result["attribution"])
            chart(go.Figure(go.Heatmap(z=np.abs(attr[:, keep]), x=t[keep], y=LEADS, colorscale="Teal",
                                      colorbar=dict(title="|IG|"))), 380)
        st.caption(f"Integrated Gradients · nullajel referencia · {steps} lépés · konvergenciaeltérés: {result['convergence_delta']:.5f}. "
                   "A színek a modell érzékenységét mutatják, nem bizonyítanak élettani ok-okozati kapcsolatot.")
    record_download(result, checkpoint if "predicted_age" in result else None, data_dir)


def result_source(demo):
    if demo:
        return demo_predictions(), "Szintetikus bemutató", True
    if not LOCAL:
        return hosted_result_source()
    candidates = sorted((PROJECT / "runs").glob("**/predictions.csv")) if (PROJECT / "runs").exists() else []
    source = st.radio("Eredmények betöltése", ["Helyi eredményfájl", "CSV feltöltése"], horizontal=True)
    if source == "CSV feltöltése":
        return uploaded_source()
    default = st.session_state.pop("next_predictions_path", None)
    if default is not None:
        st.session_state.predictions_path = default
    if candidates:
        choice = st.selectbox("Korábbi futások", ["Egyedi elérési út"] + [str(p) for p in candidates],
                              format_func=lambda p: p if p == "Egyedi elérési út" else str(Path(p).relative_to(PROJECT)))
        if choice != "Egyedi elérési út":
            st.session_state.predictions_path = choice
    path = st.text_input("Predikciók elérési útja", key="predictions_path", placeholder="/…/predictions.csv")
    if not path:
        st.info("Tölts be egy korábbi eredményt, vagy indíts kiértékelést a Modelltesztek oldalon.")
        return None, None, False
    csv_path = Path(path).expanduser()
    frame = load_predictions(csv_path)
    provenance_path = csv_path.parent / "metrics.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if isinstance(provenance, dict):
                frame.attrs["run_provenance"] = provenance
        except (ValueError, OSError):
            st.warning("A futás metrics.json fájlja nem olvasható; a CSV eredetadatai hiányosak.")
    return frame, path, bool(frame.attrs.get("synthetic", False))


def uploaded_source():
    uploaded = st.file_uploader("Predikciók CSV-fájlja", type=["csv"])
    if uploaded is None:
        st.info("Elvárt oszlopok: age, predicted_age. Ajánlott: ecg_id, patient_id, split (validation/test), sex és diagnosztikai címkék.")
        return None, None, False
    frame = load_predictions(io.BytesIO(uploaded.getvalue()))
    return frame, uploaded.name, bool(frame.attrs.get("synthetic", False))


def hosted_result_source():
    """Cloud / browser: the bundled example model's predictions or an uploaded CSV – no local paths."""
    bundled = bundled_predictions_path()
    options = (["Példamodell eredményei"] if bundled else []) + ["CSV feltöltése"]
    source = st.radio("Eredmények betöltése", options, horizontal=True)
    if source == "CSV feltöltése":
        return uploaded_source()
    frame = load_predictions(bundled)
    metrics_path = bundled.parent / "metrics.json"
    if metrics_path.is_file():
        try:
            provenance = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(provenance, dict):
                frame.attrs["run_provenance"] = provenance
        except (ValueError, OSError):
            pass
    return frame, "példamodell (predictions.csv)", bool(frame.attrs.get("synthetic", False))


def enrich_predictions(df, data_dir):
    if not data_dir or "ecg_id" not in df:
        return df
    folder = Path(data_dir).expanduser()
    from ecg_age_common import add_diagnostic_superclasses, load_ptbxl_metadata
    db = load_ptbxl_metadata(folder)
    if (folder / "scp_statements.csv").is_file():
        db = add_diagnostic_superclasses(db, folder)
    db = db.reset_index()
    db["ecg_id"] = db["ecg_id"].astype("string")
    checked = df.merge(db[["ecg_id", "age", "patient_id", "strat_fold"]], on="ecg_id", how="left",
                       suffixes=("", "_metadata"), validate="one_to_one")
    if checked.age_metadata.isna().any():
        raise ValueError("A predikciók egyes EKG-azonosítói nem találhatók a kiválasztott PTB-XL felnőtt mintájában.")
    if not np.allclose(checked.age, checked.age_metadata):
        raise ValueError("A predikciók életkorai eltérnek a kiválasztott PTB-XL metaadataitól.")
    folds = checked.strat_fold
    expected = pd.Series(np.where(folds <= 8, "train", np.where(folds == 9, "validation", "test")), index=checked.index)
    if ((checked.split != "unspecified") & (checked.split != expected)).any():
        raise ValueError("A predikciók split értékei eltérnek a PTB-XL hivatalos felosztásától.")
    if "patient_id" in df:
        left_ids = pd.to_numeric(checked.patient_id, errors="coerce")
        right_ids = pd.to_numeric(checked.patient_id_metadata, errors="coerce")
        if not np.array_equal(left_ids, right_ids):
            raise ValueError("A predikciók betegazonosítói eltérnek a metaadatoktól.")
    cols = [c for c in ["patient_id", "sex", *LABELS] if c not in df and c in db]
    enriched = df.merge(db[["ecg_id", *cols]], on="ecg_id", how="left", validate="one_to_one")
    enriched.attrs.update(df.attrs)
    if "patient_id" in enriched and enriched.patient_id.notna().any():
        if (enriched.dropna(subset=["patient_id"]).groupby("patient_id").split.nunique() > 1).any():
            raise ValueError("Betegszintű átfedés van az adathalmazrészek között; az elemzés leállt.")
    return enriched


def reports_page(demo, data_dir):
    header("Eredmények, összefüggések.", "Szűrd a mintákat, vizsgáld a becslési hibákat, és exportáld a kimutatásokat.", demo)
    try:
        raw, source_name, synthetic = result_source(demo)
        if raw is None:
            return
        if not demo and not synthetic:
            raw = enrich_predictions(raw, data_dir)
        full, correction = prepare_predictions(raw)
    except Exception as exc:
        st.error(f"Az eredményfájl nem dolgozható fel: {exc}")
        return
    if synthetic and not demo:
        st.warning("A betöltött CSV szintetikus bemutatóadatokat tartalmaz; ezek nem kutatási eredmények.")
    provenance = raw.attrs.get("run_provenance", {})
    if provenance.get("subset"):
        st.warning("RÉSZMINTÁS KIÉRTÉKELÉS · Ez a futás nem tartalmazza a teljes validációs és teszthalmazt.")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        splits = list(full.split.unique())
        split = c1.selectbox("Adathalmaz", splits, index=splits.index("test") if "test" in splits else 0)
        age_range = c2.slider("Életkor (év)", 18, 89, (18, 89))
        sex = c3.selectbox("Nem (PTB-XL kód)", ["Összes", "0", "1", "Hiányzó"], disabled="sex" not in full)
        labels = [c for c in LABELS if c in full]
        label = c4.selectbox("Diagnosztikai csoport", ["Összes", *labels])
    data = full.loc[(full.split == split) & full.age.between(*age_range)].copy()
    if "sex" in data and sex != "Összes":
        codes = pd.to_numeric(data.sex, errors="coerce")
        data = data.loc[codes.isna() if sex == "Hiányzó" else codes == int(sex)]
    if label != "Összes":
        data = data.loc[pd.to_numeric(data[label], errors="coerce") == 1]
    if data.empty:
        st.warning("A megadott szűrőkkel nincs megjeleníthető rekord.")
        return
    met = metrics(data)
    cols = st.columns(5)
    for col, label_text, val in zip(cols, ["EKG-rekordok", "MAE · átlagos abszolút hiba", "RMSE", "R²", "Pearson-r"],
                                    [str(met["n"]), number(met["mae"], " év"), number(met["rmse"], " év"),
                                     number(met["r2"]), number(met["pearson_r"])]):
        col.metric(label_text, val)
    if split != "test":
        st.caption("A megjelenített mutatók nem független teszteredmények: a kijelölt halmaz " + split + ".")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Becsült és kronológiai életkor")
        fig = px.scatter(data, x="age", y="predicted_age", opacity=.5,
                         hover_data=["ecg_id"] if "ecg_id" in data else None,
                         labels={"age": "Kronológiai életkor (év)", "predicted_age": "Becsült életkor (év)"})
        fig.add_trace(go.Scatter(x=[18, 89], y=[18, 89], mode="lines", name="Tökéletes becslés", line=dict(color=AMBER, dash="dash")))
        chart(fig)
    with right, st.container(border=True):
        st.subheader("AgeGap eloszlása")
        use_corrected = st.checkbox("Validációval korrigált AgeGap", value=correction is not None, disabled=correction is None)
        gap = "age_gap_corrected" if use_corrected else "age_gap_raw"
        chart(px.histogram(data, x=gap, nbins=35, labels={gap: "AgeGap (év)"}, color_discrete_sequence=[TEAL]))
    if correction:
        st.caption(f"Korrekció: nyers AgeGap − ({correction['intercept']:.3f} + {correction['slope']:.3f} × életkor). "
                   f"Illesztés: {correction['n_validation']} validációs rekord, a fenti szűrők alkalmazása előtt.")
    else:
        st.info("Korrigált AgeGap-hez legalább 3 validációs rekord és változó életkor kell. A nyers eredmények elérhetők.")
    groups = data.copy()
    groups["Korcsoport"] = pd.cut(groups.age, [17, 29, 39, 49, 59, 69, 79, 89],
                                  labels=["18–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80–89"])
    a, b = st.columns(2)
    with a, st.container(border=True):
        st.subheader("Teljesítmény korcsoportonként")
        by_age = subgroup_metrics(groups, "Korcsoport")
        chart(px.bar(by_age, x="Korcsoport", y="mae", hover_data=["n"], labels={"mae": "MAE (év)"}))
    with b, st.container(border=True):
        st.subheader("AgeGap diagnosztikai csoportonként")
        pieces = [data.loc[data[c] == 1, [gap]].assign(Csoport=c) for c in labels]
        if pieces and any(len(p) for p in pieces):
            chart(px.box(pd.concat(pieces), x="Csoport", y=gap, labels={gap: "AgeGap (év)"}))
            st.caption("Egy EKG több csoportban is szerepelhet. A csoportok nem függetlenek.")
        else:
            st.info("Ehhez NORM / MI / STTC / CD / HYP címkék vagy a PTB-XL metaadatai szükségesek.")
    st.subheader("Rekordszintű eredmények")
    st.dataframe(data.sort_values("abs_error", ascending=False), width="stretch", hide_index=True)
    metadata = {"source": source_name, "synthetic": synthetic, "created_at": datetime.now(timezone.utc).isoformat(),
                "filters": {"split": split, "age_range": age_range, "sex": sex, "diagnosis": label},
                "bias_correction": correction, "total_source_records": len(full), "run_provenance": provenance}
    a, b = st.columns(2)
    a.download_button("Szűrt eredmények · CSV", data.to_csv(index=False).encode("utf-8-sig"),
                      "DEMO_predictions.csv" if synthetic else "filtered_predictions.csv", "text/csv")
    b.download_button("Teljes kimutatás · ZIP", export_report(data, metadata),
                      "DEMO_report.zip" if synthetic else "tdk_report.zip", "application/zip", type="primary")
    with st.expander("Két eredményfájl összehasonlítása"):
        st.caption("Az összevetés azonos EKG-azonosítókon történik. A két modell teszthalmazának egyeznie kell.")
        files = st.file_uploader("Két predictions.csv", type="csv", accept_multiple_files=True, key="compare")
        if len(files) == 2:
            try:
                frames = [load_predictions(io.BytesIO(f.getvalue())) for f in files]
                frames = [f[f.split == "test"].copy() for f in frames]
                if any(f.empty or "ecg_id" not in f for f in frames):
                    raise ValueError("Mindkét fájlban szükséges test halmaz és ecg_id.")
                if set(frames[0].ecg_id) != set(frames[1].ecg_id):
                    raise ValueError("A teszthalmazok EKG-azonosítói eltérnek.")
                aligned = [f.set_index("ecg_id").sort_index() for f in frames]
                if not np.allclose(aligned[0].age, aligned[1].age):
                    raise ValueError("Azonos EKG-hoz eltérő életkor tartozik a két fájlban.")
                st.dataframe(pd.DataFrame([{"Fájl": f.name, **metrics(df)} for f, df in zip(files, frames)]), hide_index=True)
            except Exception as exc:
                st.error(str(exc))
        elif files:
            st.info("Pontosan két CSV-fájlt válassz.")


@st.fragment(run_every=3)
def jobs_panel():
    jobs = job_manager().list_jobs()
    if not jobs:
        st.caption("Még nincs elindított folyamat ebben az alkalmazásmenetben.")
        return
    names = {"train": "Tanítás", "evaluate": "Modellteszt", "analysis": "Diagnosztikai elemzés"}
    statuses = {"running": "Fut", "completed": "Elkészült", "failed": "Hibával leállt", "stopped": "Leállítva"}
    for job in jobs[:8]:
        with st.expander(f"{names.get(job['kind'], job['kind'])} · {statuses.get(job['status'], job['status'])} · {job['id']}", expanded=job["status"] in {"running", "failed"}):
            st.caption(f"Eredmények: {job['output_dir']}")
            st.code(job.get("log") or "A folyamat indul…", language="text")
            if job["status"] == "running" and st.button("Folyamat leállítása", key=f"stop_{job['id']}"):
                job_manager().stop(job["id"])
                st.rerun(scope="fragment")
            prediction_path = Path(job["output_dir"]) / "predictions.csv"
            if job["status"] == "completed" and prediction_path.is_file():
                st.download_button("Predikciók letöltése", prediction_path.read_bytes(), f"{job['id']}_predictions.csv",
                                   "text/csv", key=f"pred_{job['id']}")
                st.caption("A Kimutatások oldalon a Korábbi futások listájából választhatod ki ezt az eredményt.")


def start_job(kind, config):
    try:
        # Validate research splits before entering the older training/analysis CLIs.
        if kind in {"train", "analysis"}:
            if not config.get("data_dir"):
                raise ValueError("Add meg a PTB-XL adatbázis mappáját a bal oldalon.")
            from dashboard.model_service import inspect_dataset
            dataset = inspect_dataset(config["data_dir"])
            if kind == "train" and not all(dataset[f"n_{part}"] for part in ["train", "validation", "test"]):
                raise ValueError("A tanításhoz nem üres train, validation és test halmaz szükséges.")
            if kind == "analysis":
                frame = load_predictions(config.get("predictions", ""))
                if frame.attrs.get("synthetic"):
                    raise ValueError("A valós diagnosztikai elemzéshez nem használhatók bemutatóadatok.")
                if "ecg_id" not in frame or not frame.split.eq("test").any():
                    raise ValueError("Az elemzéshez ecg_id oszlop és test rekordok szükségesek.")
                frame = enrich_predictions(frame, config["data_dir"])
                _, calibration = prepare_predictions(frame)
                if calibration is None:
                    raise ValueError("Az elemzéshez legalább 3 validációs rekord és változó életkor szükséges.")
        job = job_manager().start(kind, config)
        if job["status"] == "failed":
            st.error(f"A folyamat nem indult el: {job.get('log', '')}")
        else:
            st.success(f"Elindult. Az eredmények ide kerülnek: {job['output_dir']}")
    except Exception as exc:
        st.error(f"Nem indítható a folyamat: {exc}")


def tests_page(demo, data_dir, checkpoint, device):
    header("A modell próbája.", "Ellenőrizd az adatbázist és a modellfájlt, majd mérd a teljesítményt elkülönített adatokon.", demo)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("01 / Bemenetek ellenőrzése")
        st.write("Adathalmazméret, betegszintű felosztás, modellarchitektúra és mintavétel.")
        check = st.button("Adatok és modell ellenőrzése", disabled=demo, width="stretch")
        if check:
            try:
                if not data_dir or not checkpoint:
                    raise ValueError("Add meg az adatbázis mappáját és a modellfájlt a bal oldalon.")
                from dashboard.model_service import inspect_dataset, load_model
                with st.spinner("Ellenőrzés…"):
                    report = inspect_dataset(Path(data_dir).expanduser())
                    model, info = load_model(Path(checkpoint).expanduser(), device)
                st.success("A metaadatok és a modellfájl ellenőrzése sikeres.")
                st.json({k: v for k, v in report.items() if k != "record_ids"})
                st.json(info)
                del model
                st.caption("A jelállományok teljességét és a numerikus kimenetet a kiértékelés vizsgálja.")
            except Exception as exc:
                st.error(str(exc))
    with right, st.container(border=True):
        st.subheader("02 / Kiértékelés")
        st.write("Becslés a validation (9. fold) és test (10. fold) rekordokra a kiválasztott modellel.")
        with st.form("evaluate_form"):
            batch = st.select_slider("Kötegméret", [1, 4, 8, 16, 32, 64, 128], value=16)
            limit = st.number_input("Rekordlimit halmazonként (0 = teljes)", min_value=0, max_value=100000, value=0, step=25)
            st.caption("Részminta gyors ellenőrzésre használható. A végső TDK-mutatókat a teljes teszthalmazon mérd.")
            submit = st.form_submit_button("Modellteszt indítása", type="primary", disabled=demo)
        if submit:
            start_job("evaluate", {"data_dir": data_dir, "checkpoint": checkpoint, "device": device,
                                   "batch_size": batch, "limit": limit})
    if demo:
        st.info("Valós modellteszthez válts Saját adatok módra, és add meg a PTB-XL mappát és a betanított modellt.")
    st.subheader("Folyamatok és futási napló")
    jobs_panel()


def experiments_page(demo, data_dir):
    header("Kísérletek, követhetően.", "Indíts tanítást vagy részletes diagnosztikai elemzést. Minden futás külön mappába kerül.", demo)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Új modell tanítása")
        st.caption("1–8. fold: tanítás · 9. fold: validáció · 10. fold: teszt")
        with st.form("training"):
            a, b = st.columns(2)
            epochs = a.number_input("Epochok", 1, 1000, 30)
            batch = b.selectbox("Kötegméret", [4, 8, 16, 32, 64, 128], index=2)
            lr = a.number_input("Tanulási ráta", min_value=.000001, max_value=.1, value=.001, format="%.6f")
            patience = b.number_input("Early stopping türelme", 1, 100, 7)
            seed = a.number_input("Véletlenmag", 0, 2147483647, 42)
            sampling_rate = b.selectbox("Mintavétel (Hz)", [100, 500])
            submitted = st.form_submit_button("Tanítás indítása", type="primary", disabled=demo)
        if submitted:
            start_job("train", {"data_dir": data_dir, "epochs": epochs, "batch_size": batch,
                                "lr": lr, "patience": patience, "seed": seed, "sampling_rate": sampling_rate})
        st.caption("A tanítás a rendelkezésre álló gyorsítót automatikusan választja ki. Hosszabb futás közben is használhatod a felületet.")
    with right, st.container(border=True):
        st.subheader("Diagnosztikai AgeGap-elemzés")
        st.write("Csoportstatisztikák, Mann–Whitney-próba és életkorra/nemre korrigált logisztikus regresszió.")
        with st.form("analysis"):
            predictions = st.text_input("Validation + test predikciók", placeholder="/…/predictions.csv")
            analyze = st.form_submit_button("Elemzés indítása", disabled=demo)
        if analyze:
            start_job("analysis", {"data_dir": data_dir, "predictions": predictions})
        st.caption("Az elemzés a projekt eredeti evaluate_agegap.py programját futtatja, ábrákat és táblázatokat ment.")
    histories = sorted((PROJECT / "runs").glob("**/training_history.csv")) if (PROJECT / "runs").exists() else []
    if histories:
        st.subheader("Tanulási görbe")
        history = st.selectbox("Tanítási napló", histories, format_func=lambda p: str(p.relative_to(PROJECT)))
        try:
            frame = pd.read_csv(history)
            chart(px.line(frame, x="epoch", y=[c for c in ["mae", "rmse"] if c in frame],
                          labels={"epoch": "Epoch", "value": "Validációs hiba (év)"}))
        except Exception as exc:
            st.warning(f"A tanulási görbe még nem olvasható: {exc}")
    st.subheader("Folyamatok és futási napló")
    jobs_panel()


def main():
    style()
    st.session_state.setdefault("data_dir", os.environ.get("PTBXL_DATA_DIR", ""))
    st.session_state.setdefault("checkpoint", "")
    page, demo, data_dir, checkpoint, device = sidebar()
    if page == "EKG-vizsgálat":
        ecg_page(demo, data_dir, checkpoint, device)
    elif page == "Kimutatások":
        reports_page(demo, data_dir)
    elif page == "Modelltesztek":
        tests_page(demo, data_dir, checkpoint, device)
    else:
        experiments_page(demo, data_dir)
    st.divider()
    st.caption(FOOTER)


if __name__ == "__main__":
    main()
