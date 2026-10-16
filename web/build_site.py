"""Build the static GitHub Pages site: start page, results, documentation, browser app.

    pip install markdown
    python web/build_site.py --out _site [--cloud-url https://<app>.streamlit.app]

Output (_site/):
    index.html          start page: project summary, links, results (if web/results/ exists)
    results/            figures and tables copied from web/results/ (web/collect_results.py)
    docs/*.html         README, DASHBOARD, UI_DOKUMENTACIO, EXPERIMENTS, ARCHITECTURE, CHANGELOG
    app/index.html      the ECG Age Lab running in the browser (stlite + Pyodide, no server)
    app/src/            the Python files and example data the browser app mounts

The GitHub Actions workflow (deploy/github_pages/pages.yml → .github/workflows/) runs this on every push
to main. Preview locally:  python -m http.server -d _site 8000
"""
from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import tomllib
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode

ROOT = Path(__file__).resolve().parents[1]
STLITE_VERSION = "1.9.2"          # @stlite/browser on jsDelivr; bundles Streamlit 1.50+
BROWSER_REQUIREMENTS = ["plotly==5.24.1"]   # numpy / pandas come from the Pyodide distribution
SITE_TITLE = "ECG Age Lab"
PROJECT_TITLE = "Biológiai életkor becslése 12 elvezetéses EKG-ből"
AUTHOR_LINE = "TDK-kutatás · Semmelweis Egyetem, Egészségügyi adattudomány MSc"
DOCS = [  # (source, output name, title)
    ("docs/UI_DOKUMENTACIO.md", "UI_DOKUMENTACIO", "UI dokumentáció"),
    ("README.md", "README", "Áttekintés és telepítés"),
    ("DASHBOARD.md", "DASHBOARD", "A helyi felület használata"),
    ("docs/EXPERIMENTS.md", "EXPERIMENTS", "Kísérleti protokoll"),
    ("docs/ARCHITECTURE.md", "ARCHITECTURE", "Architektúra"),
    ("CHANGELOG.md", "CHANGELOG", "Változásnapló"),
]
APP_FILES = ["dashboard_app.py", "dashboard/__init__.py", "dashboard/analytics.py"]
PTBXL_NOTE = ("Az EKG-adatok forrása a PTB-XL adatbázis (Wagner és mtsai., Scientific Data 2020; PhysioNet v1.0.3), "
              "CC BY 4.0 licenc. A becslések kutatási célúak; az EKG-életkor és az AgeGap nem diagnózis.")

CSS = """
:root{--bg:#f5f7fa;--surface:#ffffff;--ink:#162d3d;--muted:#5b6f7e;--line:#dde5ec;--accent:#087f83;
--accent-ink:#ffffff;--accent-soft:#e4f2ef;--code:#eef2f6;--warn:#fff7e6;--warn-line:#f0d9a8}
@media (prefers-color-scheme:dark){:root{--bg:#0f1a22;--surface:#16232d;--ink:#e6eef3;--muted:#9fb1bd;
--line:#26363f;--accent:#38b2ac;--accent-ink:#0b1419;--accent-soft:#16333a;--code:#1d2c36;--warn:#2b2516;--warn-line:#5a4a22}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
a{color:var(--accent)}img{max-width:100%;height:auto}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
header.top{border-bottom:1px solid var(--line);background:var(--surface)}
header.top .wrap{display:flex;align-items:center;justify-content:space-between;gap:16px;min-height:60px;flex-wrap:wrap}
.brand{font-weight:750;letter-spacing:-.04em;font-size:1.25rem;color:var(--ink);text-decoration:none}.brand span{color:var(--accent)}
nav a{color:var(--muted);text-decoration:none;margin-left:18px;font-size:.95rem}nav a:hover{color:var(--accent)}
.hero{padding:56px 0 28px}.eyebrow{font-size:.75rem;font-weight:750;letter-spacing:.16em;color:var(--accent);text-transform:uppercase}
h1{font-size:clamp(1.9rem,4vw,2.7rem);line-height:1.15;letter-spacing:-.035em;margin:.4em 0 .35em}
.lead{color:var(--muted);font-size:1.08rem;max-width:720px}
.btns{display:flex;gap:12px;flex-wrap:wrap;margin:26px 0 8px}
.btn{display:inline-block;padding:11px 18px;border-radius:9px;text-decoration:none;font-weight:600;border:1px solid var(--line);color:var(--ink);background:var(--surface)}
.btn.primary{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}
section{padding:26px 0}h2{font-size:1.45rem;letter-spacing:-.025em;margin:0 0 14px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px}
.card h3{margin:0 0 6px;font-size:1.05rem}.card p{margin:0 0 12px;color:var(--muted);font-size:.95rem}
.tag{display:inline-block;font-size:.72rem;font-weight:650;padding:3px 9px;border-radius:20px;background:var(--accent-soft);color:var(--accent);margin-bottom:10px}
.note{background:var(--warn);border:1px solid var(--warn-line);border-radius:10px;padding:14px 16px;font-size:.95rem}
figure{margin:0}figure img{border:1px solid var(--line);border-radius:10px;background:#fff}
figcaption{color:var(--muted);font-size:.88rem;margin-top:6px}
.doc{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:8px 28px 28px;margin:28px 0}
.doc table,.results table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;font-size:.92rem;margin:12px 0}
.doc th,.doc td,.results th,.results td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
.doc th,.results th{background:var(--code)}
code{background:var(--code);padding:1px 5px;border-radius:5px;font-size:.9em}
pre{background:var(--code);padding:14px;border-radius:10px;overflow-x:auto}pre code{padding:0;background:none}
footer{border-top:1px solid var(--line);margin-top:30px;padding:22px 0 40px;color:var(--muted);font-size:.88rem}
.doclist a{display:block;padding:9px 0;border-bottom:1px solid var(--line);text-decoration:none}
@media (max-width:600px){nav a{margin-left:12px}.doc{padding:4px 16px 18px}}
"""


def page(title: str, body: str, depth: int = 0) -> str:
    up = "../" * depth
    return f"""<!doctype html>
<html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><link rel="stylesheet" href="{up}assets/style.css"></head>
<body><header class="top"><div class="wrap"><a class="brand" href="{up}index.html">ECG<span> Age Lab</span></a>
<nav><a href="{up}app/">Alkalmazás</a><a href="{up}index.html#eredmenyek">Eredmények</a><a href="{up}index.html#dokumentacio">Dokumentáció</a></nav>
</div></header>{body}
<footer><div class="wrap">{html.escape(PTBXL_NOTE)}<br>{html.escape(AUTHOR_LINE)}</div></footer></body></html>"""


def render_markdown(text: str) -> str:
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "toc", "sane_lists"],
                            extension_configs={"toc": {"slugify": slugify_unicode}})
    # links between documents: foo.md / docs/FOO.md -> FOO.html (all docs live in one folder)
    names = {Path(src).name: name for src, name, _ in DOCS}
    def fix(match):
        target = match.group(1)
        base = target.split("#")[0].split("/")[-1]
        if base in names:
            anchor = target[len(target.split("#")[0]):]
            return f'href="{names[base]}.html{anchor}"'
        return match.group(0)
    return re.sub(r'href="([^"]+?\.md(?:#[^"]*)?)"', fix, out)


def stlite_app(src_files: list[str], config: dict) -> str:
    files = {name: {"url": f"./src/{name}"} for name in src_files}
    theme = {f"theme.{k}": v for k, v in config.get("theme", {}).items()}
    options = {
        "requirements": BROWSER_REQUIREMENTS,
        "entrypoint": "dashboard_app.py",
        "files": files,
        "streamlitConfig": {**theme, "client.toolbarMode": "viewer", "client.showErrorDetails": True},
    }
    return f"""<!doctype html>
<html lang="hu"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{SITE_TITLE} · böngészős változat</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.css">
<style>body{{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}}
#boot{{max-width:640px;margin:18vh auto;padding:0 20px;color:#355165;text-align:center;line-height:1.6}}
#boot b{{color:#087f83}}</style></head>
<body><div id="root"><div id="boot"><b>ECG Age Lab</b> betöltése a böngészőben…<br>
Az első indítás 20–60 másodperc is lehet (a Python-környezet letöltődik), utána gyorsabb.<br>
<small>Nem fut szerver: minden számítás a te gépeden, a böngészőben történik.</small></div></div>
<script type="module">
import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.js";
mount({json.dumps(options, ensure_ascii=False, indent=1)}, document.getElementById("root"));
</script></body></html>"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "_site"))
    p.add_argument("--cloud-url", default="", help="URL of the Streamlit Community Cloud app (optional)")
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / "assets").mkdir(parents=True)
    (out / "assets" / "style.css").write_text(CSS, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    # ---- documentation pages ------------------------------------------------------
    (out / "docs").mkdir()
    doc_links = []
    for src, name, title in DOCS:
        path = ROOT / src
        if not path.is_file():
            continue
        body = f'<main class="wrap"><article class="doc">{render_markdown(path.read_text(encoding="utf-8"))}</article></main>'
        (out / "docs" / f"{name}.html").write_text(page(f"{title} · {SITE_TITLE}", body, depth=1), encoding="utf-8")
        doc_links.append(f'<a href="docs/{name}.html">{html.escape(title)}</a>')

    # ---- browser app (stlite) -----------------------------------------------------
    src_files = list(APP_FILES)
    examples = ROOT / "web" / "examples"
    if examples.is_dir():
        src_files += [f"web/examples/{f.name}" for f in sorted(examples.iterdir())
                      if f.suffix in {".npz", ".json", ".csv"}]
    for rel in src_files:
        dst = out / "app" / "src" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    (out / "app" / "index.html").write_text(stlite_app(src_files, config), encoding="utf-8")
    has_examples = any(r.endswith("examples.npz") for r in src_files)

    # ---- results --------------------------------------------------------------------
    results_html = ('<div class="note">Valódi kísérleti eredmények még nincsenek közzétéve. Egy lefutott elemzés után: '
                    '<code>python web/collect_results.py --run runs/&lt;futás&gt;</code>, majd commit és push.</div>')
    manifest_path = ROOT / "web" / "results" / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shutil.copytree(ROOT / "web" / "results", out / "results")
        figures = "".join(
            f'<figure><img src="results/{html.escape(f["file"])}" alt="{html.escape(f["caption"])}" loading="lazy">'
            f'<figcaption>{html.escape(f["caption"])}</figcaption></figure>' for f in manifest["figures"])
        summary = (ROOT / "web" / "results" / "results_summary.md").read_text(encoding="utf-8")
        summary = re.sub(r"^# .*\n", "", summary, count=1)
        tables = " · ".join(f'<a href="results/{t}">{t}</a>' for t in manifest.get("tables", []))
        results_html = (f'<p class="lead">Forrás futás: <code>{html.escape(manifest["run"])}</code></p>'
                        f'<div class="grid">{figures}</div><div class="results">{render_markdown(summary)}</div>'
                        f'<p>Letölthető táblák: {tables}</p>')

    cloud_card = ""
    if args.cloud_url.strip():
        cloud_card = (f'<div class="card"><span class="tag">Szerveren fut</span><h3>Online alkalmazás élő modellel</h3>'
                      f'<p>Valós PTB-XL teszt-EKG-k, élő életkorbecslés és Integrated Gradients a betanított modellel.</p>'
                      f'<a class="btn" href="{html.escape(args.cloud_url.strip())}">Megnyitás</a></div>')
    examples_line = ("valós PTB-XL példák előre kiszámított becsléssel és magyarázattal, " if has_examples else "")
    body = f"""<main class="wrap">
<div class="hero"><div class="eyebrow">TDK · EKG-alapú életkorbecslés</div><h1>{html.escape(PROJECT_TITLE)}</h1>
<p class="lead">Egy 1D ResNet mélytanulási modell a nyers, 12 elvezetéses EKG-ből becsli az életkort. A becsült és a valódi
életkor különbsége – az <b>AgeGap</b> – életkorfüggő torzítástól korrigálva vethető össze a diagnosztikai csoportokkal,
az Integrated Gradients pedig megmutatja, mely elvezetésekre és hullámokra figyel a modell.</p>
<div class="btns"><a class="btn primary" href="app/">Alkalmazás megnyitása</a><a class="btn" href="docs/UI_DOKUMENTACIO.html">UI dokumentáció</a>
<a class="btn" href="#eredmenyek">Eredmények</a></div></div>
<section><h2>Három felület, egy kódbázis</h2><div class="grid">
<div class="card"><span class="tag">Böngészőben fut</span><h3>Böngészős alkalmazás</h3><p>Telepítés és szerver nélkül:
{examples_line}bemutató mód, saját predikciós CSV elemzése és exportja.</p><a class="btn" href="app/">Megnyitás</a></div>
{cloud_card}
<div class="card"><span class="tag">Saját gépen</span><h3>Teljes kutatói munkafelület</h3><p>Saját PTB-XL, tanítás, modelltesztek,
diagnosztikai elemzés. Indítás: <code>launch_dashboard.command</code>.</p><a class="btn" href="docs/DASHBOARD.html">Leírás</a></div>
</div></section>
<section id="eredmenyek"><h2>Eredmények</h2>{results_html}</section>
<section id="dokumentacio"><h2>Dokumentáció</h2><div class="card doclist">{''.join(doc_links)}</div></section>
</main>"""
    (out / "index.html").write_text(page(f"{SITE_TITLE} · {PROJECT_TITLE}", body), encoding="utf-8")
    print(f"Site: {out.resolve()} ({len(doc_links)} dokumentum, {len(src_files)} alkalmazásfájl"
          f"{', példákkal' if has_examples else ''})")


if __name__ == "__main__":
    main()
