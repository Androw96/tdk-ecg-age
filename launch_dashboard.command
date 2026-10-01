#!/bin/zsh
# Dupla kattintással indítható helyi TDK felület macOS-en.
set -eu
cd -- "${0:A:h}"

task_python=""
for task_candidate in "$PWD/.venv/bin/python" /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  if [[ -x "$task_candidate" ]] && "$task_candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
    task_python="$task_candidate"
    break
  fi
done
if [[ -z "$task_python" ]] && command -v python3 >/dev/null 2>&1; then
  task_candidate="$(command -v python3)"
  if "$task_candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))' 2>/dev/null; then
    task_python="$task_candidate"
  fi
fi
if [[ -z "$task_python" ]]; then
  print 'Python 3.11 vagy újabb szükséges. A telepítés lépései: DASHBOARD.md'
  read -r '?Nyomj Entert az ablak bezárásához…'
  exit 1
fi
if ! "$task_python" -c 'import streamlit, plotly, numpy, pandas, scipy, sklearn, matplotlib, wfdb, torch, statsmodels' >/dev/null 2>&1; then
  print 'A felülethez szükséges Python-csomagok még hiányoznak.'
  print 'A projekt könyvtárában futtasd (részletek: DASHBOARD.md):'
  print '  python3 -m venv .venv'
  print '  .venv/bin/python -m pip install -r requirements-dashboard.txt'
  read -r '?Nyomj Entert az ablak bezárásához…'
  exit 1
fi
print 'TDK EKG-labor indul: http://127.0.0.1:8501'
print 'Leállítás: Ctrl+C ebben az ablakban. A futó modellfeladatot előbb a felületen állítsd le.'
exec "$task_python" -m streamlit run dashboard_app.py --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
