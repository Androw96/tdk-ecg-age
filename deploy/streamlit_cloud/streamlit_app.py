"""Streamlit Community Cloud entry point for the ECG Age Lab.

Runs the regular dashboard_app.py in the "cloud" profile: bundled PTB-XL example
ECGs and the trained model from demo_data/, live CPU inference and Integrated
Gradients, no local paths, no training or background jobs.
See docs/UI_DOKUMENTACIO.md for the deployment steps.
"""
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("ECG_AGE_PROFILE", "cloud")
os.environ.setdefault("ECG_AGE_DEMO_DIR", str(ROOT / "demo_data"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
runpy.run_path(str(ROOT / "dashboard_app.py"), run_name="__main__")
