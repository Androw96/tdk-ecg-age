from __future__ import annotations

import ast
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ptbxl_metadata(data_dir: str | Path, min_age: int = 18, max_age: int = 89) -> pd.DataFrame:
    data_dir = Path(data_dir)
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["age"] = pd.to_numeric(db["age"], errors="coerce")
    db = db[db["age"].between(min_age, max_age, inclusive="both")].copy()
    db = db[db["strat_fold"].notna()].copy()
    db["strat_fold"] = db["strat_fold"].astype(int)
    return db


def add_diagnostic_superclasses(db: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """Map SCP statements to PTB-XL diagnostic superclasses.

    A record may belong to multiple diagnostic superclasses.
    """
    data_dir = Path(data_dir)
    scp = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    if "diagnostic" not in scp.columns or "diagnostic_class" not in scp.columns:
        raise ValueError("scp_statements.csv does not contain diagnostic / diagnostic_class columns")

    diagnostic_map: Dict[str, str] = {}
    for code, row in scp.iterrows():
        if float(row.get("diagnostic", 0) or 0) == 1 and pd.notna(row.get("diagnostic_class")):
            diagnostic_map[str(code)] = str(row["diagnostic_class"])

    out = db.copy()
    for label in SUPERCLASSES:
        out[label] = 0

    for idx, raw in out["scp_codes"].items():
        try:
            codes = ast.literal_eval(raw) if isinstance(raw, str) else dict(raw)
        except Exception:
            codes = {}
        classes = {diagnostic_map.get(str(code)) for code in codes.keys()}
        classes.discard(None)
        for label in SUPERCLASSES:
            out.at[idx, label] = int(label in classes)
    return out


class PTBXLAgeDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        data_dir: str | Path,
        sampling_rate: int = 100,
        clip_mv: Optional[float] = 5.0,
    ):
        self.metadata = metadata.copy()
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.clip_mv = clip_mv
        self.filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        if self.filename_col not in self.metadata.columns:
            raise ValueError(f"Missing {self.filename_col} in PTB-XL metadata")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item: int):
        row = self.metadata.iloc[item]
        record_path = self.data_dir / str(row[self.filename_col])
        import wfdb
        signal, fields = wfdb.rdsamp(str(record_path))
        # WFDB -> [time, channels]; PyTorch Conv1d -> [channels, time]
        x = signal.astype(np.float32).T

        # Ensure canonical 12-lead order when signal names are available.
        names = fields.get("sig_name", [])
        if names and list(names) != LEADS:
            lookup = {name: i for i, name in enumerate(names)}
            if all(lead in lookup for lead in LEADS):
                x = x[[lookup[lead] for lead in LEADS], :]

        if self.clip_mv is not None:
            x = np.clip(x, -self.clip_mv, self.clip_mv)
            x = x / float(self.clip_mv)

        y = np.float32(row["age"])
        ecg_id = int(self.metadata.index[item])
        return torch.from_numpy(x), torch.tensor(y), ecg_id


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x + identity)
        return x


class ECGAgeResNet(nn.Module):
    def __init__(self, in_leads: int = 12, dropout: float = 0.15):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualBlock1D(64, 64, 1, dropout),
            ResidualBlock1D(64, 64, 1, dropout),
            ResidualBlock1D(64, 128, 2, dropout),
            ResidualBlock1D(128, 128, 1, dropout),
            ResidualBlock1D(128, 256, 2, dropout),
            ResidualBlock1D(256, 256, 1, dropout),
            ResidualBlock1D(256, 512, 2, dropout),
            ResidualBlock1D(512, 512, 1, dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
