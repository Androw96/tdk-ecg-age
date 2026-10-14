"""Shared data, preprocessing and model code for the ECG-age TDK project.

Backward compatibility: checkpoints written by the first baseline (no
``model_config`` / ``preprocessing`` keys) load unchanged; the defaults below
reproduce the original architecture and the ±5 mV clip + /5 scaling exactly.
"""
from __future__ import annotations

import ast
import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

PREPROCESSING_MODES = ("clip5", "record_zscore", "lead_standardize")
DEFAULT_PREPROCESSING = {"mode": "clip5", "clip_mv": 5.0}
TRAIN_COHORTS = ("all", "norm")


# --------------------------------------------------------------------------- utilities
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


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- metadata
def load_ptbxl_metadata(data_dir: str | Path, min_age: int = 18, max_age: int = 89) -> pd.DataFrame:
    data_dir = Path(data_dir)
    db = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    db["age"] = pd.to_numeric(db["age"], errors="coerce")
    db = db[db["age"].between(min_age, max_age, inclusive="both")].copy()
    db = db[db["strat_fold"].notna()].copy()
    db["strat_fold"] = db["strat_fold"].astype(int)
    return db


def split_name(fold: int) -> str:
    return "train" if fold <= 8 else ("validation" if fold == 9 else "test")


def add_diagnostic_superclasses(db: pd.DataFrame, data_dir: str | Path) -> pd.DataFrame:
    """Map SCP statements to PTB-XL diagnostic superclasses.

    A record may belong to multiple diagnostic superclasses. ``n_superclasses``
    counts them; 0 means the record carries no diagnostic statement.
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
    out["n_superclasses"] = out[SUPERCLASSES].sum(axis=1).astype(int)
    return out


def strict_normal_mask(db: pd.DataFrame) -> pd.Series:
    """Records whose ONLY diagnostic superclass is NORM (no MI/STTC/CD/HYP)."""
    missing = [c for c in SUPERCLASSES if c not in db.columns]
    if missing:
        raise ValueError(f"Missing superclass columns: {missing}; call add_diagnostic_superclasses first")
    others = [c for c in SUPERCLASSES if c != "NORM"]
    return (db["NORM"] == 1) & (db[others].sum(axis=1) == 0)


def training_cohort_mask(db: pd.DataFrame, cohort: str) -> pd.Series:
    if cohort == "all":
        return pd.Series(True, index=db.index)
    if cohort == "norm":
        return strict_normal_mask(db)
    raise ValueError(f"Unknown training cohort: {cohort} (expected one of {TRAIN_COHORTS})")


# --------------------------------------------------------------------------- signals
def read_wfdb_12lead(record_path: str | Path) -> np.ndarray:
    """Read a WFDB record as float32 [12, time] in mV, canonical lead order."""
    import wfdb

    signal, fields = wfdb.rdsamp(str(record_path))
    x = signal.astype(np.float32).T
    names = list(fields.get("sig_name", []) or [])
    if names and names != LEADS:
        lookup = {name: i for i, name in enumerate(names)}
        if not all(lead in lookup for lead in LEADS):
            raise ValueError(f"{record_path}: missing standard leads, found {names}")
        x = x[[lookup[lead] for lead in LEADS], :]
    if x.shape[0] != 12:
        raise ValueError(f"{record_path}: expected 12 leads, got {x.shape[0]}")
    return np.ascontiguousarray(x)


def preprocess(x_mv: np.ndarray, preprocessing: Optional[dict] = None) -> np.ndarray:
    """Map a [12, T] mV signal to the network input.

    clip5            : clip to ±clip_mv, divide by clip_mv (original baseline)
    record_zscore    : clip, then per-record, per-lead z-score
    lead_standardize : clip, then per-lead z-score with TRAIN-set statistics
    """
    cfg = {**DEFAULT_PREPROCESSING, **(preprocessing or {})}
    mode = cfg["mode"]
    clip = cfg.get("clip_mv")
    x = np.asarray(x_mv, dtype=np.float32)
    if clip is not None:
        x = np.clip(x, -float(clip), float(clip))
    if mode == "clip5":
        out = x / float(clip) if clip is not None else x
    elif mode == "record_zscore":
        mu = x.mean(axis=1, keepdims=True)
        sd = np.maximum(x.std(axis=1, keepdims=True), 1e-2)
        out = (x - mu) / sd
    elif mode == "lead_standardize":
        if "lead_mean" not in cfg or "lead_std" not in cfg:
            raise ValueError("lead_standardize requires lead_mean / lead_std train statistics")
        mu = np.asarray(cfg["lead_mean"], dtype=np.float32)[:, None]
        sd = np.maximum(np.asarray(cfg["lead_std"], dtype=np.float32)[:, None], 1e-3)
        out = (x - mu) / sd
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode} (expected {PREPROCESSING_MODES})")
    return np.ascontiguousarray(out.astype(np.float32))


class SignalCache:
    """All PTB-XL records of one sampling rate in a single memory-mapped .npy.

    Reading ~20 000 WFDB files every epoch dominates training time on a laptop;
    the cache is built once (raw mV, float32, canonical lead order) and reused.
    """

    def __init__(self, array: np.ndarray, ecg_ids: Sequence[int]):
        self.array = array
        self.row = {int(e): i for i, e in enumerate(ecg_ids)}

    def __contains__(self, ecg_id: int) -> bool:
        return int(ecg_id) in self.row

    def get(self, ecg_id: int) -> np.ndarray:
        return np.asarray(self.array[self.row[int(ecg_id)]], dtype=np.float32)

    @staticmethod
    def paths(cache_dir: str | Path, sampling_rate: int):
        root = Path(cache_dir)
        stem = f"ptbxl_{sampling_rate}hz"
        return root / f"{stem}.npy", root / f"{stem}_ids.csv", root / f"{stem}_meta.json"

    @classmethod
    def build_or_load(cls, metadata: pd.DataFrame, data_dir: str | Path, sampling_rate: int,
                      cache_dir: str | Path, verbose: bool = True) -> "SignalCache":
        arr_path, ids_path, meta_path = cls.paths(cache_dir, sampling_rate)
        wanted = [int(i) for i in sorted(metadata.index)]
        data_dir = Path(data_dir).expanduser().resolve()
        if arr_path.exists() and ids_path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ids = pd.read_csv(ids_path)["ecg_id"].astype(int).tolist()
            if meta.get("complete") and meta.get("data_dir") == str(data_dir) and set(wanted) <= set(ids):
                arr = np.load(arr_path, mmap_mode="r")
                if verbose:
                    print(f"Signal cache loaded: {arr_path} {arr.shape}", flush=True)
                return cls(arr, ids)
            if verbose:
                print("Signal cache is stale or incomplete; rebuilding.", flush=True)

        col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        first = read_wfdb_12lead(data_dir / str(metadata.loc[wanted[0], col]))
        length = first.shape[1]
        arr_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"complete": False, "data_dir": str(data_dir), "sampling_rate": sampling_rate,
                "n": len(wanted), "length": length, "dtype": "float32", "unit": "mV",
                "lead_order": LEADS}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        arr = np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.float32, shape=(len(wanted), 12, length))
        for i, ecg_id in enumerate(wanted):
            x = first if i == 0 else read_wfdb_12lead(data_dir / str(metadata.loc[ecg_id, col]))
            if x.shape[1] != length:
                raise ValueError(f"ecg_id {ecg_id}: length {x.shape[1]} != {length}; cache needs equal lengths")
            arr[i] = x
            if verbose and (i + 1) % 2000 == 0:
                print(f"  cache {i + 1}/{len(wanted)}", flush=True)
        arr.flush()
        del arr
        pd.DataFrame({"ecg_id": wanted}).to_csv(ids_path, index=False)
        meta["complete"] = True
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if verbose:
            print(f"Signal cache written: {arr_path}", flush=True)
        return cls(np.load(arr_path, mmap_mode="r"), wanted)


def compute_lead_stats(signals: Sequence[np.ndarray], clip_mv: float = 5.0) -> dict:
    """Per-lead mean / std of clipped mV signals (use TRAIN records only)."""
    s = np.zeros(12, dtype=np.float64)
    ss = np.zeros(12, dtype=np.float64)
    n = 0
    for x in signals:
        x = np.clip(np.asarray(x, dtype=np.float64), -clip_mv, clip_mv)
        s += x.sum(axis=1)
        ss += (x ** 2).sum(axis=1)
        n += x.shape[1]
    mean = s / n
    std = np.sqrt(np.maximum(ss / n - mean ** 2, 0.0))
    return {"lead_mean": [float(v) for v in mean], "lead_std": [float(v) for v in std]}


# --------------------------------------------------------------------------- augmentation
class ECGAugment:
    """Label-preserving train-time augmentation applied to mV signals.

    Uses torch's RNG so that DataLoader workers get distinct, seeded streams.
    Amplitude scaling is deliberately mild and global (all leads together):
    QRS voltage is itself age-related, so aggressive or per-lead scaling would
    destroy signal the model should learn and break Einthoven relations.
    """

    def __init__(self, sampling_rate: int, p: float = 0.5, scale: float = 0.05,
                 noise_mv: float = 0.02, wander_mv: float = 0.1, crop_seconds: Optional[float] = None):
        self.fs = sampling_rate
        self.p = p
        self.scale = scale
        self.noise_mv = noise_mv
        self.wander_mv = wander_mv
        self.crop = int(round(crop_seconds * sampling_rate)) if crop_seconds else None

    def _hit(self) -> bool:
        return bool(torch.rand(()) < self.p)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=np.float32, copy=True)
        leads, length = x.shape
        if self.crop and self.crop < length:
            start = int(torch.randint(0, length - self.crop + 1, ()))
            x = x[:, start:start + self.crop]
            length = self.crop
        if self.scale > 0 and self._hit():
            x *= float(1.0 + (torch.rand(()) * 2 - 1) * self.scale)
        if self.wander_mv > 0 and self._hit():
            t = np.arange(length, dtype=np.float32) / self.fs
            freq = float(0.05 + torch.rand(()) * 0.45)
            phase = (torch.rand(leads) * 2 * np.pi).numpy()[:, None]
            amp = (torch.rand(leads) * self.wander_mv).numpy()[:, None]
            x += (amp * np.sin(2 * np.pi * freq * t[None, :] + phase)).astype(np.float32)
        if self.noise_mv > 0 and self._hit():
            x += torch.randn(leads, length).numpy() * self.noise_mv
        return x

    def describe(self) -> dict:
        return {"p": self.p, "scale": self.scale, "noise_mv": self.noise_mv,
                "wander_mv": self.wander_mv, "crop_samples": self.crop}


# --------------------------------------------------------------------------- dataset
class PTBXLAgeDataset(Dataset):
    """Returns (x [12, T], age, ecg_id). Original positional signature kept."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        data_dir: str | Path,
        sampling_rate: int = 100,
        clip_mv: Optional[float] = 5.0,
        preprocessing: Optional[dict] = None,
        cache: Optional[SignalCache] = None,
        augment: Optional[ECGAugment] = None,
    ):
        self.metadata = metadata.copy()
        self.data_dir = Path(data_dir)
        self.sampling_rate = sampling_rate
        self.preprocessing = {**DEFAULT_PREPROCESSING, **(preprocessing or {}), "clip_mv": clip_mv}
        self.cache = cache
        self.augment = augment
        self.filename_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
        if cache is None and self.filename_col not in self.metadata.columns:
            raise ValueError(f"Missing {self.filename_col} in PTB-XL metadata")
        self.ages = self.metadata["age"].to_numpy(dtype=np.float32)
        self.ids = self.metadata.index.to_numpy()

    def __len__(self) -> int:
        return len(self.metadata)

    def raw_signal(self, item: int) -> np.ndarray:
        ecg_id = int(self.ids[item])
        if self.cache is not None and ecg_id in self.cache:
            return self.cache.get(ecg_id)
        row = self.metadata.iloc[item]
        return read_wfdb_12lead(self.data_dir / str(row[self.filename_col]))

    def __getitem__(self, item: int):
        x = self.raw_signal(item)
        if self.augment is not None:
            x = self.augment(x)
        x = preprocess(x, self.preprocessing)
        return torch.from_numpy(x), torch.tensor(self.ages[item]), int(self.ids[item])


# --------------------------------------------------------------------------- model
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
    """1D ResNet age regressor.

    Default arguments give exactly the original baseline (8 blocks,
    64-128-256-512 channels), so old ``best_model.pt`` files load strictly.
    """

    def __init__(self, in_leads: int = 12, dropout: float = 0.15,
                 widths: Sequence[int] = (64, 128, 256, 512),
                 blocks_per_stage: Sequence[int] = (2, 2, 2, 2),
                 stem_kernel: int = 15):
        super().__init__()
        widths = [int(w) for w in widths]
        blocks_per_stage = [int(b) for b in blocks_per_stage]
        if len(widths) != len(blocks_per_stage) or min(blocks_per_stage) < 1:
            raise ValueError("widths and blocks_per_stage must have equal length and >=1 block per stage")
        self.config = {"in_leads": in_leads, "dropout": dropout, "widths": widths,
                       "blocks_per_stage": blocks_per_stage, "stem_kernel": stem_kernel}
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, widths[0], kernel_size=stem_kernel, stride=2, padding=stem_kernel // 2, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks = []
        in_ch = widths[0]
        for stage, (width, n_blocks) in enumerate(zip(widths, blocks_per_stage)):
            for b in range(n_blocks):
                stride = 2 if (stage > 0 and b == 0) else 1
                blocks.append(ResidualBlock1D(in_ch, width, stride, dropout))
                in_ch = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, 1))

    def init_output_bias(self, value: float) -> None:
        """Start predictions at the training mean age instead of 0 years."""
        with torch.no_grad():
            self.head[-1].bias.fill_(float(value))

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.pool(x)
        return self.head(x).squeeze(-1)


def build_model(config: Optional[dict] = None) -> ECGAgeResNet:
    config = dict(config or {})
    allowed = {"in_leads", "dropout", "widths", "blocks_per_stage", "stem_kernel"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown model_config keys: {sorted(unknown)}")
    return ECGAgeResNet(**config)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu"):
    """Load a checkpoint (old or new format); returns (model in eval mode, ckpt dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt.get("model_config"))
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device).eval()
    ckpt.setdefault("preprocessing", dict(DEFAULT_PREPROCESSING))
    ckpt.setdefault("sampling_rate", 100)
    return model, ckpt


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
