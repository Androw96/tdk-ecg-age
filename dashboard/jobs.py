"""Owned background processes for the local dashboard; no shell execution."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO


class JobManager:
    """Keep one live process per cached dashboard manager.

    The Streamlit app should cache this object with ``st.cache_resource`` so
    reruns and browser sessions share the same process ownership and lock.
    Only this manager's live Popen objects can be stopped.
    """

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).expanduser().resolve()
        if not self.project_dir.is_dir():
            raise ValueError("A projekt könyvtára nem található.")
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._processes: dict[str, subprocess.Popen] = {}
        self._logs: dict[str, BinaryIO] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _input_path(self, value: Any, label: str, *, directory: bool = False) -> str:
        if value is None or not str(value).strip():
            raise ValueError(f"Hiányzó útvonal: {label}.")
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = self.project_dir / path
        path = path.resolve()
        exists = path.is_dir() if directory else path.is_file()
        if not exists:
            raise ValueError(f"Nem található {label}: {path}")
        return str(path)

    @staticmethod
    def _integer(config: dict, name: str, default: int, minimum: int = 1) -> str:
        value = config.get(name, default)
        if isinstance(value, bool):
            raise ValueError(f"Érvénytelen egész szám: {name}.")
        try:
            number = int(value)
            if float(value) != number or number < minimum:
                raise ValueError
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"A(z) {name} legalább {minimum} értékű egész szám legyen.") from None
        return str(number)

    def _command(self, kind: str, config: dict, output_dir: Path) -> list[str]:
        scripts = {
            "train": "train_ecg_age.py",
            "evaluate": "predict_ecg_age.py",
            "analysis": "evaluate_agegap.py",
        }
        if kind not in scripts:
            raise ValueError(f"Ismeretlen feladattípus: {kind}")
        script = self.project_dir / scripts[kind]
        if not script.is_file():
            raise ValueError(f"A futtatóprogram nem található: {script.name}")
        command = [sys.executable, "-u", str(script)]
        command += ["--data-dir", self._input_path(config.get("data_dir"), "adatkönyvtár", directory=True)]
        command += ["--output-dir", str(output_dir)]
        if kind == "train":
            for name, default, minimum in (
                ("epochs", 30, 1), ("batch_size", 64, 1),
                ("patience", 7, 1), ("seed", 42, 0),
            ):
                command += ["--" + name.replace("_", "-"), self._integer(config, name, default, minimum)]
            sampling_rate = self._integer(config, "sampling_rate", 100)
            if sampling_rate not in {"100", "500"}:
                raise ValueError("A mintavételi frekvencia 100 vagy 500 Hz legyen.")
            try:
                lr = float(config.get("lr", 1e-3))
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                raise ValueError("A tanulási ráta véges, pozitív szám legyen.") from None
            command += ["--sampling-rate", sampling_rate, "--lr", str(lr)]
        elif kind == "evaluate":
            device = str(config.get("device", "auto"))
            if device not in {"auto", "cpu", "cuda", "mps"}:
                raise ValueError("Ismeretlen számítási eszköz.")
            command += ["--checkpoint", self._input_path(config.get("checkpoint"), "modellfájl")]
            command += ["--batch-size", self._integer(config, "batch_size", 64)]
            command += ["--device", device, "--limit", self._integer(config, "limit", 0, 0)]
        else:
            command += ["--predictions", self._input_path(config.get("predictions"), "predikciós CSV")]
        return command

    def _persist(self, job: dict) -> None:
        (Path(job["output_dir"]) / "job.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _finish(self, job_id: str, returncode: int, status: str | None = None) -> None:
        job = self._jobs[job_id]
        job.update(status=status or ("completed" if returncode == 0 else "failed"),
                   returncode=returncode, finished_at=self._now())
        handle = self._logs.pop(job_id, None)
        if handle is not None:
            handle.close()
        self._processes.pop(job_id, None)
        self._persist(job)

    def _refresh(self) -> None:
        for job_id, process in list(self._processes.items()):
            returncode = process.poll()
            if returncode is not None:
                self._finish(job_id, returncode)

    def _snapshot(self, job: dict) -> dict:
        result = {**job, "command": list(job["command"])}
        log_path = Path(job["output_dir"]) / "process.log"
        try:
            with log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                length = handle.tell()
                handle.seek(max(0, length - 16_384))
                result["log"] = handle.read().decode("utf-8", errors="replace")
                if length > 16_384:
                    result["log"] = "[… a napló utolsó 16 KB-ja …]\n" + result["log"]
        except FileNotFoundError:
            result["log"] = ""
        return result

    def start(self, kind: str, config: dict) -> dict:
        """Validate inputs and launch a job in a new, never reused runs directory."""
        with self._lock:
            self._refresh()
            if self._processes:
                raise RuntimeError("Már fut egy feladat. Várd meg, vagy állítsd le az új indítása előtt.")
            # Validate everything before creating output or launching a process.
            job_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}"
            runs_dir = self.project_dir / "runs"
            output_dir = runs_dir / f"{kind}-{job_id}"
            command = self._command(kind, config, output_dir)
            if runs_dir.resolve() != runs_dir:
                raise ValueError("A runs könyvtár nem lehet szimbolikus hivatkozás.")
            runs_dir.mkdir(exist_ok=True)
            output_dir.mkdir(exist_ok=False)
            job = {
                "id": job_id, "kind": kind, "status": "running",
                "output_dir": str(output_dir), "command": command,
                "returncode": None, "started_at": self._now(), "finished_at": None,
            }
            self._jobs[job_id] = job
            self._persist(job)
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
            handle = (output_dir / "process.log").open("ab", buffering=0)
            self._logs[job_id] = handle
            try:
                self._processes[job_id] = subprocess.Popen(
                    command, cwd=self.project_dir, env=env,
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                    shell=False, close_fds=True,
                )
            except OSError as exc:
                handle.write(f"A folyamat nem indítható: {exc}\n".encode("utf-8"))
                self._finish(job_id, -1, "failed")
            return self._snapshot(job)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return [self._snapshot(job) for job in reversed(list(self._jobs.values()))]

    def get(self, job_id: str) -> dict:
        with self._lock:
            self._refresh()
            if job_id not in self._jobs:
                raise ValueError("A feladat nem található ebben a munkamenetben.")
            return self._snapshot(self._jobs[job_id])

    def stop(self, job_id: str) -> dict:
        """Stop only an owned live process. Finished jobs are left untouched."""
        with self._lock:
            self._refresh()
            if job_id not in self._jobs:
                raise ValueError("A feladat nem található ebben a munkamenetben.")
            process = self._processes.get(job_id)
            if process is not None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    returncode = process.wait(timeout=3)
                self._finish(job_id, returncode, "stopped")
            return self._snapshot(self._jobs[job_id])
