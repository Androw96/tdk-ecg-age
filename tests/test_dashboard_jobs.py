"""Real subprocess checks using tiny fixtures, without training a model."""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from dashboard.jobs import JobManager


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tdk-jobs-")
        self.project = Path(self.temp.name).resolve()
        self.data = self.project / "data with spaces"
        self.data.mkdir()
        self.checkpoint = self.project / "model.pt"
        self.checkpoint.write_bytes(b"fixture")
        self.predictions = self.project / "predictions.csv"
        self.predictions.write_text("ecg_id,age,predicted_age,split\n", encoding="utf-8")
        self.manager = JobManager(self.project)

    def tearDown(self):
        for job in self.manager.list_jobs():
            self.manager.stop(job["id"])
        self.temp.cleanup()

    def script(self, name="train_ecg_age.py", source="print('done', flush=True)"):
        (self.project / name).write_text(source, encoding="utf-8")

    def await_finished(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = self.manager.get(job_id)
            if job["status"] != "running":
                return job
            time.sleep(0.02)
        self.fail("Fixture process did not finish")

    def test_completed_job_uses_correct_executable_environment_and_unique_output(self):
        self.script(source="import os,sys,json\nprint(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd(), 'backend': os.getenv('MPLBACKEND')}), flush=True)")
        first = self.await_finished(self.manager.start("train", {"data_dir": self.data})["id"])
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["returncode"], 0)
        self.assertEqual(first["command"][:2], [sys.executable, "-u"])
        observed = json.loads(first["log"])
        self.assertEqual(observed["cwd"], str(self.project))
        self.assertEqual(observed["backend"], "Agg")
        self.assertIn(str(self.data), observed["args"])
        self.assertEqual(Path(first["output_dir"]).parent, self.project / "runs")
        manifest = json.loads((Path(first["output_dir"]) / "job.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        second = self.await_finished(self.manager.start("train", {"data_dir": self.data})["id"])
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual(self.manager.list_jobs()[0]["id"], second["id"])

    def test_failed_job_surfaces_stderr_and_exit_code(self):
        self.script(source="import sys\nprint('fixture traceback', file=sys.stderr, flush=True)\nsys.exit(7)")
        job = self.await_finished(self.manager.start("train", {"data_dir": self.data})["id"])
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["returncode"], 7)
        self.assertIn("fixture traceback", job["log"])
        self.assertEqual(self.manager.stop(job["id"])["status"], "failed")

    def test_only_one_job_and_stop_owned_process(self):
        self.script(source="import time\nprint('ready', flush=True)\ntime.sleep(30)")
        job = self.manager.start("train", {"data_dir": self.data})
        with self.assertRaises(RuntimeError):
            self.manager.start("train", {"data_dir": self.data})
        stopped = self.manager.stop(job["id"])
        self.assertEqual(stopped["status"], "stopped")
        self.assertIsNotNone(stopped["returncode"])
        self.assertEqual(self.manager.stop(job["id"])["status"], "stopped")
        with self.assertRaises(ValueError):
            self.manager.stop("unowned-id")

    def test_concurrent_launches_are_serialized(self):
        self.script(source="import time\ntime.sleep(30)")
        barrier = threading.Barrier(2)
        outcomes = []

        def launch():
            barrier.wait()
            try:
                outcomes.append(self.manager.start("train", {"data_dir": self.data})["status"])
            except RuntimeError:
                outcomes.append("blocked")

        threads = [threading.Thread(target=launch) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        self.assertCountEqual(outcomes, ["running", "blocked"])

    def test_evaluation_and_analysis_command_contract(self):
        self.script("predict_ecg_age.py")
        self.script("evaluate_agegap.py")
        evaluation = self.await_finished(self.manager.start("evaluate", {
            "data_dir": self.data, "checkpoint": self.checkpoint,
            "batch_size": 16, "device": "cpu", "limit": 50,
        })["id"])
        for expected in ("--checkpoint", str(self.checkpoint), "--device", "cpu", "--limit", "50"):
            self.assertIn(expected, evaluation["command"])
        analysis = self.await_finished(self.manager.start("analysis", {
            "data_dir": self.data, "predictions": self.predictions,
        })["id"])
        self.assertIn(str(self.predictions), analysis["command"])

    def test_bad_inputs_create_no_outputs(self):
        self.script()
        for config in ({"epochs": 0}, {"epochs": 1.5}, {"batch_size": True},
                       {"lr": float("nan")}, {"sampling_rate": 250}, {"seed": -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.manager.start("train", {"data_dir": self.data, **config})
        with self.assertRaises(ValueError):
            self.manager.start("train", {"data_dir": self.project / "missing"})
        with self.assertRaises(ValueError):
            self.manager.start("unknown", {"data_dir": self.data})
        self.assertFalse((self.project / "runs").exists())

    def test_runs_symlink_cannot_redirect_outputs(self):
        self.script()
        with tempfile.TemporaryDirectory() as external:
            (self.project / "runs").symlink_to(external, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.manager.start("train", {"data_dir": self.data})
            self.assertEqual(list(Path(external).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
