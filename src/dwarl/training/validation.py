"""Validation runs in a child process, isolating training RNG and model state."""

import json
import os
import subprocess
import sys

from dwarl.runs.store import lock, sha256, write_json
from dwarl.training.runner_adapter import RunnerAdapter


def attach_validation(runner, run, cfg):
    def validate(path):
        iteration = int(runner.current_learning_iteration)
        if iteration == 0 or iteration % cfg.train.validation_interval:
            return
        command = [sys.executable, "-m", "dwarl", "test", "--config", str(run / "config.yaml"),
                   "--run", str(run), "--checkpoint", str(iteration), "--sim", "mujoco",
                   "--suite", "validation", "--set", f"test.max_cases={cfg.train.validation_cases}",
                   "--set", "test.case=null", "--set", "test.dataset=null", "--set", "test.show=false",
                   "--set", "test.repeats=1", "--set", f"test.seed={cfg.train.seed}"]
        directory = run / "validation"
        directory.mkdir(exist_ok=True)
        previous = set((run / "tests").glob("*/summary.json"))
        checkpoint_hash = sha256(path)
        with (directory / f"{iteration}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        if result.returncode not in (0, 1):
            print(f"Validation infrastructure failed: {directory / f'{iteration}.log'}")
            return
        reports = []
        for file in (run / "tests").glob("*/summary.json"):
            if file in previous:
                continue
            report = json.loads(file.read_text())
            if (report.get("suite") == "validation" and report["checkpoint"] == checkpoint_hash
                    and report["iteration"] == iteration and report["status"] == "completed"):
                reports.append((file.stat().st_mtime_ns, file, report))
        if not reports:
            return
        _, file, report = max(reports)
        score = report["simulators"]["mujoco"]
        if not score["complete"]:
            return
        with lock(run / ".manifest.lock"):
            manifest = json.loads((run / "checkpoint_manifest.json").read_text())
            best = manifest.get("best")
            if best and best["workload"] != report["workload"]:
                raise ValueError("validation workload changed; refusing incomparable best-checkpoint selection")
            if best is None or score["success_rate"] > best["value"]:
                manifest["best"] = {"sha256": report["checkpoint"], "iteration": iteration,
                                    "metric": "success_rate", "direction": "maximize", "ties": "earliest",
                                    "value": score["success_rate"], "workload": report["workload"],
                                    "evidence": str(file.relative_to(run))}
                write_json(run / "checkpoint_manifest.json", manifest)
    RunnerAdapter.for_runner(runner).after_save.append(validate)
