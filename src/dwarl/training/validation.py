"""Optional, isolated checkpoint checks and validation-only early stopping."""

import json
import math
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from dwarl.runs.store import lock, sha256, write_json
from dwarl.training.runner_adapter import RunnerAdapter


class EarlyStop(Exception):
    """A completed validation workload exhausted the configured patience."""


class TrainingChecks:
    def __init__(self, runner, run, cfg):
        self.runner, self.run, self.cfg = runner, Path(run), cfg
        self.state = {"version": 1, "last_checks": {}, "best_score": None,
                      "best_iteration": None, "bad_checks": 0, "stop_reason": None}

    def state_dict(self):
        return deepcopy(self.state)

    def restore(self, checkpoint, infos):
        # Checks happen after the immutable checkpoint is committed. Its sidecar
        # captures the post-check decision without rewriting any checkpoint bytes.
        path = self.run / "training_checks" / f"{checkpoint.sha256}.json"
        state = (infos or {}).get("dwarl_training")
        if path.exists():
            record = json.loads(path.read_text())
            if record["checkpoint"] != checkpoint.sha256:
                raise ValueError("training-check state belongs to a different checkpoint")
            state = record["state"]
        if state:
            if state.get("version") != 1:
                raise ValueError("unsupported training-check state")
            self.state = deepcopy(state)

    def scalar(self, tag, value, iteration):
        logger = getattr(self.runner, "logger", None)
        writer = getattr(logger, "writer", None)
        if writer is not None and value is not None:
            writer.add_scalar(tag, float(value), iteration)

    def observe_validation(self, value, iteration):
        """Patience counts completed comparable validations, not iterations."""
        early = self.cfg.train.early_stop
        if not early.enabled:
            return
        best = self.state["best_score"]
        if best is None or (value > best and value - best >= early.min_delta):
            self.state.update(best_score=value, best_iteration=iteration, bad_checks=0, stop_reason=None)
        else:
            self.state["bad_checks"] += 1
        bad = self.state["bad_checks"]
        if bad >= early.patience:
            self.state["stop_reason"] = (
                f"Validation success did not improve by {early.min_delta:g} for {bad} completed checks; "
                f"significant best={self.state['best_score']:.4f} at iteration {self.state['best_iteration']}.")
        for name, score in {"best_success": self.state["best_score"],
                            "best_iteration": self.state["best_iteration"],
                            "no_improvement_checks": bad,
                            "patience_remaining": max(0, early.patience - bad),
                            "stopped": bool(self.state["stop_reason"])}.items():
            self.scalar(f"EarlyStop/{name}", score, iteration)

    def evaluate(self, path, suite, cases, iteration):
        command = [sys.executable, "-m", "dwarl", "test", "--config", str(self.run / "config.yaml"),
                   "--run", str(self.run), "--checkpoint", str(iteration), "--sim", "mujoco",
                   "--suite", suite, "--set", f"test.max_cases={cases}",
                   "--set", "test.case=null", "--set", "test.dataset=null", "--set", "test.show=false",
                   "--set", "test.repeats=1", "--set", f"test.seed={self.cfg.train.seed}"]
        directory = self.run / suite
        directory.mkdir(exist_ok=True)
        previous = set((self.run / "tests").glob("*/summary.json"))
        checkpoint_hash = sha256(path)
        log_path = directory / f"{iteration}.log"
        print(f"Intermediate {suite}: iteration {iteration}, up to {cases} cases; log: {log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        reports = []
        if result.returncode in (0, 1):
            for file in (self.run / "tests").glob("*/summary.json"):
                if file in previous:
                    continue
                report = json.loads(file.read_text())
                if (report.get("suite") == suite and report["checkpoint"] == checkpoint_hash
                        and report["iteration"] == iteration and report["status"] == "completed"):
                    reports.append((file.stat().st_mtime_ns, file, report))
        prefix = "Validation" if suite == "validation" else "Benchmark"
        if not reports:
            self.scalar(f"{prefix}/complete", 0, iteration)
            print(f"Intermediate {suite} incomplete; excluded from selection/patience. See {log_path}", flush=True)
            return
        _, file, report = max(reports)
        score = report["simulators"]["mujoco"]
        value = score.get("success_rate")
        if not score["complete"] or value is None or not math.isfinite(value) or not 0 <= value <= 1:
            self.scalar(f"{prefix}/complete", 0, iteration)
            return
        for name in ("complete", "success_rate", "collision_rate", "completed", "expected"):
            self.scalar(f"{prefix}/{name}", score.get(name), iteration)
        if suite == "validation":
            with lock(self.run / ".manifest.lock"):
                manifest = json.loads((self.run / "checkpoint_manifest.json").read_text())
                best = manifest.get("best")
                if best and best["workload"] != report["workload"]:
                    raise ValueError("validation workload changed; refusing incomparable best-checkpoint selection")
                if best is None or value > best["value"]:
                    best = {"sha256": report["checkpoint"], "iteration": iteration,
                            "metric": "success_rate", "direction": "maximize", "ties": "earliest",
                            "value": value, "workload": report["workload"],
                            "evidence": str(file.relative_to(self.run))}
                    manifest["best"] = best
                    write_json(self.run / "checkpoint_manifest.json", manifest)
            self.scalar("Validation/best_success", best["value"], iteration)
            self.scalar("Validation/best_iteration", best["iteration"], iteration)
            self.observe_validation(value, iteration)
        print(f"Intermediate {suite}: success={value:.1%}; report: {file.parent / 'report/README.md'}", flush=True)

    def after_save(self, path):
        iteration = int(self.runner.current_learning_iteration)
        try:
            for suite in ("validation", "benchmark"):
                interval = self.cfg.train[f"{suite}_interval"]
                if (not interval or iteration == 0 or iteration % interval
                        or self.state["last_checks"].get(suite, -1) >= iteration):
                    continue
                self.state["last_checks"][suite] = iteration
                self.evaluate(path, suite, self.cfg.train[f"{suite}_cases"], iteration)
        finally:
            if self.cfg.train.validation_interval or self.cfg.train.benchmark_interval:
                digest = sha256(path)
                write_json(self.run / "training_checks" / f"{digest}.json",
                           {"checkpoint": digest, "iteration": iteration, "state": self.state_dict()})
            writer = getattr(getattr(self.runner, "logger", None), "writer", None)
            if writer is not None:
                writer.flush()
        if self.cfg.train.early_stop.enabled and self.state["stop_reason"]:
            raise EarlyStop(self.state["stop_reason"])


def attach_validation(runner, run, cfg):
    checks = TrainingChecks(runner, run, cfg)
    RunnerAdapter.for_runner(runner).after_save.append(checks.after_save)
    return checks
