#!/usr/bin/env python3
"""Bounded clearance-reward trials with validation-only selection and two levels of early stopping.

Run with the project's Python environment, e.g. uv run python scripts/iterate_policy.py RUN.
The parent run is read-only. Each trial starts a new training run on the parent's dataset.
"""

import argparse
import json
import math
import time
from pathlib import Path

from omegaconf import OmegaConf

from dwarl.config import compose_config
from dwarl.runs.store import resolve_checkpoint, resolve_run, write_json
from dwarl.training.contract import load_contract
from dwarl.workflows.test import execute as test
from dwarl.workflows.train import execute as train


def comparable_score(report, workload=None):
    if report["suite"] != "validation" or report["status"] != "completed":
        raise ValueError("selection requires completed validation evidence")
    if workload is not None and report["workload"] != workload:
        raise ValueError("validation workload changed; refusing cross-workload selection")
    score = report["simulators"]["mujoco"]
    if not score["complete"] or score["success_rate"] is None or score["collision_rate"] is None:
        raise ValueError("validation measurements are incomplete")
    return score


def execute(args):
    parent = resolve_run(args.run)
    checkpoint = resolve_checkpoint(parent, "best")
    contract = load_contract(parent)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    common = ["test.sim=mujoco", "test.suite=validation", f"test.max_cases={args.validation_cases}",
              "test.controllers=[dwarl]", "test.profile_compute=false", "test.controller_params=null",
              "test.case=null", "test.dataset=null", "test.repeats=1", f"test.seed={contract.seed}"]
    baseline_cfg = compose_config(parent / "config.yaml", common + [f"test.run={parent}", "test.checkpoint=best"])
    baseline_path, baseline = test(baseline_cfg, checkpoint)
    best = comparable_score(baseline)
    record = {"schema": "dwarl_policy_trials/1", "parent": str(parent), "parent_checkpoint": checkpoint.sha256,
              "baseline": str(baseline_path), "validation_workload": baseline["workload"],
              "selection": {"min_success_delta": args.min_delta, "collision_rate_must_not_increase": True,
                            "iteration_patience": args.iteration_patience, "maximum_trials": args.max_trials},
              "trials": [], "selected_run": str(parent), "stop_reason": None}
    write_json(output / "study.json", record)
    weights = args.weights[:args.max_trials]
    bad_trials = 0
    for index, weight in enumerate(weights, 1):
        assignments = [
            f"train.name=clearance_trial_{index}", "train.resume=null", "train.detached=false", "train.show=false",
            f"train.data.path={contract.dataset.path}", f"train.max_iterations={args.max_iterations}",
            f"train.save_interval={args.validation_interval}", f"train.validation_interval={args.validation_interval}",
            f"train.validation_cases={args.validation_cases}", "train.benchmark_interval=0",
            "train.early_stop.enabled=true", f"train.early_stop.patience={args.training_patience}",
            f"train.early_stop.min_delta={args.min_delta}", f"recipe.rewards.proximity.weight={weight}",
        ]
        cfg = compose_config(parent / "config.yaml", common + assignments)
        OmegaConf.save(cfg, output / f"trial_{index}.yaml", resolve=True)
        started = time.monotonic()
        run = train(cfg)
        elapsed = time.monotonic() - started
        manifest = json.loads((run / "checkpoint_manifest.json").read_text())
        if not manifest.get("best"):
            raise RuntimeError(f"No completed validation for {run}; excluded from selection")
        evidence = run / manifest["best"]["evidence"]
        report = json.loads(evidence.read_text())
        score = comparable_score(report, baseline["workload"])
        improved = (score["success_rate"] > best["success_rate"]
                    and score["success_rate"] - best["success_rate"] >= args.min_delta
                    and score["collision_rate"] <= best["collision_rate"])
        if improved:
            best = score
            record["selected_run"] = str(run)
            bad_trials = 0
        else:
            bad_trials += 1
        record["trials"].append({"run": str(run), "proximity_weight": weight, "best_evidence": str(evidence),
                                 "score": score, "accepted": improved, "workflow_wall_s": elapsed,
                                 "training_status": json.loads((run / "status.json").read_text())})
        if bad_trials >= args.iteration_patience:
            record["stop_reason"] = "no_improvement_patience"
        write_json(output / "study.json", record)
        print(json.dumps(record["trials"][-1], indent=2), flush=True)
        if record["stop_reason"]:
            break
    record["stop_reason"] = record["stop_reason"] or "trial_budget"
    record["next_step"] = "Verify any selected candidate twice in Gazebo on validation cases before held-out testing."
    write_json(output / "study.json", record)
    print(f"Policy iteration stopped: {record['stop_reason']}; selected run: {record['selected_run']}", flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="parent run with a selected best checkpoint")
    parser.add_argument("--output", default=f"runs/studies/policy-{time.strftime('%Y%m%d-%H%M%S')}")
    parser.add_argument("--weights", nargs="+", type=float, default=[7.5, 10.0])
    parser.add_argument("--max-trials", type=int, default=2)
    parser.add_argument("--iteration-patience", type=int, default=2)
    parser.add_argument("--training-patience", type=int, default=2)
    parser.add_argument("--min-delta", type=float, default=.03125)
    parser.add_argument("--max-iterations", type=int, default=250)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-cases", type=int, default=32)
    arguments = parser.parse_args()
    if min(arguments.max_trials, arguments.iteration_patience, arguments.training_patience,
           arguments.max_iterations, arguments.validation_interval, arguments.validation_cases) <= 0:
        parser.error("budgets, intervals and patience must be positive")
    if not 0 < arguments.min_delta <= 1:
        parser.error("min-delta must be in (0, 1]")
    if arguments.max_iterations <= arguments.validation_interval:
        parser.error("max-iterations must exceed the first validation interval")
    if any(not math.isfinite(weight) or weight <= 0 for weight in arguments.weights):
        parser.error("clearance penalty weights must be finite and positive")
    execute(arguments)
