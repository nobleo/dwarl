"""Read recorded evidence to describe state and recommend one next action."""

import json
import math
import time
from pathlib import Path

from dwarl.cli import ui
from dwarl.runs.store import resolve_checkpoint, resolve_run
from dwarl.workflows import context


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def run_info(run):
    run = resolve_run(run)
    provenance = read_json(run / "provenance.json", {})
    status = read_json(run / "status.json", {"state": "unknown"})
    if status.get("state") == "running":
        from dwarl.runs.process import process_identity

        try:
            live = status.get("process_identity") == process_identity(status["pid"])
        except (OSError, KeyError, TypeError, IndexError):
            live = False
        if not live:
            status = {**status, "state": "process_exited"}
    checkpoint, error = None, None
    try:
        chosen = resolve_checkpoint(run)
        checkpoint = {"iteration": chosen.iteration, "sha256": chosen.sha256}
    except (ValueError, OSError, KeyError) as exc:
        error = str(exc)
    reports = []
    for path in sorted((run / "tests").glob("*/summary.json")):
        try:
            value = read_json(path)
            reports.append({"directory": str(path.parent), "checkpoint": value["checkpoint"],
                            "suite": value["suite"], "status": value["status"], "passed": value["passed"]})
        except (ValueError, KeyError, OSError):
            continue
    return {"run": str(run), "robot": provenance.get("robot"), "preset": provenance.get("preset"),
            "state": status, "checkpoint": checkpoint, "checkpoint_error": error,
            "progress": read_json(run / "progress.json", {}), "reports": reports}


def inspect(robot=None, run=None):
    state = context.load()
    robot = robot or state.get("robot")
    result = {"schema": "dwarl_status/1", "robot": robot, "checks": None, "run": None, "installed": None,
              "next": ["setup"], "reason": "Select or create your robot integration."}
    if not robot:
        return result
    from dwarl.robot.diagnostics import inspect as inspect_robot

    checks = inspect_robot(robot)
    result["checks"] = checks
    if not checks["passed"]:
        result.update(next=["dwarl", "robot", "check", robot], reason="Resolve the model errors before training.")
        return result
    from dwarl.robot.profile import load_profile

    profile = load_profile(robot)
    selection = profile.dir / profile.raw["deployment"]["selected_policy"]
    if selection.is_file():
        from dwarl.artifacts.prepare import validate_bundle

        try:
            selected = read_json(selection)
            bundle = (profile.dir / selected["path"]).resolve()
            if not bundle.is_relative_to(profile.dir):
                raise ValueError("selected bundle is outside the integration")
            validate_bundle(bundle)
            result["installed"] = {"bundle": str(bundle), "checkpoint": selected["checkpoint"]}
        except (ValueError, OSError, KeyError) as exc:
            result["installed"] = {"error": str(exc)}
    reference = run or context.selected_run(robot)
    if not reference:
        result.update(next=["train", str(profile.path), "--preset", "smoke"],
                      reason="The model compiles. Exercise training and checkpointing with a small smoke run.")
        return result
    try:
        details = run_info(reference)
    except (ValueError, OSError, KeyError) as exc:
        result.update(next=["dwarl", "runs", "list"], reason=f"Selected run is unavailable: {exc}")
        return result
    result["run"] = details
    state_name = details["state"]["state"]
    if state_name in ("failed", "unknown"):
        result.update(next=["dwarl", "runs", "show", details["run"]],
                      reason="Inspect the run state and training log before continuing this run.")
        return result
    if state_name in ("running", "preparing"):
        result.update(next=["dwarl", "runs", "watch", details["run"]], reason="Training is in progress.")
        return result
    if not details["checkpoint"]:
        result.update(next=["dwarl", "runs", "show", details["run"]],
                      reason="This run has no valid completed checkpoint. Inspect its state and logs.")
        return result
    from dwarl.workflows.deploy import check_compatibility

    try:
        check_compatibility(resolve_checkpoint(reference), profile)
    except (ValueError, OSError, KeyError) as exc:
        result.update(next=["train", str(profile.path), "--preset", "smoke"],
                      reason=f"The current integration differs from the frozen run: {exc}")
        return result
    if state_name in ("interrupted", "process_exited"):
        result.update(next=["train", "--resume", details["run"]], reason="Continue from the completed checkpoint.")
        return result
    if details["preset"] == "smoke":
        result.update(next=["train", str(profile.path), "--preset", "navigation"],
                      reason="The smoke pipeline completed. Start a navigation recipe with periodic validation.")
        return result
    chosen = resolve_checkpoint(reference)
    try:
        chosen = resolve_checkpoint(reference, "best")
    except ValueError:
        pass
    pin = ["--run", details["run"], "--checkpoint", str(chosen.iteration)]
    comparable = [r for r in details["reports"] if r["checkpoint"] == chosen.sha256]
    validations = [r for r in comparable if r["suite"] == "validation"]
    heldout = [r for r in comparable if r["suite"] == "test"]
    if heldout and heldout[-1]["passed"]:
        if result["installed"] and result["installed"].get("checkpoint") == chosen.sha256:
            result.update(next=["dwarl", "report", heldout[-1]["directory"], "--open"],
                          reason="This checkpoint is installed. Review its evidence and follow the deployment guide.")
        else:
            result.update(next=["deploy", str(profile.path), *pin],
                          reason="Held-out evaluation passed for this checkpoint. Install its compatible bundle.")
    elif heldout or (validations and not validations[-1]["passed"]):
        report = (heldout or validations)[-1]
        result.update(next=["dwarl", "report", report["directory"], "--open"],
                      reason="Inspect incomplete/failed cases; use validation data for further tuning.")
    elif validations:
        result.update(next=["test", *pin, "--suite", "test"],
                      reason="Validation passed. Once the candidate is frozen, evaluate it on held-out cases.")
    else:
        result.update(next=["test", *pin, "--suite", "validation"],
                      reason="Evaluate this checkpoint on validation cases before selecting a final candidate.")
    return result


def execute(args):
    if args.command == "runs":
        if args.action == "list":
            rows, runs = [], []
            for path in sorted(Path("runs").glob("*/provenance.json"), reverse=True):
                try:
                    info = run_info(path.parent)
                    runs.append(info)
                    rows.append((path.parent.name, info["robot"], info["preset"] or "custom",
                                 info["state"]["state"], (info["checkpoint"] or {}).get("iteration")))
                except (ValueError, OSError, KeyError):
                    continue
            ui.table(["Run", "Robot", "Recipe", "State", "Checkpoint"], rows)
            if not rows:
                ui.info("No runs yet.")
                ui.next_step("setup")
            return 0, {"runs": runs}
        reference = args.run or context.selected_run()
        if not reference:
            raise ValueError("No run selected. Use `pixi run dwarl runs list` and pass a run directory.")
        if not math.isfinite(args.interval) or args.interval <= 0:
            raise ValueError("--interval must be positive and finite")
        while True:
            info = run_info(reference)
            ui.title("Run", info["run"])
            ui.table(["State", "Checkpoint", "Progress"], [(info["state"]["state"],
                     (info["checkpoint"] or {}).get("iteration"), json.dumps(info["progress"]))])
            if info["state"].get("error"):
                ui.finding("fail", info["state"]["error"])
            if info["checkpoint_error"]:
                ui.finding("check", info["checkpoint_error"])
            ui.info(f"Training log: {Path(info['run']) / 'training.log'}")
            if args.action != "watch" or args.json or info["state"]["state"] not in ("running", "preparing"):
                return 0, info
            time.sleep(args.interval)
    result = inspect(args.robot, args.run)
    if args.command == "status":
        ui.title("Status", f"Robot: {result['robot'] or 'not selected'}")
        if result["checks"]:
            for finding in result["checks"]["findings"]:
                ui.finding(finding["level"], finding["message"])
        if result["run"]:
            details = result["run"]
            ui.info(f"Run: {details['run']} · {details['state']['state']}")
            ui.info(f"Evaluations: {len(details['reports'])}")
        if result["installed"]:
            ui.info(f"Installed: {result['installed']}")
    ui.info(result["reason"])
    ui.next_step(*result["next"])
    return 0, result
