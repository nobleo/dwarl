"""Read lightweight run artifacts; never hash checkpoints on a browser refresh."""

import json
from pathlib import Path

from dwarl.web.jobs import tail


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def runs(root):
    result = []
    for path in sorted((root / "runs").glob("*/provenance.json"), reverse=True):
        provenance = read(path, {})
        state = read(path.parent / "status.json", {"state": "preparing"})
        if state.get("state") == "running":
            from dwarl.runs.process import process_identity

            try:
                if state.get("process_identity") != process_identity(state["pid"]):
                    state = {**state, "state": "process_exited"}
            except (OSError, KeyError, TypeError, IndexError):
                state = {**state, "state": "process_exited"}
        result.append({"name": path.parent.name, "robot": provenance.get("robot"),
                       "preset": provenance.get("preset"), "state": state,
                       "progress": read(path.parent / "progress.json", {}),
                       "checkpoints": len(read(path.parent / "checkpoint_manifest.json", {}).get("completed", []))})
    return result


def run_directory(root, name):
    run = (root / "runs" / name).resolve()
    if not name or not run.is_relative_to((root / "runs").resolve()) or not (run / "provenance.json").is_file():
        raise ValueError("unknown run")
    return run


def details(root, name):
    run = run_directory(root, name)
    evaluations = []
    for directory in sorted((run / "tests").glob("*"), reverse=True):
        if directory.is_dir():
            evaluations.append({"name": directory.name, "summary": read(directory / "summary.json"),
                                "report": (directory / "report/index.html").is_file(),
                                "episodes": read(directory / "episodes.json", {})})
    manifest = read(run / "checkpoint_manifest.json", {})
    best = (manifest.get("best") or {}).get("sha256")
    checkpoints = sorted(manifest.get("completed", []), key=lambda item: item["iteration"], reverse=True)
    return {"evaluations": evaluations, "output": tail(run / "training.log"),
            "checkpoints": [{**item, "best": item["sha256"] == best} for item in checkpoints]}


def history(root):
    records = []
    for file in (root / ".dwarl/web").glob("*/job.json"):
        item = read(file)
        if item:
            records.append(item)
    return sorted(records, key=lambda item: item["started_at"], reverse=True)[:100]


def case_details(root, run, evaluation, case_id, backend):
    directory = (root / "runs" / run / "tests" / evaluation).resolve()
    if (not directory.is_relative_to(root / "runs") or Path(run).name != run
            or Path(evaluation).name != evaluation):
        raise ValueError("unknown evaluation")
    rows = read(directory / "episodes.json", {}).get(backend, [])
    row = next((row for row in rows if row["case_id"] == case_id), None)
    if row is None:
        raise ValueError("unknown case")
    trace = None
    if row.get("trace_file"):
        path = (directory / row["trace_file"]).resolve()
        if path.is_relative_to(directory):
            trace = read(path)
    return directory, row, trace


def retry_arguments(root, data):
    from uuid import uuid4

    from omegaconf import OmegaConf

    directory, row, _ = case_details(root, data["run"], data["evaluation"], data["case"], data["backend"])
    summary = read(directory / "summary.json")
    if not summary:
        raise ValueError("wait for this evaluation to finish before rerunning a case")
    request = OmegaConf.load(directory / "request.yaml")
    request.run = str(directory.parents[1])
    request.checkpoint = str(summary["iteration"])
    request.dataset = str(directory / "maps")
    request.case_id = row["case_id"]
    request.sim = row["backend"]
    request.controllers = [row["controller"]]
    if (directory / "controller_overrides.yaml").is_file():
        request.controller_params = str(directory / "controller_overrides.yaml")
    path = root / ".dwarl/web/retries" / f"{uuid4().hex}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create({"test": OmegaConf.to_container(request)}), path)
    return ["test", "--config", str(path)]
