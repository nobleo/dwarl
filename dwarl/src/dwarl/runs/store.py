import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


@contextmanager
def lock(path, blocking=True):
    import fcntl

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def resolve_run(reference="latest", root=None):
    root = Path(root or Path.cwd() / "runs")
    if str(reference) != "latest":
        path = Path(reference).resolve()
        _manifest(path)
        return path
    candidates = []
    for file in root.glob("*/provenance.json"):
        try:
            data = _manifest(file.parent)
            candidates.append((data["created_ns"], file.parent.resolve()))
        except (ValueError, KeyError, OSError):
            continue
    if not candidates:
        raise FileNotFoundError(f"no valid training runs in {root}")
    return max(candidates)[1]


def _manifest(run):
    data = json.loads((run / "provenance.json").read_text())
    if data.get("schema") != "dwarl_run/1":
        raise ValueError(f"unsupported run: {run}")
    if not (run / "config.yaml").is_file() or not (run / "contract.yaml").is_file():
        raise ValueError(f"incomplete training run: {run}")
    return data


@dataclass(frozen=True)
class Checkpoint:
    run: Path
    path: Path
    iteration: int
    sha256: str
    contract: str


def resolve_checkpoint(run, selector="latest"):
    run = Path(run)
    manifest = json.loads((run / "checkpoint_manifest.json").read_text())
    if manifest.get("schema") != "dwarl_checkpoints/1":
        raise ValueError("unsupported checkpoint manifest")
    records = manifest["completed"]
    if selector == "best":
        best = manifest.get("best")
        if best is None:
            raise ValueError("no best checkpoint: no completed comparable validation workload")
        records = [r for r in records if r["sha256"] == best["sha256"]]
    elif selector != "latest":
        if not str(selector).isdigit():
            raise ValueError("checkpoint must be latest, best or an iteration number")
        records = [r for r in records if r["iteration"] == int(selector)]
    if not records:
        raise ValueError(f"no completed checkpoint {selector!r} in selected run {run}")
    record = max(enumerate(records), key=lambda item: (item[1]["iteration"], item[0]))[1]
    path = (run / record["file"]).resolve()
    if not path.is_relative_to(run.resolve()) or sha256(path) != record["sha256"]:
        raise ValueError(f"checkpoint missing or changed: {path}")
    if sha256(run / "contract.yaml") != record["contract"]:
        raise ValueError("saved training contract has changed")
    return Checkpoint(run, path, record["iteration"], record["sha256"], record["contract"])


def complete_checkpoint(run, path, iteration):
    run, path = Path(run), Path(path)
    with lock(run / ".manifest.lock"):
        manifest = json.loads((run / "checkpoint_manifest.json").read_text())
        record = {"file": str(path.relative_to(run)), "iteration": int(iteration), "sha256": sha256(path),
                  "contract": sha256(run / "contract.yaml")}
        manifest["completed"].append(record)
        write_json(run / "checkpoint_manifest.json", manifest)
    return record
