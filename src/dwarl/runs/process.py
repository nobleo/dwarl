import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf


def detach(cfg):
    directory = Path(".dwarl/processes").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    path = directory / f"{token}.yaml"
    cfg.train.detached = False
    OmegaConf.save(cfg, path, resolve=True)
    with (directory / f"{token}.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "dwarl", "train", "--config", str(path)],
                                   start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=log)
    print(f"Started PID {process.pid}; run location and progress: {log.name}")


def stop(reference):
    from dwarl.runs.store import resolve_run

    run = resolve_run(reference)
    status = json.loads((run / "status.json").read_text())
    pid = status.get("pid")
    if status["state"] != "running" or not pid:
        raise ValueError("run is not running")
    if status.get("process_identity") != process_identity(pid):
        raise ValueError("recorded process has exited; refusing to signal a reused PID")
    os.kill(pid, signal.SIGINT)


def process_identity(pid):
    start = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip() + ":" + start


def watch(reference, off=False):
    """Toggle the live view of a running training (``NavVecEnv.viewer_trigger``)."""
    from dwarl.runs.store import resolve_run

    run = resolve_run(reference or "latest")
    status = json.loads((run / "status.json").read_text())
    if status["state"] != "running":
        raise ValueError(f"{run} is not training (state: {status['state']})")
    trigger = run / "live_view"
    if off or trigger.exists():
        trigger.unlink(missing_ok=True)
        print(f"Live view off: {run}")
    else:
        trigger.touch()
        print(f"Live view on: {run} (opens within a second; close the window or repeat to stop)")
