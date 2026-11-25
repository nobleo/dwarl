"""One managed CLI process per lane; output remains on disk after the browser closes."""

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from uuid import uuid4


def command_arguments(argv):
    from dwarl.cli import parser

    if (not isinstance(argv, list) or not argv or len(argv) > 128
            or any(not isinstance(arg, str) or len(arg) > 4096 for arg in argv)):
        raise ValueError("command must be an argument array")
    if argv[0] not in ("train", "test", "deploy", "doctor", "status", "next", "report"):
        raise ValueError("unsupported dashboard command")
    try:
        args = parser().parse_args(argv)
    except SystemExit as exc:
        raise ValueError("invalid CLI arguments; see dwarl <command> --help") from exc
    if getattr(args, "detached", False):
        raise ValueError("dashboard jobs must remain attached")
    if args.command == "train":
        if any(item.partition("=")[0] == "train.detached" for item in args.set):
            raise ValueError("dashboard manages train.detached")
        argv = [*argv, "--set", "train.detached=false"]
    return [sys.executable, "-u", "-m", "dwarl", *argv,
            *([] if "--plain" in argv else ["--plain"])]


def pin_checkpoint(root, argv):
    """Resolve a moving selector before spawning a concurrent evaluation."""
    if argv[0] not in ("test", "deploy"):
        return argv
    from dwarl.cli import parser
    from dwarl.runs.store import resolve_checkpoint, resolve_run

    args = parser().parse_args(argv)
    cfg = inspect_command(root, argv, "--print-config")
    request = cfg[args.command]
    reference = request["run"]
    run = resolve_run("latest", root / "runs") if reference == "latest" else Path(reference)
    chosen = resolve_checkpoint(run, str(request["checkpoint"]))
    # Reject override routes that would defeat pinning or duplicate CLI assignments.
    if any(item.partition("=")[0] in (f"{args.command}.run", f"{args.command}.checkpoint")
           for item in args.set):
        raise ValueError("use the run/checkpoint selectors instead of --set for these inputs")
    result, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg in ("--run", "--checkpoint"):
            skip = True
        elif not arg.startswith(("--run=", "--checkpoint=")):
            result.append(arg)
    return [*result, "--run", str(chosen.run), "--checkpoint", str(chosen.iteration)]


def inspect_command(root, argv, flag):
    """Use a separate CLI process so configuration resolution matches actual execution."""
    command = command_arguments(argv)
    if flag not in command:
        command.append(flag)
    if "--json" not in command:
        command.append("--json")
    result = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL, capture_output=True,
                            text=True, timeout=90)
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise ValueError(result.stderr[-4000:] or "CLI returned no inspection result") from exc
    if result.returncode:
        raise ValueError(data.get("error", result.stderr[-4000:]))
    return data


def tail(path, size=48000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stream.read(size).decode(errors="replace"))
    except OSError:
        return ""


class Jobs:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.process = None
        self.job = None
        self.attached = False

    @classmethod
    def reconnect(cls, root):
        """Reconnect old dashboard jobs without stopping or taking shutdown ownership of them."""
        import psutil

        managers = {"training": cls(root), "evaluation": cls(root)}
        directory = (Path(root) / ".dwarl/web").resolve()
        for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
            try:
                args = process.info["cmdline"] or []
                if "-m" not in args or args[args.index("-m") + 1:args.index("-m") + 2] != ["dwarl"]:
                    continue
                if Path(process.info["cwd"] or "/").resolve() != Path(root).resolve():
                    continue
                telemetry = process.environ().get("DWARL_WEB_TELEMETRY")
                if not telemetry or not Path(telemetry).resolve().is_relative_to(directory):
                    continue
                record = Path(telemetry).parent / "job.json"
                job = json.loads(record.read_text())
                if job.get("returncode") is not None:
                    continue
                lane = "training" if job["argv"][0] == "train" else "evaluation"
                manager = managers[lane]
                manager.process = ExistingProcess(process.pid, record)
                manager.job, manager.attached = job, True
            except (psutil.Error, OSError, ValueError, KeyError, IndexError):
                continue
        return managers

    def start(self, argv, telemetry=True):
        command_arguments(argv)
        argv = pin_checkpoint(self.root, argv)
        command = command_arguments(argv)
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                raise ValueError("a dashboard command is already running")
            directory = self.root / ".dwarl/web" / uuid4().hex
            directory.mkdir(parents=True)
            log = directory / "command.log"
            env = {**os.environ, "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"}
            env["DWARL_WEB_TELEMETRY"] = str(directory / "live.json")
            with log.open("wb") as output:
                self.process = subprocess.Popen(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            self.attached = False
            self.job = {"argv": argv, "started_at": time.time(), "log": str(log),
                        "id": directory.name, "telemetry": str(directory / "live.json"), "viewer_protocol": 3}
            from dwarl.runs.store import write_json

            write_json(directory / "job.json", {**self.job, "returncode": None})
            threading.Thread(target=self.record_exit, args=(self.process, dict(self.job), directory),
                             daemon=True).start()
            self.view(telemetry)
            return self.snapshot()

    def view(self, enabled, settings=None, realtime=False):
        from dwarl.runs.store import write_json

        if self.job is None:
            return
        settings = settings or {}
        camera = {"mode": "overview" if settings.get("mode") == "overview" else "environment",
                  "environment": max(0, int(settings.get("environment", 0)))}
        # Pacing only ever slows a watched test; training keeps its full throughput.
        realtime = realtime is True and self.job["argv"][:1] == ["test"]
        write_json(Path(self.job["telemetry"]).with_suffix(".control.json"),
                   {**camera, "realtime": realtime, "expires_at": time.time() + 3 if enabled else 0})

    @staticmethod
    def record_exit(process, job, directory):
        from dwarl.runs.store import write_json

        code = process.wait()
        write_json(directory / "job.json", {**job, "returncode": code, "finished_at": time.time()})

    def snapshot(self):
        if self.job is None:
            return None
        code = self.process.poll()
        return {**self.job, "returncode": code, "running": code is None,
                "attached": self.attached, "output": tail(self.job["log"])}

    def stop(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                raise ValueError("no dashboard command is running")
            # Signal the CLI first: its backend owns simulator shutdown and report persistence.
            try:
                self.process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass

    def close(self):
        self.view(False)
        if self.attached:
            return  # The original dashboard still owns this job's lifetime.
        if self.process is None or self.process.poll() is not None:
            return
        self.stop()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            # This session/process group was created by us, never an unrelated simulator.
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()


class ExistingProcess:
    """Minimal signal/poll adapter guarded against PID reuse; no ownership of other processes."""

    def __init__(self, pid, record):
        from dwarl.runs.process import process_identity

        self.pid, self.record = pid, record
        self.identity = process_identity(pid)

    def poll(self):
        import psutil

        from dwarl.runs.process import process_identity

        try:
            if (self.identity == process_identity(self.pid)
                    and psutil.Process(self.pid).status() != psutil.STATUS_ZOMBIE):
                return None
        except (OSError, psutil.Error):
            pass
        try:
            code = json.loads(self.record.read_text()).get("returncode")
            return code if code is not None else 255
        except (OSError, ValueError):
            return 255

    def send_signal(self, sig):
        if self.poll() is None:
            os.kill(self.pid, sig)
