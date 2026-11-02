"""Managed native ROS execution. No ROS imports enter training's process."""

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from omegaconf import OmegaConf

from dwarl.ros_env import acquire_stack_lock, lease_directory
from dwarl.runs.store import write_json


def prerequisites(*, quiet=False):
    from dwarl.cli import ui
    from dwarl.ros_env import health_warnings, overlay

    missing = [name for name in ("ros2", "gz") if not shutil.which(name)]
    if missing:
        raise ValueError(f"missing {', '.join(missing)}; run DWARL through pixi (`pixi run setup`, docs/ros.md)")
    overlay()
    for package in ("dwarl_controller", "dwarl_bringup", "nav2_bringup", "ros_gz_bridge"):
        check = subprocess.run(["ros2", "pkg", "prefix", package], capture_output=True, text=True)
        if check.returncode:
            raise ValueError(f"ROS package {package} unavailable; run `pixi run setup` (or `pixi run build`)")
    findings = [{"level": "pass", "message": "ROS/Gazebo executables and required packages found"}]
    for warning in health_warnings():
        findings.append({"level": "check", "message": warning})
    if not quiet:
        for item in findings:
            ui.finding(item["level"], item["message"])
    return findings


def stop_owned(process):
    # Signal the owned group even when its leader has exited: descendants may remain.
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    process.wait()


class GazeboBackend:
    def __init__(self, contract, bundle, workload, directory, request, controller="dwarl"):
        prerequisites()
        package = {"mppi": "nav2_mppi_controller", "dwb": "dwb_core"}.get(controller)
        if package and subprocess.run(["ros2", "pkg", "prefix", package], capture_output=True).returncode:
            raise ValueError(f"ROS package {package} unavailable; install the selected Nav2 controller")
        from dwarl.runs.store import sha256

        adapter = contract.robot.ros.get("sim_launch")
        if not adapter:
            raise ValueError("robot integration has no ros.sim_launch adapter")
        location = subprocess.run(["ros2", "pkg", "prefix", "--share", adapter.package],
                                  capture_output=True, text=True)
        if location.returncode:
            raise ValueError(f"robot ROS package {adapter.package} unavailable; run `pixi run setup` "
                             "and select the integration that provides it")
        launch = Path(location.stdout.strip()) / "launch" / adapter.file
        if not launch.is_file():
            raise ValueError(f"robot spawn adapter is missing: {launch}")
        source = Path(contract.robot.profile_source)
        write_json(directory / "gazebo_stack.json", {
            "adapter": str(launch.resolve()), "adapter_sha256": sha256(launch),
            "observer_sha256": sha256(Path(__file__).with_name("ros_worker.py")),
            "ros_distro": os.environ.get("ROS_DISTRO"),
            "source_profile": str(source), "source_available": source.is_file(),
            "source_changed": sha256(source) != contract.robot.profile_sha256 if source.is_file() else None,
            "note": "Frozen MuJoCo assets do not freeze the installed ROS/Gazebo stack",
        })
        print(f"Gazebo adapter: {launch.resolve()}")
        self.contract, self.bundle, self.directory, self.request = contract, bundle, directory, request
        self.controller = controller
        self.process = None
        self.lease = None
        # Shared per-user leases coordinate DWARL consumers, including separate workspaces.
        leases = lease_directory()
        leases.mkdir(parents=True, exist_ok=True)
        for domain in range(120, 200):
            stream = (leases / str(domain)).open("a")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.domain, self.lease = domain, stream
                break
            except BlockingIOError:
                stream.close()
        if self.lease is None:
            raise RuntimeError("all managed ROS domains are in use")
        try:
            self.build_lock = acquire_stack_lock()
        except ValueError:
            self.lease.close()
            raise

    def execute_case(self, case):
        self.partial = {}
        from dwarl.datasets.generation.mapdata import MapData
        from dwarl.testing.scenarios import export_gazebo

        out = self.directory / "gazebo" / self.controller / case["id"]
        out.mkdir(parents=True)
        export_gazebo(MapData.load(self.directory / case["map_file"]), out, "world")
        write_json(out / "input.json", {"case": case, "bundle": str(self.bundle), "controller": self.controller,
                   "request": OmegaConf.to_container(self.request),
                   "robot": OmegaConf.to_container(self.contract.robot, resolve=True)})
        environment = dict(os.environ, ROS_DOMAIN_ID=str(self.domain), GZ_PARTITION=uuid4().hex,
                           ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
        with (out / "launch.log").open("w") as log:
            self.process = subprocess.Popen([sys.executable, "-m", "dwarl.testing.backends.ros_worker", str(out)],
                                            env=environment, stdout=log, stderr=log, start_new_session=True)
            # The worker leads a new session that its Gazebo keeps even when orphaned (see ros_env).
            (lease_directory() / f"{self.domain}.session").write_text(str(self.process.pid))
            try:
                self.process.wait(timeout=self.request.startup_timeout + case["time_budget"] * 10 + 60)
                if self.process.returncode:
                    raise RuntimeError(f"Gazebo worker failed ({self.process.returncode}); see {out / 'launch.log'}")
                return json.loads((out / "result.json").read_text())
            finally:
                stop_owned(self.process)
                self.process = None
                if (out / "partial.json").is_file():
                    self.partial = json.loads((out / "partial.json").read_text())

    def close(self):
        if self.process is not None:
            stop_owned(self.process)
        if self.lease is not None:
            self.lease.close()
        self.build_lock.close()
