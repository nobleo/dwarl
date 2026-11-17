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
        raise ValueError(f"missing {', '.join(missing)}; run DWARL through pixi (`pixi run setup`, web/docs/ros.md)")
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


def stop_owned(process, partition=None):
    import psutil

    # Gazebo may call setsid(), escaping both the worker's session and process
    # group. The random per-episode partition survives that detachment.
    detached = []
    if partition:
        for candidate in psutil.process_iter():
            try:
                if (candidate.uids().real == os.getuid()
                        and candidate.environ().get("GZ_PARTITION") == partition):
                    detached.append(candidate)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
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
    for sig, timeout in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        for candidate in detached:
            try:
                candidate.send_signal(sig)
            except psutil.NoSuchProcess:
                pass
        _, detached = psutil.wait_procs(detached, timeout=timeout)
        if not detached:
            break


class GazeboBackend:
    def __init__(self, contract, bundle, workload, directory, request, controller="dwarl"):
        prerequisites()
        self._routes = None  # Nav2 first plans per case, when test.unmapped_route is set
        package = {"mppi": "nav2_mppi_controller", "dwb": "dwb_core",
                   "rpp": "nav2_regulated_pure_pursuit_controller"}.get(controller)
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
        from dwarl.testing.unmapped import place_unmapped

        out = self.directory / "gazebo" / self.controller / case["id"]
        out.mkdir(parents=True)
        md = MapData.load(self.directory / case["map_file"])
        robot = OmegaConf.to_container(self.contract.robot, resolve=True)
        route, placement = case, {"route": "dataset_path"}
        source = self.request.get("unmapped_route")
        if source and int(self.request.get("unmapped_obstacles", 0)):
            if self._routes is None:
                from dwarl.testing.unmapped import recorded_routes

                self._routes = recorded_routes(source)
            if case["id"] not in self._routes:
                raise ValueError(f"test.unmapped_route: {source} has no obstacle-free Gazebo global plan "
                                 f"for selected case {case['id']}")
            route = dict(case, reference_path=self._routes[case["id"]])
            placement = {"route": "nav2_first_plan", "source": str(source)}
        unmapped = place_unmapped(md, route, int(self.request.get("unmapped_obstacles", 0)), robot,
                                  doorways=bool(self.request.get("unmapped_near_doorways", False)))
        export_gazebo(md, out, "world", unmapped)
        write_json(out / "input.json", {"case": case, "bundle": str(self.bundle), "controller": self.controller,
                   "unmapped_obstacles": unmapped.tolist(), "unmapped_placement": placement,
                   "request": OmegaConf.to_container(self.request),
                   "robot": robot})
        environment = dict(os.environ, ROS_DOMAIN_ID=str(self.domain), GZ_PARTITION=uuid4().hex,
                           ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
        self.partition = environment["GZ_PARTITION"]
        with (out / "launch.log").open("w") as log:
            self.process = subprocess.Popen([sys.executable, "-m", "dwarl.testing.backends.ros_worker", str(out)],
                                            env=environment, stdout=log, stderr=log, start_new_session=True)
            # Keep the legacy session record for diagnostics as well as the exact
            # partition used to reap Gazebo processes that detach from that session.
            (lease_directory() / f"{self.domain}.session").write_text(str(self.process.pid))
            try:
                self.process.wait(timeout=self.request.startup_timeout + case["time_budget"] * 10 + 60)
                if self.process.returncode:
                    raise RuntimeError(f"Gazebo worker failed ({self.process.returncode}); see {out / 'launch.log'}")
                return json.loads((out / "result.json").read_text())
            finally:
                stop_owned(self.process, self.partition)
                self.process = None
                if (out / "partial.json").is_file():
                    self.partial = json.loads((out / "partial.json").read_text())

    def close(self):
        if self.process is not None:
            stop_owned(self.process, self.partition)
        if self.lease is not None:
            self.lease.close()
        self.build_lock.close()
