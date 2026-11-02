"""ROS environment for managed Gazebo runs: load saved setup and overlay built workspaces at run time.

pixi activates the environment once, before a task runs; a task that just built `ros/` or the robot
workspace would otherwise launch ROS without them. Loading here makes the first run on a fresh clone work.
"""

import os
import subprocess
from pathlib import Path

SETUP_FILE = ".dwarl/setup.env"


def root():
    return Path(os.environ.get("PIXI_PROJECT_ROOT") or Path.cwd())


def load_setup():
    """Saved answers of `dwarl setup`; variables already in the environment take precedence."""
    from dwarl.workflows.setup import load_saved

    for key, value in load_saved(root() / SETUP_FILE).items():
        os.environ.setdefault(key, value)
    workspace = os.environ.get("DWARL_ROBOT_WS")
    if workspace and (Path(workspace) / "src").is_dir():
        src = str((Path(workspace) / "src").resolve())
        paths = [p for p in os.environ.get("DWARL_ROBOT_PATH", "").split(os.pathsep) if p]
        if src not in paths:
            os.environ["DWARL_ROBOT_PATH"] = os.pathsep.join([*paths, src])


def overlay():
    """Refresh built overlays, including packages added since pixi activated this process."""
    load_setup()
    for workspace in (root() / ".pixi/robot/install", root() / "ros/install"):
        script = workspace / "local_setup.sh"
        if not script.is_file():
            continue
        dump = subprocess.run(["bash", "-c", f'AMENT_CURRENT_PREFIX="{workspace}" . "{script}" >/dev/null && env -0'],
                              capture_output=True, check=True).stdout.decode()
        os.environ.update(dict(item.split("=", 1) for item in dump.split("\0") if "=" in item))


def lease_directory():
    return Path(os.environ.get("XDG_RUNTIME_DIR", f"/tmp/dwarl-{os.getuid()}")) / "dwarl-domains"


def acquire_stack_lock(*, build=False):
    """Prevent a prepared build from overlapping managed simulator use, across checkouts."""
    import fcntl

    directory = lease_directory()
    directory.mkdir(parents=True, exist_ok=True)
    stream = (directory / "build.lock").open("a")
    try:
        fcntl.flock(stream, (fcntl.LOCK_EX if build else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise ValueError("A ROS build or managed simulator is active; finish it before starting this one.") from exc
    return stream


def _lease_held(domain):
    import fcntl

    path = lease_directory() / str(domain)
    if not path.exists():
        return False
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(stream, fcntl.LOCK_UN)
        return False


def orphaned_gazebo():
    """Gazebo servers launched by a DWARL test whose test has gone (its domain lease is free).

    Each managed test runs in its own session, recorded next to its domain lease; Gazebo stays in that
    session after its parents die. A Gazebo the user started themselves is never reported.
    """
    sessions = {}
    for record in lease_directory().glob("*.session"):
        try:
            sessions[int(record.read_text())] = int(record.stem)
        except ValueError:
            continue
    result = subprocess.run(["pgrep", "-f", "gz sim server"], capture_output=True, text=True)
    orphans = []
    for pid in result.stdout.split():
        try:
            domain = sessions.get(os.getsid(int(pid)))
        except ProcessLookupError:
            continue
        if domain is not None and not _lease_held(domain):
            orphans.append((pid, domain))
    return orphans


def health_warnings():
    warnings = []
    for pid, domain in orphaned_gazebo():
        warnings.append(f"orphaned DWARL Gazebo (pid {pid}, ROS domain {domain}) slows/corrupts tests: "
                        f"stop it with `kill {pid}`")
    governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if governor.is_file() and governor.read_text().strip() == "powersave":
        warnings.append("CPU governor is 'powersave': Gazebo may run far below real time; compare results "
                        "only between runs made under the same load (check compute.json real_time_factor)")
    return warnings
