"""Common ROS build preparation for CLI shortcuts and direct invocations."""

import os
import shutil
import subprocess
import time
from pathlib import Path

from dwarl.cli import ui


def ros(robot=None):
    from dwarl.ros_env import overlay
    from dwarl.workflows.setup import load_saved

    if os.environ.get("DWARL_PREBUILT_ROS"):  # e.g. the Docker image: packages are built and sourced
        overlay()
        ui.finding("pass", "ROS packages prebuilt in this environment (DWARL_PREBUILT_ROS)")
        return None
    pixi = os.environ.get("PIXI_EXE") or shutil.which("pixi")
    root = Path(__file__).resolve().parents[3]
    if not pixi or not (root / "pyproject.toml").is_file():
        raise ValueError("ROS builds need the DWARL checkout and pixi. Run `pixi run setup` there.")
    values = load_saved()
    env = {**os.environ, **{k: os.environ.get(k, v) for k, v in values.items()}}
    standalone = None
    if robot:
        from dwarl.robot.profile import _package_name, load_profile

        profile = load_profile(robot)
        package = next((p for p in profile.path.parents if (p / "package.xml").exists()), None)
        workspace = next((p.parent for p in profile.path.parents if p.name == "src"), None)
        # An explicit robot always determines its build workspace, including standalone profiles.
        env.pop("DWARL_ROBOT_WS", None)
        env.pop("DWARL_ROBOT_PACKAGES", None)
        if package and workspace:
            env.update(DWARL_ROBOT_WS=str(workspace), DWARL_ROBOT_PACKAGES=_package_name(package / "package.xml"))
        elif package:
            standalone = package
    task = "build-robot" if env.get("DWARL_ROBOT_WS") else "build"
    commands = [[pixi, "run", task]]
    if standalone:
        commands.append([pixi, "run", "colcon", "build", "--symlink-install", "--base-paths", str(standalone),
                         "--build-base", str(root / ".pixi/robot/build"),
                         "--install-base", str(root / ".pixi/robot/install"),
                         "--cmake-args", "-DBUILD_TESTING=OFF"])
    log = Path(".dwarl/logs") / f"build-{time.time_ns()}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # Do not replace libraries while any managed Gazebo stack is using them.
    import fcntl
    from contextlib import ExitStack

    from dwarl.ros_env import acquire_stack_lock, lease_directory

    with ExitStack() as held:
        held.enter_context(acquire_stack_lock(build=True))
        for lease in sorted(lease_directory().glob("[0-9]*")):
            if not lease.name.isdigit():
                continue
            stream = held.enter_context(lease.open("a"))
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValueError("A managed Gazebo test is active; finish it before rebuilding ROS packages.") from exc
        with ui.progress(f"Preparing ROS packages · log: {log}"), log.open("w") as output:
            for command in commands:
                status = subprocess.run(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
                if status.returncode:
                    break
    if status.returncode or ui.verbose:
        ui.info("\n".join(log.read_text().splitlines()[-40:]))
    if status.returncode:
        raise ValueError(f"ROS build failed. Full log: {log.resolve()}")
    overlay()
    ui.finding("pass", "ROS packages prepared (incremental build)")
    return log
