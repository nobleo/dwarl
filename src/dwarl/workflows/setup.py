"""First-run setup: ask for the robot repository once, remember it, build and check the ROS side."""

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from dwarl.robot.profile import PROFILE_FILE

SETUP_FILE = Path(".dwarl/setup.env")
_SKIP = {"build", "install", "log", ".git", "policies", "runs", ".pixi", ".venv"}


def load_saved(path=SETUP_FILE):
    values = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.removeprefix("export ").strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = " ".join(shlex.split(value))
    return values


def save(values, path=SETUP_FILE):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Written by `dwarl setup`; loaded by pixi activation. Environment variables take precedence."]
    lines += [f"export {key}={shlex.quote(value)}" for key, value in values.items() if value]
    path.write_text("\n".join(lines) + "\n")


def integrations(workspace):
    """ROS packages under ``workspace/src`` that contain a robot profile: (package name, profile path)."""
    src = Path(workspace) / "src"
    found = []
    for profile in sorted(src.rglob(PROFILE_FILE)) if src.is_dir() else []:
        if _SKIP.intersection(profile.relative_to(src).parts):
            continue
        for parent in profile.parents:
            if (parent / "package.xml").is_file():
                from dwarl.robot.profile import _package_name

                found.append((_package_name(parent / "package.xml") or parent.name, profile))
                break
            if parent == src:
                break
    return found


def guess_workspace(saved):
    for candidate in (os.environ.get("DWARL_ROBOT_WS"), saved.get("DWARL_ROBOT_WS")):
        if candidate:
            return candidate
    parent = Path.cwd().parent
    candidates = []
    for sibling in sorted(parent.iterdir()) if parent.is_dir() else []:
        if sibling.resolve() != Path.cwd().resolve() and (sibling / "src").is_dir() and integrations(sibling):
            candidates.append(str(sibling))
    return candidates[0] if len(candidates) == 1 else ""


def ask(question, default="", interactive=True):
    if not interactive:
        return default
    from dwarl.cli import ui

    answer = ui.ask(question, default, path="repository" in question.lower())
    return os.path.expanduser(answer) if answer else default


def choose(found, interactive=True):
    if len(found) > 1 and not interactive:
        raise ValueError("Multiple integrations found; choose with --packages instead of guessing.")
    if len(found) <= 1:
        return [name for name, _ in found]
    from dwarl.cli import ui

    return [ui.choose("Select an integration", [name for name, _ in found])]


_DEPENDS = ("depend", "build_depend", "exec_depend", "buildtool_depend")
# rosdep system keys that are not ROS packages (pixi provides them or they are build tools)
_SYSTEM = {"boost", "git", "eigen", "autoconf", "automake", "libtool", "pkg-config", "cmake", "yaml-cpp"}


def missing_ros_dependencies(workspace, packages, prefixes=None):
    """ROS packages needed by ``packages`` (and the workspace packages they use) that are not installed."""
    import xml.etree.ElementTree as ET

    manifests = {}
    src = Path(workspace) / "src"
    for xml in src.rglob("package.xml"):
        if _SKIP.intersection(xml.relative_to(src).parts) or (xml.parent / "COLCON_IGNORE").exists():
            continue
        root = ET.parse(xml).getroot()
        name = root.findtext("name", "").strip()
        manifests[name] = {e.text.strip() for tag in _DEPENDS for e in root.iter(tag) if e.text}
    prefixes = prefixes if prefixes is not None else os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)

    def installed(name):
        return any(Path(p, "share/ament_index/resource_index/packages", name).exists() for p in prefixes if p)

    missing, todo, seen = set(), list(packages), set()
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for dep in manifests.get(name, ()):
            if dep in manifests:
                todo.append(dep)
            elif "-" not in dep and dep not in _SYSTEM and not installed(dep):
                missing.add(dep)
    return sorted(missing)


def _pixi(*task, env):
    pixi = os.environ.get("PIXI_EXE") or shutil.which("pixi")
    if not pixi:
        raise ValueError("pixi not found; install it from https://pixi.sh and run `pixi run setup`")
    print(f"\n$ pixi run {' '.join(task)}", flush=True)
    return subprocess.run([pixi, "run", *task], env=env).returncode


def execute(robot_ws=None, packages=None, interactive=None, build=True, example=False):
    from dwarl.cli import ui
    from dwarl.workflows import context

    interactive = sys.stdin.isatty() if interactive is None else interactive
    saved = load_saved()
    ui.title("Setup", "Select your robot once. All commands run through pixi; no manual ROS activation.")
    mode = None
    if interactive and not robot_ws and not example:
        mode = ui.choose("What would you like to do?", ["Try the included example robot", "Integrate my robot",
                                                       "Continue an existing integration"],
                         "Continue an existing integration" if context.load().get("robot") else None)
        example = mode == "Try the included example robot"
    if example:
        context.select_robot("diffbot")
        save({})
        from dwarl.robot.diagnostics import environment

        report = environment("mujoco")
        for item in report["findings"]:
            ui.finding(item["level"], item["message"])
        if not report["passed"]:
            return 2
        ui.finding("pass", "Selected diffbot")
        ui.next_step("train", "--preset", "smoke")
        if interactive and ui.ask("Run the small pipeline check now? (y/n)", "y").lower() == "y":
            return smoke("diffbot")
        return 0
    selected = context.load().get("robot")
    if mode == "Continue an existing integration" and selected and not guess_workspace(saved):
        from dwarl.robot.diagnostics import inspect, render

        report = inspect(selected)
        render(report)
        ui.next_step("dwarl", "next")
        return 0 if report["passed"] else 2
    workspace = robot_ws or ask("Robot repository (contains src/)", guess_workspace(saved), interactive)
    values = {}
    robot = None
    if workspace:
        workspace = str(Path(workspace).expanduser().resolve())
        if not (Path(workspace) / "src").is_dir():
            raise ValueError(f"{workspace} has no src/ directory")
        found = integrations(workspace)
        if not found:
            if not interactive:
                raise ValueError("No integration found. Run `pixi run dwarl robot init` first, or setup --example.")
            from dwarl.robot.init import execute as initialize

            destination = ui.ask("New integration directory", str(Path(workspace) / "src/my_robot_dwarl"), path=True)
            initialize(destination)
            found = integrations(workspace)
        if packages is None:
            default = saved.get("DWARL_ROBOT_PACKAGES", "") if saved.get("DWARL_ROBOT_WS") == workspace else ""
            packages = default.split() if default else choose(found, interactive)
        values = {"DWARL_ROBOT_WS": workspace, "DWARL_ROBOT_PACKAGES": " ".join(packages)}
        selected = [(name, profile) for name, profile in found if name in packages]
        if not selected:
            raise ValueError("Selected packages contain no DWARL integration.")
        if len(selected) > 1:
            if not interactive:
                raise ValueError("Choose one integration package for the active robot with --packages.")
            name = ui.choose("Active robot", [name for name, _ in selected])
            robot = next(profile for candidate, profile in selected if name == candidate)
        else:
            robot = selected[0][1]
        context.select_robot(robot)
    else:
        raise ValueError("Choose a robot workspace, or run `pixi run setup --example`.")
    save(values)
    print(f"Saved {SETUP_FILE}")

    missing = missing_ros_dependencies(workspace, packages) if workspace and packages else []
    if missing:
        print(f"\nwarning: your integration needs ROS packages that are not installed: {', '.join(missing)}")
        print("  add them (they become part of this checkout's pixi environment):")
        print(f"  pixi add --feature ros {' '.join('ros-jazzy-' + m.replace('_', '-') for m in missing)}")
        print("  then rerun `pixi run setup`.")

    if build:
        from dwarl.robot.diagnostics import inspect, render
        from dwarl.robot.profile import load_profile

        report = inspect(robot)
        render(report)
        if not report["passed"]:
            return 2
        profile = load_profile(robot)
        if profile.ros.get("sim_launch"):
            from dwarl.workflows.prepare import ros

            ros(robot)
            if interactive and ui.ask("Verify the Gazebo adapter now? (y/n)", "y").lower() == "y":
                from dwarl.workflows.sim_check import execute as check_sim

                checked = check_sim(robot)
                for item in checked["findings"]:
                    ui.finding(item["level"], item["message"])
                if not checked["passed"]:
                    return 2
        from dwarl.ros_env import orphaned_gazebo

        for pid, domain in orphaned_gazebo():
            question = f"Stop orphaned DWARL Gazebo pid {pid} (ROS domain {domain})? (y/n)"
            if ask(question, "y" if interactive else "n", interactive).lower().startswith("y"):
                subprocess.run(["kill", pid])

    ui.finding("pass", f"Selected integration: {robot}")
    ui.next_step("train", "--preset", "smoke")
    if build and interactive and ui.ask("Run the small pipeline check now? (y/n)", "y").lower() == "y":
        return smoke(robot)
    return 0


def smoke(robot):
    from dwarl.cli import ui

    if _pixi("train", str(robot), "--preset", "smoke", env=dict(os.environ)):
        return 2
    # Pin the smoke run even when a navigation run already exists.
    from dwarl.robot.profile import resolve_profile
    from dwarl.workflows.context import load

    run = load()["runs"][str(resolve_profile(robot))]["smoke"]
    code = _pixi("test", "--run", run, "--case", "doorway", "--set", "test.max_cases=1", env=dict(os.environ))
    if code not in (0, 1):
        return code
    ui.finding("pass", "Training, checkpointing and inference completed")
    ui.info("This tiny policy checks the pipeline; navigation failures are expected.")
    ui.next_step("train", "--preset", "navigation")
    return 0
