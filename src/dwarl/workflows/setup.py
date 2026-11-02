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
    for sibling in sorted(parent.iterdir()) if parent.is_dir() else []:
        if sibling.resolve() != Path.cwd().resolve() and (sibling / "src").is_dir() and integrations(sibling):
            return str(sibling)
    return ""


def ask(question, default="", interactive=True):
    if not interactive:
        return default
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return os.path.expanduser(answer) if answer else default


def choose(found, interactive=True):
    if len(found) <= 1 or not interactive:
        return [name for name, _ in found]
    print("Robot integrations found:")
    for index, (name, profile) in enumerate(found, 1):
        print(f"  {index}. {name}  ({profile})")
    answer = ask("Which to build (numbers, comma-separated)", "1")
    picked = [found[int(item) - 1][0] for item in answer.replace(" ", "").split(",") if item]
    return picked or [found[0][0]]


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


def execute(robot_ws=None, packages=None, interactive=None, build=True):
    interactive = sys.stdin.isatty() if interactive is None else interactive
    saved = load_saved()
    print("DWARL setup: answers are saved to .dwarl/setup.env (press Enter to accept a default).\n")

    workspace = robot_ws or ask("Robot repository (contains src/; empty for none)", guess_workspace(saved), interactive)
    values = {}
    if workspace:
        workspace = str(Path(workspace).expanduser().resolve())
        if not (Path(workspace) / "src").is_dir():
            raise ValueError(f"{workspace} has no src/ directory")
        found = integrations(workspace)
        if not found:
            print(f"warning: no {PROFILE_FILE} under {workspace}/src; see docs/integrate-robot.md")
        if packages is None:
            default = saved.get("DWARL_ROBOT_PACKAGES", "") if saved.get("DWARL_ROBOT_WS") == workspace else ""
            packages = default.split() if default else choose(found, interactive)
        values = {"DWARL_ROBOT_WS": workspace, "DWARL_ROBOT_PACKAGES": " ".join(packages)}
        for name, profile in found:
            if name in packages:
                print(f"Robot profile: {profile}")
    save(values)
    print(f"Saved {SETUP_FILE}")

    missing = missing_ros_dependencies(workspace, packages) if workspace and packages else []
    if missing:
        print(f"\nwarning: your integration needs ROS packages that are not installed: {', '.join(missing)}")
        print("  add them (they become part of this checkout's pixi environment):")
        print(f"  pixi add --feature ros {' '.join('ros-jazzy-' + m.replace('_', '-') for m in missing)}")
        print("  then rerun `pixi run setup`.")

    if build:
        env = {**os.environ, **values}
        if not values:
            env.pop("DWARL_ROBOT_WS", None)
        if _pixi("build-robot" if values else "build", env=env) or _pixi("doctor", env=env):
            print("\nSetup incomplete: fix the error above and run `pixi run setup` again.")
            return 1
        from dwarl.ros_env import orphaned_gazebo

        for pid, domain in orphaned_gazebo():
            question = f"Stop orphaned DWARL Gazebo pid {pid} (ROS domain {domain})? (y/n)"
            if ask(question, "y" if interactive else "n", interactive).lower().startswith("y"):
                subprocess.run(["kill", pid])

    robot = next((str(profile) for name, profile in integrations(workspace) if name in packages), None) \
        if workspace else None
    print("\nReady. Next:")
    print(f"  pixi run train {robot or '<robot integration>'}")
    print("  pixi run test --sim both")
    if workspace:
        print(f"  pixi run deploy {workspace}")
    print("  (or `pixi shell` and use `dwarl ...` directly)")
    return 0
