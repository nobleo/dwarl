"""Local selections, separate from robot-owned configuration and immutable run data."""

import json
from pathlib import Path

from dwarl.runs.store import lock, write_json

FILE = Path(".dwarl/context.json")


def load():
    if not FILE.exists():
        return {"schema": "dwarl_context/1", "runs": {}}
    data = json.loads(FILE.read_text())
    if data.get("schema") != "dwarl_context/1":
        raise ValueError(f"unsupported local context: {FILE}")
    return data


def select_robot(reference):
    from dwarl.robot.profile import resolve_profile

    reference = str(resolve_profile(reference))
    with lock(FILE.with_suffix(".lock")):
        state = load()
        state["robot"] = reference
        write_json(FILE, state)
    return reference


def remember_run(robot, run, preset=None):
    from dwarl.robot.profile import resolve_profile

    with lock(FILE.with_suffix(".lock")):
        state = load()
        key = str(resolve_profile(robot))
        choices = state.setdefault("runs", {}).setdefault(key, {})
        choices["smoke" if preset == "smoke" else "navigation"] = str(Path(run).resolve())
        write_json(FILE, state)


def selected_run(robot=None):
    state = load()
    if robot:
        from dwarl.robot.profile import resolve_profile

        robot = str(resolve_profile(robot))
    choices = state.get("runs", {}).get(robot or state.get("robot"), {})
    return choices.get("navigation") or choices.get("smoke")
