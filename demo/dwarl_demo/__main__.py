"""DWARL end-to-end demo: pick a robot, train, evaluate, compare with Nav2 controllers, deploy, showcase.

    docker compose run --rm demo                      # interactive (docker/README.md)
    DEMO_ROBOT=turtlebot3 DEMO_PRESET=quick DEMO_YES=1 docker compose run --rm demo

Stages (DEMO_STAGES, comma separated): setup, train, evaluate, compare, deploy, showcase.
Everything is written below /work: robots/, runs/, demo_worlds/, logs/ and the .dwarl caches.
"""

from __future__ import annotations

import json
import math
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("DWARL_WORK", "/work"))
STAGES = ["setup", "train", "evaluate", "compare", "deploy", "showcase"]
CONTROLLERS = ["dwarl", "mppi", "dwb", "rpp"]  # Nav2 controllers compared under one shared setup
PLANNER = "smac2d"                             # global planner for every controller
UNMAPPED = 3                                   # obstacles per route that are not on the Nav2 map
BOLD, DIM, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def say(text="", style=""):
    print(f"{style}{text}{RESET if style else ''}", flush=True)


def heading(text):
    say(f"\n━━ {text} " + "━" * max(0, 70 - len(text)), BOLD)


def interactive():
    return sys.stdin.isatty() and os.environ.get("DEMO_YES") != "1"


def ask(prompt, default):
    if not interactive():
        say(f"{prompt} {default}", DIM)
        return default
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def choose(title, options, default):
    """options: list of (key, description); returns a key."""
    say(title, BOLD)
    for i, (key, text) in enumerate(options, 1):
        mark = " (default)" if key == default else ""
        say(f"  {i}) {key:<11} {text}{mark}")
    while True:
        answer = ask("Choice", str([k for k, _ in options].index(default) + 1))
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        if answer in dict(options):
            return answer
        say(f"Please enter 1-{len(options)}", YELLOW)


def load_registry(work=WORK):
    """Built-in robots and presets, with /work/robots.yaml merged over them."""
    registry = yaml.safe_load((HERE / "robots.yaml").read_text())
    extra = Path(work) / "robots.yaml"
    if extra.is_file():
        override = yaml.safe_load(extra.read_text()) or {}
        local = override.get("robots") or {}
        if any(entry.get("default") for entry in local.values()):  # a local default wins
            for entry in registry["robots"].values():
                entry.pop("default", None)
        registry["robots"].update(local)
        registry["presets"].update(override.get("presets") or {})
    return registry


def default_robot(robots):
    """The robot marked `default: true` (the last one if several), else the first one listed."""
    marked = [key for key, entry in robots.items() if entry.get("default")]
    return marked[-1] if marked else next(iter(robots))


def run(cmd, log=None, check=True, env=None):
    say("$ " + " ".join(map(str, cmd)), DIM)
    if log is None:
        result = subprocess.run(list(map(str, cmd)), env=env)
    else:
        with open(log, "a") as stream:
            result = subprocess.run(list(map(str, cmd)), stdout=stream, stderr=subprocess.STDOUT, env=env)
    if check and result.returncode not in (0, 1):  # dwarl test exits 1 when navigation (not the tool) failed
        raise SystemExit(f"command failed ({result.returncode}); see {log or 'output above'}")
    return result.returncode


# ------------------------------------------------------------------------------------------------- setup
def prepare_robot(key, entry):
    """Copy the robot's integration (profile directory) to /work/robots/<key> so policies can be installed."""
    from dwarl.robot.profile import load_profile

    source = load_profile(entry["profile"])
    target = WORK / "robots" / key
    shutil.copytree(source.dir, target, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("policies", "selected_policy.yaml", "__pycache__"))
    profile = target / source.path.name
    say(f"Robot {key}: {source.id} from {source.path}")
    say(f"Integration copy: {target}")
    return profile, load_profile(profile)


def controllers_for(entry):
    return list(entry.get("controllers") or CONTROLLERS)


def planner_for(entry):
    return entry.get("planner", PLANNER)


def unmapped_for(entry, preset):
    """Robot setting first, then the preset, then the default."""
    return int(entry.get("unmapped_obstacles", preset.get("unmapped_obstacles", UNMAPPED)))


def resolve_optional(uri):
    if not uri:
        return None
    from dwarl.robot.profile import resolve_uri

    return Path(resolve_uri(uri, Path.cwd()))


def build_worlds(key, profile):
    from dwarl_demo.worlds import build_worlds as build

    out = WORK / "demo_worlds" / key
    stamp = out / "profile.sha256"
    if not (stamp.is_file() and stamp.read_text() == profile.sha256):
        say("Building demo worlds (office, warehouse) for this robot…")
        if out.exists():
            shutil.rmtree(out)
        build(out, profile.path)
        stamp.write_text(profile.sha256)
    say(f"Demo worlds: {out}  (previews in {out / 'previews'})")
    return out


# ------------------------------------------------------------------------------------------------- runs
def runs_for(robot_id):
    found = []
    for provenance in sorted((WORK / "runs").glob("*/provenance.json")):
        run_dir = provenance.parent
        try:
            data = json.loads(provenance.read_text())
            status = json.loads((run_dir / "status.json").read_text())
            manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text())
        except (OSError, ValueError):
            continue
        if data.get("robot") == robot_id and manifest.get("completed") and status.get("state") != "running":
            found.append((run_dir, status, manifest))
    return found


def checkpoint_for(manifest):
    return "best" if manifest.get("best") else "latest"


@contextmanager
def keypresses():
    if not sys.stdin.isatty():
        yield lambda: None
        return
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield lambda: sys.stdin.read(1) if select.select([sys.stdin], [], [], 0)[0] else None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def train(key, entry, profile_path, preset):
    logs = WORK / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"train-{key}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    workers = max(1, min(8, (os.cpu_count() or 2) // 2))
    cmd = ["dwarl", "train", str(profile_path), "--name", f"demo-{key}", "--num-envs", str(entry.get("num_envs", 64)),
           "--max-iterations", str(preset["iterations"]),
           "--set", f"train.validation_interval={preset['validation_interval']}",
           # validation runs on saved checkpoints, so its interval must be a multiple of the save interval
           "--set", f"train.save_interval={math.gcd(50, int(preset['validation_interval']))}",
           "--set", f"train.data.workers={workers}"]
    say("$ " + " ".join(cmd), DIM)
    say(f"Log: {log}")
    say("Keys:  v = show/hide live training view   l = recent log   q = stop early (keeps checkpoints)", YELLOW)
    with open(log, "w") as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    run_dir, state, last = None, {}, ""
    patterns = {
        "iteration": re.compile(r"Learning iteration (\d+)/(\d+)"),
        "success": re.compile(r"Episode_Termination/success:\s*([-\d.]+)"),
        "collision": re.compile(r"Episode_Termination/collision:\s*([-\d.]+)"),
        "reward": re.compile(r"Mean reward:\s*([-\d.]+)"),
        "eta": re.compile(r"ETA:\s*([\d:]+)"),
        "stage": re.compile(r"^(Training: .*|Generating .*|Dataset.*|Validation.*)$"),
        "run": re.compile(r"Training run: (\S+)"),
    }
    position = 0
    with keypresses() as key_pressed:
        while process.poll() is None:
            text = log.read_text(errors="replace")
            chunk, position = text[position:], len(text)
            for line in chunk.splitlines():
                for name, pattern in patterns.items():
                    match = pattern.search(line)
                    if match:
                        state[name] = match.groups() if name == "iteration" else match.group(1)
            if state.get("run") and run_dir is None:
                run_dir = Path(state["run"])
            if "iteration" in state:
                i, n = state["iteration"]
                status = (f"iter {int(i) + 1}/{n} | success {float(state.get('success', 0)):.2f} | "
                          f"collision {float(state.get('collision', 0)):.2f} | reward {state.get('reward', '-')} | "
                          f"ETA {state.get('eta', '-')}")
            else:
                status = state.get("stage", "starting…")
            if status != last:
                print(f"\r\033[K[train] {status[:110]}", end="", flush=True)
                last = status
            pressed = key_pressed()
            if pressed in ("v", "V") and run_dir:
                subprocess.run(["dwarl", "runs", "view", str(run_dir)], stdout=subprocess.DEVNULL)
                print("\r\033[K[train] live view toggled (the window shows one simulation shard)", flush=True)
            elif pressed in ("l", "L"):
                print("\n" + "\n".join(log.read_text(errors="replace").splitlines()[-25:]), flush=True)
            elif pressed in ("q", "Q"):
                print("\n[train] stopping after the current iteration…", flush=True)
                os.killpg(process.pid, signal.SIGINT)
            time.sleep(0.5)
    print()
    if process.returncode not in (0, 130):
        say("\n".join(log.read_text(errors="replace").splitlines()[-30:]))
        raise SystemExit(f"training failed ({process.returncode}); see {log}")
    if run_dir is None:
        raise SystemExit(f"training did not report a run directory; see {log}")
    say(f"Training run: {run_dir}", GREEN)
    say("TensorBoard: docker compose --profile monitor up tensorboard  ->  http://localhost:6006", DIM)
    return run_dir


# ------------------------------------------------------------------------------------------------- tests
def latest_test(run_dir):
    tests = sorted((run_dir / "tests").glob("*/summary.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(tests[-1].read_text()), tests[-1].parent


def table(summary, title):
    say(f"\n{title}", BOLD)
    say(f"  {'controller':<14}{'success':>9}{'collision':>11}{'time (s)':>10}{'clearance (m)':>15}{'stops':>7}")
    for name, s in summary["simulators"].items():
        label = {"gazebo": "dwarl", "mujoco": "dwarl (MuJoCo)"}.get(name, name.replace("gazebo-", ""))
        t = s.get("successful_mean_time_s")
        c = s.get("mean_min_clearance_m")
        rate = s.get("success_rate") or 0
        colour = GREEN if rate >= 0.8 else (YELLOW if rate >= 0.5 else RED)
        say(f"  {label:<14}{colour}{(s.get('success_rate') or 0):>9.2f}{RESET}{(s.get('collision_rate') or 0):>11.2f}"
            f"{(f'{t:.1f}' if t else '-'):>10}{(f'{c:.2f}' if c else '-'):>15}"
            f"{(s.get('mean_safety_stops') or 0):>7.1f}")
    for name, s in summary["simulators"].items():
        drift = s.get("mean_localization_error_m")
        label = {"gazebo": "dwarl"}.get(name, name.replace("gazebo-", ""))
        if drift is not None and drift > 0.1:
            say(f"  warning: {label} ends {drift:.2f} m from ground truth on average; the bench localizes with "
                "odometry, so this includes odometry drift (docs/manual-integration.md)", YELLOW)


def evaluate(run_dir, checkpoint, preset):
    log = WORK / "logs" / f"evaluate-{run_dir.name}.log"
    run(["dwarl", "test", "--run", run_dir, "--checkpoint", checkpoint, "--suite", "benchmark", "--sim", "mujoco",
         "--set", f"test.max_cases={preset['max_cases']}"], log=log)
    summary, directory = latest_test(run_dir)
    table(summary, "MuJoCo benchmark (canonical cases)")
    say(f"Report: {directory / 'report' / 'README.md'}", DIM)
    return summary


def compare(run_dir, checkpoint, entry, worlds, preset):
    controllers = ",".join(controllers_for(entry))
    planner, count = planner_for(entry), unmapped_for(entry, preset)
    shared = ["--set", f"test.planner={planner}"]
    baselines = resolve_optional(entry.get("baselines"))
    if baselines:
        shared += ["--set", f"test.controller_params={baselines}"]
        say(f"Shared tuning (costmap, planner, controllers): {baselines}")
    say(f"Shared Nav2 setup: global planner {planner}, same map, costmaps, limits and safety for every controller.")
    passes = []
    if os.environ.get("DEMO_CANONICAL") == "1":
        passes.append(("canonical", "canonical cases", ["--set", f"test.max_cases={preset['max_cases']}"]))
    passes.append(("on the map", "demo worlds, all obstacles on the map", ["--set", f"test.dataset={worlds}"]))
    if count:
        passes.append(("not on the map", f"demo worlds + up to {count} obstacles per route that are not on the map",
                       ["--set", f"test.dataset={worlds}", "--set", f"test.unmapped_obstacles={count}"]))
    results = {}
    for short, title, workload in passes:
        say(f"\nGazebo: {title} — {controllers}. Each case starts a fresh Gazebo + Nav2 stack (headless).")
        log = WORK / "logs" / f"compare-{run_dir.name}.log"
        run(["dwarl", "test", "--run", run_dir, "--checkpoint", checkpoint, "--suite", "benchmark", "--sim", "gazebo",
             "--controllers", controllers, *shared, *workload], log=log)
        summary, directory = latest_test(run_dir)
        table(summary, f"Gazebo: {title}")
        say(f"Report: {directory / 'report' / 'README.md'}", DIM)
        results[short] = summary
    if len(results) > 1:
        overview(results)
    return results


def overview(results):
    """Success and collision rate per controller, one column pair per pass."""
    names = list(dict.fromkeys(name for summary in results.values() for name in summary["simulators"]))
    say("\nSummary: success / collision rate per pass", BOLD)
    say("  " + f"{'controller':<14}" + "".join(f"{title:>18}" for title in results))
    for name in names:
        label = {"gazebo": "dwarl"}.get(name, name.replace("gazebo-", ""))
        cells = []
        for summary in results.values():
            s = summary["simulators"].get(name) or {}
            cells.append(f"{(s.get('success_rate') or 0):.2f} / {(s.get('collision_rate') or 0):.2f}")
        say("  " + f"{label:<14}" + "".join(f"{cell:>18}" for cell in cells))


def deploy(profile_path, run_dir, checkpoint):
    log = WORK / "logs" / f"deploy-{run_dir.name}.log"
    run(["dwarl", "deploy", profile_path, "--run", run_dir, "--checkpoint", checkpoint], log=log)
    selection = json.loads((profile_path.parent / "selected_policy.yaml").read_text())
    bundle = profile_path.parent / selection["path"]
    say(f"Installed policy bundle: {bundle}", GREEN)
    return bundle


# ------------------------------------------------------------------------------------------------- main
def main():
    registry = load_registry()
    robots, presets = registry["robots"], registry["presets"]
    heading("DWARL demo: reinforcement-learned local planner for Nav2")
    say("Train a robot-specific policy in MuJoCo, compare it with Nav2's MPPI, DWB and Regulated Pure Pursuit")
    say("in Gazebo, with obstacles that are not on the map, install it as the Nav2 controller and watch it drive.")
    key = os.environ.get("DEMO_ROBOT") or choose(
        "\nWhich robot?", [(k, v.get("label", "")) for k, v in robots.items()], default_robot(robots))
    if key not in robots:
        raise SystemExit(f"unknown robot {key!r}; choose from {', '.join(robots)}")
    entry = robots[key]

    heading("1. Setup")
    profile_path, profile = prepare_robot(key, entry)
    worlds = build_worlds(key, profile)
    existing = runs_for(profile.id)
    options = [(name, f"{p['iterations']} iterations: {p.get('label', '')}") for name, p in presets.items()]
    if existing:
        run_dir, status, manifest = existing[-1]
        options.insert(0, ("reuse", f"skip training: {run_dir.name} ({status.get('state')}, "
                                    f"iteration {status.get('iteration', '?')})"))
    preset_name = os.environ.get("DEMO_PRESET") or choose("\nTraining budget?", options, options[0][0])
    preset = presets.get(preset_name, presets["quick"])
    stages = [s.strip() for s in os.environ.get("DEMO_STAGES", ",".join(STAGES)).split(",") if s.strip()]
    if preset_name == "reuse":
        stages = [s for s in stages if s != "train"]

    run_dir = existing[-1][0] if existing and preset_name == "reuse" else None
    if "train" in stages:
        heading(f"2. Train ({preset['iterations']} iterations, headless)")
        run_dir = train(key, entry, profile_path, preset)
    if run_dir is None:
        if not existing:
            raise SystemExit("no trained run for this robot; include the train stage")
        run_dir = existing[-1][0]
    manifest = json.loads((run_dir / "checkpoint_manifest.json").read_text())
    checkpoint = checkpoint_for(manifest)
    say(f"Using {run_dir.name}, checkpoint '{checkpoint}'")

    if "evaluate" in stages:
        heading("3. Evaluate in MuJoCo")
        evaluate(run_dir, checkpoint, preset)
    gazebo = entry.get("gazebo", True)
    if "compare" in stages and gazebo:
        heading("4. Compare with Nav2 controllers in Gazebo")
        compare(run_dir, checkpoint, entry, worlds, preset)
    bundle = None
    if "deploy" in stages or "showcase" in stages:
        heading("5. Deploy the policy into the robot integration")
        bundle = deploy(profile_path, run_dir, checkpoint)
    if "showcase" in stages and gazebo:
        heading("6. Showcase: Nav2 + DWARL in Gazebo")
        from dwarl_demo.navigate import showcase

        showcase(bundle, worlds, entry, run_dir, interactive=interactive())
    heading("Done")
    say(f"Everything is in {WORK}: runs/{run_dir.name}, robots/{key}, demo_worlds/{key}, logs/")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\ninterrupted", YELLOW)
        sys.exit(130)
