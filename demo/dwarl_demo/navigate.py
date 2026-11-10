"""Visual showcase: Gazebo (GUI) + RViz + Nav2 with the installed policy, driving start -> goal.

Uses DWARL's bench.launch.py (robot adapter, map server, spawn-pose localisation, Nav2) on a demo world
whose Gazebo scene contains obstacles that are missing from the Nav2 map, then sends NavigateToPose.
The scenes are the ones of the compare stage's unmapped-obstacle pass (same world files, same
obstacles), or the demo worlds' fixed surprise obstacles if that pass has not run.

    python -m dwarl_demo.navigate <bundle> <demo_worlds_dir> [world] [controller]
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

_DRIVES = [0]  # each drive gets a fresh ROS domain: stale graph entries must not answer for a new stack
BOLD, DIM, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _stop(process):
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=timeout)
            return
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue


def surprise_scenes(worlds):
    """First start/goal pair of each demo world, with the world's fixed surprise obstacles."""
    gazebo = Path(worlds) / "gazebo"
    scenes = []
    for meta_file in sorted(gazebo.glob("*_surprise.json")):
        world = meta_file.name[:-len("_surprise.json")]
        meta = json.loads((gazebo / f"{world}.json").read_text())["pairs"][0]
        scenes.append({"world": world, "sdf": gazebo / f"{world}_surprise.sdf", "map": gazebo / f"{world}.yaml",
                       "start": meta["start"], "goal": meta["goal"], "path_length": meta["path_length"]})
    return scenes


def unmapped_scenes(run_dir):
    """First case per world of the latest Gazebo test of ``run_dir`` with unmapped obstacles."""
    import yaml

    tests = sorted(Path(run_dir).glob("tests/*/request.yaml"), key=lambda p: p.stat().st_mtime, reverse=True)
    for request_file in tests:
        request = yaml.safe_load(request_file.read_text()) or {}
        if request.get("sim") != "gazebo" or not request.get("unmapped_obstacles"):
            continue
        scenes = {}
        for case_input in sorted(request_file.parent.glob("gazebo/dwarl/*/input.json")):
            case = json.loads(case_input.read_text())["case"]
            if case["pair"] != 0 or case["name"] in scenes:
                continue
            path = case["reference_path"]
            (ax, ay), (bx, by) = path[-2], path[-1]
            length = sum(math.dist(a, b) for a, b in zip(path, path[1:], strict=False))
            scenes[case["name"]] = {"world": case["name"], "sdf": case_input.parent / "world.sdf",
                                    "map": case_input.parent / "world.yaml", "start": case["start"],
                                    "goal": [*case["goal"][:2], math.atan2(by - ay, bx - ax)],
                                    "path_length": length}
        if scenes:
            return [scenes[name] for name in sorted(scenes)]
    return []


def drive(bundle, scene, controller="dwarl", controller_params=None, planner="smac2d", timeout=None, gui=True):
    """Launch the stack on ``scene``, send one goal, report the outcome. Returns a result dict."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from std_srvs.srv import Trigger

    world = scene["world"]
    (x, y, yaw), (gx, gy, gyaw) = scene["start"], scene["goal"]
    import yaml

    v_max = float(yaml.safe_load((Path(bundle) / "deploy.yaml").read_text())["action"]["v"]["max"])
    budget = timeout or max(120.0, 4.0 * scene["path_length"] / v_max + 60.0)  # generous: includes recoveries
    logs = Path(os.environ.get("DWARL_WORK", "/work")) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"showcase-{world}-{controller}-{time.strftime('%H%M%S')}.log"
    cmd = ["ros2", "launch", "dwarl_bringup", "bench.launch.py", f"model_dir:={bundle}",
           f"world:={scene['sdf']}", f"map:={scene['map']}", f"planner:={planner}",
           f"x:={x}", f"y:={y}", f"yaw:={yaw}", f"gui:={str(gui).lower()}", f"rviz:={str(gui).lower()}",
           f"controller:={controller}", "safety:=true"]
    if controller_params:
        cmd.append(f"controller_params:={controller_params}")
    _DRIVES[0] += 1
    domain = str(int(os.environ.get("DEMO_ROS_DOMAIN_ID", "70")) + _DRIVES[0] % 30)
    env = dict(os.environ, ROS_DOMAIN_ID=domain, GZ_PARTITION=uuid4().hex)
    print(f"{DIM}$ {' '.join(cmd)}\nLaunch log: {log}{RESET}", flush=True)
    with open(log, "w") as stream:
        process = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    os.environ["ROS_DOMAIN_ID"] = env["ROS_DOMAIN_ID"]
    rclpy.init()
    node = rclpy.create_node("dwarl_demo_goal")
    result = {"world": world, "controller": controller, "outcome": "infrastructure_failure"}
    try:
        client = ActionClient(node, NavigateToPose, "navigate_to_pose")
        active = node.create_client(Trigger, "lifecycle_manager_navigation/is_active")
        print(f"Starting Gazebo, Nav2 and {controller}…", end="", flush=True)
        deadline = time.monotonic() + 240
        while True:  # Nav2 accepts goals only once every lifecycle node is active
            if active.wait_for_service(timeout_sec=1.0):
                answer = active.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(node, answer, timeout_sec=2.0)
                if answer.result() is not None and answer.result().success and client.wait_for_server(1.0):
                    break
            print(".", end="", flush=True)
            if process.poll() is not None or time.monotonic() > deadline:
                print(f"\n{RED}Navigation stack did not come up; see {log}{RESET}")
                return result
            time.sleep(1.0)
        time.sleep(3.0)  # let costmaps receive the first scans
        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = "map"
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(gx), float(gy)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(gyaw / 2), math.cos(gyaw / 2)
        feedback = {}
        print(f"\n{BOLD}Goal sent{RESET}: ({x:.1f}, {y:.1f}) -> ({gx:.1f}, {gy:.1f}), "
              f"reference route {scene['path_length']:.1f} m; obstacles not on the map are placed on that route.")
        for _ in range(3):
            future = client.send_goal_async(goal, feedback_callback=lambda m: feedback.update(f=m.feedback))
            rclpy.spin_until_future_complete(node, future, timeout_sec=30)
            handle = future.result()
            if handle is not None and handle.accepted:
                break
            time.sleep(5.0)
        if handle is None or not handle.accepted:
            print(f"{RED}Goal rejected{RESET}")
            result["outcome"] = "rejected"
            return result
        done = handle.get_result_async()
        start, shown, printed_at, tty = time.monotonic(), None, -1e9, sys.stdout.isatty()
        while not done.done():
            rclpy.spin_once(node, timeout_sec=0.5)
            f = feedback.get("f")
            if f is not None:
                elapsed = time.monotonic() - start
                line = (f"  remaining {f.distance_remaining:5.1f} m | elapsed {elapsed:5.0f} s"
                        f" | recoveries {f.number_of_recoveries}")
                if tty and line != shown:
                    print(f"\r\033[K{line}", end="", flush=True)
                    shown = line
                elif not tty and elapsed - printed_at >= 15:  # logs: one line every 15 s
                    print(line, flush=True)
                    printed_at = elapsed
            if time.monotonic() - start > budget:
                handle.cancel_goal_async()
                result["outcome"] = "timeout"
                print(f"\n{YELLOW}Timed out after {budget:.0f} s{RESET}")
                return result
            if process.poll() is not None:
                print(f"\n{RED}Navigation stack exited; see {log}{RESET}")
                return result
        status = done.result().status
        elapsed = time.monotonic() - start
        ok = status == 4  # GoalStatus.STATUS_SUCCEEDED
        result.update(outcome="success" if ok else f"failed (status {status})", wall_time_s=round(elapsed, 1))
        print(f"\n{GREEN + 'Reached the goal' if ok else RED + 'Navigation failed'}{RESET} in {elapsed:.1f} s "
              f"wall time (Nav2 status {status}).")
        return result
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if gui and sys.stdin.isatty() and os.environ.get("DEMO_YES") != "1":
            try:
                input("Press Enter to close this simulation… ")
            except EOFError:
                pass
        _stop(process)


def _pick(title, names, default):
    from dwarl_demo.__main__ import choose

    return choose(title, [(name, "") for name in names], default)


def showcase(bundle, worlds, entry, run_dir=None, interactive=True):
    from dwarl_demo.__main__ import controllers_for, planner_for, resolve_optional

    baselines = resolve_optional(entry.get("baselines"))  # the comparison's shared tuning, for every controller
    planner = planner_for(entry)
    scenes = (unmapped_scenes(run_dir) if run_dir else []) or surprise_scenes(worlds)
    others = [c for c in controllers_for(entry) if c != "dwarl"]
    gui = os.environ.get("DEMO_GUI", "1") != "0"
    results = []
    names = [scene["world"] for scene in scenes]
    pending = list(scenes)
    if interactive and gui and len(scenes) > 1:  # one scene at a time, the user picks
        pending = [scenes[names.index(_pick("Which world?", names, names[0]))]]
    while pending:
        scene = pending.pop(0)
        print(f"\n{BOLD}World: {scene['world']}{RESET} — Gazebo shows the robot and the obstacles that are not on "
              "the map; RViz shows the map, global plan and costmap.")
        results.append(drive(bundle, scene, "dwarl", baselines, planner, gui=gui))
        if interactive and gui and others:
            prompt = f"Replay this world with another controller? [n]/{'/'.join(others)}: "
            answer = input(prompt).strip().lower()
            while answer in others:
                results.append(drive(bundle, scene, answer, baselines, planner, gui=gui))
                answer = input(prompt).strip().lower()
        if interactive and gui and not pending and len(scenes) > 1:
            answer = input(f"Drive another world? [n]/{'/'.join(names)}: ").strip().lower()
            if answer in names:
                pending.append(scenes[names.index(answer)])
    print(f"\n{BOLD}Showcase results{RESET}")
    for r in results:
        print(f"  {r['world']:<10} {r['controller']:<6} {r['outcome']:<12} {r.get('wall_time_s', '-')}")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    name = args[2] if len(args) > 2 else "office"
    chosen = [scene for scene in surprise_scenes(args[1]) if scene["world"] == name]
    if not chosen:
        raise SystemExit(f"no demo world {name!r} in {args[1]}")
    drive(args[0], chosen[0], *(args[3:4] or []))
