"""Visual showcase: Gazebo (GUI) + RViz + Nav2 with the installed policy, driving start -> goal.

Uses DWARL's bench.launch.py (robot adapter, map server, spawn-pose localisation, Nav2) on a demo world
whose Gazebo scene contains obstacles that are missing from the Nav2 map, then sends NavigateToPose.

    python -m dwarl_demo.navigate <bundle> <demo_worlds_dir> [world] [pair] [controller]
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


def drive(bundle, worlds, world, pair=0, controller="dwarl", controller_params=None, timeout=None, gui=True):
    """Launch the stack, send one goal, report the outcome. Returns a result dict."""
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from std_srvs.srv import Trigger

    gazebo = Path(worlds) / "gazebo"
    meta = json.loads((gazebo / f"{world}.json").read_text())["pairs"][pair]
    (x, y, yaw), (gx, gy, gyaw) = meta["start"], meta["goal"]
    import yaml

    v_max = float(yaml.safe_load((Path(bundle) / "deploy.yaml").read_text())["action"]["v"]["max"])
    budget = timeout or max(120.0, 4.0 * meta["path_length"] / v_max + 60.0)  # generous: includes recoveries
    logs = Path(os.environ.get("DWARL_WORK", "/work")) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"showcase-{world}-{controller}-{time.strftime('%H%M%S')}.log"
    cmd = ["ros2", "launch", "dwarl_bringup", "bench.launch.py", f"model_dir:={bundle}",
           f"world:={gazebo / (world + '_surprise.sdf')}", f"map:={gazebo / (world + '.yaml')}",
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
    result = {"world": world, "pair": pair, "controller": controller, "outcome": "infrastructure_failure"}
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
              f"reference route {meta['path_length']:.1f} m; obstacles not on the map are placed on that route.")
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


def showcase(bundle, worlds, entry, interactive=True):
    from dwarl_demo.__main__ import resolve_optional

    baselines = resolve_optional(entry.get("baselines"))
    names = sorted(p.stem for p in (Path(worlds) / "scenarios").glob("*.npz"))
    gui = os.environ.get("DEMO_GUI", "1") != "0"
    results = []
    for world in names:
        print(f"\n{BOLD}World: {world}{RESET} — Gazebo shows the robot; RViz shows the map, global plan and costmap.")
        results.append(drive(bundle, worlds, world, 0, "dwarl", gui=gui))
        if interactive and gui:
            answer = input("Replay this world with another controller? [n]/dwb/rpp: ").strip().lower()
            while answer in ("dwb", "rpp", "mppi"):
                results.append(drive(bundle, worlds, world, 0, answer, baselines, gui=gui))
                answer = input("Another controller? [n]/dwb/rpp: ").strip().lower()
    print(f"\n{BOLD}Showcase results{RESET}")
    for r in results:
        print(f"  {r['world']:<10} {r['controller']:<6} {r['outcome']:<12} {r.get('wall_time_s', '-')}")
    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    drive(args[0], args[1], *(args[2:3] or ["office"]), *([int(args[3])] if len(args) > 3 else []),
          *(args[4:5] or []))
