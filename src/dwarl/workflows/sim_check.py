"""Managed, policy-independent Gazebo contract check."""

import fcntl
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from dwarl.cli import ui
from dwarl.runs.store import write_json


def execute(robot, timeout=120.):
    import numpy as np

    from dwarl.datasets.generation.mapdata import MapData
    from dwarl.robot.profile import load_profile
    from dwarl.ros_env import acquire_stack_lock, lease_directory
    from dwarl.testing.backends.gazebo import prerequisites, stop_owned
    from dwarl.testing.scenarios import export_gazebo

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be a positive finite number")
    profile = load_profile(robot)
    if not profile.ros.get("sim_launch"):
        raise ValueError("No ros.sim_launch configured. Add it with `pixi run dwarl robot init <dir> --gazebo`.")
    prerequisites()
    directory = (Path(".dwarl/checks") / f"gazebo-{time.time_ns()}").resolve()
    directory.mkdir(parents=True)
    md = MapData(np.zeros((100, 100), dtype=bool), .1, 10., 2., np.zeros((0, 6), dtype=np.float32))
    export_gazebo(md, directory, "world")
    write_json(directory / "request.json", {"ros": profile.ros, "timeout": timeout, "spawn": [1., 2., .4]})
    leases = lease_directory()
    leases.mkdir(parents=True, exist_ok=True)
    lease, domain = None, None
    for candidate in range(120, 200):
        stream = (leases / str(candidate)).open("a")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lease, domain = stream, candidate
            break
        except BlockingIOError:
            stream.close()
    if lease is None:
        raise ValueError("All managed ROS domains are busy.")
    try:
        build_lock = acquire_stack_lock()
    except ValueError:
        lease.close()
        raise
    process = None
    try:
        env = dict(os.environ, ROS_DOMAIN_ID=str(domain), GZ_PARTITION=uuid4().hex,
                   ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST")
        with (directory / "launch.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "dwarl.workflows.sim_check", str(directory)],
                                       env=env, stdout=log, stderr=log, start_new_session=True)
            (leases / f"{domain}.session").write_text(str(process.pid))
            with ui.progress(f"Checking live Gazebo interfaces · log: {directory / 'launch.log'}"):
                try:
                    process.wait(timeout=timeout + 20)
                except subprocess.TimeoutExpired:
                    pass
        result = directory / "result.json"
        if result.exists():
            report = json.loads(result.read_text())
        else:
            report = {"passed": False, "findings": [{"level": "fail", "message":
                      f"Simulator check did not finish. Inspect {directory / 'launch.log'}"}]}
        report.update(schema="dwarl_sim_check/1", directory=str(directory), profile=str(profile.path),
                      fingerprint=profile.sha256)
        write_json(directory / "result.json", report)
        return report
    finally:
        if process is not None:
            stop_owned(process)
        lease.close()
        build_lock.close()


def observe(directory):
    import rclpy
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzNode
    from nav_msgs.msg import Odometry
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener

    request = json.loads((directory / "request.json").read_text())
    ros, spawn = request["ros"], request["spawn"]
    rclpy.init()
    node = rclpy.create_node("dwarl_integration_check")
    tf = Buffer()
    listener = TransformListener(tf, node)
    state = {"clock": None, "advancing": False, "scan": None, "odom": None, "pose": None, "entities": set()}

    def clock(msg):
        value = msg.clock.sec + msg.clock.nanosec * 1e-9
        if state["clock"] is not None and value > state["clock"]:
            state["advancing"] = True
        state["clock"] = value

    def truth(msg):
        for pose in msg.pose:
            state["entities"].add(pose.name)
            if pose.name.strip("/") == ros.get("gazebo_model", "robot"):
                q = pose.orientation
                state["pose"] = [pose.position.x, pose.position.y,
                                 math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))]

    node.create_subscription(Clock, "/clock", clock, 10)
    node.create_subscription(LaserScan, ros["scan_topic"], lambda m: state.update(scan=m), qos_profile_sensor_data)
    node.create_subscription(Odometry, ros["odom_topic"], lambda m: state.update(odom=m), qos_profile_sensor_data)
    truth_node = GzNode()
    truth_node.subscribe(Pose_V, "/world/default/dynamic_pose/info", truth)
    adapter = ros["sim_launch"]
    process = subprocess.Popen(["ros2", "launch", adapter["package"], adapter["file"],
                                f"world:={directory / 'world.sdf'}", f"x:={spawn[0]}", f"y:={spawn[1]}",
                                f"yaw:={spawn[2]}", "gui:=false"])

    def checks():
        scan, odom, pose = state["scan"], state["odom"], state["pose"]
        command_type = "geometry_msgs/msg/" + ("TwistStamped" if ros.get("cmd_vel_stamped", True) else "Twist")
        command = node.resolve_topic_name(ros["cmd_vel_topic"])
        subscribers = node.get_subscriptions_info_by_topic(command)
        angle_error = abs(math.atan2(math.sin(pose[2]-spawn[2]), math.cos(pose[2]-spawn[2]))) if pose else None
        odom_pose = odom.pose.pose if odom else None
        odom_angle = (math.atan2(2*(odom_pose.orientation.w*odom_pose.orientation.z
                                  + odom_pose.orientation.x*odom_pose.orientation.y),
                                 1-2*(odom_pose.orientation.y**2+odom_pose.orientation.z**2)) if odom else None)
        return [
            (state["advancing"], "Simulation clock advances"),
            (scan is not None and bool(scan.ranges), f"LaserScan received on {ros['scan_topic']}"),
            (odom is not None, f"Odometry received on {ros['odom_topic']}"),
            (tf.can_transform(ros["odom_frame"], ros["base_frame"], rclpy.time.Time()), "Odometry-to-base TF"),
            (scan is not None and tf.can_transform(ros["base_frame"], scan.header.frame_id, rclpy.time.Time()),
             "Base-to-scan TF"),
            (any(s.topic_type == command_type for s in subscribers), f"Command subscriber: {command} ({command_type})"),
            (pose is not None, f"Ground-truth entity {ros.get('gazebo_model', 'robot')} in world/default; "
             f"observed: {', '.join(sorted(state['entities'])) or 'none'}"),
            (pose is not None and math.dist(pose[:2], spawn[:2]) < .15 and angle_error < .15,
             f"Requested spawn {spawn}; observed {pose}"),
            (odom is not None and math.hypot(odom_pose.position.x, odom_pose.position.y) < .15
             and abs(odom_angle) < .15, "Odometry starts at zero at spawn"),
        ]

    try:
        deadline = time.monotonic() + request["timeout"]
        while time.monotonic() < deadline and process.poll() is None:
            rclpy.spin_once(node, timeout_sec=.1)
            if all(passed for passed, _ in checks()):
                break
        findings = [{"level": "pass" if passed else "fail", "message": label} for passed, label in checks()]
        findings.append({"level": "check", "message": "Command endpoint checked; this check sends no motion commands."})
        write_json(directory / "result.json", {"passed": not any(f["level"] == "fail" for f in findings),
                                                "findings": findings})
    finally:
        process.terminate() if process.poll() is None else None
        node.destroy_node()
        rclpy.shutdown()
        del listener


if __name__ == "__main__":
    observe(Path(sys.argv[1]))
