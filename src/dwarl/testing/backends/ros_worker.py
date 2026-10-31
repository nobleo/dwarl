"""One fresh Gazebo/Nav2 process tree per canonical case; invoked only by GazeboBackend."""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from dwarl.datasets.generation.mapdata import MapData
from dwarl.navigation.clearance import footprint_clearance
from dwarl.runs.store import write_json
from dwarl.testing.scoring import score
from dwarl.testing.trace import BehaviorTrace


def execute(directory):
    import rclpy
    from action_msgs.msg import GoalStatus
    from geometry_msgs.msg import Twist, TwistStamped
    from gz.msgs10.pose_v_pb2 import Pose_V
    from gz.transport13 import Node as GzNode
    from lifecycle_msgs.srv import GetState
    from nav2_msgs.action import NavigateToPose
    from nav2_msgs.msg import CollisionMonitorState
    from nav_msgs.msg import Odometry
    from nav_msgs.msg import Path as RosPath
    from rclpy.action import ActionClient
    from rclpy.qos import qos_profile_sensor_data
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from tf2_ros import Buffer, TransformListener

    data = json.loads((directory / "input.json").read_text())
    case, request, robot = data["case"], data["request"], data["robot"]
    ros = robot["ros"]
    x, y, yaw = case["start"]
    show = str(request["show"]).lower()
    command = ["ros2", "launch", "dwarl_bringup", "bench.launch.py", f"model_dir:={data['bundle']}",
               f"world:={directory / 'world.sdf'}", f"map:={directory / 'world.yaml'}",
               f"x:={x}", f"y:={y}", f"yaw:={yaw}", f"gui:={show}", f"rviz:={show}", "safety:=true"]
    processes = []
    rclpy.init()
    node = rclpy.create_node("dwarl_case_observer")
    tf = Buffer()
    node.tf_listener = TransformListener(tf, node)
    state = {"clock": None, "scan": None, "odom": False, "pose": None, "stops": 0, "recoveries": 0,
             "plan": [], "clock_wall": time.monotonic(), "pose_wall": time.monotonic(), "seen_frames": set(),
             "stopping": False, "velocity": None, "command": None}
    trace, start_time, partial = None, None, None
    model_name = ros.get("gazebo_model", "robot")

    def clock(msg):
        value = msg.clock.sec + msg.clock.nanosec * 1e-9
        if value != state["clock"]:
            state.update(clock=value, clock_wall=time.monotonic())

    def truth(msg):
        for pose in msg.pose:
            state["seen_frames"].add(pose.name)
            if pose.name.strip("/") == model_name:
                p, q = pose.position, pose.orientation
                angle = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))
                state.update(pose=[p.x, p.y, angle], pose_wall=time.monotonic())

    def safety(msg):
        stopping = msg.action_type == CollisionMonitorState.STOP
        if stopping and not state["stopping"]:
            state["stops"] += 1
        if trace is not None and stopping != state["stopping"]:
            trace.event("safety_stop" if stopping else "safety_resume", state["clock"] - start_time, state["pose"])
        state["stopping"] = stopping

    def odometry(msg):
        state["odom"] = True
        state["velocity"] = (state["clock"], (msg.twist.twist.linear.x, msg.twist.twist.angular.z))

    def commanded(msg):
        twist = msg.twist if ros.get("cmd_vel_stamped", True) else msg
        state["command"] = (state["clock"], (twist.linear.x, twist.angular.z))

    def recovery(msg):
        count = msg.feedback.number_of_recoveries
        if count > state["recoveries"] and trace is not None:
            trace.event("recovery", state["clock"] - start_time, state["pose"],
                        count=count, increase=count-state["recoveries"])
        state["recoveries"] = max(state["recoveries"], count)

    def fresh(key):
        value = state[key]
        if value is None or value[0] is None or not 0 <= state["clock"] - value[0] <= .5:
            return None
        return value[1]

    node.create_subscription(Clock, "/clock", clock, 10)
    node.create_subscription(LaserScan, ros["scan_topic"], lambda msg: state.update(scan=msg.header.frame_id),
                             qos_profile_sensor_data)
    node.create_subscription(Odometry, ros["odom_topic"], odometry, qos_profile_sensor_data)
    node.create_subscription(TwistStamped if ros.get("cmd_vel_stamped", True) else Twist,
                             ros["cmd_vel_topic"], commanded, qos_profile_sensor_data)
    truth_node = GzNode()
    truth_node.subscribe(Pose_V, "/world/default/dynamic_pose/info", truth)
    node.create_subscription(CollisionMonitorState, "collision_monitor_state", safety, 10)
    node.create_subscription(RosPath, "/plan", lambda msg: state["plan"].append(
        [[p.pose.position.x, p.pose.position.y] for p in msg.poses]), 10)
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")

    def spin_until(condition, timeout):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() >= deadline:
                raise TimeoutError("ROS readiness/action wall-clock timeout: " + str({
                    key: state[key] for key in ("clock", "scan", "odom", "pose", "seen_frames")}))
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("owned ROS/Gazebo child exited; see launch.log")
            rclpy.spin_once(node, timeout_sec=.05)

    try:
        processes.append(subprocess.Popen(command))
        spin_until(lambda: state["clock"] is not None and state["pose"] is not None and state["scan"]
                   and state["odom"] and client.server_is_ready(), request["startup_timeout"])
        spin_until(lambda: tf.can_transform(ros["map_frame"], ros["base_frame"], rclpy.time.Time())
                   and tf.can_transform(ros["base_frame"], state["scan"], rclpy.time.Time()),
                   request["startup_timeout"])
        for name in ("map_server", "planner_server", "controller_server", "bt_navigator", "collision_monitor"):
            service = node.create_client(GetState, f"/{name}/get_state")
            deadline = time.monotonic() + request["startup_timeout"]
            while True:
                spin_until(service.service_is_ready, max(.1, deadline-time.monotonic()))
                future = service.call_async(GetState.Request())
                spin_until(future.done, max(.1, deadline-time.monotonic()))
                if future.result().current_state.id == 3:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{name} did not activate")
                rclpy.spin_once(node, timeout_sec=.1)
        if math.dist(state["pose"][:2], case["start"][:2]) > .1:
            raise RuntimeError("Gazebo ground-truth spawn does not match the canonical start")
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = ros["map_frame"]
        goal.pose.pose.position.x, goal.pose.pose.position.y = case["goal"]
        goal.pose.pose.orientation.w = 1.
        sent = client.send_goal_async(goal, feedback_callback=recovery)
        spin_until(sent.done, request["startup_timeout"])
        handle = sent.result()
        if not handle.accepted:
            raise RuntimeError("active Nav2 rejected the case goal")
        result = handle.get_result_async()
        start_time, wall_start = state["clock"], time.monotonic()
        trajectory, travelled, clearance, collision = [state["pose"][:2]], 0., float("inf"), False
        md = MapData.load(directory.parents[1] / case["map_file"])
        footprint = np.asarray(robot["footprint"]) if robot.get("footprint") else None
        trace = BehaviorTrace(case, pose_source="Gazebo ground truth in canonical map coordinates",
                              velocity_source=f"{ros['odom_topic']} odometry twist in its child frame",
                              command_source=f"{ros['cmd_vel_topic']} after collision monitor or direct Nav2 recovery",
                              clearance_source="sampled ground-truth 2D footprint geometry, not contact sensors")
        trace.data["freshness"] = "command/velocity are latest received samples, null after 0.5 s of simulation time"
        trace.data["sample_convention"] = "ground-truth pose samples with latest fresh ROS command/velocity"
        state["stops"], state["recoveries"] = 0, 0
        distance = float(footprint_clearance(np.asarray(state["pose"][:2]), state["pose"][2],
                                           robot["footprint_radius"], footprint, md.primitives))
        clearance = distance
        collision = distance <= 0.
        trace.sample(0., state["pose"], command=fresh("command"), velocity=fresh("velocity"), clearance=distance)
        if state["stopping"]:
            state["stops"] = 1
            trace.event("safety_stop", 0., state["pose"], already_active=True)
        partial = {"case_id": case["id"], "trajectory": trajectory, "_trace": trace.data,
                   "travelled_distance": 0., "completion_time": 0., "clearance": clearance,
                   "safety_stops": state["stops"], "recovery_actions": state["recoveries"], "collision": collision}
        write_json(directory / "partial.json", partial)
        last_sample = None
        last_snapshot = 0.
        while not result.done() and state["clock"] - start_time < case["time_budget"] and not collision:
            rclpy.spin_once(node, timeout_sec=.02)
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("ROS/Gazebo exited during the episode")
            now = time.monotonic()
            if now - min(state["clock_wall"], state["pose_wall"]) > request["stall_timeout"]:
                raise TimeoutError("simulation clock or ground-truth pose stalled")
            if now - wall_start > case["time_budget"] * 10 + request["stall_timeout"]:
                raise TimeoutError("episode wall-clock budget exhausted")
            if state["pose_wall"] == last_sample:
                continue
            last_sample = state["pose_wall"]
            pose = state["pose"]
            travelled += math.dist(trajectory[-1], pose[:2])
            trajectory.append(pose[:2])
            distance = footprint_clearance(np.asarray(pose[:2]), pose[2], robot["footprint_radius"],
                                           footprint, md.primitives)
            clearance = min(clearance, float(distance))
            collision |= distance <= 0.
            trace.sample(state["clock"] - start_time, pose, command=fresh("command"),
                         velocity=fresh("velocity"), clearance=float(distance))
            partial = {"case_id": case["id"], "trajectory": trajectory, "_trace": trace.data,
                       "travelled_distance": travelled, "completion_time": state["clock"]-start_time,
                       "clearance": clearance, "safety_stops": state["stops"],
                       "recovery_actions": state["recoveries"], "collision": collision}
            if now - last_snapshot >= .5:
                write_json(directory / "partial.json", partial)
                last_snapshot = now
        timed_out = state["clock"] - start_time >= case["time_budget"]
        succeeded = result.done() and result.result().status == GoalStatus.STATUS_SUCCEEDED
        reason = ("nav2_succeeded" if succeeded else "nav2_aborted"
                  if result.done() and result.result().status == GoalStatus.STATUS_ABORTED else
                  "nav2_canceled" if result.done() else "collision" if collision else "time_budget")
        if succeeded and math.dist(state["pose"][:2], case["goal"]) > case["tolerance"]:
            reason = "nav2_succeeded_outside_goal_tolerance"
        if not result.done():
            handle.cancel_goal_async()
        row = score(case, success=succeeded and math.dist(state["pose"][:2], case["goal"]) <= case["tolerance"],
                    collision=collision, timeout=timed_out, travelled=travelled,
                    elapsed=state["clock"]-start_time, clearance=clearance if math.isfinite(clearance) else None,
                     trajectory=trajectory, safety_stops=state["stops"], recoveries=state["recoveries"],
                     termination_reason=reason)
        row["_trace"] = trace.data
        row["diagnostics"] = {"localization": "spawn_transform_sim_odometry", "planner": "navfn",
                              "safety": True, "global_plans": state["plan"],
                              "collision_measurement": "ground_truth_2d_footprint_geometry",
                              "limitation": "sampled poses; not continuous contact reporting"}
        write_json(directory / "result.json", row)
    finally:
        if partial is not None:
            write_json(directory / "partial.json", partial)
        truth_node.unsubscribe("/world/default/dynamic_pose/info")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
