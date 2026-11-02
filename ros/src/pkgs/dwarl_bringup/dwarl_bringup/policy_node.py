"""Standalone ROS 2 node running a DWARL policy bundle via ``dwarl_runtime`` (no Nav2 needed).

For stacks with their own planner / executive: follow a ``nav_msgs/Path`` with the policy.

    ros2 run dwarl_bringup policy_node --ros-args -p model_dir:=<bundle>

Subscribes ``plan`` (nav_msgs/Path, any fixed frame), the scan and ``odom`` (measured v, w); robot pose via TF
(plan frame -> base frame); publishes cmd_vel (TwistStamped or Twist per the bundle's ``ros`` section) at the
training rate and ``~/goal_reached`` (std_msgs/Bool). Stops (zero command) on a stale scan / missing TF, when
the goal is reached or when an empty plan is received. Requires ``pip install <DWARL>/runtime``.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

from dwarl_bringup.nav import resolve_bundle


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("dwarl_policy")
        from dwarl_runtime import Policy

        p = {k: self.declare_parameter(k, v).value for k, v in dict(
            model_dir="", expected_robot_id="", scan_topic="", cmd_vel_topic="",
            base_frame="", scan_timeout=0.5, inference_threads=1).items()}  # fmt: skip
        bundle = resolve_bundle(p["model_dir"])
        self.policy = Policy(
            bundle, threads=int(p["inference_threads"]), expected_robot_id=p["expected_robot_id"] or None
        )
        ros = self.policy.spec.ros
        self.base = p["base_frame"] or ros.get("base_frame", "base_link")
        self.stamped = bool(ros.get("cmd_vel_stamped", False))
        self.scan_timeout = float(p["scan_timeout"])
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.scan: LaserScan | None = None
        self.vel = (0.0, 0.0)
        self.odom_stamp = None
        self.plan_frame = ""
        self.done = True
        self.create_subscription(LaserScan, p["scan_topic"] or ros.get("scan_topic", "scan"), self._on_scan,
                                 qos_profile_sensor_data)  # fmt: skip
        self.create_subscription(Odometry, ros.get("odom_topic", "odom"), self._on_odom, 10)
        self.create_subscription(Path, "plan", self._on_plan, 10)
        topic = p["cmd_vel_topic"] or ros.get("cmd_vel_topic", "cmd_vel")
        self.pub = self.create_publisher(TwistStamped if self.stamped else Twist, topic, 10)
        self.reached_pub = self.create_publisher(Bool, "~/goal_reached", 10)
        self.create_timer(self.policy.dt, self._tick)
        self.get_logger().info(f"policy {bundle} (robot '{self.policy.spec.robot_id}') -> {topic}, {self.policy.dt}s")

    def _on_scan(self, m: LaserScan) -> None:
        self.scan = m

    def _on_odom(self, m: Odometry) -> None:
        self.vel = (m.twist.twist.linear.x, m.twist.twist.angular.z)
        self.odom_stamp = rclpy.time.Time.from_msg(m.header.stamp)

    def _on_plan(self, m: Path) -> None:
        if not m.poses:
            self.done = True
            self.policy.reset()
            self._publish(0.0, 0.0)
            return
        if self.plan_frame and self.plan_frame != m.header.frame_id:
            self.policy.tracker = type(self.policy.tracker)()
            self.policy.reset()
        self.plan_frame = m.header.frame_id
        self.policy.set_plan([(q.pose.position.x, q.pose.position.y) for q in m.poses])
        self.done = False

    def _publish(self, v: float, w: float) -> None:
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.base
            msg.twist.linear.x, msg.twist.angular.z = v, w
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = v, w
        self.pub.publish(msg)

    def _tick(self) -> None:
        from dwarl_runtime import Scan

        if self.done:
            return
        now = self.get_clock().now()
        s = self.scan
        if self.odom_stamp is None or (now - self.odom_stamp).nanoseconds * 1e-9 > self.scan_timeout:
            self.policy.reset()
            return self._publish(0.0, 0.0)
        if s is None or (now - rclpy.time.Time.from_msg(s.header.stamp)).nanoseconds * 1e-9 > self.scan_timeout:
            self.get_logger().warn("no recent scan: stopping", throttle_duration_sec=1.0)
            self.policy.reset()
            return self._publish(0.0, 0.0)
        try:
            pose = self.tf.lookup_transform(self.plan_frame, self.base, rclpy.time.Time()).transform
            laser = self.tf.lookup_transform(self.base, s.header.frame_id, rclpy.time.Time()).transform
        except TransformException as e:
            self.get_logger().warn(f"TF: {e}", throttle_duration_sec=1.0)
            self.policy.reset()
            return self._publish(0.0, 0.0)
        q = pose.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        lt, lq = laser.translation, laser.rotation
        try:
            cmd = self.policy.step(
                Scan(np.asarray(s.ranges), s.angle_min, s.angle_increment, s.range_min, s.range_max),
                ((lt.x, lt.y, lt.z), (lq.x, lq.y, lq.z, lq.w)),
                (pose.translation.x, pose.translation.y, yaw),
                self.vel,
            )
        except (ValueError, RuntimeError) as e:
            self.get_logger().error(f"policy: {e}; stopping", throttle_duration_sec=1.0)
            self.policy.reset()
            return self._publish(0.0, 0.0)
        if cmd.goal_reached:
            self.done = True
            self.reached_pub.publish(Bool(data=True))
            return self._publish(0.0, 0.0)
        self._publish(cmd.v, cmd.w)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
