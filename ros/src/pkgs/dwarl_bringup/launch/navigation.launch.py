"""Nav2 navigation stack with the DWARL controller for *any* robot, configured from a policy bundle.

No localisation / map server (provided by the caller). Usable on the robot and in simulation.

    ros2 launch dwarl_bringup navigation.launch.py model_dir:=<bundle>          (or run:=<train run>|latest)

The bundle's deploy.yaml sets frames, footprint, scan topic, control rate, goal tolerance and the cmd_vel
interface (topic, TwistStamped or Twist). safety:=true (default) adds the collision_monitor layer:
    controller_server -cmd_vel_raw-> collision_monitor -> <cmd_vel_topic>
A STOP fails FollowPath and bt_navigator backs up and retries the policy (behavior_trees/dwarl_safety_retry.xml).
The policy output is already rate limited like in training: no velocity smoother.
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from dwarl_bringup.nav import load_deploy, render_nav2_params, render_recovery, resolve_bundle, write_params
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration as LC
from launch_ros.actions import Node, SetParameter


def _setup(context):
    get = lambda k: LC(k).perform(context)  # noqa: E731
    here = get_package_share_directory("dwarl_bringup")
    bundle = resolve_bundle(get("model_dir"))
    deploy = load_deploy(bundle)
    safety = get("safety") == "true"
    ros = deploy.get("ros") or {}
    params = write_params(
        render_nav2_params(os.path.join(here, "config", "nav2_params.yaml"), bundle, safety, get("controller"),
                           get("profile_compute") == "true", get("controller_params") or None),
        get("params_output") or tempfile.NamedTemporaryFile("w", prefix="dwarl_nav2_", suffix=".yaml", delete=False).name,
    )
    cmd_vel = get("cmd_vel_topic") or ros.get("cmd_vel_topic", "cmd_vel")
    stamped = ros.get("cmd_vel_stamped", False) if get("cmd_vel_stamped") == "" else get("cmd_vel_stamped") == "true"
    bt = (
        os.path.join(here, "behavior_trees", "dwarl_safety_retry.xml")
        if safety
        else os.path.join(
            get_package_share_directory("nav2_bt_navigator"),
            "behavior_trees",
            "navigate_to_pose_w_replanning_and_recovery.xml",
        )
    )
    recovery_monitor = False
    if safety and get("controller_params"):
        bt, recovery_monitor = render_recovery(bt, params + ".bt.xml", get("controller_params"))
    planner = os.path.join(here, "config", f"planner_{get('planner')}.yaml")
    args = ["--ros-args", "--log-level", get("log_level")]
    nodes = ["controller_server", "planner_server", "behavior_server", "bt_navigator"] + (
        ["collision_monitor"] if safety else []
    )
    actions = [
        SetParameter("use_sim_time", get("use_sim_time") == "true"),
        SetParameter("enable_stamped_cmd_vel", bool(stamped)),
        Node(
            package="nav2_controller",
            executable="controller_server",
            output="screen",
            arguments=args,
            parameters=[params],
            remappings=[("cmd_vel", "cmd_vel_raw" if safety else cmd_vel)],
        ),
        Node(
            package="nav2_planner",
            executable="planner_server",
            output="screen",
            arguments=args,
            parameters=[params, planner],
        ),
        Node(
            package="nav2_behaviors",
            executable="behavior_server",
            output="screen",
            arguments=args,
            parameters=[params],
            remappings=[("cmd_vel", "cmd_vel_raw" if safety and recovery_monitor else cmd_vel)],
        ),
        Node(
            package="nav2_bt_navigator",
            executable="bt_navigator",
            output="screen",
            arguments=args,
            parameters=[params, {"default_nav_to_pose_bt_xml": bt}],
        ),
        Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="lifecycle_manager_navigation",
            output="screen",
            parameters=[{"autostart": True, "node_names": nodes}],
        ),
    ]
    if safety:
        actions.insert(
            3,
            Node(
                package="nav2_collision_monitor",
                executable="collision_monitor",
                output="screen",
                arguments=args,
                parameters=[params, {"cmd_vel_out_topic": cmd_vel}],
            ),
        )
    return [
        LogInfo(
            msg=f"[dwarl] policy bundle {bundle} (robot '{deploy['robot']['id']}'), cmd_vel -> {cmd_vel} "
            f"({'TwistStamped' if stamped else 'Twist'}), safety={safety}"
        ),
        GroupAction(actions),
    ]


def generate_launch_description():
    args = [
        ("model_dir", "", "explicit immutable policy bundle (deploy.yaml + policy.onnx)"),
        ("use_sim_time", "false", ""),
        ("safety", "true", "collision_monitor stop + back-up-and-retry behavior tree"),
        ("planner", "navfn", "global planner: navfn | smac2d"),
        ("controller", "rl", "local controller: dwarl | mppi | dwb (matched bundle limits)"),
        ("profile_compute", "false", "publish wall-clock controller call timings"),
        ("controller_params", "", "optional controller tuning YAML"),
        ("params_output", "", "optional rendered parameter snapshot path"),
        ("cmd_vel_topic", "", "override the bundle's ros.cmd_vel_topic"),
        ("cmd_vel_stamped", "", "override the bundle's ros.cmd_vel_stamped (true|false)"),
        ("log_level", "info", ""),
    ]
    return LaunchDescription(
        [DeclareLaunchArgument(n, default_value=d, description=h) for n, d, h in args]
        + [OpaqueFunction(function=_setup)]
    )
