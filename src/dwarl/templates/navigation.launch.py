"""Launch Nav2 with the robot integration's selected policy."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def setup(context):
    integration = Path(LaunchConfiguration("integration").perform(context)).resolve()
    profile = yaml.safe_load((integration / "dwarl_robot.yaml").read_text())
    selected = profile.get("deployment", {}).get("selected_policy", "selected_policy.yaml")
    selection = (integration / selected).resolve()
    if not selection.is_relative_to(integration) or not selection.is_file():
        raise RuntimeError(f"No selected policy. Run: pixi run deploy {integration}")
    data = yaml.safe_load(selection.read_text())
    if data.get("schema") != "dwarl_selected_policy/1":
        raise RuntimeError(f"Unsupported selection: {selection}")
    bundle = (integration / data["path"]).resolve()
    if not bundle.is_relative_to(integration) or not all(
        (bundle / name).is_file() for name in ("deploy.yaml", "policy.onnx")
    ):
        raise RuntimeError(f"Selected bundle is missing or outside the integration: {bundle}")
    source = Path(get_package_share_directory("dwarl_bringup")) / "launch/navigation.launch.py"
    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(source)),
        launch_arguments={"model_dir": str(bundle),
                          "use_sim_time": LaunchConfiguration("use_sim_time").perform(context)}.items(),
    )]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("integration", default_value=str(Path(__file__).resolve().parents[1])),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        OpaqueFunction(function=setup),
    ])
