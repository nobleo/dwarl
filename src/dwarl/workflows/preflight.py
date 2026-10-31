"""Read-only selection and prerequisite checks for the primary commands."""

from dwarl.robot.profile import load_profile
from dwarl.runs.store import resolve_checkpoint, resolve_run
from dwarl.training.contract import load_contract


def execute(command, cfg):
    if command == "train":
        if cfg.train.resume:
            checkpoint = resolve_checkpoint(resolve_run(cfg.train.resume))
            load_contract(checkpoint.run)
            print(f"Resume: {checkpoint.run}, iteration {checkpoint.iteration}; episodes restart")
        else:
            if not cfg.robot:
                raise ValueError("select a robot: dwarl train <integration>")
            from dwarl.robot.build import build_xml, summarize

            profile = load_profile(cfg.robot)
            summarize(profile, build_xml(profile))
            print(f"Robot: {profile.id} ({profile.path})")
            print(f"Data: {cfg.train.data.path or 'prepare/reuse robot-sized dataset cache'}")
        print(f"Train to iteration {cfg.train.max_iterations}; publish completed checkpoints atomically")
        return
    selected = cfg[command]
    checkpoint = resolve_checkpoint(resolve_run(selected.run), selected.checkpoint)
    contract = load_contract(checkpoint.run)
    print(f"Run: {checkpoint.run}\nCheckpoint: {checkpoint.iteration} ({checkpoint.sha256})")
    if command == "deploy":
        if not cfg.robot:
            raise ValueError("select the target integration: dwarl deploy <robot>")
        from dwarl.workflows.deploy import check_compatibility

        profile = load_profile(cfg.robot)
        check_compatibility(checkpoint, profile)
        print(f"Install compatible bundle into {profile.dir / profile.raw['deployment']['policy_directory']}")
        return
    if cfg.test.suite == "benchmark" and not cfg.test.dataset:
        from dwarl.testing.scenarios import SCENARIOS

        names = sorted(SCENARIOS)
    else:
        from dwarl.training.environment.dataset import MapDataset

        split = {"validation": "val", "test": "test", "benchmark": "scenarios"}[cfg.test.suite]
        dataset = MapDataset(cfg.test.dataset or contract.dataset.path, split)
        names = sorted({m.generator for m in dataset.maps})
    if cfg.test.case and cfg.test.case not in names:
        raise ValueError(f"unknown case {cfg.test.case!r}; available: {', '.join(names)}")
    if cfg.test.sim in ("gazebo", "both"):
        import subprocess

        from dwarl.testing.backends.gazebo import prerequisites

        prerequisites()
        for controller in cfg.test.controllers:
            package = {"mppi": "nav2_mppi_controller", "dwb": "dwb_core"}.get(controller)
            if package and subprocess.run(["ros2", "pkg", "prefix", package], capture_output=True).returncode:
                raise ValueError(f"ROS package {package} unavailable")
        adapter = contract.robot.ros.get("sim_launch")
        if not adapter:
            raise ValueError("robot has no ros.sim_launch adapter")
        result = subprocess.run(["ros2", "pkg", "prefix", adapter.package], capture_output=True, text=True)
        if result.returncode:
            raise ValueError(f"robot ROS package {adapter.package} unavailable; source its built workspace")
    print(f"Prepare/reuse ONNX bundle; materialize {cfg.test.case or cfg.test.suite}; run {cfg.test.sim}")
    print(f"Controllers: {', '.join(cfg.test.controllers)}; "
          f"case cap per controller: {cfg.test.max_cases or 'all'}; compute profiling: {cfg.test.profile_compute}")
