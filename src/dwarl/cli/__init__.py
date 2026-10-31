"""Small argument adapter; workflows own execution."""

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from dwarl.config import compose_config


def parser():
    root = argparse.ArgumentParser(prog="dwarl", description="Train, test and install a robot local planner.")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("train", "test", "deploy"):
        p = commands.add_parser(name)
        p.add_argument("--config", help="YAML configuration (replaces ./dwarl.yaml)")
        p.add_argument("--set", action="append", default=[], metavar="PATH=VALUE", help="advanced schema override")
        p.add_argument("--print-config", action="store_true", help="print resolved configuration and exit")
        p.add_argument("--dry-run", action="store_true", help="check selected inputs without running")
        if name in ("train", "deploy"):
            p.add_argument("robot", nargs="?", help="robot integration directory or profile")
        if name in ("test", "deploy"):
            p.add_argument("--run", help="training run directory (default: latest)")
            p.add_argument("--checkpoint", help="latest, best or iteration number (default: latest)")
        if name == "train":
            for flag in ("num-envs", "seed", "max-iterations"):
                p.add_argument(f"--{flag}", type=int)
            for flag in ("device", "name", "resume"):
                p.add_argument(f"--{flag}")
            p.add_argument("--detached", action="store_true", default=None)
        if name == "test":
            p.add_argument("--sim", choices=("mujoco", "gazebo", "both"))
            p.add_argument("--suite", choices=("benchmark", "validation", "test"))
            p.add_argument("--case")
            p.add_argument("--repeats", type=int)
            p.add_argument("--controllers", type=lambda value: value.split(","),
                           help="Gazebo controllers: dwarl,mppi,dwb (default: dwarl)")
            p.add_argument("--profile-compute", action="store_true", default=None)
        if name in ("train", "test"):
            p.add_argument("--show", action="store_true", default=None)
    p = commands.add_parser("runs", help="list local runs")
    p.add_argument("action", choices=["list"], nargs="?", default="list")
    p = commands.add_parser("doctor", help="check simulator prerequisites")
    p.add_argument("--sim", choices=["mujoco", "gazebo"], default="mujoco")
    p = commands.add_parser("robot", help="inspect a robot integration")
    p.add_argument("action", choices=["check", "view"])
    p.add_argument("robot")
    p = commands.add_parser("stop", help="interrupt an owned training process")
    p.add_argument("run")
    p = commands.add_parser("model-benchmark", help="replay recorded ONNX inputs without ROS/Gazebo")
    p.add_argument("bundle")
    p.add_argument("observations", help="observations.npz from a profiled DWARL Gazebo episode")
    p.add_argument("--output", default="model-benchmark.json")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=100)
    return root


def app():
    command_parser = parser()
    seen = set()
    for token in sys.argv[1:]:
        flag = token.partition("=")[0]
        if flag.startswith("--") and flag != "--set":
            if flag in seen:
                command_parser.error(f"duplicate CLI assignment: {flag}")
            seen.add(flag)
    args = command_parser.parse_args()
    try:
        code = dispatch(args)
    except (ValueError, OSError, KeyError, OmegaConfBaseException) as exc:
        print(f"dwarl: {exc}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        code = 130
    raise SystemExit(code or 0)


def dispatch(args):
    if args.command not in ("train", "test", "deploy"):
        return utility(args)
    assignments = list(args.set)
    excluded = {"command", "config", "set", "print_config", "dry_run"}
    for name, value in vars(args).items():
        if name in excluded or value is None:
            continue
        key = "robot" if name == "robot" else f"{args.command}.{name}"
        assignments.append(f"{key}={json.dumps(value)}")
    explicit_keys = []
    cfg = compose_config(args.config, assignments, explicit_keys=explicit_keys)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if args.dry_run:
        from dwarl.workflows.preflight import execute

        execute(args.command, cfg)
        return
    if args.command == "train":
        if cfg.train.detached:
            from dwarl.runs.process import detach

            return detach(cfg)
        from dwarl.workflows.train import execute

        execute(cfg, explicit_keys)
    elif args.command == "test":
        from dwarl.workflows.test import execute

        _, report = execute(cfg)
        if report["status"] == "interrupted":
            return 130
        return (0 if report["passed"] else 1) if report["status"] == "completed" else 2
    else:
        from dwarl.workflows.deploy import execute

        execute(cfg)


def utility(args):
    if args.command == "model-benchmark":
        from dwarl.testing.model_benchmark import execute

        execute(args.bundle, args.observations, args.output, args.threads, args.iterations, args.warmup)
    elif args.command == "runs":
        for path in sorted(Path("runs").glob("*/status.json")):
            print(f"{path.parent}: {json.loads(path.read_text())['state']}")
    elif args.command == "doctor":
        if args.sim == "gazebo":
            from dwarl.testing.backends.gazebo import prerequisites

            prerequisites()
        else:
            import mujoco
            import onnxruntime

            print(f"MuJoCo {mujoco.__version__}; ONNX Runtime {onnxruntime.__version__}")
    elif args.command == "robot":
        from dwarl.robot.build import build_xml, summarize
        from dwarl.robot.profile import load_profile

        profile = load_profile(args.robot)
        xml = build_xml(profile)
        print(json.dumps(summarize(profile, xml).as_dict(), indent=2))
        if args.action == "view":
            import mujoco
            import mujoco.viewer

            from dwarl.robot.view import inspection_model

            model, data = inspection_model(xml)
            print("Robot inspection on a floor (simplified training geometry). Close the window to exit.", flush=True)
            mujoco.viewer.launch(model, data)
    elif args.command == "stop":
        from dwarl.runs.process import stop

        stop(args.run)
