"""The public configuration contract. Paths are resolved by the composer."""

from dataclasses import dataclass, field

from .recipe import Recipe


@dataclass
class DataConfig:
    path: str | None = None
    allow_custom: bool = False
    train_maps: int = 10000
    validation_maps: int = 500
    test_maps: int = 500
    pairs_per_map: int = 32
    workers: int = 1


@dataclass
class EarlyStopConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.01


@dataclass
class TrainConfig:
    preset: str | None = None
    name: str = "navigation"
    num_envs: int = 64
    seed: int = 42
    device: str = "auto"
    max_iterations: int = 3000
    resume: str | None = None
    # Start a new run from another run's selected checkpoint: weights whose shapes match are copied
    # (optimizer, iteration and exploration noise start fresh). Lets a recipe change build on a policy.
    init_from: str | None = None
    init_checkpoint: str = "best"  # checkpoint of init_from: best (falls back to latest), latest or an iteration
    show: bool = False
    detached: bool = False
    save_interval: int = 50
    validation_interval: int = 0
    validation_cases: int = 32
    benchmark_interval: int = 0
    benchmark_cases: int = 32
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)
    deployment_ready: bool = True
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class TestConfig:
    run: str = "latest"
    checkpoint: str = "latest"
    suite: str = "benchmark"
    sim: str = "mujoco"
    case: str | None = None
    case_id: str | None = None  # replay one exact materialized case, rather than a generator family
    show: bool = False
    repeats: int = 1
    seed: int = 42
    dataset: str | None = None
    max_cases: int | None = 32
    controllers: list[str] = field(default_factory=lambda: ["dwarl"])
    profile_compute: bool = False
    controller_params: str | None = None
    planner: str = "navfn"
    unmapped_obstacles: int = 3  # target per case; use 0 for clean navigation
    obstacle_comparison: bool = False  # clean + frozen on-route obstacles, in one Gazebo test
    unmapped_near_doorways: bool = True  # retain reachability checks without blanket doorway exclusions
    # Gazebo: place them on Nav2's first global plan recorded per case in this earlier test directory
    # (a run without unmapped obstacles), instead of on the dataset path, which Nav2 may not follow.
    unmapped_route: str | None = None
    startup_timeout: float = 120.0
    stall_timeout: float = 30.0


@dataclass
class DeployConfig:
    run: str = "latest"
    checkpoint: str = "latest"


@dataclass
class Config:
    schema: str = "dwarl/1"
    robot: str | None = None
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)
    deploy: DeployConfig = field(default_factory=DeployConfig)
    recipe: Recipe = field(default_factory=Recipe)
