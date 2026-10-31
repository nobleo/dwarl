"""The public configuration contract. Paths are resolved by the composer."""

from dataclasses import dataclass, field

from .recipe import Recipe


@dataclass
class DataConfig:
    path: str | None = None
    allow_custom: bool = False
    train_maps: int = 128
    validation_maps: int = 16
    test_maps: int = 16
    pairs_per_map: int = 32
    workers: int = 1


@dataclass
class TrainConfig:
    name: str = "navigation"
    num_envs: int = 64
    seed: int = 42
    device: str = "auto"
    max_iterations: int = 3000
    resume: str | None = None
    show: bool = False
    detached: bool = False
    save_interval: int = 50
    validation_interval: int = 200
    validation_cases: int = 32
    deployment_ready: bool = True
    data: DataConfig = field(default_factory=DataConfig)


@dataclass
class TestConfig:
    run: str = "latest"
    checkpoint: str = "latest"
    suite: str = "benchmark"
    sim: str = "mujoco"
    case: str | None = None
    show: bool = False
    repeats: int = 1
    seed: int = 42
    dataset: str | None = None
    max_cases: int | None = None
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
