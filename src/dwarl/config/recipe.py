"""Typed training recipe; function-specific manager parameters remain extensible."""

from dataclasses import dataclass, field

from dwarl.config.managers import ManagerBasedNavCfg


@dataclass
class Simulation:
    timestep: float = .005
    tile_gap: float = 1.
    pool_size: int = 300
    num_shards: int = 0
    render_fps: int = 30
    realtime: bool = False
    realtime_speed: float = 1.


@dataclass
class Environment:
    control_dt: float = .1
    episodes_per_map: int = 1
    spawn_yaw_noise: float = 3.14159
    spawn_yaw_curriculum: bool = True
    spawn_yaw_min: float = .5
    spawn_yaw_full_level: float = .6


@dataclass
class Curriculum:
    enabled: bool = True
    init_level: float = .2
    min_level: float = .1
    step: float = .05
    window: int = 200
    promote_success: float = .75
    demote_success: float = .3


@dataclass
class Encoder:
    channels: list[int] = field(default_factory=lambda: [32, 64, 64])
    kernels: list[int] = field(default_factory=lambda: [7, 5, 3])
    strides: list[int] = field(default_factory=lambda: [2, 2, 2])
    latent_dim: int = 128
    activation: str = "elu"


@dataclass
class Distribution:
    class_name: str = "GaussianDistribution"
    init_std: float = .6
    std_type: str = "scalar"


@dataclass
class Model:
    class_name: str = "dwarl.training.models.lidar_cnn:LidarCNNModel"
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    obs_normalization: bool = True
    encoder_cfg: Encoder = field(default_factory=Encoder)
    distribution_cfg: Distribution | None = None


@dataclass
class Algorithm:
    class_name: str = "PPO"
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    clip_param: float = .2
    gamma: float = .99
    lam: float = .95
    value_loss_coef: float = 1.
    entropy_coef: float = .005
    learning_rate: float = .0003
    max_grad_norm: float = 1.
    schedule: str = "adaptive"
    desired_kl: float = .01


@dataclass
class Agent:
    num_steps_per_env: int = 64
    logger: str = "tensorboard"
    obs_groups: dict[str, list[str]] = field(default_factory=dict)
    actor: Model = field(default_factory=Model)
    critic: Model = field(default_factory=Model)
    algorithm: Algorithm = field(default_factory=Algorithm)


@dataclass
class Recipe(ManagerBasedNavCfg):
    sim: Simulation = field(default_factory=Simulation)
    env: Environment = field(default_factory=Environment)
    curriculum: Curriculum = field(default_factory=Curriculum)
    agent: Agent = field(default_factory=Agent)
