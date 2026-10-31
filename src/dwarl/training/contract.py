from omegaconf import OmegaConf, open_dict

from dwarl.config.managers import apply_manager_schemas
from dwarl.robot.artifact import resolve_artifact


def build_contract(public):
    cfg = OmegaConf.create(OmegaConf.to_container(public.recipe, resolve=True))
    with open_dict(cfg):
        cfg.robot = {"profile": public.robot, "v_min": None, "v_max": None,
                     "w_max": None, "a_max": None, "alpha_max": None}
        cfg.lidar = dict.fromkeys(("num_rays", "fov_deg", "min_range", "max_range", "noise_std", "dropout_prob"))
        cfg.dataset = {"path": None, "split": "train"}
        cfg.env.num_envs = public.train.num_envs
        cfg.seed = public.train.seed
        cfg.device = public.train.device
        cfg.sim.gui = public.train.show
        cfg.agent.save_interval = public.train.save_interval
    return apply_manager_schemas(cfg)


def load_contract(run):
    return resolve_artifact(apply_manager_schemas(OmegaConf.load(run / "contract.yaml")), run)


def device(value):
    import torch

    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value
