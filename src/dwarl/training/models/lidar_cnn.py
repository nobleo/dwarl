"""Actor/critic model: RSL-RL's ``CNNModel`` with a circular 1D-CNN lidar encoder.

rsl-rl's own CNN is 2D and pads height like width, which fails on a (1, num_rays) "image". We therefore
only swap in a 1D encoder via ``CNNModel``'s ``cnns=`` hook; everything else (normaliser, MLP, distribution,
and ``as_onnx`` / ``as_jit`` export) is stock rsl-rl.

The lidar observation group is shaped (B, 1, H, num_rays), where H is the number of scan-history
entries, so rsl-rl treats it as a 2D group. The encoder uses H as its Conv1d input channels.
Exported ONNX: inputs ``obs`` (B, n_1d, normalised in-graph) + ``lidar`` (B, 1, H, num_rays) -> ``actions``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.models import CNNModel
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class LidarEncoder(nn.Module):
    """(B, 1, H, R) -> (B, latent_dim): the H stacked scans are Conv1d input channels; circular padding for a
    360 deg scan (it wraps around)."""

    output_channels = None  # flattened output (required by CNNModel)

    def __init__(self, num_rays: int, in_channels: int = 1, channels=(16, 32, 32), kernels=(5, 5, 3), strides=(2, 2, 2),
                 latent_dim: int = 64, activation: str = "elu", circular: bool = True) -> None:  # fmt: skip
        super().__init__()
        dimensions = (num_rays, in_channels, latent_dim, *channels, *kernels, *strides)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dimensions):
            raise ValueError("encoder dimensions, kernels and strides must be positive integers")
        if not channels or len(channels) != len(kernels) or len(channels) != len(strides):
            raise ValueError("encoder channels, kernels and strides must have equal nonzero lengths")
        layers: list[nn.Module] = []
        c_in, length = in_channels, num_rays
        for c, k, s in zip(channels, kernels, strides, strict=True):
            if circular and k // 2 > length:
                raise ValueError(f"circular encoder padding {k // 2} exceeds input length {length}")
            layers += [nn.Conv1d(c_in, c, k, s, padding=k // 2, padding_mode="circular" if circular else "zeros"),
                       resolve_nn_activation(activation)]  # fmt: skip
            c_in = c
            length = (length + 2 * (k // 2) - k) // s + 1
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(c_in * length, latent_dim), resolve_nn_activation(activation))
        self.output_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.flatten(1, 2)))  # (B,1,H,R) -> (B,H,R)


class LidarCNNModel(CNNModel):
    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int,
                 hidden_dims=(256, 256, 128), activation: str = "elu", obs_normalization: bool = False,
                 distribution_cfg: dict | None = None, encoder_cfg: dict | None = None) -> None:  # fmt: skip
        encoder_options = {"activation": activation, **(encoder_cfg or {})}
        cnns = {g: LidarEncoder(int(obs[g].shape[-1]), int(obs[g].shape[1] * obs[g].shape[2]),
                                **encoder_options)
                for g in obs_groups[obs_set] if obs[g].dim() == 4}  # fmt: skip
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization,
                         distribution_cfg, cnns=nn.ModuleDict(cnns))  # fmt: skip
