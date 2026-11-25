"""Typed in-process episode records; JSON and NPZ remain the reporting boundary."""

from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EpisodeStep:
    """Measurements from one completed control tick, before automatic resets.

    Scan and guidance describe the inputs that informed the action; pose and
    velocity describe its result. Clearance uses the recorder's declared method.
    Arrays are borrowed for immediate consumption by the recorder.
    """

    xy: np.ndarray
    yaw: np.ndarray
    speed: np.ndarray
    omega: np.ndarray
    command: np.ndarray
    command_delta: np.ndarray
    action: np.ndarray
    ranges: np.ndarray
    scan_angles: np.ndarray
    clearance: np.ndarray
    lidar_clearance: np.ndarray
    rewards: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeInfo:
    """Completed episode snapshot. Outcome fields describe roles, not term names."""

    map: int
    pair: int
    success: bool
    collision: bool
    timeout: bool
    stuck: bool
    terminated: bool
    termination_terms: dict[str, bool]
    steps: int
    path_length: float
    travelled: float
    completion: float
    generator: str
    episode_id: str
    repeat: int
    initial_yaw: float
    initial_sensor_state: dict[str, Any]
    time_seconds: float
    min_footprint_clearance: float
    clearance_method: str
    command_variation: float
    stalled_seconds: float
    reward_components: dict[str, float]
    interventions: int = 0
    fallback_steps: int = 0
    feasibility: str | None = None
    trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON report fields; potentially large trace arrays are written to NPZ separately."""
        return {item.name: getattr(self, item.name) for item in fields(self) if item.name != "trace"}
