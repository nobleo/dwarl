"""Defaults and validation of navigation options, without numpy or simulation imports.

Composing a configuration validates these options, and composing must work in slim environments
(the documentation build has no numpy), so the modules that use them re-export them from here.
"""

from __future__ import annotations

SURPRISE_OBSTACLES = {
    "probability": 0.0,          # share of training episodes with obstacles on the path (0 disables)
    "count": [1, 3],             # obstacles per episode, inclusive range
    "size": [0.1, 0.4],          # radius or larger half extent [m]
    "lateral": 1.0,              # centre offset from the path, as a fraction of (size + robot half width)
    "keepout": 1.0,              # free distance around start and goal [m]
    "spacing": 1.2,              # minimum distance between obstacle centres [m]
    "margin": 0.1,               # extra half width a detour must leave for the robot [m]
    "tries": 12,                 # placement candidates per obstacle
    "horizon_time": 5.0,         # look this far ahead along the path, at full speed [s] ...
    "horizon": [1.5, 3.0],       # ... clipped to this distance range [m]
    "near": 0.6,                 # full avoidance level for blockages within this fraction of the horizon
    "clear": 0.35,               # path is blocked when it passes closer than this to an obstacle [m] ...
    "ramp": 0.25,                # ... with the level rising linearly over this distance
    "release_time": 1.5,         # decay time constant once the path is clear [s]
}


def surprise_obstacles(cfg) -> dict:
    """Merge configured values over the defaults; ``cfg`` may be None or a partial mapping."""
    values = dict(SURPRISE_OBSTACLES)
    if cfg:
        unknown = set(cfg) - set(SURPRISE_OBSTACLES)
        if unknown:
            raise ValueError(f"unknown surprise_obstacles settings: {', '.join(sorted(unknown))}")
        values.update({k: list(v) if isinstance(SURPRISE_OBSTACLES[k], list) else v
                       for k, v in dict(cfg).items()})
    if not 0.0 <= float(values["probability"]) <= 1.0:
        raise ValueError("surprise_obstacles.probability must be between 0 and 1")
    lo, hi = (int(v) for v in values["count"])
    if lo < 1 or hi < lo:
        raise ValueError("surprise_obstacles.count must be [min, max] with 1 <= min <= max")
    if float(values["size"][0]) <= 0 or float(values["size"][1]) < float(values["size"][0]):
        raise ValueError("surprise_obstacles.size must be [min, max] with 0 < min <= max")
    return values
