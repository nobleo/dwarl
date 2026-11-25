"""Outcome-gated obstacle stages, independent of route-difficulty progression."""

from copy import deepcopy

import numpy as np


class AvoidanceCurriculum:
    """Stage zero is clean navigation; stages 1..5 ramp to the configured obstacle mix.

    Each decision needs 200 completed episodes per required group. Actual placements,
    rather than attempted placements, identify obstacle episodes. Windows reset after
    every decision; episodes started at an older stage cannot influence the new stage.
    """

    def __init__(self):
        self.stage = 0
        self.windows = [[], []]

    def settings(self, base):
        fraction = self.stage / 5
        return {
            **base,
            "probability": base["probability"] * fraction,
            "count": [max(1, int(np.ceil(value * fraction))) for value in base["count"]],
            "size": [base["size"][0], base["size"][0] +
                     (base["size"][1] - base["size"][0]) * fraction],
            "margin": base["margin"] + 0.2 * (1 - fraction),
        }

    def report(self, stages, obstacles, successes, collisions):
        for stage, obstacle, success, collision in zip(stages, obstacles, successes, collisions, strict=True):
            if stage == self.stage:
                window = self.windows[int(obstacle)]
                window.append((bool(success), bool(collision)))
                del window[:-200]
        groups = self.windows[:1] if self.stage == 0 else self.windows
        if not all(len(group) >= 200 for group in groups):
            return
        rates = np.asarray([np.mean(group, axis=0) for group in groups])
        if np.all(rates[:, 0] >= .75) and np.all(rates[:, 1] <= .05):
            self.stage = min(5, self.stage + 1)
        elif np.any(rates[:, 0] <= .3) or np.any(rates[:, 1] > .1):
            self.stage = max(0, self.stage - 1)
        self.windows = [[], []]

    def state(self):
        return {"stage": self.stage, "windows": deepcopy(self.windows)}

    def restore(self, state):
        self.stage = int(state["stage"])
        self.windows = deepcopy(state["windows"])
