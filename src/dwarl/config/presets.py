"""Explicit starter recipes; normal defaults do not change.

Keys are train settings; an optional ``recipe`` mapping is merged into the training recipe.
"""

PRESETS = {
    "smoke": {"name": "smoke", "preset": "smoke", "num_envs": 2, "max_iterations": 2, "save_interval": 1,
              "validation_interval": 0, "data": {"train_maps": 1, "validation_maps": 1, "test_maps": 1,
                                                   "pairs_per_map": 3}},
    "navigation": {"name": "navigation", "preset": "navigation", "validation_interval": 200},
    # Path following plus avoidance of obstacles that are not on the map (dwarl.navigation.surprises):
    # random obstacles on the reference path of 60 % of the episodes, path adherence relaxed while they
    # block the path ahead, and the avoidance level as privileged critic input.
    "avoidance": {"name": "avoidance", "preset": "avoidance", "validation_interval": 200, "recipe": {
        "env": {"surprise_obstacles": {"probability": 0.6}},
        "observations": {"privileged": {"terms": {
            "avoidance": {"func": "dwarl.navigation.observations:avoidance"}}}},
    }},
}
