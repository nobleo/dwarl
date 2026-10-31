from copy import deepcopy

from dwarl.testing.scoring import score


class MujocoBackend:
    def __init__(self, contract, bundle, workload, directory, request):
        from dwarl.artifacts.exporter import OnnxPolicy
        from dwarl.training.environment.nav_env import NavVecEnv

        cfg = deepcopy(contract)
        cfg.env.num_envs, cfg.device = 1, "cpu"
        cfg.sim.gui, cfg.sim.realtime = request.show, request.show
        cfg.curriculum.enabled = False
        cfg.dataset.path, cfg.dataset.split = str(directory / "maps"), workload["split"]
        self.env = NavVecEnv(cfg, split=workload["split"], sequential=True)
        try:
            self.policy = OnnxPolicy(bundle / "policy.onnx")
            self.env.record = True
        except BaseException:
            self.env.close()
            raise

    def execute_case(self, case):
        import torch

        env = self.env
        self.partial = {"case_id": case["id"], "trajectory": [case["start"][:2]]}
        obs = env.reset_case(case["map"], case["pair"], case["start"][2], case["seed"])
        with torch.inference_mode():
            for _ in range(int(round(case["time_budget"] / env.control_dt)) + 1):
                obs, _, done, _ = env.step(self.policy(obs))
                if bool(done[0]):
                    episode = env.last_episode_info[0]
                    trajectory = [case["start"][:2], *episode.trace["xy"].tolist()]
                    return score(case, success=episode.success, collision=episode.collision,
                                 timeout=episode.timeout, travelled=episode.travelled, elapsed=episode.time_seconds,
                                 clearance=episode.min_footprint_clearance, trajectory=trajectory,
                                 safety_stops=0, recoveries=0)
                xy, _ = env.world.robot_pose()
                self.partial["trajectory"].append(xy[0].tolist())
                self.partial["travelled_distance"] = float(env.recorder.travelled[0])
                self.partial["completion_time"] = float(env.episode_length_buf[0]) * env.control_dt
        raise RuntimeError("environment exceeded the materialized episode budget")

    def close(self):
        self.env.close()
