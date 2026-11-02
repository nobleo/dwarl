from copy import deepcopy

from dwarl.testing.scoring import score
from dwarl.testing.trace import BehaviorTrace


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
        trace = BehaviorTrace(case, pose_source="MuJoCo base pose in canonical map coordinates",
                              velocity_source="MuJoCo base linear-forward/angular-z velocity",
                              command_source="rate-limited policy velocity applied to the drive; no safety filter",
                              clearance_source="2D footprint geometry; physical contact clamps clearance to zero")
        self.partial = {"case_id": case["id"], "trajectory": [case["start"][:2]], "_trace": trace.data,
                        "safety_stops": 0, "recovery_actions": 0}
        obs = env.reset_case(case["map"], case["pair"], case["start"][2], case["seed"])
        xy, yaw = env.world.robot_pose()
        speed, omega = env.world.robot_velocity(yaw)
        trace.sample(0., [*xy[0].tolist(), float(yaw[0])], velocity=(float(speed[0]), float(omega[0])),
                     clearance=float(env._episode_clearance(xy, yaw)[0]))
        with torch.inference_mode():
            for _ in range(int(round(case["time_budget"] / env.control_dt)) + 1):
                obs, _, done, _ = env.step(self.policy(obs))
                if bool(done[0]):
                    episode = env.last_episode_info[0]
                    trajectory = [case["start"][:2], *episode.trace["xy"].tolist()]
                    raw = episode.trace
                    trace.sample(episode.time_seconds, [*raw["xy"][-1].tolist(), float(raw["yaw"][-1])],
                                 command=tuple(raw["cmd"][-1]), velocity=(raw["v"][-1], raw["w"][-1]),
                                 clearance=raw["clearance"][-1])
                    row = score(case, success=episode.success, collision=episode.collision,
                                 timeout=episode.timeout, travelled=episode.travelled, elapsed=episode.time_seconds,
                                 clearance=episode.min_footprint_clearance, trajectory=trajectory,
                                  safety_stops=0, recoveries=0, stuck=episode.stuck,
                                  termination_reason=",".join(k for k, v in episode.termination_terms.items() if v))
                    row["_trace"] = trace.data
                    return row
                raw = env.recorder.traces[0][-1]
                elapsed = float(env.episode_length_buf[0]) * env.control_dt
                trace.sample(elapsed, [*raw["xy"].tolist(), raw["yaw"]], command=tuple(raw["cmd"]),
                             velocity=(raw["v"], raw["w"]), clearance=raw["clearance"])
                xy, _ = env.world.robot_pose()
                self.partial["trajectory"].append(xy[0].tolist())
                self.partial["travelled_distance"] = float(env.recorder.travelled[0])
                self.partial["completion_time"] = float(env.episode_length_buf[0]) * env.control_dt
                self.partial["clearance"] = float(env.recorder.min_clearance[0])
        raise RuntimeError("environment exceeded the materialized episode budget")

    def close(self):
        self.env.close()
