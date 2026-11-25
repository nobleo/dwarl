"""Isolated browser-test workspace; never launches a real training or evaluation workload."""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from dwarl.runs.store import write_json
from dwarl.web.server import Dashboard

TOKEN = "dwarl-browser-test"


def run():
    with tempfile.TemporaryDirectory(prefix="dwarl-web-test-") as temporary:
        root = Path(temporary)
        training = root / "runs/example"
        report = training / "tests/check/report"
        report.mkdir(parents=True)
        write_json(training / "provenance.json", {"schema": "dwarl_run/1", "robot": "diffbot"})
        write_json(training / "status.json", {"state": "finished"})
        write_json(training / "checkpoint_manifest.json", {"completed": [], "best": None})
        write_json(report.parent / "summary.json", {"status": "completed", "passed": True, "suite": "validation",
                   "iteration": 1, "checkpoint": "example-hash", "simulators": {}, "workload": "example"})
        (report / "index.html").write_text(
            '<main><h1>Example results</h1><img src="route.png" alt="Route plot"></main>')
        Image.new("RGB", (100, 100), "purple").save(report / "route.png")
        writer = EventFileWriter(str(training / "checkpoints"))
        for step in range(3):
            writer.add_event(event_pb2.Event(step=step, wall_time=time.time(), summary=summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag="Train/mean_reward", simple_value=float(step))])))
        writer.close()
        with Dashboard(root, port=18765) as server:
            server.token = TOKEN
            manager = server.jobs["training"]
            job_dir = root / ".dwarl/web/scene"
            job_dir.mkdir(parents=True)
            live = job_dir / "live.json"
            script = """
import time, numpy as np
from types import SimpleNamespace
from dwarl.simulation.world import TiledWorld
from dwarl.web.telemetry import Publisher
world = TiledWorld(1, pool_size=4)
world.forward()
env = SimpleNamespace(world=world, episode_length_buf=[0], control_dt=.1,
    path_command=SimpleNamespace(path=np.array([[[0.,0.],[1.,1.]]]), path_len=[2],
    command=np.array([[[.5,.5]]])))
publisher = Publisher.from_environment()
started = time.monotonic()
try:
    while True:
        t = time.monotonic() - started
        xy = np.array([[.6 * np.sin(t), .3 * np.cos(t)]])
        world.set_robot_pose([0], xy, np.array([t]))
        world.data.time = t
        world.forward()
        publisher.training(env, xy, [t])
        time.sleep(.03)
except KeyboardInterrupt:
    pass
finally:
    publisher.close()
"""
            with (job_dir / "command.log").open("w") as log:
                manager.process = subprocess.Popen([sys.executable, "-c", script],
                    env={**os.environ, "MUJOCO_GL": "disable", "DISPLAY": "", "WAYLAND_DISPLAY": "",
                         "DWARL_WEB_TELEMETRY": str(live)},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
            manager.job = {"argv": ["train", "diffbot"], "id": "scene", "started_at": time.time(),
                           "log": str(job_dir / "command.log"), "telemetry": str(live), "viewer_protocol": 3}
            manager.view(False)
            print(json.dumps({"ready": True}), flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    run()
