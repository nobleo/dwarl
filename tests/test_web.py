"""Dashboard boundaries: real HTTP, concurrent children, checkpoint pinning and exact replay."""

import http.client
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from dwarl.runs.store import complete_checkpoint, write_json
from dwarl.web import evidence
from dwarl.web.jobs import Jobs, command_arguments, pin_checkpoint
from dwarl.web.server import Dashboard
from dwarl.web.telemetry import Publisher


@pytest.fixture
def dashboard(tmp_path):
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<h1>Dashboard</h1>")
    with Dashboard(tmp_path, static=static) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield server
        finally:
            server.shutdown()
            thread.join()


def request(server, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=100)
    connection.request("POST" if body is not None else "GET", path,
                       json.dumps(body) if body is not None else None,
                       headers=headers or {})
    response = connection.getresponse()
    status, payload = response.status, response.read()
    connection.close()
    return status, payload


def test_local_api_requires_session_and_rejects_foreign_origins(dashboard):
    assert request(dashboard, "/")[0] == 200
    assert request(dashboard, "/api/state")[0] == 403
    assert request(dashboard, "/studio/training/")[0] == 403
    assert request(dashboard, "/studio/training/state",
                   headers={"Upgrade": "websocket", "Connection": "Upgrade"})[0] == 403
    headers = {"X-DWARL-Token": dashboard.token}
    status, payload = request(dashboard, "/api/state", headers=headers)
    assert status == 200 and json.loads(payload)["jobs"] == {"training": None, "evaluation": None}
    assert request(dashboard, "/api/state", headers={
        "Cookie": f"{dashboard.cookie_name}={dashboard.token}"})[0] == 200
    assert request(dashboard, "/api/state", headers={**headers, "Origin": "https://foreign.example"})[0] == 403
    assert request(dashboard, "/", headers={"Host": "foreign.example"})[0] == 403
    assert request(dashboard, "/%2e%2e/secret")[0] == 404
    assert request(dashboard, "/api/command", {"argv": ["setup"]}, headers)[0] == 400


def test_cli_preflight_and_robot_readiness(dashboard):
    headers = {"X-DWARL-Token": dashboard.token}
    status, payload = request(dashboard, "/api/readiness", {"robot": "diffbot"}, headers)
    assert status == 200 and json.loads(payload)["passed"], payload
    status, payload = request(dashboard, "/api/preview",
                              {"argv": ["train", "diffbot", "--preset", "smoke"]}, headers)
    assert status == 200, payload
    result = json.loads(payload)
    assert result["config"]["train"]["max_iterations"] == 2
    assert not list((dashboard.root / "runs").glob("*"))


def test_commands_are_allowlisted_and_stay_attached():
    for argv in (["web"], ["train", "--detached"], ["train", "--set", "train.detached=true"], "train"):
        with pytest.raises(ValueError):
            command_arguments(argv)
    command = command_arguments(["train", "diffbot"])
    assert command[:4] == [sys.executable, "-u", "-m", "dwarl"]
    assert "train.detached=false" in command


def test_training_and_evaluation_can_run_and_stop_independently(tmp_path, monkeypatch):
    # Real children: event file confirms the signal handler is installed before Stop.
    ready = tmp_path / "ready"
    script = ("import signal,time,pathlib; "
              "signal.signal(signal.SIGINT, lambda *_: exit(130)); "
              f"pathlib.Path({str(ready)!r}).touch(); "
              "print('ready',flush=True); time.sleep(60)")
    monkeypatch.setattr("dwarl.web.jobs.command_arguments", lambda _: [sys.executable, "-u", "-c", script])
    training, evaluation = Jobs(tmp_path), Jobs(tmp_path)
    try:
        training.start(["status"])
        evaluation.start(["status"])
        with pytest.raises(ValueError, match="already running"):
            training.start(["status"])
        # Wait for output using the child-created file rather than a fixed startup delay.
        import time
        deadline = time.monotonic() + 5
        while ("ready" not in training.snapshot()["output"] or "ready" not in evaluation.snapshot()["output"]):
            assert time.monotonic() < deadline
            threading.Event().wait(.01)
        evaluation.stop()
        assert evaluation.process.wait(timeout=5) == 130
        assert training.snapshot()["running"]
        training.stop()
        assert training.process.wait(timeout=5) == 130
        assert len(evidence.history(tmp_path)) == 2
    finally:
        training.close()
        evaluation.close()


def test_latest_is_pinned_and_partial_files_are_not_choices(tmp_path):
    run = tmp_path / "runs/example"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text("schema: dwarl/1\n")
    (run / "contract.yaml").write_text("contract\n")
    write_json(run / "provenance.json", {"schema": "dwarl_run/1", "created_ns": 1})
    write_json(run / "checkpoint_manifest.json", {"schema": "dwarl_checkpoints/1", "completed": [], "best": None})
    checkpoint = run / "model_1.pt"
    checkpoint.write_bytes(b"complete")
    complete_checkpoint(run, checkpoint, 1)
    (run / "model_999.pt").write_bytes(b"partial")
    argv = pin_checkpoint(tmp_path, ["test", "--run", str(run), "--checkpoint", "latest"])
    assert argv[-1] == "1"
    assert [cp["iteration"] for cp in evidence.details(tmp_path, "example")["checkpoints"]] == [1]
    checkpoint.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="changed"):
        pin_checkpoint(tmp_path, argv)


def test_telemetry_is_opt_in_rate_limited_and_nonfatal(tmp_path, monkeypatch):
    monkeypatch.delenv("DWARL_WEB_TELEMETRY", raising=False)
    assert Publisher.from_environment() is None
    publisher = Publisher(tmp_path / "live.json")
    publisher.publish(lambda: pytest.fail("disabled view must not build a snapshot"))
    write_json(tmp_path / "live.control.json", {"expires_at": time.time() + 3})
    publisher.next_control = 0
    publisher.publish(lambda: {"pose": [1, 2, 0]})
    publisher.publish(lambda: pytest.fail("rate limit must avoid snapshot work"))
    assert json.loads((tmp_path / "live.json").read_text())["pose"] == [1, 2, 0]
    write_json(tmp_path / "live.control.json", {"expires_at": 0})
    publisher.next_control = 0
    publisher.next_update = 0
    publisher.publish(lambda: pytest.fail("expired view must not build a snapshot"))
    Publisher(tmp_path / "missing/live.json").publish(lambda: {"pose": []})


def test_reports_require_session_and_do_not_escape_evaluation(dashboard):
    directory = dashboard.root / "runs/example/tests/check/report"
    directory.mkdir(parents=True)
    (directory / "index.html").write_text("<h1>Result</h1>")
    (directory / "outside.json").symlink_to(dashboard.root / "static/index.html")
    cookie = {"Cookie": f"{dashboard.cookie_name}={dashboard.token}"}
    prefix = "/reports/example/tests/check/report/"
    assert request(dashboard, prefix + "index.html")[0] == 403
    assert request(dashboard, prefix + "index.html", headers=cookie)[0] == 200
    assert request(dashboard, prefix + "outside.json", headers=cookie)[0] == 400
    assert request(dashboard, prefix + "%2e%2e/secret", headers=cookie)[0] == 400


def test_documentation_search_json_and_nested_navigation(dashboard):
    directory = dashboard.static / "docs/search"
    directory.mkdir(parents=True)
    index = directory / "search_index.json"
    index.write_text('{"docs": [{"title": "Train"}]}')
    (directory / "index.html").write_text("<h1>Search</h1>")
    status, payload = request(dashboard, "/docs/search/search_index.json")
    assert status == 200 and json.loads(payload)["docs"][0]["title"] == "Train"
    assert request(dashboard, "/docs/search")[0] == 301
    assert request(dashboard, "/docs/search/")[0] == 200


def test_native_metrics_read_real_event_files(dashboard):
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    run = dashboard.root / "runs/example"
    run.mkdir(parents=True)
    write_json(run / "provenance.json", {"schema": "dwarl_run/1"})
    writer = EventFileWriter(str(run / "checkpoints"))
    for step in range(3):
        writer.add_event(event_pb2.Event(step=step, wall_time=time.time(), summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag="Train/mean_reward", simple_value=float(step))])))
    writer.close()
    status, payload = request(dashboard, "/api/metrics?run=example", headers={"X-DWARL-Token": dashboard.token})
    assert status == 200, payload
    assert json.loads(payload)["series"] == [{"tag": "Train/mean_reward", "points": [[0, 0.], [1, 1.], [2, 2.]]}]


def test_studio_streams_live_world_without_host_rendering_and_expires(tmp_path):
    # Exercise the actual Studio server in a process with no display or GPU rendering.
    code = '''
import json, time, http.client, socket
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
def no_renderer(*args, **kwargs):
    raise AssertionError("host must not render")
mujoco.Renderer = no_renderer
from dwarl.simulation.world import ShardedWorld
from dwarl.web.telemetry import Publisher
root = Path(__import__("sys").argv[1])
world = ShardedWorld(2, 2, pool_size=4)
world.forward()
env = SimpleNamespace(world=world, episode_length_buf=[0, 0], control_dt=.1,
    path_command=SimpleNamespace(path=np.array([[[0.,0.],[1.,1.]], [[0.,0.],[-1.,1.]]]), path_len=[2, 2],
    command=np.array([[[.5,.5]], [[-.5,.5]]])))
publisher = Publisher(root / "live.json")
publisher.training(env, np.array([[0.,0.]]), [0.])
assert publisher.renderer is None and not (root / "live.jpg").exists()
(root / "live.control.json").write_text(json.dumps({"expires_at": time.time()+10}))
publisher.next_control = 0
publisher.training(env, np.array([[0.,0.]]), [0.])
deadline = time.monotonic() + 15
while True:
    publisher.next_frame = 0
    publisher.training(env, np.array([[0.,0.]]), [0.])
    state = json.loads((root / "live.render.json").read_text())
    assert not state.get("error"), state
    if state.get("port"):
        break
    assert time.monotonic() < deadline
    time.sleep(.02)
view = publisher.renderer
port = state["port"]
connection = http.client.HTTPConnection("127.0.0.1", port)
connection.request("GET", "/")
response = connection.getresponse()
assert response.status == 200
assert b"web_client" in response.read()
connection.close()
assert world.shards[0].data.time == 0
assert not (root / "live.jpg").exists()
# Stream the second shard, including a runtime obstacle change. No mj_step occurs
# in Studio: its displayed state must be exactly the source's state.
source = world.shards[1]
source.write_map(0, np.array([[0, 2., 1., 0., .3, .4]]), 1.)
source.set_robot_pose([0], np.array([[.25, .75]]), np.array([.4]))
source.data.time = 12.5
source.forward()
(root / "live.control.json").write_text(json.dumps({"expires_at": time.time()+10, "environment": 1}))
publisher.next_control = publisher.next_frame = 0
publisher.training(env, np.array([[0.,0.], [.25,.75]]), [0., .4])
deadline = time.monotonic() + 5
while view.viewer.data.time != 12.5:
    assert time.monotonic() < deadline, view.error
    time.sleep(.02)
assert view.viewer.model is not source.model
np.testing.assert_allclose(view.viewer.data.qpos, source.data.qpos)
np.testing.assert_allclose(view.viewer.model.geom_pos, source.model.geom_pos)
assert len(view.viewer.extra_geoms) >= 3
assert json.loads((root / "live.render.json").read_text())["shard_offset"] == 1
before = source.data.qpos.copy()
# Inspection/editing the display cannot modify the simulation's owned state.
view.viewer.data.qpos[0] += 1
np.testing.assert_array_equal(source.data.qpos, before)
(root / "live.control.json").write_text('{"expires_at":0}')
# No publisher.training calls: lease must expire during a long policy update too.
view.thread.join(timeout=5)
assert not view.thread.is_alive(), "Studio kept running after expiry"
with socket.socket() as sock:
    assert sock.connect_ex(("127.0.0.1", port)) != 0
publisher.next_control = 0
publisher.training(env, np.array([[0.,0.]]), [0.])
assert publisher.renderer is None
world.close()
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True,
                            env={**os.environ, "DISPLAY": "", "WAYLAND_DISPLAY": "", "MUJOCO_GL": "disable"},
                            timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
