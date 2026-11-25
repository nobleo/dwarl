"""MuJoCo Studio bridge: immutable live snapshots, browser-side graphics, no physics stepping."""

import contextlib
import copy
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
from mujoco.studio import launch_thread, messages, plugin, ux, viewer_protocol, web_viewer
from mujoco.studio.dear_imgui import dear_imgui as imgui
from mujoco.studio.web import headless_ui, web_server

from dwarl.simulation.viewer import SimViewer

# MuJoCo 4.0's WebViewer binds to :: internally; keep its transport private behind
# the dashboard's authenticated HTTP/WebSocket proxy. Serialize constructor patches.
_BIND_LOCK = threading.Lock()
_GEOMETRY = ("geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_type")
# Studio logs every browser connection at INFO; the run's output keeps its warnings and errors.
logging.getLogger("WebServer").setLevel(logging.WARNING)
# MuJoCo 4.0.1 destroys a closed viewer's Dear ImGui context from whichever thread garbage-collects
# it. ImGui's current context is process-global, so this crashed the next viewer (and the run)
# after a few viewer toggles. google-deepmind/mujoco@3538515 adds HeadlessUi.close(), called on
# the viewer thread; until a release ships it, reuse one context for every viewing session.
_SHARE_UI = not hasattr(headless_ui.HeadlessUi, "close")
_HeadlessUi = headless_ui.HeadlessUi
_UI = None


def _shared_ui(title, assets_dir=""):
    global _UI
    if _UI is None:
        _UI = _HeadlessUi(title, assets_dir)
    return _UI


@dataclass(frozen=True)
class LiveSnapshot(messages.Snapshot):
    model: object
    state: np.ndarray
    paths: list
    lookahead: np.ndarray
    origins: np.ndarray
    focus: tuple
    tile_size: float


class LiveScene(plugin.ViewerPlugin):
    """Studio owns its display copy; UI inspection never edits the training world."""

    def __init__(self):
        self.source = None
        self.focus = None
        self.scene = None

    @messages.handler
    def live(self, snapshot: LiveSnapshot):
        viewer = self.viewer
        if snapshot.model is not self.source:
            camera = copy.copy(viewer.camera)
            viewer.dispatch(messages.ModelEvent(model=snapshot.model))
            viewer.camera = camera
            self.source = snapshot.model
            self.scene = mujoco.MjvScene(viewer.model, maxgeom=1024)
        viewer.dispatch(messages.StateSnapshot(
            state=snapshot.state, state_sig=int(mujoco.mjtState.mjSTATE_INTEGRATION)))
        markers = SimViewer(SimpleNamespace(env_origins=snapshot.origins), launch=False)
        for index, path in snapshot.paths:
            if len(path):
                markers.set_episode(index, path[0], path[-1], path)
        markers.set_lookahead(snapshot.lookahead)
        markers._rebuild_scene(self.scene)
        viewer.extra_geoms = [self.scene.geoms[i] for i in range(self.scene.ngeom)]
        if self.focus != snapshot.focus:
            index, overview, _ = snapshot.focus
            span = float(np.ptp(snapshot.origins, axis=0).max()) + snapshot.tile_size
            viewer.camera.lookat[:] = [0., 0., 0.] if overview else [*snapshot.origins[index], 0.]
            viewer.camera.distance = 1.1 * (span if overview else snapshot.tile_size)
            viewer.camera.elevation = -60
            viewer.camera.azimuth = 90
            self.focus = snapshot.focus


class Viewport(plugin.ViewerPlugin):
    """Dark scene with only an orientation gizmo: no Studio menus, inspector or physics controls."""

    def __init__(self):
        self.ux_state = ux.UxState()

    @messages.handler(priority=messages.Priority.CRITICAL)
    def theme(self, _: plugin.ViewerInitEvent):
        ux.setup_theme(ux.GuiTheme.DARK)

    @messages.handler
    def gizmo(self, _: messages.BuildGuiEvent):
        size = imgui.GetIO().DisplaySize
        viewer = self.viewer
        ux.view_gizmo_gui(viewer.model, viewer.data, viewer.camera, self.ux_state, [0., 0., size.x, size.y])
        viewer.camera_index = self.ux_state.camera_index


class LeasedViewer(web_viewer.WebViewer):
    """Expire even while the simulation thread is busy with a long PPO update."""

    def __init__(self, *args, control_path, **kwargs):
        self.control_path = control_path
        self.next_check = 0.
        super().__init__(*args, **kwargs)

    def prepare_next_frame(self):
        if time.monotonic() >= self.next_check:
            self.next_check = time.monotonic() + .25
            try:
                active = json.loads(self.control_path.read_text()).get("expires_at", 0) > time.time()
            except (OSError, ValueError):
                active = False
            if not active:
                self.dispatch(messages.ExitEvent())
                return False
        return super().prepare_next_frame()


class StudioView:
    def __init__(self, control_path):
        self.session = uuid.uuid4().hex
        self.port = None
        self.viewer = None
        self.error = None
        self.model = None
        self.world = None
        self.endpoint, self.sender = launch_thread.make_thread_endpoints()
        self.control_path = control_path
        self.thread = threading.Thread(target=self._run, name="dwarl-studio", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            bind = web_server.bind_public_socket
            shared = patch.object(headless_ui, "HeadlessUi", _shared_ui) if _SHARE_UI else contextlib.nullcontext()
            # No GPU/display context is created by the web backend.
            with (_BIND_LOCK, shared,
                  patch.object(web_server, "bind_public_socket", lambda _host, port: bind("127.0.0.1", port))):
                viewer = LeasedViewer(
                    viewer_protocol.ViewerConfig(gfx="web", title="DWARL · Live simulation"),
                    self.endpoint, plugins=[Viewport(), LiveScene()], control_path=self.control_path)
            self.port = viewer._http_port
            self.viewer = viewer
            viewer_protocol.run_viewer_loop(viewer)
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.port = None

    def sync(self, env, settings):
        if self.error:
            raise RuntimeError(self.error)
        if not self.thread.is_alive():
            raise RuntimeError("Studio session closed; toggle the viewer off and on to reconnect")
        index = max(0, min(env.world.num_envs - 1, int(settings.get("environment", 0))))
        world, local, offset = env.world, index, 0
        if hasattr(world, "shards"):
            shard = int(world._shard_of[index])
            offset = int(world.bounds[shard])
            world, local = world.shards[shard], index - offset
        if world is not self.world or any(
                not np.array_equal(getattr(world.model, name), getattr(self.model, name)) for name in _GEOMETRY):
            self.model = copy.copy(world.model)
            self.world = world
            floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
            if floor >= 0:
                # Studio's Filament renderer reads unmarked builtin colours as linear; marking the
                # display copy sRGB shows the same dark floor as the classic viewer.
                self.model.tex_colorspace[floor] = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
        overview = settings.get("mode") == "overview"
        indices = range(world.num_envs) if overview else [local]
        paths = [(i, env.path_command.path[offset + i, :env.path_command.path_len[offset + i]].copy())
                 for i in indices]
        points = env.path_command.command[offset:offset + world.num_envs] + world.env_origins[:, None]
        signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(world.model, signature))
        mujoco.mj_getState(world.model, world.data, state, signature)
        # Drain UI requests: DWARL owns policy execution, resets, and physics.
        self.sender.get_viewer_events()
        self.sender.get_viewer_snapshots()
        self.sender.send_to_viewer(LiveSnapshot(
            self.model, state, paths, (points.reshape(-1, 2) if overview else points[local]).copy(),
            world.env_origins.copy(), (local, overview, offset), world.tile_size))
        return {"port": self.port, "session": self.session, "environments": env.world.num_envs, "shard_offset": offset,
                "shard_environments": world.num_envs}

    @property
    def expired(self):
        return not self.thread.is_alive() and self.error is None

    def close(self):
        if self.thread.is_alive():
            self.sender.send_to_viewer(messages.ExitEvent())
            self.thread.join(timeout=5)
        self.sender.close()
