"""Passive MuJoCo viewer with start/goal/path markers drawn into ``user_scn``."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from dwarl.simulation.world import TiledWorld

START_RGBA = np.array([0.1, 0.85, 0.2, 0.8], dtype=np.float32)
GOAL_RGBA = np.array([0.95, 0.1, 0.1, 0.8], dtype=np.float32)
PATH_RGBA = np.array([1.0, 0.6, 0.0, 0.9], dtype=np.float32)
LOOKAHEAD_RGBA = np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float32)
_EYE = np.eye(3).ravel()


class SimViewer:
    """Wraps ``mujoco.viewer.launch_passive``. Rendering is throttled to ``render_fps``; with
    ``realtime`` the caller's control loop is slowed to wall-clock time (x ``speed``)."""

    def __init__(
        self,
        world: TiledWorld,
        render_fps: float = 30.0,
        realtime: bool = False,
        speed: float = 1.0,
        max_path_segments: int = 60,
        marker_height: float = 0.03,
        launch: bool = True,
    ) -> None:
        self.world = world
        self.render_dt = 1.0 / render_fps
        self.realtime, self.speed = realtime, speed
        self.max_path_segments = max_path_segments
        self.z = marker_height
        self._last_render = 0.0
        self._last_step = time.perf_counter()
        self._markers: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._lookahead: np.ndarray | None = None
        if not launch:
            self.handle = None
            return
        self.handle = mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False,
                                                   show_right_ui=False)  # fmt: skip
        n = world.num_envs
        with self.handle.lock():
            cam = self.handle.cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.0, 0.0, 0.0]
            span = float(np.ptp(world.env_origins, axis=0).max()) + world.tile_size
            cam.distance = 1.1 * span
            cam.elevation = -70.0 if n > 1 else -60.0
            cam.azimuth = 90.0

    @property
    def running(self) -> bool:
        return self.handle is not None and self.handle.is_running()

    def set_episode(self, env: int, start: np.ndarray, goal: np.ndarray, path: np.ndarray) -> None:
        """Store tile-local start/goal/path for an env; drawn on the next sync."""
        o = self.world.env_origins[env]
        self._markers[env] = (start + o, goal + o, path + o)

    def set_lookahead(self, points_world: np.ndarray | None) -> None:
        self._lookahead = points_world

    def _rebuild_scene(self, scn: mujoco.MjvScene | None = None, start_index: int = 0) -> None:
        """Write markers into ``scn`` (default: the viewer's user scene; also usable for offscreen renders)."""
        scn = self.handle.user_scn if scn is None else scn
        i = start_index
        cap = scn.maxgeom
        for start, goal, path in self._markers.values():
            if i + 2 >= cap:
                break
            mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([0.12, 0.005, 0]),
                                np.array([start[0], start[1], self.z]), _EYE, START_RGBA)  # fmt: skip
            mujoco.mjv_initGeom(scn.geoms[i + 1], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.12, 0, 0]),
                                np.array([goal[0], goal[1], 0.12]), _EYE, GOAL_RGBA)  # fmt: skip
            i += 2
            stride = max(1, int(np.ceil((len(path) - 1) / self.max_path_segments)))
            pts = path[::stride]
            if not np.allclose(pts[-1], path[-1]):
                pts = np.vstack([pts, path[-1:]])
            for a, b in zip(pts[:-1], pts[1:], strict=False):
                if i >= cap:
                    break
                g = scn.geoms[i]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), _EYE, PATH_RGBA)
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.015,
                                     np.array([a[0], a[1], self.z]), np.array([b[0], b[1], self.z]))  # fmt: skip
                i += 1
        if self._lookahead is not None:
            for p in self._lookahead:
                if i >= cap:
                    break
                mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.05, 0, 0]),
                                    np.array([p[0], p[1], 0.2]), _EYE, LOOKAHEAD_RGBA)  # fmt: skip
                i += 1
        scn.ngeom = i

    def sync(self, control_dt: float) -> None:
        if self.realtime:
            target = control_dt / max(self.speed, 1e-6)
            elapsed = time.perf_counter() - self._last_step
            if elapsed < target:
                time.sleep(target - elapsed)
            self._last_step = time.perf_counter()
        now = time.perf_counter()
        if now - self._last_render < self.render_dt:
            return
        self._last_render = now
        if not self.running:
            return
        with self.handle.lock():
            self._rebuild_scene()
        self.handle.sync()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
