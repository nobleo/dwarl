"""Tiled multi-robot MuJoCo world.

One ``MjModel``/``MjData`` contains ``num_envs`` square tiles laid out on a grid. Every tile owns a
pool of ``pool_size`` static geoms (in a per-tile body) and one diff-drive robot attached with the
prefix ``env{i}/``. Maps are swapped at runtime by rewriting the pool geoms' type/pos/size directly
in the compiled model (see :mod:`dwarl.simulation.maps`), so the model/data pointers never change and
the passive viewer keeps working. Tiles are enclosed by border walls, so robots never interact.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import mujoco
import numpy as np

MAP_GROUP = 1  # geom group of map obstacles (lidar only hits this group)
ROBOT_GROUP = 2


def default_robot_xml() -> Path:
    return Path(str(resources.files("dwarl") / "robots" / "diffbot" / "diffbot.xml"))


def resolve_robot_xml(robot_xml: str | Path | None) -> Path:
    """None -> builtin diffbot model, else a path to a (generated) robot MJCF."""
    return Path(robot_xml) if robot_xml else default_robot_xml()


@dataclass
class RobotParams:
    """Robot geometry read from the compiled model (built from a robot profile, :mod:`dwarl.robot`).

    Naming contract: free joint ``root`` on body ``base``; driven wheel bodies/joints ``left_wheel`` /
    ``right_wheel`` with velocity actuators ``left_motor`` / ``right_motor`` and tire geoms ``left_tire``;
    optional footprint geom ``chassis``; site ``lidar`` (rays are cast horizontally from its position).
    """

    wheel_radius: float
    wheel_separation: float
    footprint_radius: float  # from the ``chassis`` geom; the env prefers the profile's value
    lidar_offset: np.ndarray  # lidar site position in the base frame (x, y, z)
    spawn_height: float  # base origin height when the wheels touch the ground
    drive_type: str = "diff"

    @property
    def lidar_height(self) -> float:
        return float(self.lidar_offset[2] + self.spawn_height)


class TiledWorld:
    def __init__(
        self,
        num_envs: int,
        tile_size: float = 8.0,
        tile_gap: float = 1.0,
        pool_size: int = 300,
        max_obstacle_height: float = 2.0,
        robot_xml: str | Path | None = None,
        timestep: float = 0.005,
        fixed_map: tuple[np.ndarray, float] | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.tile_size = tile_size
        self.fixed_map = fixed_map
        self.pool_size = pool_size = len(fixed_map[0]) if fixed_map is not None else pool_size
        robot_xml = resolve_robot_xml(robot_xml)

        spec = mujoco.MjSpec()
        spec.modelname = "dwarl_world"
        spec.option.timestep = timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        # the per-tile obstacle bodies have static BVHs from compile time; geoms are moved at runtime,
        # so midphase must be disabled (broadphase + per-geom bounding spheres stay exact).
        if fixed_map is None:
            spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 720
        spec.visual.quality.shadowsize = 2048

        cols = math.ceil(math.sqrt(num_envs))
        rows = math.ceil(num_envs / cols)
        spacing = tile_size + tile_gap
        origins = []
        for i in range(num_envs):
            r, c = divmod(i, cols)
            origins.append(((c - (cols - 1) / 2) * spacing, ((rows - 1) / 2 - r) * spacing))
        self.env_origins = np.asarray(origins, dtype=np.float64)
        extent = max(cols, rows) * spacing

        tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                               rgb1=[0.82, 0.82, 0.82], rgb2=[0.74, 0.74, 0.74])  # fmt: skip
        mat = spec.add_material(name="grid", texrepeat=[1, 1], texuniform=True)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
        del tex
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[extent, extent, 0.1],
                                material="grid", group=0, contype=1, conaffinity=1)  # fmt: skip
        spec.worldbody.add_light(pos=[0, 0, 30], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                                 castshadow=0, diffuse=[0.7, 0.7, 0.7])

        half = tile_size / 2.0
        for i, (ox, oy) in enumerate(origins):
            tb = spec.worldbody.add_body(name=f"tile{i}", pos=[ox, oy, 0.0])
            for k in range(pool_size):
                if fixed_map is not None:
                    # Compile detailed, immutable maps at their actual positions. Keeping
                    # midphase/BVH enabled is essential for tens of thousands of 1 cm cells.
                    (kind, x, y, yaw, hx, hy), height = fixed_map[0][k], fixed_map[1]
                    cylinder = int(kind) == 1
                    tb.add_geom(
                        name=f"tile{i}/obs{k}",
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER if cylinder else mujoco.mjtGeom.mjGEOM_BOX,
                        pos=[x, y, height / 2], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)],
                        size=[hx, height / 2, 0] if cylinder else [hx, hy, height / 2],
                        group=MAP_GROUP, rgba=[0.45, 0.45, 0.5, 1.0],
                    )
                    continue
                # spread the initial geoms over the whole tile volume so that the compiled body
                # bounding volume covers every position a pool geom can take later on.
                corner = k % 2
                x = -half if corner == 0 else half
                z = 0.0 if corner == 0 else max_obstacle_height
                tb.add_geom(name=f"tile{i}/obs{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01],
                            pos=[x, x, z], group=MAP_GROUP, rgba=[0.45, 0.45, 0.5, 1.0])  # fmt: skip
            frame = spec.worldbody.add_frame(pos=[ox, oy, 0.0])
            robot = mujoco.MjSpec.from_file(str(robot_xml))
            spec.attach(robot, prefix=f"env{i}/", frame=frame)
            # robot subtree vs this tile's obstacles, evaluated in C every mj_step (cheap to poll)
            spec.add_sensor(name=f"env{i}/collision", type=mujoco.mjtSensor.mjSENS_CONTACT,
                            objtype=mujoco.mjtObj.mjOBJ_XBODY, objname=f"env{i}/base",
                            reftype=mujoco.mjtObj.mjOBJ_BODY, refname=f"tile{i}",
                            intprm=[1, 0, 1])  # data="found", reduce=none, num=1  # fmt: skip

        self.spec = spec
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index()

    # ------------------------------------------------------------------ indexing
    def _has(self, kind: str, name: str) -> bool:
        try:
            getattr(self.model, kind)(name)
            return True
        except KeyError:
            return False

    def _index(self) -> None:
        m = self.model
        n = self.num_envs
        self.drive_type = "diff"
        self.qpos_adr = np.array([m.joint(f"env{i}/root").qposadr[0] for i in range(n)])
        self.qvel_adr = np.array([m.joint(f"env{i}/root").dofadr[0] for i in range(n)])
        wheels, motors = ("left_wheel", "right_wheel"), ("left_motor", "right_motor")
        # drive wheel dofs / actuators: (N, 2) [left, right] - ctrl = wheel rad/s
        self.wheel_qvel_adr = np.array([[m.joint(f"env{i}/{j}").dofadr[0] for j in wheels] for i in range(n)])
        self.actuator_ids = np.array([[m.actuator(f"env{i}/{a}").id for a in motors] for i in range(n)])
        # every non-root robot dof (wheels, passive casters, suspension), zeroed on reset
        root_dofs = {int(a) + k for a in self.qvel_adr for k in range(6)}
        self.robot_dofs = []
        self.robot_qpos = []  # (qpos adr, default value) of non-root joints
        for i in range(n):
            base = m.body(f"env{i}/base").id
            dofs, qps = [], []
            for j in range(m.njnt):
                if m.body_rootid[m.jnt_bodyid[j]] == m.body_rootid[base] and int(m.jnt_dofadr[j]) not in root_dofs:
                    nd = {0: 6, 1: 3, 2: 1, 3: 1}[int(m.jnt_type[j])]
                    dofs += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + nd))
                    nq = {0: 7, 1: 4, 2: 1, 3: 1}[int(m.jnt_type[j])]
                    qps += [(int(m.jnt_qposadr[j]) + k, float(m.qpos0[int(m.jnt_qposadr[j]) + k])) for k in range(nq)]
            self.robot_dofs.append(np.asarray(dofs, dtype=np.int64))
            self.robot_qpos.append(qps)
        self.lidar_site = np.array([m.site(f"env{i}/lidar").id for i in range(n)])
        self.collision_adr = np.array([m.sensor(f"env{i}/collision").adr[0] for i in range(n)])
        self.base_body = np.array([m.body(f"env{i}/base").id for i in range(n)])
        self.pool_geoms = np.array([[m.geom(f"tile{i}/obs{k}").id for k in range(self.pool_size)] for i in range(n)])

        self.geom_tile = np.full(m.ngeom, -1, dtype=np.int64)
        for i in range(n):
            self.geom_tile[self.pool_geoms[i]] = i
        # robot geoms -> env index (every geom whose root body is a robot base)
        root_of_geom = m.body_rootid[m.geom_bodyid]
        base_to_env = {int(b): i for i, b in enumerate(self.base_body)}
        self.geom_env = np.array([base_to_env.get(int(r), -1) for r in root_of_geom], dtype=np.int64)

        # robot geometry from the model (in the base frame, at the default configuration)
        from dwarl.robot.model import inspect_model

        geometry = inspect_model(m, "env0/")
        self.robot = RobotParams(
            wheel_radius=geometry.wheel_radius,
            wheel_separation=geometry.wheel_separation,
            footprint_radius=geometry.footprint_radius,
            lidar_offset=geometry.lidar_offset,
            spawn_height=geometry.spawn_height,
        )

    # ------------------------------------------------------------------ state access
    def robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Tile-local xy (N, 2) and yaw (N,)."""
        qp = self.data.qpos
        xy = np.stack([qp[self.qpos_adr], qp[self.qpos_adr + 1]], -1) - self.env_origins
        w, x, y, z = (qp[self.qpos_adr + 3 + k] for k in range(4))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return xy, yaw

    def robot_upright(self) -> np.ndarray:
        """z component of the base z axis (1 = upright)."""
        qp = self.data.qpos
        w, x, y = (qp[self.qpos_adr + 3 + k] for k in range(3))
        return 1 - 2 * (x * x + y * y)

    def robot_velocity(self, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward speed and yaw rate (N,) each."""
        qv = self.data.qvel
        vx, vy = qv[self.qvel_adr], qv[self.qvel_adr + 1]
        return vx * np.cos(yaw) + vy * np.sin(yaw), qv[self.qvel_adr + 5]

    def set_robot_pose(self, env_ids: np.ndarray, xy: np.ndarray, yaw: np.ndarray) -> None:
        d = self.data
        for k, e in enumerate(env_ids):
            a, v = self.qpos_adr[e], self.qvel_adr[e]
            z = self.robot.spawn_height + 0.001
            d.qpos[a : a + 3] = [self.env_origins[e, 0] + xy[k, 0], self.env_origins[e, 1] + xy[k, 1], z]
            d.qpos[a + 3 : a + 7] = [np.cos(yaw[k] / 2), 0.0, 0.0, np.sin(yaw[k] / 2)]
            d.qvel[v : v + 6] = 0.0
            d.qvel[self.robot_dofs[e]] = 0.0
            for adr, val in self.robot_qpos[e]:
                d.qpos[adr] = val
            d.ctrl[self.actuator_ids[e]] = 0.0

    # ------------------------------------------------------------------ simulation
    def set_ctrl(self, wheels: np.ndarray) -> None:
        """Drive wheel speeds (N, 2) [rad/s]."""
        self.data.ctrl[self.actuator_ids.ravel()] = wheels.ravel()

    def step(self, nsub: int) -> np.ndarray:
        """Advance ``nsub`` physics steps; returns bool (N,) any robot/obstacle contact during them."""
        m, d, adr = self.model, self.data, self.collision_adr
        hits = np.zeros(self.num_envs)
        for _ in range(nsub):
            mujoco.mj_step(m, d)
            hits += d.sensordata[adr]
        return hits > 0

    def forward(self, env_ids: np.ndarray | None = None) -> None:
        mujoco.mj_forward(self.model, self.data)

    def close(self) -> None:
        pass

    def write_map(self, env: int, primitives: np.ndarray, height: float) -> None:
        if self.fixed_map is not None:
            if height != self.fixed_map[1] or not np.array_equal(primitives, self.fixed_map[0]):
                raise ValueError("cannot swap an immutable high-resolution map")
            return
        from dwarl.simulation.maps import write_tile_map

        write_tile_map(self, env, primitives, height)

    def cast_rays(self, yaw: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
        """Planar rays from each lidar site against map geoms. Returns (N, R) distances, -1 = no hit."""
        m, d = self.model, self.data
        # mj_step integrates qpos after computing site/geom transforms. Refresh
        # those transforms so the scan origin and yaw describe the same instant.
        mujoco.mj_kinematics(m, d)
        R = len(angles)
        group = np.zeros(6, dtype=np.uint8)
        group[MAP_GROUP] = 1
        geomid = np.zeros(R, dtype=np.int32)
        dist = np.zeros(R, dtype=np.float64)
        out = np.empty((self.num_envs, R), dtype=np.float64)
        a = angles[None] + yaw[:, None]
        vecs = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], -1)  # (N, R, 3)
        pnts = d.site_xpos[self.lidar_site]
        for e in range(self.num_envs):
            mujoco.mj_multiRay(m, d, pnts[e], vecs[e].ravel(), group, 1, -1, geomid, dist, None, R, max_range)
            dist[geomid < 0] = -1.0
            out[e] = dist
        return out

    def collided_envs(self) -> np.ndarray:
        """Bool (N,) - robot geoms in contact with a map obstacle in the current contact list."""
        out = np.zeros(self.num_envs, dtype=bool)
        ncon = self.data.ncon
        if ncon == 0:
            return out
        g = self.data.contact.geom[:ncon]
        e1, e2 = self.geom_env[g[:, 0]], self.geom_env[g[:, 1]]
        t1, t2 = self.geom_tile[g[:, 0]], self.geom_tile[g[:, 1]]
        hit = np.concatenate([e1[t2 >= 0], e2[t1 >= 0]])
        hit = hit[hit >= 0]
        out[hit] = True
        return out


def default_num_shards(num_envs: int) -> int:
    """Physical cores are the useful parallelism for MuJoCo; hyperthreads add little."""
    cores = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(num_envs, cores))


class ShardedWorld:
    """``num_envs`` robots split over ``num_shards`` independent :class:`TiledWorld` s, stepped in threads.

    Each shard is a small, separate MjModel/MjData, so physics, contacts and ray casts only see that
    shard's tiles (the single-world cost grows ~O(N^2) through lidar rays hitting every tile's pool).
    MuJoCo releases the GIL inside ``mj_step``/``mj_multiRay``, so shards run truly in parallel.
    Exposes the :class:`TiledWorld` state API used by the env (no ``model``/``data``: no viewer).
    """

    def __init__(self, num_envs: int, num_shards: int, **kw) -> None:
        self.num_envs = num_envs
        num_shards = max(1, min(num_shards, num_envs))
        sizes = [len(c) for c in np.array_split(np.arange(num_envs), num_shards)]
        self._pool = ThreadPoolExecutor(num_shards, thread_name_prefix="dwarl-shard")
        try:
            self.shards: list[TiledWorld] = list(self._pool.map(lambda n: TiledWorld(n, **kw), sizes))
        except BaseException:
            self._pool.shutdown(wait=True, cancel_futures=True)
            raise
        self.bounds = np.concatenate([[0], np.cumsum(sizes)])
        self._slices = [slice(self.bounds[i], self.bounds[i + 1]) for i in range(num_shards)]
        s0 = self.shards[0]
        self.tile_size, self.pool_size = s0.tile_size, s0.pool_size
        self.robot, self.drive_type = s0.robot, s0.drive_type
        self.env_origins = np.concatenate([s.env_origins for s in self.shards])
        self._shard_of = np.repeat(np.arange(num_shards), sizes)

    def _map(self, fn) -> list:
        """Run ``fn(shard, env_slice)`` on every shard in parallel."""
        return list(self._pool.map(fn, self.shards, self._slices))

    def _local(self, env: int) -> tuple[TiledWorld, int]:
        k = int(self._shard_of[env])
        return self.shards[k], int(env - self.bounds[k])

    # ---- simulation
    def set_ctrl(self, wheels: np.ndarray) -> None:
        for s, sl in zip(self.shards, self._slices, strict=True):
            s.set_ctrl(wheels[sl])

    def step(self, nsub: int) -> np.ndarray:
        return np.concatenate(self._map(lambda s, _: s.step(nsub)))

    def forward(self, env_ids: np.ndarray | None = None) -> None:
        shards = self.shards if env_ids is None else [self.shards[k] for k in np.unique(self._shard_of[env_ids])]
        list(self._pool.map(lambda s: s.forward(), shards))

    def cast_rays(self, yaw: np.ndarray, angles: np.ndarray, max_range: float) -> np.ndarray:
        return np.concatenate(self._map(lambda s, sl: s.cast_rays(yaw[sl], angles, max_range)))

    def write_map(self, env: int, primitives: np.ndarray, height: float) -> None:
        s, i = self._local(env)
        s.write_map(i, primitives, height)

    # ---- state access
    def robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        xy, yaw = zip(*(s.robot_pose() for s in self.shards), strict=True)
        return np.concatenate(xy), np.concatenate(yaw)

    def robot_upright(self) -> np.ndarray:
        return np.concatenate([s.robot_upright() for s in self.shards])

    def robot_velocity(self, yaw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v, w = zip(*(s.robot_velocity(yaw[sl]) for s, sl in zip(self.shards, self._slices, strict=True)), strict=True)
        return np.concatenate(v), np.concatenate(w)

    def set_robot_pose(self, env_ids: np.ndarray, xy: np.ndarray, yaw: np.ndarray) -> None:
        env_ids = np.asarray(env_ids)
        for k, s in enumerate(self.shards):
            sel = self._shard_of[env_ids] == k
            if sel.any():
                s.set_robot_pose(env_ids[sel] - self.bounds[k], xy[sel], yaw[sel])

    def collided_envs(self) -> np.ndarray:
        return np.concatenate([s.collided_envs() for s in self.shards])

    def close(self) -> None:
        self._pool.shutdown(wait=True)
