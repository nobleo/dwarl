"""Opt-in, bounded-rate browser telemetry. Never required for a workflow to succeed."""

import json
import os
import time
from pathlib import Path


class Publisher:
    def __init__(self, path, snapshots=True):
        self.path = Path(path)
        self.snapshots = snapshots
        self.next_update = 0.
        self.next_frame = 0.
        self.next_control = 0.
        self.last_step = 0.
        self.control = {}
        self.renderer = None
        self.render_failed = False

    def enabled(self):
        now = time.monotonic()
        if now >= self.next_control:
            self.next_control = now + .25
            try:
                self.control = json.loads(self.path.with_suffix(".control.json").read_text())
            except (OSError, ValueError):
                self.control = {}
        active = self.control.get("expires_at", 0) > time.time()
        if not active:
            self.close()
            self.render_failed = False
        return active

    @classmethod
    def from_environment(cls, snapshots=True):
        path = os.environ.get("DWARL_WEB_TELEMETRY")
        return cls(path, snapshots) if path else None

    def publish(self, build):
        if not self.enabled():
            return
        now = time.monotonic()
        if now < self.next_update:
            return
        self.next_update = now + .25
        try:
            data = {**build(), "updated_at": time.time()}
            # Ephemeral view only: atomic replacement without fsync on the training hot path.
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, allow_nan=False))
            temporary.replace(self.path)
        except (OSError, ValueError, TypeError):
            pass

    def pace(self, step_s):
        """Hold a watched step to wall-clock time when the viewer asks for real-time playback."""
        now = time.perf_counter()
        if self.control.get("realtime") is True:
            delay = self.last_step + step_s - now
            if delay > 0:
                time.sleep(delay)
                now += delay
        self.last_step = now

    def training(self, env, xy, yaw):
        def snapshot():
            path = env.path_command.path[0, :env.path_command.path_len[0]]
            return {"kind": "training", "label": "Environment 0", "pose": [*xy[0].tolist(), float(yaw[0])],
                    "path": path.tolist(), "time_s": float(env.episode_length_buf[0]) * env.control_dt}
        if self.snapshots:
            self.publish(snapshot)
        if not self.enabled():
            return
        self.pace(env.control_dt)
        if self.render_failed or time.monotonic() < self.next_frame:
            return
        self.next_frame = time.monotonic() + 1 / 20
        status = self.path.with_suffix(".render.json")
        try:
            if self.renderer is not None and self.renderer.expired:
                self.close()
            if self.renderer is None:
                from dwarl.web.render import StudioView

                self.renderer = StudioView(self.path.with_suffix(".control.json"))
            state = self.renderer.sync(env, self.control)
            temporary = status.with_suffix(".tmp")
            temporary.write_text(json.dumps({"updated_at": time.time(), "backend": "studio", **state}))
            temporary.replace(status)
        except Exception as exc:
            self.close()
            self.render_failed = True
            try:
                status.write_text(json.dumps({"error": f"MuJoCo rendering unavailable: {exc}"}))
            except OSError:
                pass

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
            try:
                status = self.path.with_suffix(".render.json")
                previous = json.loads(status.read_text()) if status.exists() else {}
                status.write_text(json.dumps({**previous, "port": None}))
            except (OSError, ValueError):
                pass
