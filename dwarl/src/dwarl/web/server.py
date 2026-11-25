"""Loopback dashboard serving workflows, metrics, MuJoCo Studio and shared documentation."""

import json
import mimetypes
import secrets
import subprocess
import sys
import webbrowser
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from dwarl.web import evidence
from dwarl.web.jobs import Jobs, inspect_command, pin_checkpoint, tail
from dwarl.web.metrics import Metrics

STATIC = Path(__file__).parent / "static"


class Dashboard(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, root, port=0, static=STATIC):
        super().__init__(("127.0.0.1", port), Handler)
        self.root, self.static = Path(root).resolve(), Path(static).resolve()
        self.token = secrets.token_urlsafe(32)
        self.jobs = Jobs.reconnect(self.root)
        self.metrics = Metrics()

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_port}"

    @property
    def cookie_name(self):
        return f"dwarl-{self.server_port}"

    def server_close(self):
        try:
            for jobs in self.jobs.values():
                jobs.close()
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def respond(self, status, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        if self.headers.get("X-DWARL-Token") == self.server.token:
            self.send_header("Set-Cookie",
                             f"{self.server.cookie_name}={self.server.token}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        # Reject foreign Host/Origin headers (including DNS rebinding).
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.respond(403, {"error": "use the dashboard's printed localhost URL"})
            return False
        origin = self.headers.get("Origin")
        if origin and origin != self.server.origin:
            self.respond(403, {"error": "foreign origin"})
            return False
        cookie = SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        stored = cookie.get(self.server.cookie_name)
        cookie_token = stored.value if stored else None
        if self.path.startswith("/api/") and (
                self.headers.get("X-DWARL-Token") or cookie_token) != self.server.token:
            self.respond(403, {"error": "invalid dashboard token; reopen the printed URL"})
            return False
        if self.path.startswith(("/reports/", "/studio/")):
            if cookie_token != self.server.token:
                self.respond(403, {"error": "open the dashboard first"})
                return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        url = urlsplit(self.path)
        try:
            if url.path == "/api/state":
                jobs = {name: manager.snapshot() for name, manager in self.server.jobs.items()}
                self.respond(200, {"api_version": 3, "runs": evidence.runs(self.server.root), "jobs": jobs,
                                  "docs": (self.server.static / "docs/index.html").is_file(),
                                  "context": evidence.read(self.server.root / ".dwarl/context.json", {}),
                                  "history": evidence.history(self.server.root),
                                  "render": {name: evidence.read(Path(job["telemetry"]).with_suffix(".render.json"))
                                             if job else None for name, job in jobs.items()},
                                  "live": {name: evidence.read(job["telemetry"]) if job else None
                                           for name, job in jobs.items()}})
            elif url.path == "/api/run":
                self.respond(200, evidence.details(self.server.root, parse_qs(url.query).get("name", [""])[0]))
            elif url.path == "/api/history":
                name = parse_qs(url.query).get("id", [""])[0]
                item = next((item for item in evidence.history(self.server.root) if item["id"] == name), None)
                if item is None:
                    raise ValueError("unknown command")
                self.respond(200, {**item, "output": tail(self.server.root / ".dwarl/web" / name / "command.log")})
            elif url.path == "/api/metrics":
                name = parse_qs(url.query).get("run", [""])[0]
                run = evidence.run_directory(self.server.root, name)
                self.respond(200, self.server.metrics.read(run))
            elif url.path.startswith("/studio/"):
                from dwarl.web.studio_proxy import proxy

                proxy(self)
            elif url.path.startswith("/reports/"):
                relative = Path(unquote(url.path.removeprefix("/reports/")))
                parts = relative.parts
                if len(parts) < 4 or parts[1] != "tests" or ".." in parts:
                    raise ValueError("invalid report path")
                directory = (self.server.root / "runs" / parts[0] / "tests" / parts[2]).resolve()
                path = (self.server.root / "runs" / relative).resolve()
                if (not directory.is_relative_to(self.server.root / "runs")
                        or not path.is_relative_to(directory) or not path.is_file()
                        or path.suffix not in (".html", ".png", ".svg", ".json", ".md")):
                    raise ValueError("unknown report artifact")
                self.respond(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "text/plain")
            else:
                path = (self.server.static / unquote(url.path).lstrip("/")).resolve()
                if url.path == "/":
                    path = self.server.static / "index.html"
                if path.is_dir() and path.is_relative_to(self.server.static):
                    if not url.path.endswith("/"):
                        self.send_response(301)
                        self.send_header("Location", url.path + "/")
                        self.end_headers()
                        return
                    path = path / "index.html"
                if not path.is_relative_to(self.server.static) or not path.is_file():
                    self.respond(404, {"error": "not found"})
                    return
                self.respond(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        except (ValueError, OSError) as exc:
            self.respond(400, {"error": str(exc)})

    def do_POST(self):
        if not self.authorized():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("invalid request size")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            if self.path == "/api/command":
                argv = data.get("argv")
                lane = "training" if isinstance(argv, list) and argv[:1] == ["train"] else "evaluation"
                self.respond(200, self.server.jobs[lane].start(argv, telemetry=data.get("telemetry") is True))
            elif self.path == "/api/stop":
                if data.get("lane") not in self.server.jobs:
                    raise ValueError("choose training or evaluation")
                self.server.jobs[data["lane"]].stop()
                self.respond(200, {"status": "interrupt_requested"})
            elif self.path == "/api/view":
                if data.get("lane") not in self.server.jobs:
                    raise ValueError("choose training or evaluation")
                self.server.jobs[data["lane"]].view(data.get("enabled") is True, data.get("camera"),
                                                    realtime=data.get("realtime") is True)
                self.respond(200, {"enabled": data.get("enabled") is True})
            elif self.path == "/api/preview":
                argv = data.get("argv")
                # Validate the allowlist before inspecting/pinning.
                from dwarl.web.jobs import command_arguments

                command_arguments(argv)
                if argv[0] not in ("train", "test", "deploy"):
                    raise ValueError("preview supports train, test and deploy")
                pinned = pin_checkpoint(self.server.root, argv)
                preview = inspect_command(self.server.root, pinned, "--dry-run")
                self.respond(200, {"argv": pinned, **preview})
            elif self.path == "/api/readiness":
                reference = data.get("robot") or evidence.read(
                    self.server.root / ".dwarl/context.json", {}).get("robot")
                if not isinstance(reference, str) or not reference:
                    raise ValueError("No robot selected. Run pixi run setup or enter an integration path.")
                result = subprocess.run([sys.executable, "-m", "dwarl", "robot", "check", reference, "--json"],
                                        cwd=self.server.root, stdin=subprocess.DEVNULL, capture_output=True,
                                        text=True, timeout=90)
                self.respond(200, json.loads(result.stdout))
            elif self.path == "/api/case":
                _, row, trace = evidence.case_details(self.server.root, data["run"], data["evaluation"],
                                                       data["case"], data["backend"])
                self.respond(200, {"episode": row, "trace": trace})
            elif self.path == "/api/retry":
                argv = evidence.retry_arguments(self.server.root, data)
                self.respond(200, self.server.jobs["evaluation"].start(argv, telemetry=True))
            else:
                self.respond(404, {"error": "unknown operation"})
        except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as exc:
            self.respond(400, {"error": str(exc)})

def serve(port=8765, open_browser=True):
    from dwarl.cli import ui

    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")
    if not (STATIC / "index.html").is_file():
        raise ValueError("Dashboard assets missing. In the source checkout run: pixi run web-build")
    with Dashboard(Path.cwd(), port) as server:
        url = f"{server.origin}/#token={server.token}"
        ui.info(f"Dashboard: {url}")
        ui.info("Ctrl+C stops the dashboard and its commands. Closing the browser suspends rendering.")
        if open_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
