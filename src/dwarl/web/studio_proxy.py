"""Authenticated, same-origin HTTP and WebSocket transport for MuJoCo Studio."""

import http.client
import select
import socket
import time
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

from dwarl.web import evidence

# Studio's WASM client uses origin-root endpoints. Scope those to this iframe's
# lane while leaving the upstream assets intact (including worker-relative URLs).
_SCOPE = b"""<script>
(() => {
  const prefix = location.pathname.replace(/[^/]*$/, '').replace(/\\/$/, '');
  const scoped = value => {
    const url = new URL(value, location.href);
    if (url.host === location.host && !url.pathname.startsWith(prefix + '/'))
      url.pathname = prefix + url.pathname;
    return url.href;
  };
  const fetchOriginal = window.fetch;
  window.fetch = (input, options) => fetchOriginal(
    input instanceof Request ? new Request(scoped(input.url), input) : scoped(input), options);
  const Socket = window.WebSocket;
  window.WebSocket = class extends Socket {
    constructor(url, protocols) { super(scoped(url), ...(protocols === undefined ? [] : [protocols])); }
  };
})();
</script>"""


def proxy(handler):
    url = urlsplit(handler.path)
    _, _, lane, relative = url.path.split("/", 3)
    manager = handler.server.jobs.get(lane)
    job = manager.snapshot() if manager else None
    if not job or not job["running"] or job.get("viewer_protocol") != 3:
        raise ValueError("no active Studio job")
    live = Path(job["telemetry"])
    control = evidence.read(live.with_suffix(".control.json"), {})
    state = evidence.read(live.with_suffix(".render.json"), {})
    port = state.get("port")
    if (control.get("expires_at", 0) <= time.time() or state.get("backend") != "studio"
            or not isinstance(port, int) or not 0 < port < 65536):
        raise ValueError("Studio is starting or viewing is disabled")
    target = "/" + relative + ("?" + url.query if url.query else "")
    if handler.headers.get("Upgrade", "").lower() == "websocket":
        relay_websocket(handler, port, target)
        return
    with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=30)) as upstream:
        upstream.request("GET", target)
        response = upstream.getresponse()
        body = response.read()
        content_type = response.getheader("Content-Type", "application/octet-stream")
        if response.status == 200 and relative in ("", "index.html"):
            body = body.replace(b"<head>", b"<head>" + _SCOPE, 1)
        handler.respond(response.status, body, content_type)


def relay_websocket(handler, port, target):
    """Relay raw RFC6455 bytes after the upstream handshake, without decoding frames."""
    with socket.create_connection(("127.0.0.1", port), timeout=10) as upstream:
        headers = {name: value for name, value in handler.headers.items()
                   if name.lower() in ("upgrade", "connection", "sec-websocket-key",
                                       "sec-websocket-version", "sec-websocket-protocol",
                                       "sec-websocket-extensions")}
        headers["Host"] = f"127.0.0.1:{port}"
        upstream.sendall((f"GET {target} HTTP/1.1\r\n" +
                          "".join(f"{name}: {value}\r\n" for name, value in headers.items()) + "\r\n").encode())
        # Flush the complete handshake before entering the bidirectional tunnel.
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = upstream.recv(4096)
            if not chunk:
                raise OSError("Studio closed the WebSocket handshake")
            response.extend(chunk)
            if len(response) > 65536:
                raise OSError("oversized Studio handshake")
        handler.connection.sendall(response)
        handler.close_connection = True
        if not response.startswith(b"HTTP/1.1 101"):
            return
        upstream.settimeout(10)
        handler.connection.settimeout(10)
        try:
            while True:
                ready, _, _ = select.select([upstream, handler.connection], [], [], 10)
                if not ready:
                    continue
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    (handler.connection if source is upstream else upstream).sendall(data)
        except OSError:
            return
