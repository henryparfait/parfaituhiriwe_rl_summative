"""
Live 3D visualisation + small HTTP API for the Kagugu Follow-Up environment.

Architecture (also the integration path to the mobile app):

    KaguguFollowUpEnv --render_state()--> LiveViewer (Python, stdlib only)
                                             |  GET /api/state      -> current snapshot (JSON)
                                             |  GET /api/recommend  -> agent's recommended next action
                                             |  GET /api/health     -> liveness check
                                             v
                         browser: environment/viewer/index.html (three.js 3D scene + HUD)

The same /api/recommend contract is what the CHW / nurse mobile app would call:
it sends the current caseload and receives the next recommended follow-up action.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

VIEWER_HTML = Path(__file__).resolve().parent / "viewer" / "index.html"


class LiveViewer:
    def __init__(self, port: int = 8765, open_browser: bool = True):
        self._state: dict = {}
        self._lock = threading.Lock()
        self._recommender: Callable[[], dict] | None = None
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the terminal clean for agent logs
                pass

            def _send(self, code: int, body: bytes, ctype: str):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    self._send(200, VIEWER_HTML.read_bytes(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    with viewer._lock:
                        body = json.dumps(viewer._state).encode()
                    self._send(200, body, "application/json")
                elif path == "/api/recommend":
                    rec = viewer._recommender() if viewer._recommender else {"error": "no agent attached"}
                    self._send(200, json.dumps(rec).encode(), "application/json")
                elif path == "/api/health":
                    self._send(200, b'{"status": "ok"}', "application/json")
                else:
                    self._send(404, b"not found", "text/plain")

        # try a few ports in case one is busy
        last_err = None
        for p in range(port, port + 10):
            try:
                self._server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
                self.port = p
                break
            except OSError as e:
                last_err = e
        else:
            raise RuntimeError(f"Could not open a local port for the viewer: {last_err}")

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self.port}/"
        if open_browser:
            webbrowser.open(self.url)

    def update(self, state: dict) -> None:
        with self._lock:
            self._state = state

    def set_recommender(self, fn: Callable[[], dict]) -> None:
        self._recommender = fn

    def close(self) -> None:
        try:
            self._server.shutdown()
            self._server.server_close()
        except Exception:
            pass
