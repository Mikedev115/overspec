from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from overspec import eval as ev

FAKE_SERVER = Path(__file__).resolve().parent / "fake_server.py"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(url: str, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def test_run_eval_scores_real_http_roundtrip():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(FAKE_SERVER), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        assert _wait_health(url), "fake server ไม่ขึ้น"
        items = ev.load_items()[:3]
        report = ev.run_eval(url, items)

        assert report.total == 3
        assert report.passed >= 1          # fake ตอบ "56" → thai-01 ผ่าน
        assert {r.tag for r in report.results} == {"thai"}
        assert report.as_dict()["by_tag"]["thai"]["total"] == 3
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
