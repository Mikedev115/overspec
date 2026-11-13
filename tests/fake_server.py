"""server ปลอมสำหรับเทสต์ supervise — พิมพ์ log บรรทัดเดียวกับ llama.cpp แล้วตอบ HTTP

ใช้: python tests/fake_server.py --port 8123 [--allocated 4458.03]
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (ชื่อตาม http.server)
        if self.path == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path == "/completion":
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            body = json.dumps({"content": "56",
                               "timings": {"predicted_per_second": 12.5}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, *args):  # ปิด log ของ http.server ไม่ให้ปนกับ log ที่เราตรวจ
        return


def main(argv: list[str]) -> int:
    port = 8123
    allocated = 4458.03
    if "--port" in argv:
        port = int(argv[argv.index("--port") + 1])
    if "--allocated" in argv:
        allocated = float(argv[argv.index("--allocated") + 1])

    print("load_tensors:        CUDA0 model buffer size =  %6.2f MiB" % allocated, flush=True)
    server = HTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
