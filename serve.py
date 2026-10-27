"""Dev server statis untuk GestureFlow.

Cara pakai: jalankan `python serve.py` untuk port 8000 atau
`python serve.py 8080` untuk memilih port lain.
Server ini menggunakan threading dan IPv4 eksplisit karena server
`python -m http.server` bawaan pernah mengalami transfer file besar yang
terputus pada Windows.
"""

import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class GestureFlowHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class GestureFlowRequestHandler(SimpleHTTPRequestHandler):
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "application/javascript",
        ".json": "application/json",
        ".bin": "application/octet-stream",
        ".wasm": "application/wasm",
        ".tflite": "application/octet-stream",
        ".data": "application/octet-stream",
        ".binarypb": "application/octet-stream",
    }

    def log_message(self, format_string, *args):
        print(f"{self.client_address[0]} - {format_string % args}", flush=True)


def main():
    script_dir = Path(__file__).resolve().parent
    serve_dir = script_dir / "web" / "src"
    if not serve_dir.is_dir():
        print(f"Error: serve directory not found: {serve_dir}", file=sys.stderr)
        return 1

    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    except ValueError:
        print("Error: port must be an integer.", file=sys.stderr)
        return 2

    handler = partial(GestureFlowRequestHandler, directory=str(serve_dir))
    try:
        server = GestureFlowHTTPServer(("127.0.0.1", port), handler)
    except OSError as exc:
        print(f"Error: could not start server on 127.0.0.1:{port}: {exc}", file=sys.stderr)
        return 1

    print(f"Serving GestureFlow at http://127.0.0.1:{port}/")
    print(f"Directory: {serve_dir}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
