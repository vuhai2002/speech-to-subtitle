"""python -m webui : start the local web UI on 127.0.0.1 and open the browser."""
import argparse
import os
import socket
import threading
import webbrowser

from .server import create_app

DEFAULT_STATE = os.path.join("out", "webui")
DEFAULT_ENV = ".env"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Local web UI for speech-to-subtitle")
    ap.add_argument("--host", default=os.getenv("WEBUI_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("WEBUI_PORT", "8000")))
    ap.add_argument("--no-open", dest="open", action="store_false")
    ap.set_defaults(open=True)
    return ap.parse_args(argv)


def find_free_port(host: str, start: int, tries: int = 20) -> int:
    """Return the first free TCP port at or after `start` (so a busy 8000 falls back to 8001,
    8002, ...). If none of the range is free, return `start` and let uvicorn report the error."""
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return start


def main():
    import uvicorn
    a = parse_args()
    port = find_free_port(a.host, a.port)
    if port != a.port:
        print(f"Port {a.port} is in use, using {port} instead.", flush=True)
    if a.open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{a.host}:{port}")).start()
    uvicorn.run(app=create_app(DEFAULT_STATE, DEFAULT_ENV), host=a.host, port=port, log_level="info")


if __name__ == "__main__":
    main()
