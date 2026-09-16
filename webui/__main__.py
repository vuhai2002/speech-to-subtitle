"""python -m webui : start the local web UI on 127.0.0.1 and open the browser."""
import argparse
import os
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


def main():
    import uvicorn
    a = parse_args()
    app = create_app(DEFAULT_STATE, DEFAULT_ENV)
    if a.open:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
