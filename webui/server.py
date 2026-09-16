"""FastAPI app: serve the single-page UI and the JSON/SSE APIs. Thin layer over webui.* modules."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from . import backends, settings as settings_mod
from .jobs import JobManager

STATIC = Path(__file__).resolve().parent / "static"
SECRET_KEYS = {f["key"] for b in backends.BACKENDS.values() for f in b["fields"] if f["secret"]}
_TERMINAL_STATUSES = {"done", "error", "stopped"}


def create_app(state_dir: str, env_path: str) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    jm = JobManager(state_dir)
    has_gpu = backends.detect_gpu()
    # MAI/Router subprocesses read their API keys via os.getenv(...) from the process
    # environment; they do not load .env themselves (unlike the Vertex script). Load the
    # saved values into this process so JobManager's subprocess env (os.environ + step.env)
    # actually carries keys entered in Settings.
    os.environ.update(settings_mod.read_env(env_path))

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/config")
    def config():
        return {"backends": backends.BACKENDS, "has_gpu": has_gpu}

    @app.get("/api/files")
    def files(dir: str):
        p = Path(dir)
        exts = {".mp3", ".m4a", ".wav", ".mp4", ".mkv", ".webm", ".aac", ".flac"}
        items = sorted(str(f) for f in p.glob("*") if f.suffix.lower() in exts) if p.is_dir() else []
        return {"dir": dir, "files": items}

    @app.get("/api/settings")
    def get_settings():
        env = settings_mod.read_env(env_path)
        values = {k: (settings_mod.masked(v) if k in SECRET_KEYS else v) for k, v in env.items()}
        return {"values": values}

    @app.post("/api/settings")
    async def post_settings(req: Request):
        updates = await req.json()
        applied = {k: str(v) for k, v in updates.items() if v not in ("", None)}
        settings_mod.write_env(env_path, applied)
        os.environ.update(applied)  # take effect immediately, no server restart needed
        return {"ok": True}

    @app.post("/api/test/{backend}")
    async def test(backend: str, req: Request):
        values = await req.json()
        ok, msg = backends.test_backend(backend, values)
        return {"ok": ok, "message": msg}

    @app.post("/api/jobs")
    async def start_job(req: Request):
        b = await req.json()
        if not backends.output_allowed(b["backend"], b["output"], has_gpu):
            return JSONResponse({"error": "This output needs a GPU; choose .txt or the MAI backend."}, status_code=400)
        job = jm.start_from(b["backend"], b["output"], b["source"], has_gpu, b.get("model"),
                            b.get("workers"), b.get("prompt"))
        return {"id": job.id}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.__dict__ for j in jm.list()]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        j = jm.get(job_id)
        return j.__dict__ if j else JSONResponse({"error": "not found"}, status_code=404)

    @app.post("/api/jobs/{job_id}/stop")
    def stop(job_id: str):
        jm.stop(job_id)
        return {"ok": True}

    @app.get("/api/jobs/{job_id}/download")
    def download(job_id: str, name: str):
        j = jm.get(job_id)
        if not j:
            return JSONResponse({"error": "not found"}, status_code=404)
        # Contain the resolved path inside out_dir: `name` is caller-controlled, so
        # "../../etc/passwd" or an absolute path must not be able to escape it.
        base = Path(j.out_dir).resolve()
        target = (base / name).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            return JSONResponse({"error": "invalid file"}, status_code=404)
        return FileResponse(str(target), filename=target.name)

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str):
        j = jm.get(job_id)
        if not j:
            # Guard before subscribe(): an unknown job never emits a sentinel, so entering
            # the loop below on a bad id would block a threadpool thread forever.
            return JSONResponse({"error": "not found"}, status_code=404)
        q = jm.subscribe(job_id)

        def gen():
            for ln in (j.log if j else []):
                yield _sse("log", ln)
            if j and j.status in _TERMINAL_STATUSES:
                # Job already finished before this client subscribed: it will never emit
                # another sentinel, so looping on q.get() below would hang forever. Send
                # the final status and stop instead of entering the streaming loop.
                yield _sse("status", j.status)
                return
            while True:
                kind, payload = q.get()
                if kind == "log" and payload is None:
                    yield _sse("status", jm.get(job_id).status)
                    break
                yield _sse(kind, payload)

        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


def _sse(kind: str, payload) -> str:
    return f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
