"""Backend registry, output gating, and subprocess command builder.

Pure logic: no network, no torch. Given a backend + output choice, produce the
ordered chain of subprocess steps that reuse the existing pipeline CLIs.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Step:
    label: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


BACKENDS: dict[str, dict] = {
    "router": {
        "label": "Router (OpenAI-compatible)",
        "has_prompt": True,
        "needs_gpu_for_srt": True,
        "video_ok": True,
        "fields": [
            {"key": "ROUTER_BASE_URL", "label": "Base URL", "secret": False, "placeholder": "https://<router-host>/v1"},
            {"key": "ROUTER_API_KEY", "label": "API key", "secret": True, "placeholder": "sk-..."},
            {"key": "TRANSCRIBE_MODEL", "label": "Model", "secret": False, "placeholder": "ag/gemini-3.8-flash"},
        ],
    },
    "mai": {
        "label": "MAI-Transcribe-2 (OpenRouter)",
        "has_prompt": False,
        "needs_gpu_for_srt": False,
        "video_ok": True,
        "fields": [
            {"key": "OPENROUTER_API_KEY", "label": "OpenRouter API key", "secret": True, "placeholder": "sk-or-v1-..."},
            {"key": "MAI_MODEL", "label": "Model", "secret": False, "placeholder": "microsoft/mai-transcribe-2"},
            {"key": "MAI_LANGUAGE", "label": "Language", "secret": False, "placeholder": "vi"},
        ],
    },
    "vertex": {
        "label": "Gemini (Vertex AI)",
        "has_prompt": True,
        "needs_gpu_for_srt": True,
        "video_ok": False,
        "fields": [
            {"key": "GOOGLE_CLOUD_PROJECT", "label": "Project ID", "secret": False, "placeholder": "my-gcp-project"},
            {"key": "GOOGLE_CLOUD_LOCATION", "label": "Location", "secret": False, "placeholder": "us-central1"},
            {"key": "GCS_BUCKET_NAME", "label": "GCS bucket", "secret": False, "placeholder": "my-bucket"},
            {"key": "GOOGLE_APPLICATION_CREDENTIALS", "label": "Service account key path", "secret": False, "placeholder": "./service-account-key.json"},
            {"key": "GEMINI_MODEL", "label": "Model", "secret": False, "placeholder": "gemini-2.5-pro"},
        ],
    },
}

_MODULE = {
    "mai": "transcribe.mai_transcribe",
    "router": "transcribe.chunked_transcribe",
}


def output_allowed(backend: str, output: str, has_gpu: bool) -> bool:
    if output == "txt":
        return True
    return not BACKENDS[backend]["needs_gpu_for_srt"] or has_gpu


def _transcribe_step(backend: str, input_path: str, out_dir: str, model, workers, prompt_file) -> Step:
    argv = [sys.executable, "-m", f"{_MODULE[backend]}.run_pipeline",
            "--input", input_path, "--out-dir", out_dir]
    if model:
        argv += ["--model", model]
    if workers:
        argv += ["--workers", str(workers)]
    if prompt_file and BACKENDS[backend]["has_prompt"]:
        argv += ["--prompt-file", prompt_file]
    return Step("transcribe", argv)


def build_steps(backend, output, input_path, out_dir, *, model=None, workers=None, prompt_file=None) -> list[Step]:
    if backend == "vertex":
        return _vertex_steps(output, input_path, out_dir, model, prompt_file)
    steps = [_transcribe_step(backend, input_path, out_dir, model, workers, prompt_file)]
    if output == "srt":
        steps.append(Step("build srt", [sys.executable, "-m", f"{_MODULE[backend]}.build_srt", "--out-dir", out_dir]))
    return steps


def _vertex_steps(output, input_path, out_dir, model, prompt_file) -> list[Step]:
    """Vertex is a dir-batch script; stage the single file into out_dir/_in and run it there.
    For srt, follow with realign on the produced txt + the staged audio."""
    stage = f"{out_dir}/_in"
    argv = [sys.executable, "transcribe/batch_transcribe_vertex.py", "--mp3-dir", stage, "--txt-dir", out_dir]
    if model:
        argv += ["--model", model]
    if prompt_file:
        argv += ["--prompt-file", prompt_file]
    steps = [Step("stage", [sys.executable, "-m", "webui.stage", input_path, stage]),
             Step("transcribe", argv)]
    if output == "srt":
        steps.append(Step("align", [sys.executable, "-m", "realign.run_batch",
                                     "--txt-dir", out_dir, "--audio-dir", stage,
                                     "--out-dir", out_dir, "--window-sec", "1500"]))
    return steps


import subprocess


def gpu_name() -> str:
    """Return the CUDA device name (e.g. 'NVIDIA GeForce RTX 3050 Ti'), or '' if no GPU.
    Runs torch in a subprocess so the web server itself never imports torch."""
    code = "import torch;print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        return out.stdout.strip()
    except Exception:
        return ""


def _probe_get(url: str, api_key: str, timeout: int = 20) -> tuple[int, str]:
    """GET `url` with a Bearer token and return (status_code, short_message).
    urllib raises HTTPError for 4xx/5xx, so translate that to its status code
    instead of an exception; any transport failure (DNS, TLS, timeout, bad URL)
    returns status 0 with the error text. Network call - not unit tested."""
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + api_key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return (r.status, f"HTTP {r.status}")
    except urllib.error.HTTPError as e:
        return (e.code, f"HTTP {e.code}")
    except Exception as e:                                # noqa: BLE001
        return (0, str(e)[:160])


def _classify_probe(backend: str, status: int, message: str) -> tuple[bool, str]:
    """Map an auth-probe status to (ok, human message). Pure - unit tested.

    401 means the key was rejected. A 403 from the Router means the key WAS
    accepted but the server forbids listing models (9router does this); the key
    still works for transcription, so treat it as success with a note. Status 0
    is a transport error (the message carries the detail)."""
    if 200 <= status < 300:
        return (True, "Key accepted.")
    if status == 401:
        return (False, "Invalid API key (HTTP 401).")
    if status == 403 and backend == "router":
        return (True, "Key accepted (this server does not allow listing models, but transcription works).")
    if status == 0:
        return (False, "Could not reach the server: " + message)
    return (False, f"Unexpected response ({message}).")


def probe_backend(backend: str, values: dict) -> tuple[bool, str]:
    """Light auth/connectivity probe. Returns (ok, message).

    MAI/Router probe an auth-required endpoint so an invalid key actually fails
    (OpenRouter's /models is public, so it must not be used to verify a key).
    Vertex only checks that the service-account file exists (verifying it needs
    the Google libs, which this environment does not carry)."""
    if backend == "mai":
        key = values.get("OPENROUTER_API_KEY", "")
        if not key:
            return (False, "No API key set.")
        # /api/v1/key requires auth (401 without a valid key); /models is public.
        return _classify_probe(backend, *_probe_get("https://openrouter.ai/api/v1/key", key))
    if backend == "router":
        base = values.get("ROUTER_BASE_URL", "").rstrip("/")
        key = values.get("ROUTER_API_KEY", "")
        if not base:
            return (False, "No Base URL set.")
        if not key:
            return (False, "No API key set.")
        return _classify_probe(backend, *_probe_get(base + "/models", key))
    if backend == "vertex":
        key = values.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        return (bool(key) and Path(key).exists(), "credentials file found" if key else "no credentials path")
    return (False, "unknown backend")
