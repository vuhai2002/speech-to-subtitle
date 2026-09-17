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
            {"key": "ROUTER_BASE_URL", "label": "Base URL", "secret": False},
            {"key": "ROUTER_API_KEY", "label": "API key", "secret": True},
            {"key": "TRANSCRIBE_MODEL", "label": "Model", "secret": False},
        ],
    },
    "mai": {
        "label": "MAI-Transcribe-2 (OpenRouter)",
        "has_prompt": False,
        "needs_gpu_for_srt": False,
        "video_ok": True,
        "fields": [
            {"key": "OPENROUTER_API_KEY", "label": "OpenRouter API key", "secret": True},
            {"key": "MAI_MODEL", "label": "Model", "secret": False},
            {"key": "MAI_LANGUAGE", "label": "Language", "secret": False},
        ],
    },
    "vertex": {
        "label": "Gemini (Vertex AI)",
        "has_prompt": True,
        "needs_gpu_for_srt": True,
        "video_ok": False,
        "fields": [
            {"key": "GOOGLE_CLOUD_PROJECT", "label": "Project ID", "secret": False},
            {"key": "GOOGLE_CLOUD_LOCATION", "label": "Location", "secret": False},
            {"key": "GCS_BUCKET_NAME", "label": "GCS bucket", "secret": False},
            {"key": "GOOGLE_APPLICATION_CREDENTIALS", "label": "Service account key path", "secret": False},
            {"key": "GEMINI_MODEL", "label": "Model", "secret": False},
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


def test_backend(backend: str, values: dict) -> tuple[bool, str]:
    """Light auth/connectivity probe. Returns (ok, message). Network call - not unit tested."""
    try:
        if backend == "mai":
            import urllib.request
            req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                         headers={"Authorization": "Bearer " + values.get("OPENROUTER_API_KEY", "")})
            with urllib.request.urlopen(req, timeout=20) as r:
                return (r.status == 200, f"HTTP {r.status}")
        if backend == "router":
            import urllib.request
            base = values.get("ROUTER_BASE_URL", "").rstrip("/")
            req = urllib.request.Request(base + "/models",
                                         headers={"Authorization": "Bearer " + values.get("ROUTER_API_KEY", "")})
            with urllib.request.urlopen(req, timeout=20) as r:
                return (r.status == 200, f"HTTP {r.status}")
        if backend == "vertex":
            key = values.get("GOOGLE_APPLICATION_CREDENTIALS", "")
            return (bool(key) and Path(key).exists(), "credentials file found" if key else "no credentials path")
    except Exception as e:                                # noqa: BLE001
        return (False, str(e)[:200])
    return (False, "unknown backend")
