# Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local browser UI that runs the existing transcribe/align pipeline (pick audio, choose backend, run, watch progress, download `.srt`/`.txt`).

**Architecture:** A small FastAPI server (`webui/`) serves a single static app-shell page and drives the existing CLIs as subprocesses. The UI only orchestrates; it never reimplements pipeline logic. Bind to `127.0.0.1` only.

**Tech Stack:** Python 3.10+, FastAPI + uvicorn (UI-only deps in `requirements-webui.txt`), vanilla HTML/CSS/JS (no build step), pytest.

**Spec:** `docs/webui-design.md`

## Global Constraints

- Every source file under ~200 lines; split by responsibility if larger.
- All comments, docstrings, log strings, and docs in English. Plain ASCII punctuation.
- The UI orchestrates existing modules via subprocess; the only change to existing pipeline code is an optional `--prompt-file` argument (backward compatible).
- Bind the server to `127.0.0.1` only; no auth. Secrets are read/written in `.env` (gitignored).
- Subprocesses run with `cwd = repo root` and `sys.executable` as the interpreter (same venv).
- UI runtime state lives under the already-gitignored `out/webui/` (`jobs.json` + `out/webui/<job-id>/`).
- Follow the repo pytest convention (`pytest.ini`, `pythonpath = .`, `testpaths = tests`). Do not import `torch`/`fastapi` at module top level in code paths that pure-logic tests import.

---

### Task 1: Backend registry, output gating, command builder

**Files:**
- Create: `webui/__init__.py` (empty package marker with a one-line docstring)
- Create: `webui/backends.py`
- Test: `tests/webui/__init__.py` (empty), `tests/webui/test_backends.py`

**Interfaces:**
- Produces:
  - `@dataclass Step(label: str, argv: list[str], env: dict[str, str])`
  - `BACKENDS: dict[str, dict]` (metadata: `label`, `has_prompt: bool`, `needs_gpu_for_srt: bool`, `fields: list[dict]`)
  - `output_allowed(backend: str, output: str, has_gpu: bool) -> bool`
  - `build_steps(backend: str, output: str, input_path: str, out_dir: str, *, model: str | None = None, workers: int | None = None, prompt_file: str | None = None) -> list[Step]`

- [ ] **Step 1: Write the failing test**

```python
# tests/webui/test_backends.py
import sys
from webui.backends import BACKENDS, output_allowed, build_steps, Step


def test_registry_shape():
    assert set(BACKENDS) == {"vertex", "router", "mai"}
    assert BACKENDS["mai"]["needs_gpu_for_srt"] is False
    assert BACKENDS["router"]["needs_gpu_for_srt"] is True
    assert BACKENDS["mai"]["has_prompt"] is False


def test_output_gating():
    assert output_allowed("mai", "txt", has_gpu=False)
    assert output_allowed("mai", "srt", has_gpu=False)          # native timestamps
    assert output_allowed("router", "txt", has_gpu=False)
    assert not output_allowed("router", "srt", has_gpu=False)   # needs GPU
    assert output_allowed("router", "srt", has_gpu=True)


def test_build_steps_mai_srt():
    steps = build_steps("mai", "srt", "a.m4a", "out/webui/j1", model="microsoft/mai-transcribe-2", workers=3)
    assert [s.label for s in steps] == ["transcribe", "build srt"]
    assert steps[0].argv == [sys.executable, "-m", "transcribe.mai_transcribe.run_pipeline",
                             "--input", "a.m4a", "--out-dir", "out/webui/j1",
                             "--model", "microsoft/mai-transcribe-2", "--workers", "3"]
    assert steps[1].argv == [sys.executable, "-m", "transcribe.mai_transcribe.build_srt",
                             "--out-dir", "out/webui/j1"]


def test_build_steps_router_txt_with_prompt():
    steps = build_steps("router", "txt", "a.m4a", "o", prompt_file="/tmp/p.txt")
    assert len(steps) == 1
    assert "--prompt-file" in steps[0].argv
    assert steps[0].argv[steps[0].argv.index("--prompt-file") + 1] == "/tmp/p.txt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_backends.py -v`
Expected: FAIL (`ModuleNotFoundError: webui`)

- [ ] **Step 3: Write minimal implementation**

```python
# webui/backends.py
"""Backend registry, output gating, and subprocess command builder.

Pure logic: no network, no torch. Given a backend + output choice, produce the
ordered chain of subprocess steps that reuse the existing pipeline CLIs.
"""
import sys
from dataclasses import dataclass, field


@dataclass
class Step:
    label: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


BACKENDS: dict[str, dict] = {
    "vertex": {
        "label": "Gemini (Vertex AI)",
        "has_prompt": True,
        "needs_gpu_for_srt": True,
        "fields": [
            {"key": "GOOGLE_CLOUD_PROJECT", "label": "Project ID", "secret": False},
            {"key": "GOOGLE_CLOUD_LOCATION", "label": "Location", "secret": False},
            {"key": "GCS_BUCKET_NAME", "label": "GCS bucket", "secret": False},
            {"key": "GOOGLE_APPLICATION_CREDENTIALS", "label": "Service account key path", "secret": False},
            {"key": "GEMINI_MODEL", "label": "Model", "secret": False},
        ],
    },
    "router": {
        "label": "Router (OpenAI-compatible)",
        "has_prompt": True,
        "needs_gpu_for_srt": True,
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
        "fields": [
            {"key": "OPENROUTER_API_KEY", "label": "OpenRouter API key", "secret": True},
            {"key": "MAI_MODEL", "label": "Model", "secret": False},
            {"key": "MAI_LANGUAGE", "label": "Language", "secret": False},
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
```

Also create `webui/__init__.py`:

```python
"""Local web UI that orchestrates the transcribe/align pipeline. See docs/webui-design.md."""
```

And a tiny staging helper `webui/stage.py` (copies one file into a dir), referenced by the vertex chain:

```python
"""Copy a single audio file into a staging directory (used to feed the dir-batch Vertex script)."""
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    src, dst_dir = sys.argv[1], sys.argv[2]
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, Path(dst_dir) / Path(src).name)
    print(f"[stage] {src} -> {dst_dir}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_backends.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/__init__.py webui/backends.py webui/stage.py tests/webui/__init__.py tests/webui/test_backends.py
git commit -m "feat(webui): backend registry, output gating, command builder"
```

---

### Task 2: Settings read/write to .env

**Files:**
- Create: `webui/settings.py`
- Test: `tests/webui/test_settings.py`

**Interfaces:**
- Consumes: `BACKENDS` from `webui.backends`
- Produces:
  - `read_env(path: str) -> dict[str, str]`
  - `write_env(path: str, updates: dict[str, str]) -> None` (upsert keys, preserve other lines/comments)
  - `masked(value: str) -> str` (show last 4 chars)

- [ ] **Step 1: Write the failing test**

```python
# tests/webui/test_settings.py
from webui.settings import read_env, write_env, masked


def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nEXISTING=1\nOPENROUTER_API_KEY=old\n", encoding="utf-8")
    write_env(str(p), {"OPENROUTER_API_KEY": "sk-new", "MAI_MODEL": "microsoft/mai-transcribe-2"})
    env = read_env(str(p))
    assert env["OPENROUTER_API_KEY"] == "sk-new"
    assert env["MAI_MODEL"] == "microsoft/mai-transcribe-2"
    assert env["EXISTING"] == "1"                 # untouched
    assert "# comment" in p.read_text(encoding="utf-8")   # comment preserved


def test_masked():
    assert masked("sk-or-v1-abcd1234") == "************1234"
    assert masked("") == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_settings.py -v`
Expected: FAIL (`ModuleNotFoundError: webui.settings`)

- [ ] **Step 3: Write minimal implementation**

```python
# webui/settings.py
"""Read and write simple KEY=VALUE lines in a .env file, preserving comments and unrelated keys."""
from pathlib import Path


def read_env(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        out[k.strip()] = v.strip()
    return out


def write_env(path: str, updates: dict[str, str]) -> None:
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    seen = set()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k = s.split("=", 1)[0].strip()
        if k in updates:
            lines[i] = f"{k}={updates[k]}"
            seen.add(k)
    for k, v in updates.items():
        if k not in seen:
            lines.append(f"{k}={v}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def masked(value: str) -> str:
    if not value:
        return ""
    tail = value[-4:]
    return "*" * max(0, len(value) - 4) + tail
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_settings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/settings.py tests/webui/test_settings.py
git commit -m "feat(webui): .env settings read/write with key masking"
```

---

### Task 3: Job stage parser

**Files:**
- Create: `webui/stages.py`
- Test: `tests/webui/test_stages.py`

**Interfaces:**
- Produces: `stage_from_line(line: str) -> str | None` (returns a coarse stage label when a known marker is seen, else None)

- [ ] **Step 1: Write the failing test**

```python
# tests/webui/test_stages.py
from webui.stages import stage_from_line


def test_markers():
    assert stage_from_line("10:46:19 [1/4] ffmpeg -> mono 16kHz") == "prepare"
    assert stage_from_line("10:46:42 [2/4] VAD + plan") == "plan"
    assert stage_from_line("10:47:39 [3/4] transcribe 11 chunks") == "transcribe"
    assert stage_from_line("10:49:02 [4/4] done: 11/11") == "assemble"
    assert stage_from_line("10:50:55 align per chunk (MMS, cuda)") == "align"
    assert stage_from_line("[stage] a.m4a -> out/_in") == "stage"
    assert stage_from_line("some unrelated line") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_stages.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# webui/stages.py
"""Map a pipeline log line to a coarse stage label, using markers the CLIs already print."""

_MARKERS = [
    ("[1/4]", "prepare"),
    ("[2/4]", "plan"),
    ("[3/4]", "transcribe"),
    ("[4/4]", "assemble"),
    ("[stage]", "stage"),
    ("align", "align"),
    ("gom cue", "align"),
    ("build srt", "align"),
]


def stage_from_line(line: str) -> str | None:
    low = line.lower()
    for marker, stage in _MARKERS:
        if marker in line or marker in low:
            return stage
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_stages.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/stages.py tests/webui/test_stages.py
git commit -m "feat(webui): coarse stage detection from pipeline log markers"
```

---

### Task 4: JobManager (run subprocess chain, capture, stop, history)

**Files:**
- Create: `webui/jobs.py`
- Test: `tests/webui/test_jobs.py`

**Interfaces:**
- Consumes: `Step` from `webui.backends`, `stage_from_line` from `webui.stages`
- Produces:
  - `@dataclass Job(id, backend, output, source, out_dir, status, stage, log: list[str], results: dict, created: float)`
  - `class JobManager(state_dir: str)` with:
    - `start(backend, output, source, steps: list[Step]) -> Job`
    - `get(job_id) -> Job | None`
    - `list() -> list[Job]` (newest first)
    - `stop(job_id) -> None`
    - `subscribe(job_id) -> queue.Queue` (yields `("log", str)` / `("status", str)` events; `None` sentinel on completion)

- [ ] **Step 1: Write the failing test**

```python
# tests/webui/test_jobs.py
import sys
import time
from webui.backends import Step
from webui.jobs import JobManager


def _fake_step(label, out_lines):
    code = "; ".join([f"print({line!r})" for line in out_lines])
    return Step(label, [sys.executable, "-c", code])


def test_job_runs_and_captures(tmp_path):
    jm = JobManager(str(tmp_path))
    job = jm.start("mai", "txt", "a.m4a", [_fake_step("transcribe", ["10:00 [1/4] ffmpeg", "done"])])
    for _ in range(100):
        if jm.get(job.id).status in ("done", "error"):
            break
        time.sleep(0.05)
    j = jm.get(job.id)
    assert j.status == "done"
    assert any("[1/4]" in ln for ln in j.log)
    assert j.stage == "prepare"


def test_history_persists(tmp_path):
    jm = JobManager(str(tmp_path))
    job = jm.start("mai", "txt", "a.m4a", [_fake_step("t", ["hi"])])
    for _ in range(100):
        if jm.get(job.id).status in ("done", "error"):
            break
        time.sleep(0.05)
    jm2 = JobManager(str(tmp_path))                 # reload from disk
    assert any(x.id == job.id for x in jm2.list())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_jobs.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# webui/jobs.py
"""Run a chain of subprocess Steps for one job, capture stdout, stream events, persist history.

One job runs at a time in the first version, but each is tracked and stored so the Jobs view
survives a restart. State lives under a gitignored directory (out/webui by default).
"""
import json
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .backends import Step
from .stages import stage_from_line

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    id: str
    backend: str
    output: str
    source: str
    out_dir: str
    status: str = "queued"          # queued | running | done | error | stopped
    stage: str = ""
    log: list[str] = field(default_factory=list)
    results: dict = field(default_factory=dict)
    created: float = field(default_factory=time.time)


class JobManager:
    def __init__(self, state_dir: str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._subs: dict[str, list[queue.Queue]] = {}
        self._load()

    # ---- history persistence ----
    def _index(self) -> Path:
        return self.state_dir / "jobs.json"

    def _load(self) -> None:
        if self._index().exists():
            for d in json.loads(self._index().read_text(encoding="utf-8")):
                self._jobs[d["id"]] = Job(**d)

    def _save(self) -> None:
        rows = [asdict(j) for j in sorted(self._jobs.values(), key=lambda x: x.created, reverse=True)]
        self._index().write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- api ----
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda x: x.created, reverse=True)

    def subscribe(self, job_id: str) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        self._subs.setdefault(job_id, []).append(q)
        return q

    def _emit(self, job: Job, kind: str, payload) -> None:
        for q in self._subs.get(job.id, []):
            q.put((kind, payload))

    def start(self, backend: str, output: str, source: str, steps: list[Step]) -> Job:
        job = Job(id=uuid.uuid4().hex[:8], backend=backend, output=output, source=source,
                  out_dir=str(self.state_dir / uuid.uuid4().hex[:8]))
        self._jobs[job.id] = job
        self._save()
        threading.Thread(target=self._run, args=(job, steps), daemon=True).start()
        return job

    def stop(self, job_id: str) -> None:
        p = self._procs.get(job_id)
        if p and p.poll() is None:
            p.terminate()

    def _run(self, job: Job, steps: list[Step]) -> None:
        job.status = "running"
        self._emit(job, "status", job.status)
        try:
            for step in steps:
                job.log.append(f"=== {step.label} ===")
                self._emit(job, "log", job.log[-1])
                proc = subprocess.Popen(step.argv, cwd=str(PROJECT_ROOT), env=self._env(step),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace", bufsize=1)
                self._procs[job.id] = proc
                for line in proc.stdout:                 # streaming read
                    line = line.rstrip("\n")
                    job.log.append(line)
                    st = stage_from_line(line)
                    if st:
                        job.stage = st
                        self._emit(job, "status", job.status)
                    self._emit(job, "log", line)
                proc.wait()
                if proc.returncode != 0:
                    job.status = "stopped" if proc.returncode < 0 else "error"
                    break
            else:
                job.status = "done"
        except Exception as e:                            # noqa: BLE001 - surface any launch failure
            job.log.append(f"[webui] {e}")
            job.status = "error"
        finally:
            self._procs.pop(job.id, None)
            job.results = self._collect(job)
            self._save()
            self._emit(job, "status", job.status)
            self._emit(job, "log", None)                  # sentinel: stream closed

    def _env(self, step: Step) -> dict:
        import os
        return {**os.environ, **step.env}

    def _collect(self, job: Job) -> dict:
        out = Path(job.out_dir)
        files = [str(p.relative_to(self.state_dir)) for p in out.glob("*.srt")] if out.exists() else []
        files += [str(p.relative_to(self.state_dir)) for p in out.glob("raw_transcript.txt")] if out.exists() else []
        return {"files": files}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_jobs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/jobs.py tests/webui/test_jobs.py
git commit -m "feat(webui): JobManager runs subprocess chain, captures output, persists history"
```

---

### Task 5: Add `--prompt-file` to the two Gemini entrypoints

**Files:**
- Modify: `transcribe/chunked_transcribe/config.py` (make `load_prompt` accept an optional file)
- Modify: `transcribe/chunked_transcribe/run_pipeline.py` (argparse `--prompt-file`, thread through)
- Modify: `transcribe/batch_transcribe_vertex.py` (argparse `--prompt-file`, resolve helper)
- Test: `tests/webui/test_prompt_file.py`

**Interfaces:**
- Produces:
  - `chunked_transcribe.config.load_prompt(prompt_file: str | None = None) -> str`
  - `batch_transcribe_vertex.resolve_prompt(prompt_file: str | None) -> str`

- [ ] **Step 1: Write the failing test**

```python
# tests/webui/test_prompt_file.py
from transcribe.chunked_transcribe import config as ct_config


def test_load_prompt_default_nonempty():
    assert len(ct_config.load_prompt()) > 20            # falls back to the Vertex script prompt


def test_load_prompt_from_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("CUSTOM PROMPT", encoding="utf-8")
    assert ct_config.load_prompt(str(p)) == "CUSTOM PROMPT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_prompt_file.py -v`
Expected: FAIL (`load_prompt() takes 0 positional arguments`)

- [ ] **Step 3: Write minimal implementation**

In `transcribe/chunked_transcribe/config.py`, change `load_prompt`:

```python
def load_prompt(prompt_file: str | None = None) -> str:
    """Return the transcription prompt. If prompt_file is given, read it; otherwise read
    TRANSCRIBE_PROMPT from the Vertex script (single source of truth)."""
    if prompt_file:
        return Path(prompt_file).read_text(encoding="utf-8")
    src = VERTEX_SCRIPT.read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TRANSCRIBE_PROMPT" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"TRANSCRIBE_PROMPT not found in {VERTEX_SCRIPT}")
```

In `transcribe/chunked_transcribe/run_pipeline.py`, add the arg and thread it into `run`:
- Add parameter `prompt_file: str | None = None` to `run(...)` and replace `prompt = config.load_prompt()` with `prompt = config.load_prompt(prompt_file)`.
- In `main()`, add `ap.add_argument("--prompt-file", default=None)` and pass `a.prompt_file` to `run(...)`.

In `transcribe/batch_transcribe_vertex.py`, add near the prompt constant:

```python
def resolve_prompt(prompt_file):
    """Return TRANSCRIBE_PROMPT, or the contents of prompt_file when provided."""
    if prompt_file:
        with open(prompt_file, encoding="utf-8") as f:
            return f.read()
    return TRANSCRIBE_PROMPT
```

- Add `parser.add_argument("--prompt-file", default=None)` and use `resolve_prompt(args.prompt_file)` wherever `TRANSCRIBE_PROMPT` is currently passed to the model call.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_prompt_file.py -v`
Expected: PASS

Also confirm no regression to existing behavior:
Run: `.venv-realign/Scripts/python.exe -c "import transcribe.chunked_transcribe.run_pipeline, transcribe.batch_transcribe_vertex"`
Expected: no error.

- [ ] **Step 5: Commit**

```bash
git add transcribe/chunked_transcribe/config.py transcribe/chunked_transcribe/run_pipeline.py transcribe/batch_transcribe_vertex.py tests/webui/test_prompt_file.py
git commit -m "feat(transcribe): optional --prompt-file to override the transcription prompt"
```

---

### Task 6: GPU detection + backend test probes

**Files:**
- Modify: `webui/backends.py` (add `detect_gpu()` and `test_backend(...)`)
- Test: `tests/webui/test_backends.py` (add a test for `detect_gpu` return type)

**Interfaces:**
- Produces:
  - `detect_gpu() -> bool` (runs a short subprocess: `python -c "import torch;print(torch.cuda.is_available())"`; returns False on any failure)
  - `test_backend(backend: str, values: dict[str, str]) -> tuple[bool, str]` (light connectivity/auth probe; network - not unit tested)

- [ ] **Step 1: Write the failing test**

```python
# add to tests/webui/test_backends.py
from webui.backends import detect_gpu

def test_detect_gpu_returns_bool():
    assert isinstance(detect_gpu(), bool)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_backends.py::test_detect_gpu_returns_bool -v`
Expected: FAIL (`cannot import name 'detect_gpu'`)

- [ ] **Step 3: Write minimal implementation**

Append to `webui/backends.py`:

```python
import subprocess


def detect_gpu() -> bool:
    try:
        out = subprocess.run([sys.executable, "-c", "import torch;print(torch.cuda.is_available())"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() == "True"
    except Exception:
        return False


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
```

Add `from pathlib import Path` at the top of `backends.py` if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_backends.py -v`
Expected: PASS (all backend tests)

- [ ] **Step 5: Commit**

```bash
git add webui/backends.py tests/webui/test_backends.py
git commit -m "feat(webui): GPU detection and per-backend connectivity probes"
```

---

### Task 7: FastAPI server and routes

**Files:**
- Create: `webui/server.py`
- Create: `requirements-webui.txt`
- Test: `tests/webui/test_server.py`

**Interfaces:**
- Consumes: `BACKENDS`, `output_allowed`, `build_steps`, `detect_gpu`, `test_backend` from `webui.backends`; `JobManager` from `webui.jobs`; `read_env`, `write_env`, `masked` from `webui.settings`
- Produces: `create_app(state_dir: str, env_path: str) -> FastAPI`

- [ ] **Step 1: Add the dependency file and write the failing test**

Create `requirements-webui.txt`:

```
fastapi>=0.110
uvicorn>=0.29
```

Install into the dev venv:

```bash
.venv-realign/Scripts/python.exe -m pip install -r requirements-webui.txt
```

```python
# tests/webui/test_server.py
from fastapi.testclient import TestClient
from webui.server import create_app


def _client(tmp_path):
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=sk-test\n", encoding="utf-8")
    return TestClient(create_app(str(tmp_path / "state"), str(tmp_path / ".env")))


def test_index_served(tmp_path):
    r = _client(tmp_path).get("/")
    assert r.status_code == 200 and "speech-to-subtitle" in r.text


def test_backends_and_gpu(tmp_path):
    c = _client(tmp_path)
    r = c.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert "mai" in body["backends"]
    assert isinstance(body["has_gpu"], bool)


def test_settings_roundtrip(tmp_path):
    c = _client(tmp_path)
    c.post("/api/settings", json={"MAI_MODEL": "microsoft/mai-transcribe-2"})
    r = c.get("/api/settings")
    assert r.json()["values"]["MAI_MODEL"] == "microsoft/mai-transcribe-2"
    # secrets are masked in the GET
    assert r.json()["values"]["OPENROUTER_API_KEY"].endswith("test")
    assert "*" in r.json()["values"]["OPENROUTER_API_KEY"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_server.py -v`
Expected: FAIL (`ModuleNotFoundError: webui.server`)

- [ ] **Step 3: Write minimal implementation**

```python
# webui/server.py
"""FastAPI app: serve the single-page UI and the JSON/SSE APIs. Thin layer over webui.* modules."""
import json
import queue
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse

from . import backends, settings as settings_mod
from .jobs import JobManager

STATIC = Path(__file__).resolve().parent / "static"
SECRET_KEYS = {f["key"] for b in backends.BACKENDS.values() for f in b["fields"] if f["secret"]}


def create_app(state_dir: str, env_path: str) -> FastAPI:
    app = FastAPI()
    jm = JobManager(state_dir)
    has_gpu = backends.detect_gpu()

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
        settings_mod.write_env(env_path, {k: str(v) for k, v in updates.items() if v not in ("", None)})
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
        job = jm.get_placeholder() if False else None  # noqa - see start below
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
        path = Path(j.out_dir) / name
        return FileResponse(str(path), filename=Path(name).name)

    @app.get("/api/jobs/{job_id}/events")
    def events(job_id: str):
        q = jm.subscribe(job_id)

        def gen():
            j = jm.get(job_id)
            for ln in (j.log if j else []):
                yield _sse("log", ln)
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
```

Add a convenience method to `JobManager` so the server does not build steps itself. In `webui/jobs.py`, add:

```python
    def start_from(self, backend, output, source, has_gpu, model, workers, prompt) -> Job:
        from .backends import build_steps
        out_dir = str(self.state_dir / uuid.uuid4().hex[:8])
        prompt_file = None
        if prompt:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            prompt_file = str(Path(out_dir) / "prompt.txt")
            Path(prompt_file).write_text(prompt, encoding="utf-8")
        steps = build_steps(backend, output, source, out_dir, model=model, workers=workers, prompt_file=prompt_file)
        job = Job(id=uuid.uuid4().hex[:8], backend=backend, output=output, source=source, out_dir=out_dir)
        self._jobs[job.id] = job
        self._save()
        threading.Thread(target=self._run, args=(job, steps), daemon=True).start()
        return job
```

(Remove the stray `get_placeholder` line in `start_job`; it is illustrative only - the real call is `jm.start_from(...)`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/server.py webui/jobs.py requirements-webui.txt tests/webui/test_server.py
git commit -m "feat(webui): FastAPI server with config, settings, jobs, SSE routes"
```

---

### Task 8: Entry point (`python -m webui`) + GPU-aware startup

**Files:**
- Create: `webui/__main__.py`
- Test: `tests/webui/test_args.py`

**Interfaces:**
- Produces: `parse_args(argv: list[str]) -> argparse.Namespace` (fields: `host`, `port`, `open`)

- [ ] **Step 1: Write the failing test**

```python
# tests/webui/test_args.py
from webui.__main__ import parse_args


def test_defaults():
    a = parse_args([])
    assert a.host == "127.0.0.1" and a.port == 8000 and a.open is True


def test_no_open_and_port():
    a = parse_args(["--no-open", "--port", "9001"])
    assert a.open is False and a.port == 9001
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_args.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# webui/__main__.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui/test_args.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add webui/__main__.py tests/webui/test_args.py
git commit -m "feat(webui): python -m webui entry point"
```

---

### Task 9: Front end (app shell, three views, SSE)

**Files:**
- Create: `webui/static/index.html`, `webui/static/style.css`, `webui/static/app.js`

**Interfaces:**
- Consumes the routes from Task 7: `GET /api/config`, `GET /api/files`, `GET|POST /api/settings`, `POST /api/test/{backend}`, `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/stop`, `GET /api/jobs/{id}/events`, `GET /api/jobs/{id}/download?name=...`

This task is not TDD (no JS test framework); it ends with a manual browser smoke test.

- [ ] **Step 1: Write `index.html` (app shell + three views)**

```html
<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>speech-to-subtitle</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <header class="topbar">
    <div class="brand">speech-to-subtitle</div>
    <div class="status"><span id="gpu" class="badge">GPU: ...</span></div>
  </header>
  <div class="shell">
    <nav class="sidebar">
      <button data-view="run" class="nav active">Run</button>
      <button data-view="jobs" class="nav">Jobs</button>
      <button data-view="settings" class="nav">Settings</button>
      <a class="nav" href="https://github.com/vuhai2002/speech-to-subtitle" target="_blank">Docs</a>
    </nav>
    <main class="content">
      <section id="view-run" class="view"></section>
      <section id="view-jobs" class="view hidden"></section>
      <section id="view-settings" class="view hidden"></section>
    </main>
  </div>
  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Write `style.css` (modern dark theme matching the banner)**

```css
:root{--bg:#0b1220;--panel:#132537;--panel2:#0e2033;--line:#2a4a66;--text:#e2e8f0;
--muted:#94a3b8;--accent:#22d3ee;--accent2:#34d399;--danger:#f87171;}
*{box-sizing:border-box}body{margin:0;font-family:'Segoe UI',Roboto,Arial,sans-serif;background:var(--bg);color:var(--text)}
.topbar{display:flex;justify-content:space-between;align-items:center;padding:12px 20px;border-bottom:1px solid var(--line)}
.brand{font-weight:700;font-size:18px}
.badge{background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:4px 12px;font-size:13px;color:var(--muted)}
.shell{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 53px)}
.sidebar{border-right:1px solid var(--line);padding:16px 10px;display:flex;flex-direction:column;gap:6px}
.nav{background:none;border:none;color:var(--muted);text-align:left;padding:10px 14px;border-radius:10px;font-size:15px;cursor:pointer;text-decoration:none}
.nav:hover{background:var(--panel)}.nav.active{background:var(--panel);color:var(--text)}
.content{padding:24px 28px;max-width:900px}
.view.hidden{display:none}
h2{margin:0 0 16px}
.card{background:var(--panel2);border:1px solid var(--line);border-radius:14px;padding:18px;margin-bottom:16px}
label{display:block;font-size:13px;color:var(--muted);margin:10px 0 4px}
input,select,textarea{width:100%;background:var(--bg);border:1px solid var(--line);border-radius:10px;color:var(--text);padding:10px 12px;font-size:14px}
textarea{min-height:120px;font-family:ui-monospace,monospace}
.row{display:flex;gap:10px;flex-wrap:wrap}
.btn{background:linear-gradient(90deg,var(--accent),var(--accent2));color:#06263a;font-weight:700;border:none;border-radius:10px;padding:11px 18px;cursor:pointer}
.btn.ghost{background:var(--panel);color:var(--text);border:1px solid var(--line)}
.btn.stop{background:var(--danger);color:#3a0d0d}
.chip{display:inline-block;background:var(--panel);border:1px solid var(--line);border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px}
.chip.ok{border-color:var(--accent2);color:var(--accent2)}
.log{background:#06101c;border:1px solid var(--line);border-radius:10px;padding:12px;height:320px;overflow:auto;font-family:ui-monospace,monospace;font-size:12.5px;white-space:pre-wrap}
table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);font-size:14px}
.seg{display:flex;gap:6px}.seg button{flex:1}
.muted{color:var(--muted);font-size:13px}
```

- [ ] **Step 3: Write `app.js` (routing, forms, SSE, test buttons)**

```javascript
const $ = (s, r=document) => r.querySelector(s);
let CFG = {backends:{}, has_gpu:false};

async function boot(){
  CFG = await (await fetch('/api/config')).json();
  $('#gpu').textContent = CFG.has_gpu ? 'GPU: available' : 'GPU: none (MAI only for .srt)';
  document.querySelectorAll('.nav[data-view]').forEach(b => b.onclick = () => show(b.dataset.view));
  renderRun(); renderSettings(); show('run');
}
function show(v){
  document.querySelectorAll('.view').forEach(s => s.classList.add('hidden'));
  document.querySelectorAll('.nav[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view===v));
  $('#view-'+v).classList.remove('hidden');
  if(v==='jobs') renderJobs();
}
function backendOptions(){
  return Object.entries(CFG.backends).map(([k,b]) => `<option value="${k}">${b.label}</option>`).join('');
}
function renderRun(){
  $('#view-run').innerHTML = `
    <h2>Run</h2>
    <div class="card">
      <label>Audio folder on this machine</label>
      <div class="row"><input id="dir" placeholder="e.g. C:/audio"><button class="btn ghost" onclick="listFiles()">List</button></div>
      <div id="files" class="muted" style="margin-top:8px">Enter a folder and click List.</div>
    </div>
    <div class="card">
      <label>Backend</label>
      <select id="backend" onchange="onBackend()">${backendOptions()}</select>
      <div id="promptWrap"><label>Prompt override (optional)</label><textarea id="prompt" placeholder="Leave empty to use the default prompt"></textarea></div>
      <label>Output</label>
      <div class="seg"><label class="chip"><input type="radio" name="out" value="srt" checked> .srt</label>
      <label class="chip"><input type="radio" name="out" value="txt"> .txt</label></div>
      <div id="gate" class="muted"></div>
    </div>
    <button class="btn" onclick="startJob()">Run</button>`;
  onBackend();
}
async function listFiles(){
  const dir = $('#dir').value.trim();
  const r = await (await fetch('/api/files?dir='+encodeURIComponent(dir))).json();
  $('#files').innerHTML = r.files.length
    ? r.files.map(f => `<label style="display:block"><input type="radio" name="file" value="${f}"> ${f}</label>`).join('')
    : 'No audio files found.';
}
function onBackend(){
  const b = CFG.backends[$('#backend').value];
  $('#promptWrap').style.display = b.has_prompt ? 'block' : 'none';
  const srtOk = !b.needs_gpu_for_srt || CFG.has_gpu;
  $('#gate').textContent = srtOk ? '' : 'This backend needs a GPU for .srt. Without one, choose .txt or the MAI backend.';
  if(!srtOk){ document.querySelector('input[name=out][value=txt]').checked = true; }
}
async function startJob(){
  const file = document.querySelector('input[name=file]:checked');
  if(!file) return alert('Pick an audio file first.');
  const body = {backend:$('#backend').value, output:document.querySelector('input[name=out]:checked').value,
                source:file.value, prompt:$('#prompt') ? $('#prompt').value : ''};
  const r = await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){ return alert((await r.json()).error); }
  const {id} = await r.json();
  show('jobs'); openJob(id);
}
async function renderJobs(){
  const {jobs} = await (await fetch('/api/jobs')).json();
  $('#view-jobs').innerHTML = `<h2>Jobs</h2><div class="card"><table><thead><tr>
    <th>Status</th><th>Backend</th><th>Output</th><th>File</th></tr></thead><tbody>
    ${jobs.map(j => `<tr onclick="openJob('${j.id}')" style="cursor:pointer">
      <td><span class="chip ${j.status==='done'?'ok':''}">${j.status}</span></td>
      <td>${j.backend}</td><td>${j.output}</td><td>${(j.source||'').split(/[\\/]/).pop()}</td></tr>`).join('')}
    </tbody></table></div><div id="jobDetail"></div>`;
}
function openJob(id){
  $('#jobDetail').innerHTML = `<div class="card"><div class="row" style="justify-content:space-between">
    <div id="jstage" class="muted">stage: ...</div><button class="btn stop" onclick="stopJob('${id}')">Stop</button></div>
    <div id="results"></div><div class="log" id="log"></div></div>`;
  const es = new EventSource('/api/jobs/'+id+'/events');
  const log = $('#log');
  es.addEventListener('log', e => { log.textContent += JSON.parse(e.data)+'\n'; log.scrollTop = log.scrollHeight; });
  es.addEventListener('status', async e => {
    const s = JSON.parse(e.data);
    const j = await (await fetch('/api/jobs/'+id)).json();
    $('#jstage').textContent = 'stage: '+(j.stage||s)+'  ('+j.status+')';
    if(['done','error','stopped'].includes(j.status)){ es.close(); showResults(id, j); }
  });
}
function showResults(id, j){
  const files = (j.results && j.results.files) || [];
  $('#results').innerHTML = files.map(f =>
    `<a class="btn ghost" href="/api/jobs/${id}/download?name=${encodeURIComponent(f.split(/[\\/]/).slice(1).join('/'))}">Download ${f.split(/[\\/]/).pop()}</a>`).join(' ');
}
async function stopJob(id){ await fetch('/api/jobs/'+id+'/stop',{method:'POST'}); }
function renderSettings(){
  $('#view-settings').innerHTML = '<h2>Settings</h2>' + Object.entries(CFG.backends).map(([k,b]) => `
    <div class="card"><h3>${b.label}</h3>
    ${b.fields.map(f => `<label>${f.label}</label><input data-key="${f.key}" type="${f.secret?'password':'text'}">`).join('')}
    <div class="row" style="margin-top:12px"><button class="btn" onclick="saveBackend('${k}')">Save</button>
    <button class="btn ghost" onclick="testBackend('${k}')">Test</button><span id="test-${k}" class="muted"></span></div></div>`).join('');
  loadSettings();
}
async function loadSettings(){
  const {values} = await (await fetch('/api/settings')).json();
  document.querySelectorAll('#view-settings input[data-key]').forEach(i => { if(values[i.dataset.key]) i.placeholder = values[i.dataset.key]; });
}
function collect(k){
  const o = {};
  document.querySelectorAll(`#view-settings .card`)[Object.keys(CFG.backends).indexOf(k)]
    .querySelectorAll('input[data-key]').forEach(i => { if(i.value) o[i.dataset.key] = i.value; });
  return o;
}
async function saveBackend(k){
  await fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect(k))});
  $('#test-'+k).textContent = 'saved';
}
async function testBackend(k){
  $('#test-'+k).textContent = 'testing...';
  const r = await (await fetch('/api/test/'+k,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collect(k))})).json();
  $('#test-'+k).textContent = (r.ok?'OK - ':'FAIL - ')+r.message;
}
boot();
```

- [ ] **Step 4: Manual smoke test**

Run: `.venv-realign/Scripts/python.exe -m webui --no-open`
Then open `http://127.0.0.1:8000` in a browser and verify: the app shell renders (sidebar + top bar + GPU badge), the three views switch, Settings shows the three backend cards, and entering a folder + List shows audio files. (A full run is exercised in Task 10.)

- [ ] **Step 5: Commit**

```bash
git add webui/static/index.html webui/static/style.css webui/static/app.js
git commit -m "feat(webui): app-shell front end with Run, Jobs, Settings views and SSE"
```

---

### Task 10: Packaging, docs, and end-to-end smoke

**Files:**
- Create: `webui/README.md`
- Modify: `.env.example` (add `WEBUI_HOST` / `WEBUI_PORT`)
- Modify: `README.md` (add a "Web UI (optional)" section)

- [ ] **Step 1: Add env vars to `.env.example`**

Append:

```
# === WEB UI (webui, optional) ===
# Host/port for the local UI (python -m webui). Defaults: 127.0.0.1 : 8000
WEBUI_HOST=
WEBUI_PORT=
```

- [ ] **Step 2: Write `webui/README.md`**

```markdown
# webui

Local browser UI for the pipeline. Pick an audio file, choose a backend, run, watch progress,
and download `.srt`/`.txt`. It orchestrates the existing CLIs as subprocesses; it does not
reimplement any pipeline logic. See `docs/webui-design.md`.

## Run

    pip install -r requirements-webui.txt
    python -m webui           # opens http://127.0.0.1:8000

Bind stays on 127.0.0.1 (local only, no auth). API keys are entered in Settings and saved to
`.env` (gitignored). Runtime state and outputs live under `out/webui/`.

## Notes
- `.srt` from the Router/Vertex backends needs an NVIDIA GPU (MMS alignment). MAI produces
  `.srt` from native timestamps without a GPU.
- Prompt override applies to the Gemini backends (Vertex, Router); MAI is a pure ASR endpoint.
```

- [ ] **Step 3: Add a section to the root `README.md`**

Insert after the Quickstart section:

```markdown
## Web UI (optional)

A local browser UI wraps the pipeline (pick audio, choose backend, run, download subtitles):

    pip install -r requirements-webui.txt
    python -m webui

It runs on 127.0.0.1 only. See `webui/README.md` and `docs/webui-design.md`.
```

- [ ] **Step 4: Full test suite + end-to-end smoke**

Run the webui unit tests:

Run: `.venv-realign/Scripts/python.exe -m pytest tests/webui -q`
Expected: all PASS.

End-to-end (uses a real MAI key in `.env`, ~$0.002): start `python -m webui --no-open`, open the browser, Settings -> MAI -> paste key -> Test (expect OK), Run -> pick a short clip -> output `.srt` -> watch the log reach "done" -> Download the `.srt`. Confirm the file opens with correct cues.

- [ ] **Step 5: Commit**

```bash
git add webui/README.md .env.example README.md
git commit -m "docs(webui): package README, root README section, env vars"
```

---

## Self-Review

**Spec coverage:** app shell + 3 views (Task 9); subprocess execution (Task 4); backend x output matrix + gating (Tasks 1, 6); prompt override via `--prompt-file` (Task 5); Test probes (Task 6); job model + SSE + history under `out/webui/` (Tasks 3, 4, 7); settings in `.env` (Task 2); deps in `requirements-webui.txt` + `python -m webui` + `WEBUI_HOST/PORT` (Tasks 7, 8, 10); loopback-only + no new keys (Tasks 7, 8); testing plan (Tasks 1-8). All spec sections map to a task.

**Placeholder scan:** no "TBD"/"add error handling"-style steps; every code step has real code. The one illustrative stray line in Task 7's `start_job` is called out and replaced by `start_from`.

**Type consistency:** `Step(label, argv, env)` used consistently (Tasks 1, 4, 7); `build_steps(...)` signature matches its call in `JobManager.start_from` (Tasks 1, 7); `JobManager.subscribe` yields `(kind, payload)` consumed identically by the SSE route (Tasks 4, 7); `load_prompt(prompt_file=None)` defined in Task 5 and relied on nowhere earlier.
