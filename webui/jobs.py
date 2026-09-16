"""Run a chain of subprocess Steps for one job, capture stdout, stream events, persist history.

One job runs at a time in the first version, but each is tracked and stored so the Jobs view
survives a restart. State lives under a gitignored directory (out/webui by default).
"""
from __future__ import annotations

import json
import os
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
        self._stopped: set[str] = set()      # job ids stopped via stop(), so _run can tell
                                              # a user-requested stop apart from a real failure
        self._load()

    # ---- history persistence ----
    def _index(self) -> Path:
        return self.state_dir / "jobs.json"

    def _load(self) -> None:
        if not self._index().exists():
            return
        try:
            rows = json.loads(self._index().read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, ValueError):
            return                            # corrupt history: start fresh instead of raising
        for d in rows:
            self._jobs[d["id"]] = Job(**d)

    def _save(self) -> None:
        # Write to a temp file then atomically replace, so a crash mid-write never leaves
        # jobs.json half-written (which would make the next _load fail).
        rows = [asdict(j) for j in sorted(self._jobs.values(), key=lambda x: x.created, reverse=True)]
        index = self._index()
        tmp = index.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, index)

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
        job_id = uuid.uuid4().hex[:8]
        job = Job(id=job_id, backend=backend, output=output, source=source,
                  out_dir=str(self.state_dir / job_id))       # dir derivable from the job id
        self._jobs[job.id] = job
        self._save()
        threading.Thread(target=self._run, args=(job, steps), daemon=True).start()
        return job

    def stop(self, job_id: str) -> None:
        p = self._procs.get(job_id)
        if p and p.poll() is None:
            self._stopped.add(job_id)        # record intent before terminate() races _run()
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
                if job.id in self._stopped:
                    # Explicit stop request: do not trust the returncode sign to mean "killed"
                    # (terminate() gives 1 on Windows, not a negative signal number like POSIX).
                    job.status = "stopped"
                    break
                if proc.returncode != 0:
                    job.status = "error"
                    break
            else:
                job.status = "done"
        except Exception as e:                            # noqa: BLE001 - surface any launch failure
            job.log.append(f"[webui] {e}")
            job.status = "error"
        finally:
            self._procs.pop(job.id, None)
            self._stopped.discard(job.id)
            job.results = self._collect(job)
            self._save()
            self._emit(job, "status", job.status)
            self._emit(job, "log", None)                  # sentinel: stream closed

    def _env(self, step: Step) -> dict:
        return {**os.environ, **step.env}

    def _collect(self, job: Job) -> dict:
        # Names are paths relative to out_dir (posix separators): the download route joins
        # this directly onto job.out_dir with an is_relative_to containment check, so a
        # relative subpath like "srt/x.srt" resolves safely there.
        out = Path(job.out_dir)
        if not out.exists():
            return {"files": []}
        # .srt collected recursively: Vertex writes to out_dir/srt/<stem>.srt, other
        # backends write out_dir/<stem>.srt directly.
        files = [str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*.srt")]
        # .txt collected only at the top level (raw_transcript.txt / <stem>.txt), so
        # per-chunk transcripts under chunks/ and the prompt override under _meta/ are
        # never surfaced as downloadable results.
        files += [p.name for p in out.glob("*.txt")]
        return {"files": sorted(set(files))}

    def start_from(self, backend, output, source, has_gpu, model, workers, prompt) -> Job:
        """Build the step chain for one run and start it. Kept on JobManager (rather than
        in the server route) so the server stays a thin HTTP layer over webui.* logic."""
        from .backends import build_steps
        job_id = uuid.uuid4().hex[:8]
        out_dir = str(self.state_dir / job_id)
        prompt_file = None
        if prompt:
            # Keep the prompt out of out_dir's top level: for Vertex, out_dir doubles as
            # the realign --txt-dir, so a stray prompt.txt there would be picked up by
            # run_batch as a transcript, and by _collect's top-level *.txt glob.
            meta_dir = Path(out_dir) / "_meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            prompt_file = str(meta_dir / "prompt.txt")
            Path(prompt_file).write_text(prompt, encoding="utf-8")
        steps = build_steps(backend, output, source, out_dir, model=model, workers=workers, prompt_file=prompt_file)
        job = Job(id=job_id, backend=backend, output=output, source=source, out_dir=out_dir)
        self._jobs[job.id] = job
        self._save()
        threading.Thread(target=self._run, args=(job, steps), daemon=True).start()
        return job
