"""Run a chain of subprocess Steps for one job, capture stdout, stream events, persist history.

One job runs at a time in the first version, but each is tracked and stored so the Jobs view
survives a restart. State lives under a gitignored directory (out/webui by default).
"""
from __future__ import annotations

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
