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
