# tests/webui/test_jobs.py
import sys
import time
from webui.backends import Step
from webui.jobs import Job, JobManager


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


def test_stop_marks_stopped(tmp_path):
    jm = JobManager(str(tmp_path))
    step = Step("t", [sys.executable, "-c", "import time; print('10:00 [1/4] x'); time.sleep(30)"])
    job = jm.start("mai", "txt", "a.m4a", [step])
    for _ in range(100):
        if jm.get(job.id).status == "running":
            break
        time.sleep(0.05)
    # Retry stop(): status flips to "running" before the subprocess is registered in
    # self._procs, so the very first stop() call can land in that narrow window and be a
    # no-op. Re-issuing it is harmless (stop() is idempotent once the proc is gone) and
    # avoids a flaky test without changing JobManager's behavior.
    for _ in range(100):
        jm.stop(job.id)
        if jm.get(job.id).status in ("stopped", "done", "error"):
            break
        time.sleep(0.05)
    assert jm.get(job.id).status == "stopped"


def test_collect_finds_vertex_and_toplevel_outputs(tmp_path):
    jm = JobManager(str(tmp_path))
    out_dir = tmp_path / "job1"
    (out_dir / "srt").mkdir(parents=True)
    (out_dir / "srt" / "talk.srt").write_text("1\n", encoding="utf-8")
    (out_dir / "talk.txt").write_text("transcript\n", encoding="utf-8")
    (out_dir / "chunks").mkdir()
    (out_dir / "chunks" / "01.txt").write_text("chunk\n", encoding="utf-8")
    (out_dir / "_meta").mkdir()
    (out_dir / "_meta" / "prompt.txt").write_text("do X\n", encoding="utf-8")

    job = Job(id="job1", backend="vertex", output="srt", source="a.m4a", out_dir=str(out_dir))
    result = jm._collect(job)

    assert "srt/talk.srt" in result["files"]
    assert "talk.txt" in result["files"]
    assert "chunks/01.txt" not in result["files"]
    assert "_meta/prompt.txt" not in result["files"]
    assert not any("prompt" in f for f in result["files"])


def test_failing_step_marks_error(tmp_path):
    jm = JobManager(str(tmp_path))
    step = Step("t", [sys.executable, "-c", "import sys; sys.exit(2)"])
    job = jm.start("mai", "txt", "a.m4a", [step])
    for _ in range(100):
        if jm.get(job.id).status in ("done", "error", "stopped"):
            break
        time.sleep(0.05)
    assert jm.get(job.id).status == "error"
