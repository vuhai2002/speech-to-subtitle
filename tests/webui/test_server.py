import os
from pathlib import Path

from fastapi.testclient import TestClient
from webui.server import create_app
from webui.jobs import Job, JobManager


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


def test_download_and_traversal(tmp_path):
    # Seed a finished job directly (no need to run a real subprocess): register it with a
    # JobManager pointed at the state dir, then create the app so it loads that same job.
    state = tmp_path / "state"
    jm = JobManager(str(state))
    j = Job(id="abcd1234", backend="mai", output="txt", source="a.m4a", out_dir=str(state / "abcd1234"))
    Path(j.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(j.out_dir) / "out.srt").write_text("SUB", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("TOP", encoding="utf-8")
    jm._jobs[j.id] = j
    jm._save()

    c = _client(tmp_path)
    ok = c.get(f"/api/jobs/{j.id}/download", params={"name": "out.srt"})
    assert ok.status_code == 200 and ok.text == "SUB"

    traversal = c.get(f"/api/jobs/{j.id}/download", params={"name": "../../secret.txt"})
    assert traversal.status_code == 404

    missing_job = c.get("/api/jobs/doesnotexist/download", params={"name": "out.srt"})
    assert missing_job.status_code == 404

    missing_job_events = c.get("/api/jobs/doesnotexist/events")
    assert missing_job_events.status_code == 404


def test_upload_saves_file(tmp_path):
    c = _client(tmp_path)
    r = c.post("/api/upload", files={"file": ("clip.wav", b"RIFFdata", "audio/wav")})
    assert r.status_code == 200
    d = r.json()
    assert d["name"] == "clip.wav"
    assert Path(d["path"]).is_file()
    assert Path(d["path"]).read_bytes() == b"RIFFdata"


def test_env_loaded_into_process(tmp_path):
    # MAI/Router subprocesses read their keys via os.getenv(...) from the process
    # environment, not from .env directly. create_app must load .env into os.environ so a
    # key saved in Settings actually reaches those subprocesses; distinctive values here so
    # this doesn't depend on (or get masked by) whatever is already in the real environment.
    env_path = tmp_path / ".env"
    env_path.write_text("OPENROUTER_API_KEY=sk-bridgetest\n", encoding="utf-8")
    try:
        c = TestClient(create_app(str(tmp_path / "state"), str(env_path)))
        assert os.environ.get("OPENROUTER_API_KEY") == "sk-bridgetest"

        r = c.post("/api/settings", json={"MAI_MODEL": "microsoft/mai-transcribe-2"})
        assert r.status_code == 200
        assert os.environ.get("MAI_MODEL") == "microsoft/mai-transcribe-2"
    finally:
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("MAI_MODEL", None)
