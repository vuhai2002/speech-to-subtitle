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
