# tests/webui/test_settings.py
from webui.settings import read_env, write_env


def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\nEXISTING=1\nOPENROUTER_API_KEY=old\n", encoding="utf-8")
    write_env(str(p), {"OPENROUTER_API_KEY": "sk-new", "MAI_MODEL": "microsoft/mai-transcribe-2"})
    env = read_env(str(p))
    assert env["OPENROUTER_API_KEY"] == "sk-new"
    assert env["MAI_MODEL"] == "microsoft/mai-transcribe-2"
    assert env["EXISTING"] == "1"                 # untouched
    assert "# comment" in p.read_text(encoding="utf-8")   # comment preserved
