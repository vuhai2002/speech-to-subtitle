from transcribe.chunked_transcribe import config as ct_config


def test_load_prompt_default_nonempty():
    assert len(ct_config.load_prompt()) > 20            # falls back to the Vertex script prompt


def test_load_prompt_from_file(tmp_path):
    p = tmp_path / "prompt.txt"
    p.write_text("CUSTOM PROMPT", encoding="utf-8")
    assert ct_config.load_prompt(str(p)) == "CUSTOM PROMPT"
