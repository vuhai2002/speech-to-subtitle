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
