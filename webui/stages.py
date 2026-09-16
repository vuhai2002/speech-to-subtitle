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
