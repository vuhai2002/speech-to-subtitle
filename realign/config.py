"""Re-align pipeline parameters - production standard.
See plans .../reports/subtitle-timing-standards-research.md for the source of each number.
"""
# cue text
CPL_MAX = 42                      # chars / line (counted after NFC, by grapheme)
LINES_MAX = 2
CHAR_MAX = CPL_MAX * LINES_MAX    # 84 - reference only; actual cue-break rule uses _fits_lines by CPL (see cue_builder)

# reading speed
CPS_MAX = 15.0                    # hard ceiling
CPS_TARGET = 13.0                 # design target

# duration (seconds)
DUR_MIN = 1.5
DUR_MAX = 7.0
FLOOR = 0.3                       # absolute floor

# segmentation
PAUSE_SPLIT = 0.6                 # medium pause >= -> PREFERRED cue-break point (soft, does NOT force a split)
LONG_PAUSE = 2.0                  # long pause >= -> hard segment split (like sentence end)
ISOLATION_GAP = 8.0               # first/last cue of 1-2 words separated from neighbor by >= this threshold -> boundary misalignment, merge back
SENTENCE_END = ".!?:;"            # word-final char -> sentence break

# gap between two cues
GAP_MIN = 0.084                   # 2 frames @24fps (anti-flicker)
PAUSE_GAP = 0.5                   # < -> butt together; >= -> keep as is

# VAD
VAD_PAD = 0.3
VAD_TRUST_GAP = 2.0               # VAD deviates from alignment > threshold -> trust alignment

# audio / model
SAMPLE_RATE = 16000
EMIT_CHUNK_SEC = 20.0

# paths (overridable via CLI run_batch --audio-dirs)
AUDIO_DIRS = [r"Y:\run-script-1", r"Y:\run-script-2", r"Y:\run-script-3"]
