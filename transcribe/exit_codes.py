"""Exit codes shared with the desktop app, which turns them into a readable job failure.

0  done.
3  INCOMPLETE: at least one chunk failed every retry or hit a 403. manifest.json is still written
   and names the chunks; the transcript has a hole and must not become a subtitle.
4  NO_ALIGNED_WORDS: no word got a timestamp / no cue could be built / (Router build_srt) a merged chunk with
   speech has no word MMS can align. No .srt is written.
5  AUDIO_CANCELLED: the mono mix still cancels the voice after the polarity flip (polarity.json names the
   stretches). Nothing is transcribed.
"""
OK = 0
INCOMPLETE = 3
NO_ALIGNED_WORDS = 4
AUDIO_CANCELLED = 5


class NoAlignedWords(RuntimeError):
    """build() found nothing to write. Raised before any .srt file is created."""
