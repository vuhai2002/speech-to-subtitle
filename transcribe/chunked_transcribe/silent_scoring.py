"""Keep or drop the sentences of a chunk VAD hears as silent, by how well MMS aligns them to the audio.

Over music or near-silence Gemini may write real words (closing words, an MC, chanting, lyrics) or invent
text, and VAD cannot tell which. build_srt aligns the chunk's words with MMS and keeps only the sentences that
fit the audio well. A sentence's score is the mean MMS score of its aligned words (realign.align_words gives
each word the mean of its token scores) - the measure behind the 0.45 / 0.5 thresholds the admin set by ear.
"""
import re

MAX_WORDS = 40
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+")


def split_sentences(text: str) -> list[list[str]]:
    """Sentences at . ! ? ; followed by whitespace; a sentence longer than MAX_WORDS is cut into pieces of at
    most MAX_WORDS words. Joining the pieces gives back text.split() exactly."""
    out = []
    for piece in _SENTENCE_END.split(text.strip()):
        ws = piece.split()
        for i in range(0, len(ws), MAX_WORDS):
            out.append(ws[i:i + MAX_WORDS])
    return [s for s in out if s]


def score_sentences(sentences: list[list[str]], aligned: list[dict], threshold: float) -> list[dict]:
    """One result per sentence: {"words", "text", "start", "end", "score", "kept"}.

    `aligned` is MMS output for the flattened sentence words, one entry per word in order (start=None when a
    word could not be aligned). The score is rounded to 3 decimals and THAT value is compared with the
    threshold, so what the trace shows is what was judged. A sentence with no aligned word has score None and
    is dropped. Raises ValueError when the counts differ: a silent mismatch would score every later sentence
    on the wrong words.
    """
    expected = sum(len(s) for s in sentences)
    if len(aligned) != expected:
        raise ValueError(f"score_sentences: {len(aligned)} aligned words for {expected} sentence words")
    out, i = [], 0
    for s in sentences:
        ws = aligned[i:i + len(s)]
        i += len(s)
        timed = [w for w in ws if w.get("start") is not None]
        score = round(sum(w.get("score") or 0.0 for w in timed) / len(timed), 3) if timed else None
        out.append({"words": ws, "text": " ".join(s),
                    "start": timed[0]["start"] if timed else None, "end": timed[-1]["end"] if timed else None,
                    "score": score, "kept": score is not None and score >= threshold})
    return out
