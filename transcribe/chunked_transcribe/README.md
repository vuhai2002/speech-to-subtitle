# chunked_transcribe

Transcribes long audio into text by **splitting it into roughly 10-minute chunks** and calling a Gemini model through an OpenAI-compatible router (9router, the Antigravity path). It adds an **audio-anchored self-check** to catch omissions and hallucinated words, plus a **two-pass comparison** step to catch wrong-word errors as well.

> For the design rationale, empirical measurements, dead-end approaches that were ruled out, and batch operations notes (quota/rate limits): see `docs/chunked-transcribe-notes.md`.

## This is an experimental path, not the official step 1

The official step 1 of the repo is `transcribe/batch_transcribe_vertex.py` (Gemini via **Vertex**, billed to Google Cloud). This package goes through **9router/Antigravity** using an OAuth session, with two things to be aware of:

- **Account risk:** this is a subscription login session routed through a proxy, and Google may ban it (we already hit one account that got a blanket 403). Use a secondary account and treat it as disposable at any time.
- **Why chunking is required:** sending a whole long file (roughly 25 minutes or more) makes the model **silently drop** a segment without raising an error (verified: an 88-minute file and a 27-minute chunk both dropped the same ~85-second segment; 10-minute chunks fixed it). Sending a whole file as base64 also exceeds Cloudflare's 100 MB body limit.

## Why there is a self-check step

The model can **omit** a segment or **hallucinate** words while the transcript still reads fluently, so inspecting the text alone won't catch it. Neither the MMS alignment score nor the fraction of aligned words catches omissions. The way to catch them is to **anchor to the audio**:

- `quality_check`: forced alignment (MMS) with inserted `*` tokens (placeholders for audio that has no matching text). Any `*` cell that contains several seconds of speech (per VAD) but no text -> suspected **omission**. Any word that lands in a VAD silence -> suspected **hallucination**.
- `compare_passes`: transcribe twice (two runs or two models) and diff the strings. Where the two passes agree, trust it; only the differing spots need a listen. This method also reaches **wrong-word errors** that MMS is blind to (a wrong word still matches the sound).

Neither is a guarantee. They narrow down what you have to listen to; they don't replace listening.

## Running

Required environment variables (do not hardcode; see `.env.example` at the repo root):

```
ROUTER_BASE_URL   e.g. https://<router-host>/v1
ROUTER_API_KEY    router api key
TRANSCRIBE_MODEL  optional, default ag/gemini-3.8-flash
ROUTER_WORKERS    optional, number of chunks sent in parallel, default 3
```

Chunks are sent **in parallel** (default 3 workers, change with `--workers` or `ROUTER_WORKERS`). Each chunk **retries up to 3 times** on error, empty response, `finish != stop`, or too-low character density, with a pause between attempts. On a **403** (banned account) that chunk stops without retrying, and the whole file is marked "incomplete" so it can be rerun later (rather than crashing the whole batch).

Run from the repo root via `.venv-realign` (which has torch + silero + MMS, like realign):

```bash
.venv-realign/Scripts/python.exe -m transcribe.chunked_transcribe.run_pipeline --input "audio.m4a" --out-dir out-ct
```

```bash
.venv-realign/Scripts/python.exe -m transcribe.chunked_transcribe.quality_check --out-dir out-ct
```

Produce the `.srt` (per-chunk alignment + realign cue grouping, NOT whole-file window_align). Files are named after the original audio:

```bash
.venv-realign/Scripts/python.exe -m transcribe.chunked_transcribe.build_srt --out-dir out-ct
```

Transcribe a second pass (different model), then compare the two passes and export an HTML view:

```bash
.venv-realign/Scripts/python.exe -m transcribe.chunked_transcribe.run_pipeline --input "audio.m4a" --out-dir out-ct-pro --model ag/gemini-3.1-pro-low
```

```bash
.venv-realign/Scripts/python.exe -m transcribe.chunked_transcribe.compare_passes --a out-ct --b out-ct-pro
```

```bash
.venv-realign/Scripts/python.exe -m transcribe.chunked_transcribe.render_compare_html --a out-ct --b out-ct-pro --out compare.html --a-name "3.8-flash" --b-name "3.1-pro"
```

## Output in `--out-dir`

| File | Contents |
|---|---|
| `raw_transcript.txt` | Joined transcript (chunks with no speech dropped) |
| `<audio name>.srt` | Subtitles (after running `build_srt`) |
| `manifest.json` | Per-chunk status: timing, word count, number of attempts, finish reason, joined or dropped |
| `qc_report.json` | Self-check results: omissions, needs-a-listen, hallucinations, edge repeats |
| `plan.json` | Chunk plan + speech regions (VAD) |
| `mono16k.mp3`, `chunks/` | Mono 16 kHz audio and the individual chunks |

## Parameters

All in `config.py`, adjustable via environment variables or by editing the file. Notable ones:

- `CHUNK_TARGET_SEC = 600` (roughly 10-minute chunks, cut at a silence).
- `MIN_SPEECH_CHUNK_SEC = 30`: chunks with less speech than this are not joined (avoids the model hallucinating over music/silence).
- `OMISSION_FLAG_SEC = 5` / `OMISSION_REVIEW_SEC = 2`: thresholds for flagging an omission and for needing a listen.

## About the model

`ag/gemini-3.8-flash` (default) is in fact `ag/gemini-3.8-flash-medium` (same base model, medium thinking level). It transcribed a full 100-minute file, 10/10 chunks on the first attempt. `ag/gemini-3.1-pro-low` is slightly better on the Vietnamese FLEURS benchmark but slower; on real recordings we saw no clear difference.

## Known limitations

- Only verified on 2 recordings (single speaker, clean audio). Not yet tried on multi-speaker recordings, long chanting, or noisy audio.
- The self-check catches omissions and hallucinations, but **not** wrong-word errors (use `compare_passes` for those).
- Where both passes make the same mistake, `compare_passes` can't see it either.
- The transcript has **no timestamps**. Producing subtitles still requires the `realign` step, and you should align **per chunk** and add the offset rather than using `window_align` on the whole file (it assigns wrong times when the end of the file has a long non-speech segment).
