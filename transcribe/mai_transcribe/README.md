# mai_transcribe

Transcribes long audio using **MAI-Transcribe-2** (Microsoft) through OpenRouter's transcription endpoint. It splits into roughly 10-minute chunks, calls them in parallel, then builds the `.srt` **directly from MAI's per-word timestamps** (no MMS forced alignment, no GPU needed).

> For why this approach was chosen, empirical measurements against the Gemini+MMS path, and operational notes: see `docs/mai-transcribe-notes.md`.

## This is an experimental / fallback path

The official step 1 of the repo is still `transcribe/batch_transcribe_vertex.py` (Gemini via Vertex) + `realign/` (MMS). This package is a verified alternative:

- **Different from the 9router path (`chunked_transcribe`):** MAI is a dedicated ASR that **bills per second of audio** (~$0.10/hour), is fully legitimate, and carries **no account-ban risk** like the Antigravity OAuth session.
- **Skips the MMS step:** MAI returns per-word timestamps (with punctuation), so `build_srt` groups cues directly, with no forced alignment on GPU.
- **Still requires chunking:** OpenRouter limits requests to 25 MB (about 13 minutes of WAV; mono-16k mp3 is well under), and chunking also guards against the silent omissions seen on the Gemini path.

## Running

Required environment variables (do not hardcode; see `.env.example` at the repo root):

```
OPENROUTER_API_KEY   OpenRouter api key (format sk-or-v1-...)
MAI_MODEL            optional, default microsoft/mai-transcribe-2
MAI_LANGUAGE         optional, default vi
MAI_WORKERS          optional, number of chunks sent in parallel, default 3
```

Transcribe (from the repo root). We reuse `.venv` for convenience, but this step **does not need GPU/torch** - it only needs `ffmpeg` and `silero-vad` for chunking:

```bash
.venv/Scripts/python.exe -m transcribe.mai_transcribe.run_pipeline --input "audio.m4a" --out-dir out-mai
```

Build the `.srt` from MAI's native timestamps (also GPU-free):

```bash
.venv/Scripts/python.exe -m transcribe.mai_transcribe.build_srt --out-dir out-mai
```

To cross-check the text against another transcript (e.g. the Gemini pass), reuse the tooling from `chunked_transcribe`:

```bash
.venv/Scripts/python.exe -m transcribe.chunked_transcribe.compare_passes --a out-gemini --b out-mai
```

## Output in `--out-dir`

| File | Contents |
|---|---|
| `raw_transcript.txt` | Joined transcript (chunks with no speech dropped) |
| `mai_words.json` | Per-word timestamps for the whole file `[{w,start,end,score}]` with chunk offsets applied - input for `build_srt` |
| `<audio name>.srt` | Subtitles (after running `build_srt`) |
| `manifest.json` | Per-chunk status: word count, cost, joined or dropped |
| `plan.json` | Chunk plan + speech regions (VAD) |
| `mono16k.mp3`, `chunks/` | Mono 16 kHz audio, the individual mp3 chunks, `NN.json` (raw MAI JSON), `NN.txt` |

## Parameters

In `config.py` (adjustable via environment variables). The **chunking** part reuses `chunked_transcribe.audio_utils` + `chunked_transcribe.config` (a single source), so constants like `CHUNK_TARGET_SEC` are edited there.

## Known limitations

- Verified on a single-speaker, clean-audio recording (97% agreement with Gemini, equivalent SRT). **Not yet tried** on chanting, multi-speaker recordings, or noisy audio.
- If a word is transcribed wrong, the timestamp is still present, so it's still worth running `compare_passes` against an independent pass when you need certainty.
- Verbatim: MAI captures filler words ("uh", "ah") and repeated words. If you want cleaner subtitles, consider adjusting the style at the model layer (not done here).
