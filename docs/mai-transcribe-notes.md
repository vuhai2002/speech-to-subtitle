# mai_transcribe - design notes and empirical results

A record of why the `transcribe/mai_transcribe/` package exists, the actual measurements against the Gemini+MMS path, and how to call the API correctly. Goal: so we don't have to re-test next time we need it. Read alongside `transcribe/mai_transcribe/README.md` (how to run) and `docs/chunked-transcribe-notes.md` (the 9router path, which shares the chunking).

## Context

MAI-Transcribe-2 is Microsoft's ASR (Azure Fast Transcription, enhancedMode), supporting about 60 languages including Vietnamese, returning per-word timestamps, at a very low price. It's available on OpenRouter through a dedicated transcription endpoint. Tried because: (1) it bills per second of audio, is legitimate, and carries **no account-ban risk** like the Antigravity OAuth session; (2) it returns per-word timestamps, so it can **skip the MMS step** (no GPU needed).

## Conclusion (single-speaker, clean-audio recording)

MAI is good enough to serve as a **fallback transcription + timing path**, with quality equivalent to Gemini+MMS. It's also a cross-check: an independent, dedicated ASR produces nearly the same result, confirming that the repo's Gemini+realign flow is sound.

## How to call the API correctly

- **Dedicated transcription endpoint**: `POST https://openrouter.ai/api/v1/audio/transcriptions`, multipart. **Do NOT** use `chat/completions`: chat bills base64 as text tokens, roughly 40x more expensive (same reason OpenRouter chat was ruled out in `chunked-transcribe-notes.md`).
- Fields: `file=@chunk.mp3`, `model=microsoft/mai-transcribe-2`, `response_format=verbose_json`, `timestamp_granularities[]=word`, `language=vi`.
- Returns `verbose_json`: `text` (the full transcript), `words[]` = `{word,start,end}` (timestamps in seconds, **with punctuation** in `word`, e.g. `"Phật."`, `"kiến,"`), `segments[]` (one segment for the whole chunk), `usage` = `{seconds, cost}`.
- **Limit of 25 MB per request** (about 13 minutes of WAV). Mono-16k mp3 is about 4 MB per 10 minutes, so there's plenty of room; still split into 10-minute chunks to stay consistent with the 9router path and to guard against omissions.

## Design decisions and why

- **Reuse `chunked_transcribe.audio_utils` for chunking** (ffmpeg mono16k + VAD + cut plan). A single source of truth for chunking; MAI only swaps the transcription backend and the timing source.
- **Build the `.srt` directly from MAI's native timestamps, dropping MMS entirely.** MAI returns per-word timestamps with punctuation, fed straight into `realign.cue_builder` (a `.!?:;` at the end of a word marks a sentence break; already present in the MAI transcript). No forced alignment on GPU needed. This package's `build_srt` does not need torch.
- **Drop chunks with almost no speech** (reusing the `MIN_SPEECH_CHUNK_SEC` threshold from `chunked_transcribe`), to avoid the model hallucinating over music/silence.

## Empirical measurements

File "Chánh Tinh Tấn - Phương Pháp Thiền Định - Phần 1" (106 minutes, 11 chunks), `microsoft/mai-transcribe-2`, 3 parallel workers:

| Measure | Result |
|---|---|
| Transcribe whole file (11 chunks) | **32 seconds** (3 parallel workers) |
| Cost for whole file | **$0.1778** (exactly $0.10/hour) |
| Word count | 21,069 (Gemini: 21,136, 0.3% difference) |
| Words missing a timestamp | 0 |
| Auto-detected language | vi |

The $0.10/hour price is confirmed by `usage.cost`: a 60-second clip was billed $0.001667 = $0.10/hour.

### Text cross-check against Gemini 3.8-flash (`compare_passes`)

Word-for-word agreement **97.0%**. In 106 minutes, only **4 spots differing by 4+ words**, and **none of them an omission** (all "different words"). The per-chunk word counts differ by under 1% -> MAI does not silently omit. All four differing spots are in hard-to-hear passages / repeated words:

| Timestamp | Gemini | MAI |
|---|---|---|
| 16:29 | lãng lãng lãng lãng | lảng lảng lảng lảng (only differs in hỏi/ngã tone) |
| 52:25 | thấy sao kỳ | mới sau kì hôm |
| 99:44 | nhật kim ngộ nha | nhặt chiếu nhội hay (both unclear) |
| 102:28 | dòm dòm dòm dòm | vòng vòng vòng vòng |

### SRT quality: MAI native vs Gemini+MMS

Both fed through `realign.cue_builder` (one using MMS timestamps, the other MAI timestamps):

| Measure | Gemini + MMS | MAI native |
|---|---|---|
| Cue count | 1731 | 1746 |
| Last cue ends | 106:24 | 106:24 |
| CPS > 15 (too fast) | 21.4% | 20.1% |
| Median CPS | 12.1 | 12.1 |
| Mean cue duration | 3.29s | 3.29s |
| Longest line | 42 chars | 42 chars |
| Overlapping cues | 0 | 0 |

Cross-checking timestamps at 5:00 / 30:00 / 105:00: the two agree to within about 0.1 second. MAI's native timestamps are good enough for subtitles; no MMS needed.

## Compared to the 9router path (chunked_transcribe)

| | chunked_transcribe (9router) | mai_transcribe (OpenRouter) |
|---|---|---|
| Backend | Gemini via Antigravity OAuth | MAI-Transcribe-2 (Azure) |
| Cost | free (subscription) | ~$0.10/hour |
| Account-ban risk | yes (403, weekly quota, bans up to 7 days) | no |
| Per-word timestamps | no (needs MMS align) | built in (skips MMS) |
| Needs GPU | yes (MMS step) | no |
| Chunking | shared `audio_utils` | shared `audio_utils` |

## Limitations and untried cases

- Only verified on **1 recording**, single speaker, clean audio. **Not yet tried** on chanting, multi-speaker recordings, or noisy audio.
- MAI is fairly **verbatim**: it captures filler words ("uh", "ah") and repeated words. Good for timing accuracy, but if you want tighter subtitles, consider a clean style at the model layer (not done).
- If a word is transcribed **wrong**, the timestamp is still there; for certainty it's still worth running `compare_passes` against an independent pass (e.g. the Gemini one).
- It **does cost real money** (unlike free Antigravity), in exchange for eliminating account-ban risk. The whole corpus of roughly 1,700 files x ~1.5 hours ~ 2,550 hours x $0.10 ~ **$255**.
- MAI's diarization (speaker separation) errors out when audio is longer than 15 minutes in the preview - this package **does not use** diarization, so it's unaffected.
