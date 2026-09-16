# chunked_transcribe - design notes and empirical results

A record of why the `transcribe/chunked_transcribe/` package is designed the way it is, the actual measurements, and the approaches that were tried and ruled out. Read alongside `transcribe/chunked_transcribe/README.md` (how to run) and `docs/system-architecture.md` (realign).

## Context

The task is bulk transcription of 1-2 hour Vietnamese lectures. The repo's official path is Gemini via **Vertex** (`transcribe/batch_transcribe_vertex.py`, billed to Google Cloud). This package is an **experimental** path via **9router/Antigravity** (free through an OAuth subscription), created while experimenting with using an Antigravity Pro account.

## Design decisions and why

- **Split into roughly 10-minute chunks, cut at a silence (VAD).** Sending a whole long file makes the model **silently drop** a segment while the transcript still reads fluently. Verified on "Ai cũng nghĩ mình đúng" (88 minutes): both the whole-file send and a 27-minute chunk dropped exactly the same ~85-second segment (13:29-14:54), at a spot where the phrase "rất là đáng sợ" repeats, causing the model to skip ahead. 10-minute chunks transcribed it in full. This error does not depend on the thinking level (the high variant still dropped it) or the encoding (both mono 16k and the original encoding dropped it when sent long).
- **Mono 16kHz.** The base64 is about 6.6x smaller than the original while still transcribing the hard segment in full (an A/B test on a 10-minute chunk gave identical results). Each chunk is about 2.4 MB of base64.
- **Drop chunks with almost no speech (under 30 seconds of VAD).** On music/silence, the model **hallucinates** content (the last chunk of one lecture: 0 seconds of speech, yet the model wrote 842 words of a story that isn't in the lecture).
- **Audio-anchored self-check (MMS + star token).** The MMS alignment score and the fraction of aligned words do **not** catch omissions (the chunk that dropped 85 seconds still scored 0.83 on MMS). Inserting a `*` token (a placeholder cell for audio that has no matching text) exposes the gap: the `*` cell contains several seconds of speech with no text. Thresholds: flag an omission from 5 seconds, needs a listen from 2 seconds (a slow chant/singing passage can run to 3.3 seconds, so 2-5 seconds is only "needs a listen").
- **Compare two independent passes (`compare_passes`).** MMS+star is blind to **wrong-word** errors (a wrong word still matches the sound). Transcribe twice (two models or two runs) and diff the strings: trust where they agree, only listen where they differ. This reaches wrong-word errors too. Blind spot: where both passes make the same mistake.
- **3 parallel workers + retry, 403 as a soft stop.** See the operations section below.

## Empirical measurements

File "Chánh Tinh Tấn - Phương Pháp Thiền Định - Phần 2" (100 minutes, 10 chunks):

| | 3.8-flash | 3.1-pro-low |
|---|---|---|
| Prep (ffmpeg whole file + VAD), one-time | ~181s | ~170s |
| Transcribe 10 chunks (sequential) | ~436s (~44s/chunk) | ~665s (~66s/chunk) |
| Total pipeline | ~617s | ~851s |
| Self-check MMS (separate, GPU) | ~120s | ~120s |
| Chunks succeeding on first attempt | 10/10 | 10/10 |
| Omission / hallucination / edge repeat (MMS+star) | 0 / 0 / 0 | 0 / 0 / 0 |

Two-pass comparison (3.8-flash vs 3.1-pro): word-for-word agreement **97.6%**, only **11 spots differing by 4+ words**, with neither side dropping a large segment. The differing spots are where the speaker talks quietly/fast and the two models guess differently.

## Models

- `ag/gemini-3.8-flash` **is** `ag/gemini-3.8-flash-medium` (confirmed from the 9router v0.5.75 source: both point to the upstream `gemini-3.8-flash-medium(medium)`). Medium thinking level. This is the default and has transcribed a full 100-minute file.
- `-high` / `-low` are genuinely different thinking levels. The high variant spends more thinking tokens but does not fix the omission problem when sending the whole file.
- `ag/gemini-3.1-pro-low` edges ahead on Vietnamese FLEURS (2.5% vs 3.5% WER) but is slower; on real recordings we saw no clear difference. The Pro-High variant (`ag/gemini-pro-agent`) **does not accept audio**.

## Approaches tried and ruled out

- **Sending the whole file in one request:** silent omissions (above), and the base64 of the original file (about 150 MB) exceeds Cloudflare's 100 MB body limit (HTTP 413).
- **OpenRouter `google/gemini-2.5-pro` (chat/completions endpoint):** it transcribes, but OpenRouter **bills the base64 string as text tokens**: an 8.6-minute chunk was billed roughly 522,000 tokens instead of the roughly 13,000 real audio tokens (about 40x more expensive). Not suitable for bulk runs. If you need 2.5-pro, go through Vertex (sent by GCS link, billed by real audio tokens). Note: this only applies to the **chat** endpoint; OpenRouter also has a **dedicated transcription** endpoint (`/audio/transcriptions`) billed per second of audio - that is the `transcribe/mai_transcribe/` path (MAI-Transcribe-2), see `docs/mai-transcribe-notes.md`.
- **`realign.window_align` on the whole file to get timing:** it assigns wrong times when the end of the file has a long non-speech segment (it splits words in proportion to audio duration). On the test file, about 1,950 words got placed after the point where speech ended, and the last window's score dropped to 0.07. To produce subtitles, align **per 10-minute chunk** and add the offset; do not use window_align on the whole file. (This is also a concern for subtitles already shipped to production - see the open questions section.)

## Batch operations (rate limit / quota / ban)

- **This is OAuth over a proxy, with a risk that Google bans the account.** We hit one account that got a blanket 403 (every model, both text and audio) with no clear reason. Use a secondary account.
- **Antigravity Pro (from 3/2026):** refreshes quota **weekly**, and when the quota runs out it bans for up to **7 days (168 hours)**. Many people report a week-long ban after 20-30 minutes of heavy use. Pro's requests-per-minute is not clearly published; only known to be roughly 5x the free tier (free is 5/minute).
- **A batch of 1,700 files x roughly 10 chunks = roughly 17,000 requests**, almost certainly over a one-week quota. Parallelism doesn't help this, it only burns the quota faster. What's needed: split the batch by week, use several Pro accounts + round-robin, or buy credits.
- **9router:** has round-robin across multiple accounts + backoff. On a 429/403 it locks the (account, model) pair for a while (429 ramps from 1s up to 4 minutes; 403 is fixed). So don't pile on many parallel workers: one 429 drags the whole group on that model into a lock.
- **Chosen worker count:** Pro can handle **3-5 workers/account** in terms of speed; the default is **3** (`ROUTER_WORKERS`). To go faster, add accounts, don't raise the worker count on a single account.

## Limitations and open questions

- Only verified on **2 recordings**, both a single speaker with clean audio. Not yet tried on multi-speaker recordings, long chanting, or noisy audio.
- The transcript has **no timestamps**; producing subtitles still requires realign (per-chunk alignment).
- **Real parallel timing not yet measured** (every run during development was sequential).
- **No per-file/per-chunk resume yet.** If a 403 ban hits mid-batch, a rerun re-transcribes from the start of the file. This is needed for long batches.
- **Production subtitle concern:** scanning `out/words` shows roughly 116/1,699 files with signs of words placed after the last speech or a sharp drop in the tail-window score. Part of this may be the whole-file window_align bug (batches from 4 onward ran `--window-sec 1500`), part may be transcripts misaligned to audio or VAD dropping chanting. Not yet investigated in depth; the user chose to defer it.
