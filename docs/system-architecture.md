# realign architecture

Documentation for the internals of the `realign/` package: how data flows through the steps, the cue-grouping rules, how long files are handled, and the lessons learned from real runs. For how to run it, see `README.md`.

## Goals and constraints

- Assign timing to an already-correct transcript. **Do not change the text**, only the timing and how it is split into cues.
- Vietnamese only. Do not filter out files by score, and do not fall back to the old srt when alignment is poor.
- Alignment is very resource-heavy (a few minutes of GPU per file), while cue grouping is nearly instant. So the alignment result is **cached** as `words.json`, whereas cue grouping can be rerun any number of times when tuning the rules.
- Runs on a 4 GB GPU laptop, and can resume after being interrupted partway.

## Modules

| Module | Role | Needs GPU/torch |
|---|---|---|
| `config.py` | All thresholds (see the table at the end of this document) | No |
| `run_batch.py` | Orchestration: pairing, copying audio, VAD, align, segment, write manifest | Yes |
| `pair_audio_srt.py` | Pair the transcript with the mp3 by normalized name, report leftover files | No |
| `srt_to_txt.py` | Read a `.txt` or `.srt` into a list of words | No |
| `align_words.py` | MMS_FA forced alignment for a whole file | Yes |
| `window_align.py` | Windowed forced alignment for long files | Yes |
| `vad_speech_region.py` | silero-vad: find the first and last speech regions | Yes |
| `segment_cues.py` | words.json -> cues -> clamp -> srt (has its own CLI) | No |
| `cue_builder.py` | Group words into cues by the subtitle rules | No |
| `cue_clamp.py` | Adjust cue timing: ordering, gaps, duration, VAD | No |
| `text_metrics.py` | Count Vietnamese characters, split a cue into 2 lines | No |
| `srt_writer.py` | Cue -> SRT text | No |

Each module has a same-named test file in `tests/realign/test_<module>.py` (except `config.py`).

## Processing flow for one file (`run_batch.process_one`)

```
pair (key = accent-stripped, lowercased name)
  |
  |  out/words/<stem>.json already exists and not --force ?  -- yes --> skip align
  v
copy mp3 from Drive -> out/_tmp/<stem>.mp3   (3 attempts, 2s pause)
  |
words_from_file(transcript)                 list of words, preserving case + diacritics + attached punctuation
  |
speech_region(tmp)  (VAD, None on error)
  |
align() or window_align() (when --window-sec > 0)
  |
write out/words/<stem>.json, delete temp file (even if align fails)
  |
segment(words.json) -> out/srt/<stem>.srt
  |
1 manifest.csv row (status=error on failure, batch continues)
```

`<stem>` is the transcript filename without its extension. The MMS and VAD models are loaded once for the whole batch, and are still loaded even when every file is already cached.

## Data contract

- **Word**: `{"w": str, "start": float|None, "end": float|None, "score": float|None}`. `w` is the display form. `None` means the word could not be aligned (e.g. a token consisting only of digits or symbols, which normalizes to empty).
- **words.json** (`out/words/<stem>.json`): `{"key": str, "audio": str, "vad": [first, last] | null, "words": [Word, ...]}`.
- **Cue**: `{"text": str, "start": float, "end": float}`. `text` contains at most one `\n` character (2 lines).
- **SpeechRegion**: `(first_speech_sec, last_speech_sec)` or `None`.
- **manifest.csv**: `key, srt_name, audio_name, status, words, aligned, cues, mean_score, vad, error`.

## Pairing transcript with audio

- `normalize_key`: NFD, strip diacritics, lowercase, `đ` -> `d`, collapse whitespace. Used only internally to pair the two lists of filenames.
- Only `.mp3` files are indexed. If two mp3s share a key, keep the one seen first (in the order of the directories passed in).
- Files that can't be paired are written to `out/unpaired-realign.md`.

## Alignment

- `load_audio`: read with soundfile (avoids torchaudio.load, which would require torchcodec), mix to mono, resample to 16 kHz.
- `normalize_word`: uroman romanize -> NFD strip diacritics -> lowercase, `đ` -> `d` -> keep only `[a-z']`, because the MMS tokenizer only has the Latin alphabet. Alignment uses this form, while the json stores the original form.
- **Chunked emission**: the audio is passed through the model in `EMIT_CHUNK_SEC` = 20s chunks and then concatenated, to fit a 4 GB GPU. If the last chunk is shorter than 0.5s it is merged into the previous one, because a too-short chunk makes the wav2vec2 conv raise `Kernel size can't be greater than actual input size`.
- A word's `score` is the mean of its tokens' scores. `mean_score` in the manifest is the mean over the words that aligned.

### Long files: `window_align`

The forced_align step (dynamic programming over a frames x tokens matrix) needs RAM proportional to the **product** of the frame count and token count. A 1-2 hour lecture once demanded 36-65 GB of RAM and hit OOM.

The solution:

1. Compute the emission once for the whole file (cheap).
2. Split into `K = round(duration / target_window_sec)` windows.
3. The cut point is the **middle of a silence** (a run of blank frames at least 0.3s long) nearest to the even-split point, so as not to cut mid-word.
4. Split the word list in proportion to the frames at each cut point.
5. Align each window, then add the time offset.

The memory per window drops to roughly total / K^2. With `--window-sec 1500`, the longest lecture (57 GB when aligning the whole file) drops to about 2 GB per window, aligning 99.9% of the words.

Step 4 assumes a uniform speaking rate through the file. If it isn't uniform, the error concentrates on the words around that window's cut point and does not spread to other windows. On manual QA of the 4 join points of the longest lecture, the timing did not run backwards and the text was continuous across the joins.

## Cue grouping (`cue_builder.build_cues`)

1. `fill_word_times`: a word without timing takes `start` = the previous word's `end`, and `end` = the next word's `start` (at least +0.05s).
2. `_split_segments`: **hard split** when the previous word ends with one of `.!?:;`, or on a pause of `LONG_PAUSE` = 2.0s or more.
3. `_segment_to_cues`: if a segment fits in 2 lines x 42 characters it becomes 1 cue. Otherwise, find the smallest number of cues k that allows a **balanced** split (`_fill_k`):
   - Each group targets roughly total characters / k.
   - Prefer a break once 60% of the target is reached and a `, ; :` mark or a `PAUSE_SPLIT` = 0.6s pause is encountered.
   - Once 100% of the target is reached, break immediately.
4. `_enforce_min_display`: extend a cue to meet `DUR_MIN` = 1.5s and a maximum reading speed of `CPS_MAX` = 15 characters/second, without exceeding `DUR_MAX` = 7s and without touching the next cue (leaving `GAP_MIN`). If there isn't room, merge into the previous cue; if that isn't possible, merge into the next cue; if that still isn't possible, leave it as is and guarantee a minimum of `FLOOR` = 0.3s.
5. `_deisolate_boundaries`: if the first or last cue in the file has only 1-2 words and is `ISOLATION_GAP` = 8s or more from the adjacent cue, treat it as a misalignment at the boundary and merge its text into the adjacent cue.
6. `wrap_two_lines`: split into 2 lines of at most 42 characters each. Priority order: valid, then two lines of roughly equal length, then a longer bottom line, then a break after a comma.

Character counting (`char_count`): normalize to NFC then ignore combining marks, so an accented character counts as 1 character.

### Why the rules are as they are

These rules were tuned after a pilot, based on errors users found while watching the videos:

- Long sentences got cut into a stubby tail (e.g. a cue with only "chuyện khác."). Fix: balanced splitting, preferring breaks at commas.
- Fragment cues of just a few words appeared (e.g. a lone "Nam", or "ưu ái và"). Two causes:
  - A pause of just 0.6-1.1s used to force a hard split. Fix: `PAUSE_SPLIT` is now only a preferred break point; a hard split happens only at sentence end or a pause of 2s or more.
  - A word at the start or end of the file got aligned far off. Fix: added the `_deisolate_boundaries` step.
- A 43-character line was once produced. Fix: both the group-splitting branch and the cue-merging branch must pass the `_fits_lines` check.

## Cue timing adjustment (`cue_clamp.clamp_cues`)

- Sort by `start`. Each cue lasts at least `FLOOR` and at most `DUR_MAX`.
- VAD, if present:
  - Push the first cue's `start` up to the moment speech begins (never earlier, still keeping the cue at least `DUR_MIN` long).
  - Trim the last cue's `end` to the moment speech ends + `VAD_PAD` = 0.3s.
  - The purpose is to drop the music and chanting in the intro and outro.
- If the gap to the next cue is less than `PAUSE_GAP` = 0.5s (including overlap): extend `end` close to the next cue's `start`, leaving `GAP_MIN` = 0.084s (2 frames at 24fps), so the subtitles don't flicker.

## Parameter table (`config.py`)

| Parameter | Value | Meaning |
|---|---|---|
| `CPL_MAX` | 42 | Max characters per line |
| `LINES_MAX` | 2 | Max lines per cue |
| `CHAR_MAX` | 84 | Reference only. The real rule uses `_fits_lines` per CPL |
| `CPS_MAX` | 15.0 | Max reading speed (characters/second) |
| `CPS_TARGET` | 13.0 | Reference only, not yet used in code |
| `DUR_MIN` / `DUR_MAX` | 1.5 / 7.0 | Cue duration (seconds) |
| `FLOOR` | 0.3 | Absolute minimum duration |
| `PAUSE_SPLIT` | 0.6 | Preferred (soft) break at a pause |
| `LONG_PAUSE` | 2.0 | Pause that forces a split |
| `ISOLATION_GAP` | 8.0 | Gap at which a 1-2 word boundary cue is treated as misaligned |
| `SENTENCE_END` | `.!?:;` | Sentence-ending characters |
| `GAP_MIN` | 0.084 | Minimum gap between two cues |
| `PAUSE_GAP` | 0.5 | Below this threshold, pull the two cues together |
| `VAD_PAD` | 0.3 | Padding after speech ends |
| `VAD_TRUST_GAP` | 2.0 | Reference only, not yet used in code |
| `SAMPLE_RATE` | 16000 | Sample rate of the MMS model |
| `EMIT_CHUNK_SEC` | 20.0 | Emission chunk length |
| `AUDIO_DIRS` | `Y:\run-script-1..3` | Default mp3 directories when `--audio-dir` is not passed |

Source of these numbers: the `subtitle-timing-standards-research.md` report (Netflix VN, BBC, ESIST).

## Lessons from real runs

- **A uniformly low mean_score across a whole file is usually due to audio quality**, not a code bug. The `run-script-4` batch had a median of 0.73, while other batches were 0.76-0.84. Lectures with musical performances or music scored very low. Below 0.4, open the video and check by eye.
- **Stale cache**: if you re-transcribe but forget to delete `out/words/<stem>.json`, the new srt still uses the old text's timing.
- **manifest is overwritten** on every run. If a run crashes partway, you lose the previous run's rows too. Copy the manifest right after each batch.
- **Wrong transcript content** (Gemini mis-transcribed) breaks alignment or causes OOM. The fix is to re-transcribe, not to change code.
- **A background run in an agent session gets stopped by the system** after a while. Long batches must run in the user's own terminal.
- The `Y:` drive is a Google Drive mount and can drop the network. `copy_with_retry` retries 3 times. If it still fails, that file is recorded as `error` and the batch continues; rerunning the batch will redo the failed file automatically (since there's no cache yet).
