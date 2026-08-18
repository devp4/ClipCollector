# ClipCollector

Turns a long streamer VOD into short, clippable, caption-ready vertical/horizontal clips —
automatically finds the good moments (jumpscares, rage, funny lines, chat interactions),
cuts them, burns in single-word pop-in captions, and can reframe them to 9:16 for
TikTok/Reels/Shorts.

This is the **cloud-GPU build**: no transcript caching (you're not expected to reprocess the
same VOD twice on a rented box), GPU-accelerated Whisper + ffmpeg encoding by default, and every
tunable value lives in one file, `config.py`.

## Quickstart

**Fresh cloud box, nothing downloaded yet:**
```powershell
.\pipeline.ps1 -Url "https://www.youtube.com/watch?v=..."
```

**Already have the video locally:**
```powershell
.\pipeline.ps1 -VideoPath "downloads/some_vod.webm"
```

Either way, this runs the whole thing end to end: download (if `-Url`) → spike detection →
transcription → Gemini clip detection → horizontal clips with burned-in captions → vertical 9:16
clips. Everything lands in a folder named after the video, e.g. `some_vod/clips/` and
`some_vod/clips_vertical/`.

Prefer to run the two stages yourself (e.g. to inspect the manifest between steps)?
```
python main.py downloads/some_vod.webm
python run_vertical_batch.py some_vod/clips_manifest.json
```

---

## One-time setup

**1. Install dependencies:**
```
pip install faster-whisper google-genai librosa scipy numpy tqdm python-dotenv opencv-python yt-dlp
```
> `opencv-python` is only needed if you use `config.LAYOUT_MODE = "split_screen"`. If you hit
> `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`, you have the broken
> `opencv-python==5.0.0.93` release — fix with `pip install "opencv-python<5.0.0"`.

**2. Install ffmpeg** with NVENC support (needed for `config.VIDEO_ENCODER = "h264_nvenc"`, the
default). Confirm with `ffmpeg -encoders | grep nvenc`. If your box's ffmpeg build doesn't have
it, set `config.VIDEO_ENCODER = "libx264"` to fall back to CPU encoding.

**3. Get a Gemini API key** and put it in a `.env` file next to `main.py`:
```
echo "GEMINI_API_KEY=your_actual_key_here" > .env
```

**4. Confirm your GPU is visible** to faster-whisper/CTranslate2 — `nvidia-smi` should show the
card, and CUDA/cuDNN need to be installed on the box (most GPU cloud VM images ship with these
already; if not, faster-whisper's docs have setup notes).

---

## Files

| File | Role |
|---|---|
| `config.py` | **Every tunable value lives here**, grouped by the file that consumes it. Start here. |
| `main.py` | The main pipeline: spike detection → transcription → Gemini → merge/backfill → extract + caption. |
| `audio_analyzer.py` | Volume-spike detection, scream/laugh-vs-bass classification, video duration/resolution helpers. |
| `prompts.py` | The prompt sent to Gemini for clip detection. |
| `captions.py` | Generates `.ass` subtitle files (single-word pop-in, or multi-word plain-white lines). |
| `text_cleanup.py` | ASR-typo correction + full punctuation stripping, applied to caption text only. |
| `vertical_reframe.py` | 9:16 conversion logic (blurred-background and split-screen layouts). |
| `run_vertical_batch.py` | Standalone runner for vertical conversion — reads what `main.py` wrote, doesn't touch Whisper/Gemini/spike detection. |
| `getVideo.py` | Downloads a VOD via `yt-dlp` given a URL. |
| `pipeline.ps1` | End-to-end orchestrator: optionally downloads, then runs `main.py` then `run_vertical_batch.py`. |

`vertical_reframe.py` and `run_vertical_batch.py` are intentionally decoupled from `main.py` —
`main.py` never imports them. If vertical reframing ever misbehaves, ignore/delete those two
files and the rest of the pipeline is unaffected.

---

## `config.py` — the only file you should need to edit day-to-day

Organized into one section per consuming file. The settings you'll actually want to touch for a
cloud-GPU run:

**Whisper (main.py section)**
| Setting | Default | Notes |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `"large-v3"` | `"large-v3-turbo"` is a faster, distilled variant if speed matters more than the last bit of accuracy. |
| `WHISPER_DEVICE` | `"cuda"` | Set to `"cpu"` if you're not on a GPU box. |
| `WHISPER_COMPUTE_TYPE` | `"float16"` | GPU-appropriate. Use `"int8"` for CPU. |

**ffmpeg encoding (shared section)**
| Setting | Default | Notes |
|---|---|---|
| `VIDEO_ENCODER` | `"h264_nvenc"` | GPU-accelerated encode. Set to `"libx264"` for CPU fallback. |
| `NVENC_PRESET` | `"p5"` | `p1` (fastest) to `p7` (best quality). |
| `NVENC_CQ` | `"23"` | Lower = higher quality/bigger file. |

**Everything else** (chunking, clip-length bounds, Gemini rate limiting, spike-detection
thresholds, caption styling/timing, facecam layout, download settings) is in `config.py` too,
grouped by file — see the comments there.

---

## Caption styles

`config.CAPTION_STYLE`:

- **`"single_word_pop"` (default)** — one word on screen at a time, fade + scale-overshoot
  bounce as each word is spoken. Bold white text, black outline. No color highlighting.
- **`"line_highlight"`** — short lines of a few words at once, plain white, no per-word
  highlighting (the earlier yellow current-word highlight has been fully removed — the color
  machinery for it no longer exists in this codebase).

Both run every caption word through `text_cleanup.py`: **all punctuation is stripped** (commas,
periods, `?`, `!`, quotes — apostrophes inside contractions like "don't" are kept), and known ASR
typos get corrected via `config.WORD_CORRECTIONS`.

---

## What changed from the local/testing build

This is a from-scratch cleanup pass, not just new features. If you're comparing against an older
copy:

- **No transcript caching.** Every run re-transcribes from scratch — there's no `FORCE_RETRANSCRIBE`
  flag, no cache-check, no `Segment`/`Word` stand-in classes for reloading old data. `transcript.json`
  is still written (needed by `run_vertical_batch.py`), just always fresh.
- **Yellow caption highlight fully removed**, not just defaulted off — `ENABLE_HIGHLIGHT_COLOR`
  and `HIGHLIGHT_COLOR` no longer exist anywhere in the code.
- **All config centralized** into `config.py`, organized by consuming file, instead of scattered
  `# --- Configuration ---` blocks at the top of each module.
- **`UNIVERSAL_PROMPT` removed** from `prompts.py` — it was never actually imported/used anywhere
  (only `STREAMER_SPECIFIC_PROMPT` was wired in), so it was dead code. Easy to re-add if you want
  a switchable prompt style later.
- **`get_padded_clip_windows()` and `plot_volume_profile()` removed** from `audio_analyzer.py`
  (along with the `matplotlib` dependency they needed) — these only existed for the
  `python audio_analyzer.py` standalone debug/visualization path, unused by the actual pipeline.
  If you relied on eyeballing the RMS threshold graph while tuning `SPIKE_THRESHOLD_MULTIPLIER`,
  flag it and this can come back.
- **`vertical_reframe.py`'s `loosen_detection_params()` debug helper removed** — a testing-only
  tool for probing face-detection sensitivity, not used by the actual pipeline.
- **GPU-first defaults**: Whisper on `cuda`/`float16`/`large-v3`, ffmpeg encoding via
  `h264_nvenc`. CPU fallback is a one-line config change in each case, not a code change.
- **`getVideo.py` and `pipeline.ps1` now actually connect** — `getVideo.py` accepts a URL via CLI
  and prints the downloaded path; `pipeline.ps1` can take `-Url` and download first, instead of
  only accepting a path to a file you'd already downloaded some other way.

---

## Known limitations / things to watch

- **Gemini free-tier rate limits**: `main.py` paces chunk calls (`config.GEMINI_CALL_DELAY_SEC`)
  and backs off on 429s, but processing many VODs back-to-back can still hit the daily cap.
- **`classify_spike`'s scream-vs-bass heuristic** (`audio_analyzer.py`) uses rough, untuned
  spectral thresholds (`config.CLASSIFY_VOICE_CENTROID_MIN` etc.). Worth revisiting if the
  coverage-backfill step starts pulling in loud game SFX instead of real reactions — this is
  a good candidate for a real ML audio classifier if you want to invest GPU time here.
- **Face auto-detection (`split_screen` mode only)** uses an old-school Haar cascade — can be
  unreliable depending on lighting/angle/footage quality. `config.MANUAL_FACECAM_BOX` is the
  more reliable option; `vertical_reframe.py` also has `save_debug_detection_frames()` if you
  want to see what the detector is picking up. A GPU-based face detector (RetinaFace, a small
  YOLO-face model) would be a meaningfully more reliable upgrade here, not yet implemented.
- **No caching means every run re-pays the full transcription + Gemini cost**, even for a VOD
  you already processed. Intentional per project requirements, but worth knowing if you ever do
  want to iterate on caption styling for the *same* VOD repeatedly — consider a lightweight
  "skip re-transcription for this run" flag if that becomes a common workflow.
