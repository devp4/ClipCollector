"""
config.py

Every tunable value in the project lives here, grouped by the file that
consumes it. Nothing in this file does real work -- it's just constants
(plus loading the .env file, since GEMINI_API_KEY is also just a setting).
Edit this file to tune behavior; you shouldn't need to touch the actual
pipeline logic in the other files for day-to-day adjustments.
"""

import os
import dotenv

dotenv.load_dotenv()


# =============================================================================
# main.py
# =============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Video path: overridden by a CLI arg if you run `python main.py <path>`
# (this is what pipeline.ps1 does). Edit this if you want a plain
# `python main.py` with no args to still do something useful.
DEFAULT_VIDEO_PATH = "downloads/caseoh_vod.webm"

# --- Whisper transcription ---
# Cloud-GPU defaults. For CPU-only, use: model="medium", device="cpu",
# compute_type="int8". "large-v3-turbo" is a distilled variant of large-v3
# that's noticeably faster with similar quality if you want to try it.
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"

# --- Gemini ---
GEMINI_MODEL = "gemini-3.5-flash"

# --- Captions ---
CAPTIONS_ENABLED = True
# "single_word_pop": one word on screen at a time, fade + bounce scale-in.
# "line_highlight": short lines of a few plain-white words at once.
CAPTION_STYLE = "single_word_pop"

# --- Chunking (so long VODs don't blow Gemini's context/attention) ---
CHUNK_DURATION_SEC = 20 * 60      # 20 minutes per chunk (target, before boundary snapping)
CHUNK_OVERLAP_SEC = 60            # overlap safety net -- bigger than MAX_CLIP_DURATION below
CHUNK_BOUNDARY_SEARCH_SEC = 60    # how far to search for a quiet point to actually cut a chunk

# --- Clip shaping ---
MIN_CLIP_DURATION = 12.0
MAX_CLIP_DURATION = 45.0
MERGE_GAP_SEC = 8.0                # clips whose windows are within this gap get merged
MIN_GEMINI_CONFIDENCE = 0.5

# Guaranteed-clip threshold from raw audio spikes (bypasses Gemini)
GUARANTEED_RMS_SCORE = 0.20

# Coverage safety net: after merging Gemini + guaranteed clips, any spike
# scoring above this (but below GUARANTEED_RMS_SCORE) that Gemini didn't
# turn into a clip gets backfilled as its own padded clip -- as long as it
# isn't classified as pure game SFX/bass.
COVERAGE_BACKFILL_RMS_SCORE = 0.15

# --- Gemini rate limiting ---
GEMINI_CALL_DELAY_SEC = 5.0
GEMINI_MAX_RETRIES = 4
GEMINI_BACKOFF_BASE_SEC = 10.0


# =============================================================================
# ffmpeg encoding (shared by main.py and vertical_reframe.py)
# =============================================================================

# "h264_nvenc" (default -- GPU-accelerated encode, needs an NVIDIA GPU/driver
# with NVENC support, which an RTX 3090 has) or "libx264" (CPU fallback).
VIDEO_ENCODER = "h264_nvenc"

# NVENC settings (only used when VIDEO_ENCODER == "h264_nvenc").
# Presets are p1 (fastest/lowest quality) to p7 (slowest/highest quality).
NVENC_PRESET = "p5"
NVENC_CQ = "23"                    # lower = higher quality/bigger file, ~18-28 is a reasonable range

# libx264 settings (only used when VIDEO_ENCODER == "libx264").
X264_PRESET = "medium"

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"


def ffmpeg_video_codec_args():
    """Returns the ffmpeg CLI args for whichever encoder VIDEO_ENCODER selects."""
    if VIDEO_ENCODER == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", NVENC_PRESET, "-cq", NVENC_CQ]
    return ["-c:v", "libx264", "-preset", X264_PRESET]


def ffmpeg_audio_codec_args():
    return ["-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE]


# =============================================================================
# audio_analyzer.py
# =============================================================================

PRE_PAD_DEFAULT = 5.0
POST_PAD_DEFAULT = 5.0

# Spike detection (detect_loud_spikes)
SPIKE_THRESHOLD_MULTIPLIER = 8.0   # how many std-devs above the mean RMS counts as a "spike"
SPIKE_MIN_RMS_FLOOR = 0.12         # absolute floor, regardless of the stream's overall volume
SPIKE_HOP_LENGTH = 512
SPIKE_MIN_GAP_SEC = 5.0            # minimum gap enforced between two detected spikes

# Spike classification (classify_spike) -- rough, untuned heuristic
# separating scream/laugh from bass/impact SFX. Worth revisiting if the
# coverage-backfill step in main.py starts pulling in game SFX instead of
# real reactions.
CLASSIFY_WINDOW_SEC = 0.5
CLASSIFY_VOICE_CENTROID_MIN = 1800
CLASSIFY_VOICE_ZCR_MIN = 0.08
CLASSIFY_IMPACT_CENTROID_MAX = 1200


# =============================================================================
# captions.py
# =============================================================================

FONT_NAME = "Arial Black"
BASE_COLOR = (255, 255, 255)       # white -- captions are plain white, no color highlighting
OUTLINE_COLOR = (0, 0, 0)          # black stroke
OUTLINE_WIDTH = 3
SHADOW = 0
BOLD = True
UPPERCASE = True
ALIGNMENT = 2                      # ASS alignment code: 2 = bottom-center

# "line_highlight" style
LINE_FONTSIZE_FRACTION = 0.055     # font size as a fraction of video height
LINE_MARGIN_V_FRACTION = 0.16      # bottom margin as a fraction of video height
MAX_WORDS_PER_LINE = 5
MAX_GAP_BETWEEN_WORDS_SEC = 0.6    # start a new caption line (or hide a lingering word) after a pause this long
TRAILING_HOLD_SEC = 0.15           # keep the last word of a line on screen briefly after it ends

# "single_word_pop" style
POP_FONTSIZE_FRACTION = 0.065      # font size as a fraction of video height
POP_MARGIN_V_FRACTION = 0.16
POP_START_SCALE = 60               # starting scale %, before the pop-in animation
POP_OVERSHOOT_SCALE = 115          # peak scale % during the bounce overshoot
POP_SETTLE_SCALE = 100             # resting scale % once the bounce settles
POP_OVERSHOOT_MS = 120             # time to animate from START_SCALE to OVERSHOOT_SCALE
POP_SETTLE_MS = 220                # time to animate from OVERSHOOT_SCALE down to SETTLE_SCALE
POP_FADE_IN_MS = 60
POP_TRAILING_HOLD_SEC = 0.1        # keep the last word on screen briefly after it ends


# =============================================================================
# text_cleanup.py
# =============================================================================

# Common ASR mishears seen in streamer transcripts. Match is case-insensitive
# against the whole token (after punctuation is stripped). Add to this as
# you spot new recurring errors.
#
# NOTE: some of these are inherently ambiguous outside of full context --
# e.g. "Chad" really could be someone's name in a different streamer's
# chat. Review/curate this list per streamer if that's a concern.
WORD_CORRECTIONS = {
    "chad": "Chat",
}


# =============================================================================
# vertical_reframe.py
# =============================================================================

TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920

# Face detection (only relevant if run_vertical_batch.py's LAYOUT_MODE is
# "split_screen" AND MANUAL_FACECAM_BOX is left as None)
FACE_DETECTION_SAMPLES = 12          # how many frames to sample across the clip/video
FACE_BOX_PADDING_FRACTION = 0.35     # extra room around the detected face (headroom + chest)
MIN_DETECTION_FRACTION = 0.25        # need faces found in at least this fraction of sampled frames

# Set this to a local file path to bypass cv2.data.haarcascades entirely --
# useful if your installed opencv wheel is missing the bundled data files
# (a known issue in some releases, e.g. opencv-python 5.0.0.93). Download:
# https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
CASCADE_XML_OVERRIDE = None

FACECAM_PANE_FRACTION = 0.45         # split_screen layout: facecam pane gets this fraction of the vertical height


# =============================================================================
# run_vertical_batch.py
# =============================================================================

# "blurred_background" (default, no facecam detection needed at all) or
# "split_screen" (dedicated enlarged facecam pane + gameplay pane).
LAYOUT_MODE = "blurred_background"

# Only used when LAYOUT_MODE == "split_screen". Set to (x, y, width, height)
# in source-video pixel coordinates to skip auto-detection and use a fixed
# box for every clip -- recommended over auto-detection, which can be
# unreliable depending on footage quality/angle/lighting. Run:
#     python -c "import vertical_reframe as vr; vr.save_reference_frame('downloads/your_vod.webm', 60)"
# to grab a sample frame and read off the facecam's pixel box from it.
MANUAL_FACECAM_BOX = None


# =============================================================================
# getVideo.py
# =============================================================================

DOWNLOAD_DIR = "downloads"
YT_DLP_FORMAT = "bestvideo+bestaudio/best"
