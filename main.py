import os
import re
import sys
import json
import time
import subprocess
from faster_whisper import WhisperModel
from google import genai
from tqdm import tqdm

import config
from audio_analyzer import (
    detect_loud_spikes,
    get_spike_details,
    snap_to_low_energy,
    find_chunk_split_point,
    get_video_duration,
    get_video_resolution,
    format_seconds_to_readable,
    format_seconds_to_filename,
)
import captions
from prompts import STREAMER_SPECIFIC_PROMPT

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_VIDEO_PATH


def format_timestamp(seconds):
    """Helper to convert seconds to SRT timestamp format (00:00:00,000)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def sanitize_filename(name, max_length=60):
    """
    Strips/replaces characters that are illegal (or awkward) in filenames on
    Windows, macOS, and Linux -- e.g. Gemini-generated titles like
    'Bathtub Sleep Paralysis?!' will otherwise crash ffmpeg/open() on Windows.
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "clip"
    return name[:max_length]


def save_transcript(json_path, txt_path, segments, duration):
    """
    Writes the transcript in two forms: a JSON file (start/end/text/words
    per segment -- consumed by run_vertical_batch.py) and a plain-text file
    (for humans). Always written fresh, no caching/reuse across runs.
    """
    data = {
        "duration": duration,
        "segments": [
            {
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "words": [{"start": w.start, "end": w.end, "word": w.word} for w in (s.words or [])],
            }
            for s in segments
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg.start:.1f}s -> {seg.end:.1f}s]: {seg.text}\n")


def flatten_words(segments):
    """
    Builds a flat, time-sorted list of word dicts ({"start","end","text"})
    across all segments, for slicing into per-clip caption data.
    """
    words = []
    for seg in segments:
        for w in (seg.words or []):
            text = (w.word or "").strip()
            if text:
                words.append({"start": w.start, "end": w.end, "text": text})
    words.sort(key=lambda w: w["start"])
    return words


def generate_srt_subtitles(segments, clip_start, clip_end, output_srt_path):
    subtitles = []
    sub_index = 1
    for segment in segments:
        if segment.end >= clip_start and segment.start <= clip_end:
            rel_start = max(0.0, segment.start - clip_start)
            rel_end = min(clip_end - clip_start, segment.end - clip_start)
            if rel_start < rel_end:
                subtitles.append(
                    f"{sub_index}\n{format_timestamp(rel_start)} --> {format_timestamp(rel_end)}\n{segment.text.strip()}\n"
                )
                sub_index += 1
    with open(output_srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(subtitles))


def chunk_transcript_and_spikes(segments, spike_details, total_duration, times, rms):
    """
    Splits the transcript + spike list into overlapping time windows so each
    Gemini call only has to reason about ~20 minutes of content at a time.

    Two layers of protection against slicing a moment in half at a boundary:
    1. Each internal boundary is snapped to the nearest genuinely quiet point
       (via find_chunk_split_point) within CHUNK_BOUNDARY_SEARCH_SEC, instead
       of always cutting at a fixed clock time.
    2. On top of that, CHUNK_OVERLAP_SEC (bigger than MAX_CLIP_DURATION) is
       still applied as a safety net for streams that are loud almost
       everywhere, where step 1 can't find a truly quiet spot.

    Returns a list of (chunk_start, chunk_end, transcript_text, spikes_json_str).
    """
    chunks = []
    chunk_start = 0.0
    while chunk_start < total_duration:
        target_end = chunk_start + config.CHUNK_DURATION_SEC
        if target_end < total_duration:
            chunk_end = find_chunk_split_point(
                times, rms, target_end, search_radius=config.CHUNK_BOUNDARY_SEARCH_SEC
            )
            chunk_end = max(chunk_start + 1.0, min(chunk_end, total_duration))
        else:
            chunk_end = total_duration

        chunk_text = ""
        for seg in segments:
            if seg.end >= chunk_start and seg.start <= chunk_end:
                chunk_text += f"[{seg.start:.1f}s -> {seg.end:.1f}s]: {seg.text}\n"

        chunk_spikes = [
            s for s in spike_details
            if chunk_start <= s["peak"] <= chunk_end
        ]

        if chunk_text.strip():
            chunks.append((chunk_start, chunk_end, chunk_text, json.dumps(chunk_spikes)))

        if chunk_end >= total_duration:
            break
        chunk_start = max(0.0, chunk_end - config.CHUNK_OVERLAP_SEC)

    return chunks


def call_gemini_for_chunk(client, transcript_text, spikes_json_str):
    """
    Calls Gemini for one chunk and safely parses the JSON response.
    Retries with exponential backoff specifically on rate-limit errors
    (429 / RESOURCE_EXHAUSTED); other errors fail fast without burning retries.
    """
    prompt = STREAMER_SPECIFIC_PROMPT.format(
        transcript_text=transcript_text,
        audio_spikes_json=spikes_json_str,
    )

    for attempt in range(config.GEMINI_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
            )
            raw_json = response.text.strip().replace("```json", "").replace("```", "")
            clips = json.loads(raw_json)
            if isinstance(clips, list):
                return clips
            print("Gemini returned non-list JSON, skipping chunk.")
            return []
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"Gemini response parse failed (attempt {attempt + 1}): {e}")
            return []  # not a rate-limit issue, no point retrying the same prompt
        except Exception as e:
            msg = str(e)
            is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower()
            if is_rate_limit and attempt < config.GEMINI_MAX_RETRIES:
                backoff = config.GEMINI_BACKOFF_BASE_SEC * (2 ** attempt)
                print(f"Gemini rate limit hit (attempt {attempt + 1}), backing off {backoff:.0f}s...")
                time.sleep(backoff)
                continue
            print(f"Gemini call failed: {e}")
            return []
    return []


def build_guaranteed_clips(spike_details, video_duration):
    """
    Turns high-confidence, voice-classified loud spikes into clips directly,
    bypassing Gemini. Bass/impact-only spikes are skipped since they're much
    more likely to be game SFX than an actual scream/reaction.
    """
    guaranteed = []
    for s in spike_details:
        if s["score"] > config.GUARANTEED_RMS_SCORE and s["type"] == "voice_scream_or_laugh":
            start = max(0.0, s["peak"] - config.PRE_PAD_DEFAULT)
            end = s["peak"] + config.POST_PAD_DEFAULT
            if video_duration:
                end = min(end, video_duration)
            guaranteed.append({
                "start": start,
                "end": end,
                "title": f"loud_moment_{format_seconds_to_filename(s['peak'])}",
                "category": "jumpscare",
                "confidence": 1.0,
                "source": "spike",
            })
            print(f"-> Force-locked clip at {format_seconds_to_readable(s['peak'])} ({s['peak']:.1f}s) (Score: {s['score']:.4f})")
    return guaranteed


def normalize_and_pad_clip(clip, video_duration):
    """Applies padding (for Gemini clips), clamps to bounds and duration limits."""
    start = clip["start"]
    end = clip["end"]

    if clip.get("source") != "spike" and (end - start) < 6.0:
        start = start - config.PRE_PAD_DEFAULT
        end = end + config.POST_PAD_DEFAULT

    start = max(0.0, start)
    if video_duration:
        end = min(end, video_duration)

    duration = end - start
    if duration < config.MIN_CLIP_DURATION:
        end = start + config.MIN_CLIP_DURATION
    elif duration > config.MAX_CLIP_DURATION:
        end = start + config.MAX_CLIP_DURATION

    clip["start"] = start
    clip["end"] = end
    return clip


def find_covering_clip(peak, clips):
    """Returns the first clip whose [start, end] window contains this peak, or None."""
    return next((c for c in clips if c["start"] <= peak <= c["end"]), None)


def print_spike_coverage_report(spike_details, clips):
    """
    Prints, for every raw audio spike, whether it ended up inside a final
    clip or fell through the cracks.
    """
    print("\nSpike coverage report:")
    for s in spike_details:
        match = find_covering_clip(s["peak"], clips)
        if match:
            status = f"covered by '{match.get('title', '?')}' [{format_seconds_to_readable(match['start'])}-{format_seconds_to_readable(match['end'])}]"
        else:
            status = "NOT COVERED"
        print(f"  {format_seconds_to_readable(s['peak'])} (score={s['score']:.4f}, type={s['type']}) -> {status}")


def backfill_uncovered_spikes(spike_details, clips, video_duration):
    """
    Finds spikes that scored above COVERAGE_BACKFILL_RMS_SCORE but weren't
    absorbed into any Gemini or guaranteed clip, and turns them into their
    own padded clips -- unless they're classified as impact_or_bass.
    """
    fallback_clips = []
    for s in spike_details:
        if find_covering_clip(s["peak"], clips):
            continue
        if s["score"] < config.COVERAGE_BACKFILL_RMS_SCORE:
            continue
        if s["type"] == "impact_or_bass":
            continue

        start = max(0.0, s["peak"] - config.PRE_PAD_DEFAULT)
        end = s["peak"] + config.POST_PAD_DEFAULT
        if video_duration:
            end = min(end, video_duration)
        fallback_clips.append({
            "start": start,
            "end": end,
            "title": f"backfilled_moment_{format_seconds_to_filename(s['peak'])}",
            "category": "jumpscare",
            "confidence": 0.6,
            "source": "spike_fallback",
        })
        print(
            f"-> Backfilling uncovered spike at {format_seconds_to_readable(s['peak'])} "
            f"(score={s['score']:.4f}, type={s['type']}) -- Gemini didn't flag it, adding anyway."
        )
    return fallback_clips


def merge_overlapping_clips(clips, gap=None):
    """
    Sorts clips by start time and merges any whose windows overlap or sit
    within `gap` seconds of each other, to avoid near-duplicate outputs from
    the spike-detector and Gemini both flagging the same moment.
    """
    gap = gap if gap is not None else config.MERGE_GAP_SEC
    if not clips:
        return []

    clips = sorted(clips, key=lambda c: c["start"])
    merged = [clips[0]]

    for current in clips[1:]:
        last = merged[-1]
        if current["start"] <= last["end"] + gap:
            last["end"] = max(last["end"], current["end"])
            if last.get("source") == "spike" and current.get("source") != "spike":
                last["title"] = current["title"]
                last["category"] = current.get("category", last.get("category"))
                last["source"] = current.get("source")
            last["confidence"] = max(last.get("confidence", 0), current.get("confidence", 0))
        else:
            merged.append(current)

    return merged


def main():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("Set the GEMINI_API_KEY environment variable (.env) before running.")

    print(f"Processing video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found at '{VIDEO_PATH}'. Check the path/arg you passed.")

    video_duration = get_video_duration(VIDEO_PATH)
    video_width, video_height = get_video_resolution(VIDEO_PATH)

    # 1. Detect loud spikes + classify them (scream/laugh vs. bass/impact)
    times, rms, mean_rms, threshold, spikes, y, sr = detect_loud_spikes(VIDEO_PATH)
    spike_details = get_spike_details(y, sr, times, rms, spikes)

    guaranteed_clips = build_guaranteed_clips(spike_details, video_duration)

    print(f"\n[1/4] Transcribing locally with faster-whisper ({config.WHISPER_MODEL_SIZE} on {config.WHISPER_DEVICE})...")
    video_basename = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    video_root = video_basename  # e.g. "caseoh_vod" -- everything for this video lands in here
    os.makedirs(video_root, exist_ok=True)
    transcript_json_path = f"{video_root}/transcript.json"
    transcript_txt_path = f"{video_root}/transcript.txt"

    model = WhisperModel(config.WHISPER_MODEL_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)
    segments_gen, info = model.transcribe(VIDEO_PATH, word_timestamps=True)
    segments = list(segments_gen)
    whisper_duration = info.duration

    timestamps = 0.0
    with tqdm(total=info.duration, unit="audio sec") as pbar:
        for seg in segments:
            pbar.update(seg.end - timestamps)
            timestamps = seg.end
        if timestamps < info.duration:
            pbar.update(info.duration - timestamps)

    save_transcript(transcript_json_path, transcript_txt_path, segments, whisper_duration)
    print(f"Transcript saved to {transcript_json_path} and {transcript_txt_path}")

    total_duration = video_duration or whisper_duration
    print(f"Total duration: {total_duration / 60:.2f} minutes.")

    all_words = flatten_words(segments)
    if config.CAPTIONS_ENABLED and not all_words:
        print("Warning: no word-level timestamps available -- captions will be skipped.")

    print("\n[2/4] Chunking transcript + spikes and analyzing with Gemini...")
    client = genai.Client(api_key=config.GEMINI_API_KEY)
    chunks = chunk_transcript_and_spikes(segments, spike_details, total_duration, times, rms)
    print(f"Split into {len(chunks)} chunk(s) (~{config.CHUNK_DURATION_SEC / 60:.0f} min target, boundaries snapped to quiet points).")

    gemini_clips = []
    for idx, (chunk_start, chunk_end, chunk_text, spikes_json_str) in enumerate(chunks):
        print(f"  Analyzing chunk {format_seconds_to_readable(chunk_start)} - {format_seconds_to_readable(chunk_end)}...")
        raw_clips = call_gemini_for_chunk(client, chunk_text, spikes_json_str)
        for c in raw_clips:
            try:
                start, end = float(c["start"]), float(c["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if end <= start:
                continue
            if c.get("confidence", 1.0) < config.MIN_GEMINI_CONFIDENCE:
                continue
            gemini_clips.append({
                "start": start,
                "end": end,
                "title": c.get("title", "clip"),
                "category": c.get("category", "unknown"),
                "confidence": c.get("confidence", 1.0),
                "source": "gemini",
            })

        if idx < len(chunks) - 1:
            time.sleep(config.GEMINI_CALL_DELAY_SEC)

    print(f"Gemini returned {len(gemini_clips)} candidate clip(s) across all chunks.")

    print("\n[3/4] Merging spike + Gemini clips, padding, and deduping...")
    all_clips = guaranteed_clips + gemini_clips
    all_clips = [normalize_and_pad_clip(c, total_duration) for c in all_clips]
    final_clips = merge_overlapping_clips(all_clips)

    fallback_clips = backfill_uncovered_spikes(spike_details, final_clips, total_duration)
    if fallback_clips:
        fallback_clips = [normalize_and_pad_clip(c, total_duration) for c in fallback_clips]
        final_clips = merge_overlapping_clips(final_clips + fallback_clips)

    print_spike_coverage_report(spike_details, final_clips)

    for c in final_clips:
        c["start"] = max(0.0, snap_to_low_energy(times, rms, c["start"], search_radius=1.5, mode="min"))
        c["end"] = snap_to_low_energy(times, rms, c["end"], search_radius=1.5, mode="min")
        if total_duration:
            c["end"] = min(c["end"], total_duration)

    print(f"Final clip count after merge/dedup: {len(final_clips)}")

    manifest_path = f"{video_root}/clips_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_path": VIDEO_PATH,
            "transcript_json_path": transcript_json_path,
            "clips": final_clips,
        }, f, ensure_ascii=False, indent=2)
    print(f"Clip manifest saved to {manifest_path} (used by run_vertical_batch.py)")

    print(f"\n[4/4] Extracting videos, generating .srt captions, and burning in {config.CAPTION_STYLE} captions...")
    clips_dir = f"{video_root}/clips"
    os.makedirs(clips_dir, exist_ok=True)
    for i, clip in enumerate(final_clips):
        start = clip["start"]
        end = clip["end"]
        title = sanitize_filename(clip.get("title", f"clip_{i+1}"))
        timestamp_tag = format_seconds_to_filename(start)
        base_filename = f"{clips_dir}/{i+1}_{timestamp_tag}_{title}"
        output_video = f"{base_filename}.mp4"
        output_srt = f"{base_filename}.srt"
        output_ass = f"{base_filename}.ass"

        print(
            f"Extracting Clip {i+1} [{clip.get('category', 'n/a')}, source={clip.get('source')}]: "
            f"{format_seconds_to_readable(start)} -> {format_seconds_to_readable(end)} "
            f"({start:.1f}s to {end:.1f}s)..."
        )

        clip_words = [w for w in all_words if w["end"] >= start and w["start"] <= end]
        has_captions = False
        if config.CAPTIONS_ENABLED and clip_words:
            has_captions = captions.generate_captions(
                clip_words, start, end, output_ass,
                video_width=video_width, video_height=video_height,
            )

        # -ss BEFORE -i for fast seeking (avoids decoding from file start every time)
        if has_captions:
            escaped_ass = captions.escape_ffmpeg_filter_path(os.path.abspath(output_ass))
            # NOTE: once any -map is used, ffmpeg disables automatic stream
            # selection -- a plain -vf filter's output does NOT get
            # auto-mapped in that case. -filter_complex with an explicit
            # [vout] label + an explicit -map "[vout]" avoids that trap.
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-nostats",
                "-ss", str(start), "-i", VIDEO_PATH, "-to", str(end - start),
                "-filter_complex", f"[0:v]ass='{escaped_ass}'[vout]",
                "-map", "[vout]", "-map", "0:a",
            ] + config.ffmpeg_video_codec_args() + config.ffmpeg_audio_codec_args() + [output_video]
        else:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-nostats",
                "-ss", str(start), "-i", VIDEO_PATH, "-to", str(end - start),
                "-map", "0:v", "-map", "0:a",
            ] + config.ffmpeg_video_codec_args() + config.ffmpeg_audio_codec_args() + [output_video]

        subprocess.run(cmd)
        generate_srt_subtitles(segments, start, end, output_srt)

    print("Clips successfully generated with proper context padding!")


if __name__ == "__main__":
    main()
