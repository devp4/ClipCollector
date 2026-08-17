"""
run_vertical_batch.py

Standalone runner: reads the clip manifest + transcript that main.py
writes out (<video_name>/clips_manifest.json and <video_name>/transcript.json)
and renders a 9:16 vertical version of every clip via vertical_reframe.py.

Completely independent of main.py's extraction loop -- this only reads
files main.py already wrote, it doesn't re-run Whisper, Gemini, or spike
detection.

Usage:
    python run_vertical_batch.py caseoh_vod/clips_manifest.json
"""

import os
import re
import sys
import json

import config
import vertical_reframe as vr


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_words(transcript_json_path):
    """Flattens word-level timing out of the transcript JSON."""
    with open(transcript_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    words = []
    for seg in data["segments"]:
        for w in seg.get("words", []):
            text = (w.get("word") or "").strip()
            if text:
                words.append({"start": w["start"], "end": w["end"], "text": text})
    words.sort(key=lambda w: w["start"])
    return words


def sanitize_filename(name, max_length=60):
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name or "clip")[:max_length]


def format_seconds_to_filename(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_vertical_batch.py <path_to_clips_manifest.json>")
        sys.exit(1)

    manifest_path = sys.argv[1]
    manifest = load_manifest(manifest_path)
    video_path = manifest["video_path"]
    transcript_json_path = manifest["transcript_json_path"]
    clips = manifest["clips"]

    if not os.path.exists(video_path):
        print(f"Error: source video not found at '{video_path}'. Update the path or re-run main.py.")
        sys.exit(1)
    if not os.path.exists(transcript_json_path):
        print(f"Error: transcript not found at '{transcript_json_path}'. Re-run main.py for this video.")
        sys.exit(1)

    all_words = load_words(transcript_json_path)
    if not all_words:
        print("Warning: transcript has no word-level timing -- captions will be skipped on vertical clips.")

    print(f"Loaded {len(clips)} clip(s) from manifest, {len(all_words)} words from transcript.")

    # Vertical clips land alongside the manifest, in the same per-video
    # folder main.py created.
    video_root = os.path.dirname(os.path.abspath(manifest_path))
    output_dir = os.path.join(video_root, "clips_vertical")
    os.makedirs(output_dir, exist_ok=True)

    # Only the split-screen layout needs a facecam position at all.
    shared_facecam_box = None
    if config.LAYOUT_MODE == "split_screen":
        shared_facecam_box = config.MANUAL_FACECAM_BOX
        if shared_facecam_box is not None:
            print(f"Using manual facecam box: {shared_facecam_box}")
        elif clips:
            probe_start = clips[0]["start"]
            probe_end = clips[min(2, len(clips) - 1)]["end"]
            print(f"Detecting facecam position (probing {probe_start:.1f}s - {probe_end:.1f}s)...")
            shared_facecam_box = vr.detect_facecam_box(video_path, probe_start, probe_end)
            if shared_facecam_box is None:
                print("Could not find a stable facecam position -- clips will fall back to the blurred-background layout.")

    for i, clip in enumerate(clips):
        start, end = clip["start"], clip["end"]
        title = sanitize_filename(clip.get("title", f"clip_{i+1}"))
        timestamp_tag = format_seconds_to_filename(start)
        output_path = os.path.join(output_dir, f"{i+1}_{timestamp_tag}_{title}_vertical.mp4")

        clip_words = [w for w in all_words if w["end"] >= start and w["start"] <= end]

        print(f"[{i+1}/{len(clips)}] Rendering vertical clip: {start:.1f}s -> {end:.1f}s ({clip.get('title', '')})")

        if config.LAYOUT_MODE == "blurred_background":
            vr.render_blurred_background_clip(video_path, start, end, output_path, clip_words=clip_words)
        else:
            try:
                vr.render_vertical_clip(
                    video_path, start, end, output_path,
                    facecam_box=shared_facecam_box, clip_words=clip_words,
                )
            except ValueError as e:
                print(f"  Skipped: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
