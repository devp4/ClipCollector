"""
vertical_reframe.py

Converts a horizontal (16:9-ish) stream VOD clip into a 9:16 vertical clip
for TikTok/Reels/Shorts. Two layouts (see config.LAYOUT_MODE):

  "blurred_background" -- whole frame centered, blurred copy of itself
      filling the bars above/below. No facecam detection needed.

  "split_screen" -- dedicated, enlarged facecam pane on top + gameplay
      pane on the bottom (gameplay always shows its full width, letterboxed
      with blur rather than cropped, so action at the edges never gets cut
      off). Needs a facecam position -- see config.MANUAL_FACECAM_BOX.

This module is intentionally standalone and NOT imported by main.py.
Call render_blurred_background_clip / render_vertical_clip directly (see
run_vertical_batch.py, or the __main__ block below for a single-clip
smoke test). Keeping it separate means you can adopt/drop this feature
without touching the horizontal pipeline at all.
"""

import os
import subprocess
import cv2
import numpy as np

import config
import captions


def _sample_frames(video_path, start_sec=None, end_sec=None, num_samples=None):
    """Grabs `num_samples` evenly-spaced frames from [start_sec, end_sec] (or the whole video)."""
    num_samples = num_samples or config.FACE_DETECTION_SAMPLES
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int((start_sec or 0.0) * fps)
    end_frame = int((end_sec if end_sec is not None else frame_count / fps) * fps)
    end_frame = min(end_frame, frame_count - 1)
    start_frame = max(0, min(start_frame, end_frame))

    frames = []
    if end_frame <= start_frame:
        cap.release()
        return frames

    positions = np.linspace(start_frame, end_frame, num=num_samples, dtype=int)
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)

    cap.release()
    return frames


def _detect_faces_in_frame(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return [tuple(int(v) for v in f) for f in faces]  # (x, y, w, h)


def _load_cascade():
    cascade_path = config.CASCADE_XML_OVERRIDE or (cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        print(
            f"Warning: face cascade XML not found at '{cascade_path}'.\n"
            "  This happens with some broken opencv-python/opencv-contrib-python wheels "
            "(e.g. 5.0.0.93 ships with an empty cv2/data directory -- see "
            "github.com/opencv/opencv-python/issues/1244). Fix options:\n"
            '    1) pip install "opencv-python<5.0.0"   (recommended -- known-good data files)\n'
            "    2) Download the file yourself and set config.CASCADE_XML_OVERRIDE to its path:\n"
            "       https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
            "haarcascade_frontalface_default.xml"
        )
        return None
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print(f"Warning: face cascade classifier loaded from '{cascade_path}' but is empty/invalid.")
        return None
    return cascade


def detect_facecam_box(video_path, start_sec=None, end_sec=None, num_samples=None):
    """
    Samples frames across the given time range (or the whole video) and
    looks for a face at a *consistent* pixel position -- since a facecam
    overlay is static, real detections should cluster tightly. Returns a
    padded (x, y, w, h) box in source pixel coordinates, or None if no
    stable facecam position could be found.
    """
    num_samples = num_samples or config.FACE_DETECTION_SAMPLES
    cascade = _load_cascade()
    if cascade is None:
        return None

    frames = _sample_frames(video_path, start_sec, end_sec, num_samples)
    if not frames:
        print("Warning: could not sample any frames for face detection.")
        return None

    all_detections = []
    for frame in frames:
        all_detections.extend(_detect_faces_in_frame(frame, cascade))

    if len(all_detections) < max(2, int(num_samples * config.MIN_DETECTION_FRACTION)):
        print(f"Face detection found too few consistent hits ({len(all_detections)}/{len(frames)} frames) -- no stable facecam position.")
        return None

    # Cluster by center point (simple grid-snap clustering: static overlays
    # should land in the same ~30px bucket almost every time)
    centers = np.array([[x + w / 2, y + h / 2] for (x, y, w, h) in all_detections])
    buckets = {}
    for i, (cx, cy) in enumerate(centers):
        key = (round(cx / 30), round(cy / 30))
        buckets.setdefault(key, []).append(i)

    best_key = max(buckets, key=lambda k: len(buckets[k]))
    best_indices = buckets[best_key]

    if len(best_indices) < max(2, int(len(frames) * config.MIN_DETECTION_FRACTION)):
        print(f"Largest face-position cluster only has {len(best_indices)}/{len(frames)} frames -- not confident enough.")
        return None

    boxes = np.array([all_detections[i] for i in best_indices])
    x, y, w, h = np.median(boxes, axis=0)

    pad_w = w * config.FACE_BOX_PADDING_FRACTION
    pad_h = h * config.FACE_BOX_PADDING_FRACTION
    x -= pad_w
    y -= pad_h * 1.5  # extra padding above (headroom) vs below (chest)
    w += pad_w * 2
    h += pad_h * 2.5

    print(f"Detected stable facecam box: x={x:.0f} y={y:.0f} w={w:.0f} h={h:.0f} (from {len(best_indices)}/{len(frames)} sampled frames)")
    return (int(x), int(y), int(w), int(h))


def save_debug_detection_frames(video_path, start_sec=None, end_sec=None,
                                 output_dir="debug_face_frames", num_samples=None):
    """
    Diagnostic helper: samples frames the same way detect_facecam_box does,
    runs the detector on each, and saves every frame with any detected
    face(s) drawn in green -- so you can actually SEE what the detector is
    (or isn't) picking up.
    """
    num_samples = num_samples or config.FACE_DETECTION_SAMPLES
    cascade = _load_cascade()
    if cascade is None:
        return

    frames = _sample_frames(video_path, start_sec, end_sec, num_samples)
    if not frames:
        print("Could not sample any frames -- check start_sec/end_sec and the video path.")
        return

    os.makedirs(output_dir, exist_ok=True)
    total_detections = 0
    for i, frame in enumerate(frames):
        detections = _detect_faces_in_frame(frame, cascade)
        total_detections += len(detections)
        annotated = frame.copy()
        for (x, y, w, h) in detections:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 3)
        out_path = os.path.join(output_dir, f"frame_{i:02d}_{len(detections)}_faces.png")
        cv2.imwrite(out_path, annotated)
        print(f"  frame {i:02d}: {len(detections)} face(s) detected -> {out_path}")

    print(f"\n{total_detections} total detections across {len(frames)} frames. Saved to '{output_dir}/'.")


def compute_crop_rect(source_w, source_h, target_aspect_w, target_aspect_h, center_x, center_y):
    """
    Computes the largest crop rectangle matching the target aspect ratio,
    centered at (center_x, center_y), clamped to stay within the source
    frame bounds. Used for the GAMEPLAY pane in "cover" mode.
    """
    desired_ratio = target_aspect_w / target_aspect_h

    crop_h = source_h
    crop_w = crop_h * desired_ratio
    if crop_w > source_w:
        crop_w = source_w
        crop_h = crop_w / desired_ratio

    crop_x = center_x - crop_w / 2
    crop_y = center_y - crop_h / 2
    crop_x = max(0, min(crop_x, source_w - crop_w))
    crop_y = max(0, min(crop_y, source_h - crop_h))

    crop_w = int(crop_w) - (int(crop_w) % 2)
    crop_h = int(crop_h) - (int(crop_h) % 2)
    crop_x = int(crop_x) - (int(crop_x) % 2)
    crop_y = int(crop_y) - (int(crop_y) % 2)

    return crop_x, crop_y, crop_w, crop_h


def fit_box_to_aspect(box, target_aspect_w, target_aspect_h, source_w, source_h):
    """
    Expands the given (x, y, w, h) box -- keeping its center fixed -- just
    enough to match the target aspect ratio, WITHOUT ballooning up to the
    size of the source frame. Used for the FACECAM pane: a tight zoom on
    the actual small facecam region, not a crop sized off the whole frame.
    """
    x, y, w, h = box
    desired_ratio = target_aspect_w / target_aspect_h
    current_ratio = w / h

    if current_ratio < desired_ratio:
        new_h = h
        new_w = h * desired_ratio
    else:
        new_w = w
        new_h = w / desired_ratio

    new_w = min(new_w, source_w)
    new_h = min(new_h, source_h)

    cx, cy = x + w / 2, y + h / 2
    new_x = cx - new_w / 2
    new_y = cy - new_h / 2
    new_x = max(0, min(new_x, source_w - new_w))
    new_y = max(0, min(new_y, source_h - new_h))

    new_w = int(new_w) - (int(new_w) % 2)
    new_h = int(new_h) - (int(new_h) % 2)
    new_x = int(new_x) - (int(new_x) % 2)
    new_y = int(new_y) - (int(new_y) % 2)

    return new_x, new_y, new_w, new_h


def build_vertical_filter_complex(source_w, source_h, facecam_box, ass_path=None,
                                   target_w=None, target_h=None,
                                   facecam_fraction=None, gameplay_mode="contain"):
    """
    Builds the ffmpeg -filter_complex string for the split_screen layout:
    facecam cropped tightly into the top pane, gameplay into the bottom.

    gameplay_mode:
      "contain" (default) -- full width of the gameplay frame always
          visible, letterboxed with a blurred copy of itself. Nothing at
          the edges ever gets cropped out.
      "cover" -- crops gameplay to fill the pane completely (can lose
          action at the edges that got cropped).
    """
    target_w = target_w or config.TARGET_WIDTH
    target_h = target_h or config.TARGET_HEIGHT
    facecam_fraction = facecam_fraction if facecam_fraction is not None else config.FACECAM_PANE_FRACTION

    face_pane_h = int(target_h * facecam_fraction)
    face_pane_h -= face_pane_h % 2
    game_pane_h = target_h - face_pane_h
    game_pane_h -= game_pane_h % 2

    fx, fy, fw, fh = facecam_box
    face_center_x = fx + fw / 2
    fcx, fcy, fcw, fch = fit_box_to_aspect((fx, fy, fw, fh), target_w, face_pane_h, source_w, source_h)

    filter_parts = [
        f"[0:v]crop={fcw}:{fch}:{fcx}:{fcy},scale={target_w}:{face_pane_h}[face]",
    ]

    if gameplay_mode == "cover":
        if face_center_x < source_w / 2:
            game_center_x = source_w * 0.55
        else:
            game_center_x = source_w * 0.45
        game_center_y = source_h / 2
        gcx, gcy, gcw, gch = compute_crop_rect(source_w, source_h, target_w, game_pane_h, game_center_x, game_center_y)
        filter_parts.append(f"[0:v]crop={gcw}:{gch}:{gcx}:{gcy},scale={target_w}:{game_pane_h}[game]")
    else:
        filter_parts.append(
            f"[0:v]scale={target_w}:{game_pane_h}:force_original_aspect_ratio=increase,"
            f"crop={target_w}:{game_pane_h},gblur=sigma=25[gamebg]"
        )
        filter_parts.append(
            f"[0:v]scale={target_w}:{game_pane_h}:force_original_aspect_ratio=decrease[gamefg]"
        )
        filter_parts.append("[gamebg][gamefg]overlay=(W-w)/2:(H-h)/2[game]")

    filter_parts.append("[face][game]vstack=inputs=2[stacked]")

    if ass_path:
        escaped = captions.escape_ffmpeg_filter_path(ass_path)
        filter_parts.append(f"[stacked]ass='{escaped}'[vout]")
    else:
        filter_parts.append("[stacked]copy[vout]")

    return ";".join(filter_parts)


def build_blurred_background_filter_complex(ass_path=None, target_w=None, target_h=None):
    """
    Whole frame centered in the vertical canvas (scaled to fit the width),
    with a heavily blurred, cropped-to-fill copy of the same frame behind
    it as background. Preserves the entire original frame.
    """
    target_w = target_w or config.TARGET_WIDTH
    target_h = target_h or config.TARGET_HEIGHT

    filter_parts = [
        f"[0:v]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
        f"crop={target_w}:{target_h},gblur=sigma=25[bg]",
        f"[0:v]scale={target_w}:-2:force_original_aspect_ratio=decrease[fg]",
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[stacked]",
    ]

    if ass_path:
        escaped = captions.escape_ffmpeg_filter_path(ass_path)
        filter_parts.append(f"[stacked]ass='{escaped}'[vout]")
    else:
        filter_parts.append("[stacked]copy[vout]")

    return ";".join(filter_parts)


def _generate_ass_for_clip(clip_words, clip_start, clip_end, output_path, target_w, target_h):
    """Shared helper: generates captions (per config.CAPTION_STYLE) sized for the vertical canvas."""
    if not clip_words:
        return None
    ass_path = os.path.splitext(output_path)[0] + ".ass"
    ok = captions.generate_captions(
        clip_words, clip_start, clip_end, ass_path,
        video_width=target_w, video_height=target_h,
    )
    return os.path.abspath(ass_path) if ok else None


def _run_ffmpeg(video_path, clip_start, clip_end, filter_complex, output_path):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostats",
        "-ss", str(clip_start), "-i", video_path, "-to", str(clip_end - clip_start),
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "0:a",
    ] + config.ffmpeg_video_codec_args() + config.ffmpeg_audio_codec_args() + [output_path]
    subprocess.run(cmd, check=True)


def render_blurred_background_clip(video_path, clip_start, clip_end, output_path,
                                    clip_words=None, target_w=None, target_h=None):
    """
    "Whole frame centered, blurred copy of itself filling the bars" layout.
    Doesn't touch face detection at all.
    """
    target_w = target_w or config.TARGET_WIDTH
    target_h = target_h or config.TARGET_HEIGHT
    abs_ass_path = _generate_ass_for_clip(clip_words, clip_start, clip_end, output_path, target_w, target_h)

    filter_complex = build_blurred_background_filter_complex(
        ass_path=abs_ass_path, target_w=target_w, target_h=target_h,
    )
    _run_ffmpeg(video_path, clip_start, clip_end, filter_complex, output_path)


def render_vertical_clip(video_path, clip_start, clip_end, output_path,
                          facecam_box=None, clip_words=None,
                          target_w=None, target_h=None,
                          facecam_fraction=None,
                          allow_blurred_fallback=True,
                          gameplay_mode="contain"):
    """
    Full pipeline for one clip: use the given facecam box (or try to
    detect one), generate captions sized for the vertical canvas, and
    render in one ffmpeg pass.

    Pass facecam_box=(x,y,w,h) to use a manually-specified box (recommended
    -- see save_reference_frame()). Passing facecam_box=None attempts
    auto-detection first; if that fails and allow_blurred_fallback is True
    (default), falls back to the blurred-background layout instead of
    raising.
    """
    target_w = target_w or config.TARGET_WIDTH
    target_h = target_h or config.TARGET_HEIGHT
    abs_ass_path = _generate_ass_for_clip(clip_words, clip_start, clip_end, output_path, target_w, target_h)

    if facecam_box is None:
        facecam_box = detect_facecam_box(video_path, clip_start, clip_end)

    if facecam_box is None:
        if not allow_blurred_fallback:
            raise ValueError(
                "No stable facecam position detected/provided for this clip, and "
                "allow_blurred_fallback=False. Pass an explicit facecam_box=(x,y,w,h)."
            )
        print("  No facecam box available -- falling back to blurred-background layout.")
        filter_complex = build_blurred_background_filter_complex(
            ass_path=abs_ass_path, target_w=target_w, target_h=target_h,
        )
    else:
        cap = cv2.VideoCapture(video_path)
        source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        filter_complex = build_vertical_filter_complex(
            source_w, source_h, facecam_box,
            ass_path=abs_ass_path,
            target_w=target_w, target_h=target_h, facecam_fraction=facecam_fraction,
            gameplay_mode=gameplay_mode,
        )

    _run_ffmpeg(video_path, clip_start, clip_end, filter_complex, output_path)
    return facecam_box


def save_reference_frame(video_path, at_sec, output_path="facecam_reference_frame.png"):
    """
    Saves a single frame as a PNG so you can open it in any image viewer/
    editor and read off the facecam's pixel coordinates (x, y, width,
    height) for config.MANUAL_FACECAM_BOX.
    """
    frames = _sample_frames(video_path, at_sec, at_sec + 0.1, num_samples=1)
    if not frames:
        print(f"Could not grab a frame at {at_sec}s.")
        return None
    cv2.imwrite(output_path, frames[0])
    h, w = frames[0].shape[:2]
    print(f"Saved {output_path} ({w}x{h}). Open it and note the facecam's pixel box as (x, y, width, height).")
    return output_path


if __name__ == "__main__":
    # Example / smoke test: reframe one clip window from config.DEFAULT_VIDEO_PATH.
    VIDEO_PATH = config.DEFAULT_VIDEO_PATH
    CLIP_START = 643.6
    CLIP_END = 663.8

    box = detect_facecam_box(VIDEO_PATH, CLIP_START, CLIP_END)
    if box is None:
        print("No facecam detected -- try widening the sample window or pass a manual box.")
    else:
        out_path = "clips_vertical_test/vertical_test.mp4"
        os.makedirs("clips_vertical_test", exist_ok=True)
        render_vertical_clip(VIDEO_PATH, CLIP_START, CLIP_END, out_path, facecam_box=box)
        print(f"Rendered {out_path}")
