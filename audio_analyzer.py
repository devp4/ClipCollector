import os
import subprocess
import librosa
import numpy as np
from scipy.signal import find_peaks

import config


def format_seconds_to_readable(seconds):
    """
    Converts raw seconds into HH:MM:SS.ss format. Always includes the hours
    component (even if 00) so timestamps stay unambiguous when scanning
    logs/filenames for a multi-hour VOD.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def format_seconds_to_filename(seconds):
    """
    Same as format_seconds_to_readable but filesystem-safe (no colons/dots),
    e.g. 1h23m45s -- handy for embedding a timestamp in a clip filename.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def get_video_resolution(video_path):
    """
    Uses ffprobe to get (width, height) of the video stream. Used to scale
    burned-in caption font size/margins correctly regardless of source
    resolution. Returns (None, None) if it can't be determined.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        width_str, height_str = result.stdout.strip().split("x")
        return int(width_str), int(height_str)
    except (ValueError, TypeError):
        print("Warning: could not determine video resolution via ffprobe.")
        return None, None


def get_video_duration(video_path):
    """Uses ffprobe to get the exact duration of a video/audio file in seconds."""
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except (ValueError, TypeError):
        print("Warning: could not determine video duration via ffprobe.")
        return None


def detect_loud_spikes(video_path):
    """
    Scans the audio track for volume spikes. Returns the raw signal (y, sr) too,
    so callers can run spectral classification on spikes without reloading audio.
    """
    print(f"Scanning audio track for extreme volume spikes (screams/loud noises) for video {video_path}...")

    temp_audio = "temp_audio.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
        temp_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    y, sr = librosa.load(temp_audio, sr=22050, mono=True)

    if os.path.exists(temp_audio):
        os.remove(temp_audio)

    hop_length = config.SPIKE_HOP_LENGTH
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    mean_rms = np.mean(rms)
    std_rms = np.std(rms)
    threshold = max(mean_rms + (std_rms * config.SPIKE_THRESHOLD_MULTIPLIER), config.SPIKE_MIN_RMS_FLOOR)

    # scipy's find_peaks automatically finds the true local maximum of clusters!
    min_distance_samples = int((sr / hop_length) * config.SPIKE_MIN_GAP_SEC)
    spike_indices, _ = find_peaks(rms, height=threshold, distance=min_distance_samples)

    spike_timestamps = times[spike_indices].tolist()

    print(f"\nFiltered down to {len(spike_timestamps)} major volume spikes.")
    for t in spike_timestamps:
        idx = np.argmin(np.abs(times - t))
        readable_t = format_seconds_to_readable(t)
        print(f"Loudness spike detected at {readable_t} ({t:.1f}s) with an RMS energy score of {rms[idx]:.4f}")

    return times, rms, mean_rms, threshold, spike_timestamps, y, sr


def classify_spike(y, sr, t):
    """
    Cheap heuristic to separate screams/laughter from bass-heavy game SFX
    or impact sounds. Screams and laughter tend to have higher spectral
    centroid (brighter, more "vocal") and higher zero-crossing rate than
    low-frequency thuds/explosions.

    Returns one of: "voice_scream_or_laugh", "impact_or_bass", "unknown"
    """
    window = config.CLASSIFY_WINDOW_SEC
    center_sample = int(t * sr)
    half_window = int(window * sr)
    start = max(0, center_sample - half_window)
    end = min(len(y), center_sample + half_window)

    clip = y[start:end]
    if len(clip) < 512:
        return "unknown"

    zcr = np.mean(librosa.feature.zero_crossing_rate(clip)[0])
    centroid = np.mean(librosa.feature.spectral_centroid(y=clip, sr=sr)[0])

    if centroid > config.CLASSIFY_VOICE_CENTROID_MIN and zcr > config.CLASSIFY_VOICE_ZCR_MIN:
        return "voice_scream_or_laugh"
    elif centroid < config.CLASSIFY_IMPACT_CENTROID_MAX:
        return "impact_or_bass"
    return "unknown"


def get_spike_details(y, sr, times, rms, spike_timestamps):
    """
    Builds a structured list of spike info (peak time, RMS score, and a
    rough audio-type classification) for downstream scoring and for
    sending to Gemini as structured JSON instead of prose.
    """
    details = []
    for t in spike_timestamps:
        idx = np.argmin(np.abs(times - t))
        score = float(rms[idx])
        spike_type = classify_spike(y, sr, t)
        details.append({
            "peak": round(float(t), 2),
            "score": round(score, 4),
            "type": spike_type
        })
    return details


def snap_to_low_energy(times, rms, target_time, search_radius=2.0, mode="min"):
    """
    Nudges a proposed clip boundary to the nearest local low-energy (quiet)
    point within +/- search_radius seconds, so cuts land in natural pauses
    instead of mid-word or mid-laugh.
    """
    mask = (times >= target_time - search_radius) & (times <= target_time + search_radius)
    if not np.any(mask):
        return target_time

    window_times = times[mask]
    window_rms = rms[mask]

    if mode == "min":
        best_idx = np.argmin(window_rms)
    else:
        best_idx = np.argmax(window_rms)

    return float(window_times[best_idx])


def find_chunk_split_point(times, rms, target_time, search_radius=60.0):
    """
    Used specifically when slicing a long transcript into analysis chunks
    for Gemini (as opposed to trimming a final clip's start/end). Instead of
    always cutting at a fixed clock time -- which risks slicing a scream,
    laugh, or punchline exactly in half between two chunks -- this searches
    up to `search_radius` seconds around the target time for the quietest
    moment and proposes splitting there instead.
    """
    return snap_to_low_energy(times, rms, target_time, search_radius=search_radius, mode="min")
