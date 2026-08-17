"""
Generates "TikTok-style" burned-in captions.

Two styles:
- generate_single_word_pop_ass: one word on screen at a time, popping in
  with a fade + scale-overshoot bounce (the default, see config.CAPTION_STYLE).
- generate_word_highlight_ass: short lines of a few words at once, plain
  white (no color highlighting -- that was removed; see the module
  docstring in the project history if you ever want it back).
"""

import config
import text_cleanup


def _ass_time(seconds):
    """Formats seconds as ASS timestamp: H:MM:SS.CC (centiseconds)."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    centis = int(round((secs - int(secs)) * 100))
    if centis == 100:
        centis = 0
        secs = int(secs) + 1
    return f"{hours}:{minutes:02d}:{int(secs):02d}.{centis:02d}"


def _ass_style_color(rgb, alpha=0):
    """Style-line colors include an alpha byte: &HAABBGGRR."""
    r, g, b = rgb
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def _ass_escape(text):
    """Escapes characters that have special meaning inside ASS text fields."""
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\n", "\\N")
    return text


def _display_token(word_text):
    token = text_cleanup.clean_caption_token(word_text)
    token = token.upper() if config.UPPERCASE else token
    return _ass_escape(token)


def group_words_into_lines(words, max_words=None, max_gap=None):
    """
    Splits a flat list of word dicts ({"start", "end", "text"}) into short
    caption lines: breaks on a long pause between words, or once a line hits
    max_words, whichever comes first.
    """
    max_words = max_words if max_words is not None else config.MAX_WORDS_PER_LINE
    max_gap = max_gap if max_gap is not None else config.MAX_GAP_BETWEEN_WORDS_SEC

    lines = []
    current = []
    prev_end = None

    for w in words:
        if current and (len(current) >= max_words or (prev_end is not None and w["start"] - prev_end > max_gap)):
            lines.append(current)
            current = []
        current.append(w)
        prev_end = w["end"]

    if current:
        lines.append(current)

    return lines


def build_ass_header(video_width, video_height, fontsize_fraction, margin_v_fraction):
    fontsize = max(18, int((video_height or 1080) * fontsize_fraction))
    margin_v = max(10, int((video_height or 1080) * margin_v_fraction))
    play_res_x = video_width or 1920
    play_res_y = video_height or 1080

    primary = _ass_style_color(config.BASE_COLOR)
    outline = _ass_style_color(config.OUTLINE_COLOR)
    secondary = _ass_style_color(config.BASE_COLOR)  # unused by our approach, required field
    back = _ass_style_color((0, 0, 0), alpha=0x80)
    bold_flag = -1 if config.BOLD else 0

    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{config.FONT_NAME},{fontsize},{primary},{secondary},{outline},{back},{bold_flag},0,0,0,100,100,0,0,1,{config.OUTLINE_WIDTH},{config.SHADOW},{config.ALIGNMENT},20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def _clip_relative_words(clip_words, clip_start, clip_end):
    """Converts absolute-timestamped words to clip-relative, dropping anything outside the clip."""
    clip_duration = max(0.0, clip_end - clip_start)
    words = []
    for w in clip_words:
        rel_start = max(0.0, w["start"] - clip_start)
        rel_end = min(clip_duration, w["end"] - clip_start)
        if rel_end <= rel_start:
            continue
        text = w["text"].strip()
        if not text:
            continue
        words.append({"start": rel_start, "end": rel_end, "text": text})
    return words, clip_duration


def generate_word_highlight_ass(clip_words, clip_start, clip_end, output_path,
                                 video_width=None, video_height=None):
    """
    Writes an .ass file for one clip: short lines of a few words at a time,
    plain white, no per-word color highlighting. `clip_words` is a list of
    word dicts with absolute {"start", "end", "text"} timestamps.

    Returns True if any caption lines were written, False if there were no
    words in range (caller should skip adding the -vf filter in that case).
    """
    words, clip_duration = _clip_relative_words(clip_words, clip_start, clip_end)
    if not words:
        return False

    lines = group_words_into_lines(words)

    events = []
    for line_idx, line_words in enumerate(lines):
        display_tokens = [_display_token(w["text"]) for w in line_words]
        next_line_start = lines[line_idx + 1][0]["start"] if line_idx + 1 < len(lines) else None

        line_start = line_words[0]["start"]
        natural_end = line_words[-1]["end"] + config.TRAILING_HOLD_SEC
        # Cap at the next line's start no matter what, so a line that only
        # broke because it hit MAX_WORDS_PER_LINE (not because of an actual
        # pause) can never overlap the line right after it.
        line_end = min(natural_end, next_line_start) if next_line_start is not None else natural_end
        line_end = min(line_end, clip_duration)
        line_end = max(line_end, line_start + 0.05)

        text = " ".join(display_tokens)
        events.append(f"Dialogue: 0,{_ass_time(line_start)},{_ass_time(line_end)},Default,,0,0,0,,{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(build_ass_header(video_width, video_height, config.LINE_FONTSIZE_FRACTION, config.LINE_MARGIN_V_FRACTION))
        f.write("\n".join(events))
        f.write("\n")

    return True


def generate_single_word_pop_ass(clip_words, clip_start, clip_end, output_path,
                                  video_width=None, video_height=None):
    """
    ONE word on screen at a time, popping in with a quick fade + scale-
    overshoot "bounce" as each word is spoken, then replaced by the next
    word. Matches the common CapCut/Reels-style single-word caption look.

    Same input/output contract as generate_word_highlight_ass.
    """
    words, clip_duration = _clip_relative_words(clip_words, clip_start, clip_end)
    if not words:
        return False

    events = []
    for i, w in enumerate(words):
        event_start = w["start"]
        natural_end = w["end"] + config.POP_TRAILING_HOLD_SEC
        if i + 1 < len(words):
            next_start = words[i + 1]["start"]
            gap = next_start - w["end"]
            if gap > config.MAX_GAP_BETWEEN_WORDS_SEC:
                # Big pause before the next word -- hold briefly, then let
                # the screen go blank until the next word actually pops in,
                # rather than leaving this word glued to the screen.
                event_end = natural_end
            else:
                # Normal inter-word gap -- extend seamlessly so there's no
                # distracting blank flash.
                event_end = next_start
        else:
            event_end = min(clip_duration, natural_end)
        event_end = max(event_end, event_start + 0.05)

        token = _display_token(w["text"])

        # Pop-in animation: start small + invisible, fade in fast, overshoot
        # past 100% scale, then settle back down -- a cheap but convincing
        # "bounce" using two chained \t transforms.
        anim = (
            f"\\fscx{config.POP_START_SCALE}\\fscy{config.POP_START_SCALE}"
            f"\\fad({config.POP_FADE_IN_MS},0)"
            f"\\t(0,{config.POP_OVERSHOOT_MS},\\fscx{config.POP_OVERSHOOT_SCALE}\\fscy{config.POP_OVERSHOOT_SCALE})"
            f"\\t({config.POP_OVERSHOOT_MS},{config.POP_OVERSHOOT_MS + config.POP_SETTLE_MS},"
            f"\\fscx{config.POP_SETTLE_SCALE}\\fscy{config.POP_SETTLE_SCALE})"
        )
        text = f"{{{anim}}}{token}"

        events.append(f"Dialogue: 0,{_ass_time(event_start)},{_ass_time(event_end)},Default,,0,0,0,,{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(build_ass_header(video_width, video_height, config.POP_FONTSIZE_FRACTION, config.POP_MARGIN_V_FRACTION))
        f.write("\n".join(events))
        f.write("\n")

    return True


def generate_captions(clip_words, clip_start, clip_end, output_path, video_width=None, video_height=None):
    """Dispatches to the right generator based on config.CAPTION_STYLE."""
    if config.CAPTION_STYLE == "line_highlight":
        return generate_word_highlight_ass(clip_words, clip_start, clip_end, output_path, video_width, video_height)
    return generate_single_word_pop_ass(clip_words, clip_start, clip_end, output_path, video_width, video_height)


def escape_ffmpeg_filter_path(path):
    """
    Escapes a filesystem path for safe use inside an ffmpeg -vf filter
    argument (e.g. ass=<path>). Windows drive-letter colons and backslashes
    both need escaping inside a filtergraph string, and the whole path
    needs its own single quotes.
    """
    path = path.replace("\\", "/")
    path = path.replace(":", "\\:")
    path = path.replace("'", "\\'")
    return path
