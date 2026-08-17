import sys
import os
import yt_dlp

import config


def download_video(url, output_dir=None):
    """
    Downloads `url` with yt-dlp and returns the path to the downloaded
    file. Uses extract_info's returned metadata to determine the actual
    output filename (yt-dlp's outtmpl is a template, not a fixed name).
    """
    output_dir = output_dir or config.DOWNLOAD_DIR
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        # restrictfilenames=True converts spaces/special chars to
        # underscores and strips non-ASCII characters, so downstream
        # scripts never have to deal with awkward filenames.
        "restrictfilenames": True,
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "format": config.YT_DLP_FORMAT,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python getVideo.py <video_url>")
        sys.exit(1)

    path = download_video(sys.argv[1])
    # Printed on its own so a calling script (pipeline.ps1) can capture it.
    print(path)
