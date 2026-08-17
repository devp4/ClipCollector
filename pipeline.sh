#!/bin/bash

# ==========================================
# 1. SYSTEM SETUP & DEPENDENCIES
# ==========================================
echo "[1/4] Checking system dependencies..."

# Check and install ffmpeg if missing
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg not found. Installing via apt-get..."
    apt-get update && apt-get install -y ffmpeg
else
    echo "ffmpeg is already installed."
fi

# Generate requirements.txt on the fly based on the project README
echo "[2/4] Setting up Python dependencies..."
cat <<EOT > requirements.txt
faster-whisper
google-genai
librosa
scipy
numpy
tqdm
python-dotenv
opencv-python<5.0.0
yt-dlp
EOT

# Install the packages
pip install -r requirements.txt --quiet
echo "Python dependencies installed successfully."

# ==========================================
# 2. ARGUMENT PARSING
# ==========================================
# Allows using either --url "https..." OR --path "downloads/video.webm"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path|-VideoPath) VideoPath="$2"; shift ;;
        --url|-Url) Url="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# ==========================================
# 3. PIPELINE EXECUTION
# ==========================================
echo "[3/4] Starting ClipCollector Pipeline..."

if [ -n "$Url" ]; then
    echo "Downloading video from: $Url..."
    # Run getVideo.py and capture the exact file path it prints at the end
    VideoPath=$(python3 getVideo.py "$Url" | tail -n 1 | tr -d '\r')
    
    if [ $? -ne 0 ]; then
        echo "Error: getVideo.py failed to download the video."
        exit 1
    fi
    echo "Downloaded successfully to: $VideoPath"
elif [ -z "$VideoPath" ]; then
    echo "Error: You must provide either --path <local_file> or --url <video_url>."
    echo "Usage: ./pipeline.sh --url 'https://youtube.com/...' OR ./pipeline.sh --path 'downloads/video.webm'"
    exit 1
fi

if [ ! -f "$VideoPath" ]; then
    echo "Error: Video file not found at '$VideoPath'."
    exit 1
fi

# Extract the base folder name (e.g., "caseoh_vod" from "downloads/caseoh_vod.webm")
VideoName=$(basename "$VideoPath" | cut -d. -f1)
ManifestPath="$VideoName/clips_manifest.json"

# Run main extraction and transcription
echo "Starting main.py for: $VideoPath..."
python3 main.py "$VideoPath"

if [ $? -ne 0 ]; then
    echo "Error: main.py failed. Stopping pipeline."
    exit 1
fi

# ==========================================
# 4. VERTICAL CONVERSION
# ==========================================
echo "[4/4] Starting Vertical Batch Conversion..."

if [ -f "$ManifestPath" ]; then
    echo "Found manifest at: $ManifestPath"
    python3 run_vertical_batch.py "$ManifestPath"
    echo -e "\n=========================================="
    echo "PIPELINE COMPLETE!"
    echo "Check the '$VideoName/clips/' and '$VideoName/clips_vertical/' folders."
    echo "=========================================="
else
    echo "Error: Could not find generated manifest file at $ManifestPath. Did main.py fail to find clips?"
    exit 1
fi