# --- USAGE ---
# Local file you already have:
#     .\pipeline.ps1 -VideoPath "downloads/some_vod.webm"
# Download from a URL first (e.g. a fresh cloud box with nothing local yet):
#     .\pipeline.ps1 -Url "https://www.youtube.com/watch?v=..."
# --------------

param(
    [string]$VideoPath,
    [string]$Url
)

if ($Url) {
    Write-Host "Downloading video from: $Url..." -ForegroundColor Cyan
    $downloadOutput = python getVideo.py $Url
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error: getVideo.py failed. Stopping pipeline." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    # getVideo.py prints the downloaded file's path as its last line of output.
    $VideoPath = ($downloadOutput | Select-Object -Last 1).Trim()
    Write-Host "Downloaded to: $VideoPath" -ForegroundColor Green
} elseif (-not $VideoPath) {
    Write-Host "Error: pass either -VideoPath <local file> or -Url <video url>." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $VideoPath)) {
    Write-Host "Error: video file not found at '$VideoPath'." -ForegroundColor Red
    exit 1
}

# Automatically derive the folder base name (e.g. "caseoh_vod" from "downloads/caseoh_vod.webm")
$VideoName = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)
$ManifestPath = "$VideoName/clips_manifest.json"

Write-Host "Starting main.py pipeline for: $VideoPath..." -ForegroundColor Cyan
python main.py $VideoPath

if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: main.py failed. Stopping pipeline." -ForegroundColor Red
    exit $LASTEXITCODE
}

if (Test-Path $ManifestPath) {
    Write-Host "`nFound manifest: $ManifestPath" -ForegroundColor Green
    Write-Host "Starting vertical batch conversion..." -ForegroundColor Cyan

    python run_vertical_batch.py $ManifestPath

    Write-Host "`nPipeline complete! Check $VideoName/clips/ and $VideoName/clips_vertical/." -ForegroundColor Green
} else {
    Write-Host "Error: Could not find generated manifest file at $ManifestPath" -ForegroundColor Red
}
