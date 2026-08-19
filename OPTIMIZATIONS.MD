# GPU Optimization Roadmap for ClipCollector

Here are the prioritized improvements and architectural upgrades to leverage the RTX 3090 (24GB VRAM) for the pipeline.

---

## High-Priority / Immediate Wins

### 1. Whisper `large-v3` on GPU
* **Benefit:** Switch from CPU `medium` pass to GPU `float16` for `large-v3`. Yields better accuracy and significantly faster transcription.
* **Alternative:** Use `large-v3-turbo` if speed takes priority, which is a distilled variant offering high speed with near-identical quality.

### 2. Hardware-Accelerated FFmpeg Encoding (`NVENC`)
* **Benefit:** Swap `-c:v libx264` for `-c:v h264_nvenc` (or `hevc_nvenc`) across extraction, caption-burning, and vertical-reframe commands.
* **Impact:** Offloads CPU-bound clip exports (7-8 clips per VOD across horizontal and vertical formats) directly onto the GPU hardware encoder.

---

## Medium-Priority / Reliability Upgrades

### 3. Replace Haar Cascade with a Real GPU Face Detector
* **Benefit:** Upgrade past the reliability flaws of Haar cascade by running a lightweight GPU model like **RetinaFace** or a **YOLO-face** variant.
* **Impact:** Makes facecam auto-detection for split-screen layouts trustworthy instead of relying on fallback hardcoded manual boxes.

### 4. Machine Learning Audio Event Classifier
* **Benefit:** Replace the rough spectral-centroid/ZCR heuristic in `classify_spike()` with an actual audio tagging model or lightweight CNN.
* **Impact:** Improves guaranteed-clip and coverage-backfill logic by reliably distinguishing genuine streamer reactions from loud game explosions or bass drops.

### 5. Local LLM as a Gemini Supplement
* **Benefit:** Run a quantized 8-13B local model on the 24GB VRAM card for coarse initial scans ("is anything happening in this chunk?").
* **Impact:** Cuts API call volume and mitigates rate limits, routing only promising chunks to Gemini for final judgment calls.

---

## Advanced Enhancements

### 6. Voice Activity Detection & Diarization
* **Benefit:** Integrate tools like **Silero VAD** or `pyannote`.
* **Impact:** Separates the streamer's voice from background game audio bleed, tightening word timestamps and preventing spurious Whisper hallucinations during chaotic gameplay.

### 7. Prosody/Tone-Based "Hype" Detection
* **Benefit:** Introduce a speech emotion/prosody model as a third signal alongside RMS volume spike detection.
* **Impact:** Catches excited, panicked, or scared speech even at lower volumes (e.g., a tense, hushed whisper), providing a complementary signal to raw volume.

### 8. Smarter Gameplay Reframing (Saliency / Object Detection)
* **Benefit:** Replace fixed center-crop reframing with a lightweight saliency or object-detection pass.
* **Impact:** Dynamically biases the crop/zoom toward active on-screen action frame by frame.

### 9. Scene-Cut Detection
* **Benefit:** Implement GPU-accelerated scene detection using PySceneDetect.
* **Impact:** Flags clip boundaries that awkwardly straddle hard cut transitions, resolving frozen caption/scene mismatch bugs.

---

## Recommended Build Order

1. **Phase 1:** Whisper `large-v3` + `NVENC` encoding (Instant, low-effort architectural wins).
2. **Phase 2:** Face detector upgrade (Directly resolves known manual layout bottlenecks).
3. **Phase 3:** Audio classifier & local-LLM prefilter (Deeper pipeline enhancements once baseline performance is validated).
"""

# Write to file
with open("GPU_Optimization_Roadmap.md", "w") as f:
    f.write(markdown_content)