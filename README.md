# Search Stream

A local academic prototype for finding moments in a video using either a natural-language description or visual similarity to one reference face. It samples frames, scores them, and shows the strongest matches with timestamps.

## Features

- Text-to-frame semantic retrieval using OpenAI CLIP (`openai/clip-vit-base-patch32`).
- Single-reference face similarity using MTCNN face detection and FaceNet embeddings pretrained on VGGFace2.
- Adjustable frame sampling, frame limit, and score threshold.
- Local Streamlit interface; no video upload to a hosted application.
- Temporary video file is removed after a search finishes.

## Requirements

- Python 3.10 or later (Python 3.11 is a good choice).
- FFmpeg-compatible codecs through OpenCV. H.264 MP4 is the most reliable input format.
- Internet access on the first run to download model weights. Later runs can use the cached weights offline.
- 8 GB RAM recommended. A CUDA GPU is optional; CPU inference works but can be slow.

## Run on Windows

Open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, run the venv's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit opens the interface in a browser. Upload a short video, choose a mode, tune the settings, and click **Search video**. Start with a 10-30 second clip while checking installation and model downloads.

## How it works

1. OpenCV samples frames at the selected interval, up to the chosen frame limit.
2. **Description search:** CLIP encodes the description and each sampled frame into a shared feature space; the app ranks frames by image-text similarity.
3. **Face similarity:** MTCNN detects faces, FaceNet encodes the reference and detected faces, then cosine similarity ranks the closest sampled faces.
4. Results include a preview, score, and timestamp so a reviewer can seek to the original moment.

The CLIP score is a relative softmax over the sampled frames. It is not a calibrated probability. Face similarity is also not identity verification; its threshold must be calibrated for the chosen footage and may yield false matches or misses.

## Privacy and responsible use

Use only footage and reference images you are authorized to process, and obtain consent where required. Keep this prototype to controlled coursework and consent-based demonstrations. It does not identify a person by name, keep an identity gallery, or make decisions about people. Do not treat results as evidence or as a basis for enforcement or other consequential action. Review every result manually. The app stores neither search results nor user uploads; the uploaded video is written to a temporary file for decoding and removed when the search ends. Model weights are cached by their libraries.

## Limitations

- Sparse frame sampling can miss a short event between sampled frames.
- Motion blur, occlusion, camera angle, lighting, compression, and demographic differences can reduce face similarity reliability.
- CLIP can confuse visually similar concepts and depends on prompt wording.
- Long or high-resolution videos can use substantial memory and processing time.
- This prototype does not provide multi-camera tracking, identity management, audit logs, access control, or production security controls.

## Project structure

```text
search-stream/
  app.py          Streamlit interface and video search pipelines
  requirements.txt
  README.md
  PROJECT_REPORT.md
```
