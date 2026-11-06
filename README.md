# Search Stream

**Find the moment. Skip the scroll.** Search sampled video frames using a visual description or compare them with one consented reference photo. Search Stream is a local Streamlit coursework prototype that returns ranked moments with timestamps.

## What it does

- Finds scenes and objects with CLIP text-to-image similarity.
- Compares faces in sampled frames with one reference using MTCNN and FaceNet.
- Lets you tune frame intervals, frame limits, and match thresholds.
- Displays preview frames, relative scores, and source timestamps.
- Processes video locally and removes the temporary decoding file after each search.

## Requirements

- Python 3.10–3.12 (Python 3.12 recommended).
- Internet on first run to download model weights.
- H.264 MP4 is the most reliable video format.
- 8 GB RAM recommended; CPU works, though a GPU can speed up inference.

## Run on Windows

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, use the environment Python directly:

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

Upload a short clip, choose **Description search** or **Reference face similarity**, adjust settings, and click **Search video**. Start with a 10–30 second clip. Model weights are cached after the first download.

## How it works

OpenCV samples frames at the selected interval. Description search embeds the prompt and frames with CLIP and ranks their similarity. Face search detects faces, embeds them with FaceNet, and ranks cosine similarity against the reference. The CLIP score is relative across sampled frames, not a calibrated probability. Face similarity is not identity verification; thresholds can produce false matches or misses. Review every result manually.

## Privacy and responsible use

Use only footage you are authorized to process and reference images used with consent. This prototype does not identify people by name, maintain an identity gallery, or make consequential decisions. Video is written temporarily for decoding and removed after a search; model weights remain cached by their libraries.

## Limitations

Sparse sampling can miss brief events. Blur, occlusion, lighting, camera angle, compression, and prompt wording affect results. Long or high-resolution videos can require substantial time and memory. This is a coursework prototype, not a production system.
