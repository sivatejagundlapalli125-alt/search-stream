"""Search Stream: local semantic video retrieval and consent-based face similarity demo."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Search Stream", page_icon="🎞️", layout="wide")


@st.cache_resource(show_spinner="Loading CLIP model (first run downloads model weights)...")
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor

    name = "openai/clip-vit-base-patch32"
    return (CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name), torch)


@st.cache_resource(show_spinner="Loading face detector and embedding model...")
def load_face_models():
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN

    detector = MTCNN(keep_all=True, device="cuda:0" if torch.cuda.is_available() else "cpu")
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(detector.device)
    return detector, embedder, torch


def extract_frames(path: str, interval: float, max_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open this video. Try an MP4 encoded with H.264.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps else 0
    if not duration:
        cap.release()
        raise ValueError("Could not read video duration or frame rate.")
    step = max(1, int(fps * interval))
    frames = []
    for frame_no in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if ok:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_no / fps, Image.fromarray(rgb)))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames, duration


def semantic_search(frames, query, threshold):
    model, processor, torch = load_clip()
    images = [img for _, img in frames]
    inputs = processor(text=[query], images=images, return_tensors="pt", padding=True)
    with torch.inference_mode():
        out = model(**inputs)
        scores = out.logits_per_image[:, 0].softmax(dim=0).cpu().numpy()
    return [(frames[i][0], frames[i][1], float(scores[i])) for i in range(len(frames)) if scores[i] >= threshold]


def face_search(frames, reference, threshold):
    detector, embedder, torch = load_face_models()
    ref_faces = detector(reference)
    if ref_faces is None:
        raise ValueError("No face detected in the reference image. Use a clear, front-facing photo.")
    if ref_faces.ndim == 3:
        ref_faces = ref_faces.unsqueeze(0)
    with torch.inference_mode():
        ref_vec = embedder(ref_faces[:1].to(detector.device))
        ref_vec = torch.nn.functional.normalize(ref_vec, dim=1)
    matches = []
    for timestamp, image in frames:
        faces = detector(image)
        if faces is None:
            continue
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)
        with torch.inference_mode():
            vectors = torch.nn.functional.normalize(embedder(faces.to(detector.device)), dim=1)
            similarities = torch.mm(vectors, ref_vec.T).squeeze(1).cpu().numpy()
        best = float(similarities.max())
        if best >= threshold:
            matches.append((timestamp, image, best))
    return matches


st.title("🎞️ Search Stream")
st.caption("Search video frames by description or compare them with a reference face. Processing runs locally after model setup.")
st.warning("Academic prototype only. Use footage and face images only with appropriate consent and authorization. Similarity scores are not identity verification and can be wrong.")

with st.sidebar:
    st.header("Analysis settings")
    interval = st.slider("Sample a frame every (seconds)", 0.5, 10.0, 2.0, 0.5)
    max_frames = st.slider("Maximum frames to analyze", 20, 500, 120, 20)
    st.caption("A larger frame limit may take longer and use more memory.")

video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
if video:
    st.video(video)
    mode = st.radio("Search mode", ["Description search", "Reference face similarity"], horizontal=True)
    if mode == "Description search":
        query = st.text_input("Describe the person, object, or scene", placeholder="e.g., a person carrying a red backpack")
        threshold = st.slider("Minimum relative score", 0.0, 1.0, 0.05, 0.01, help="CLIP scores here are relative across sampled frames, not calibrated probabilities.")
        reference = None
    else:
        reference_upload = st.file_uploader("Reference face photo (use only with consent)", type=["jpg", "jpeg", "png"])
        reference = Image.open(reference_upload).convert("RGB") if reference_upload else None
        threshold = st.slider("Minimum cosine similarity", 0.0, 1.0, 0.65, 0.01, help="Tune on your own test clips; there is no universally reliable threshold.")
        query = "face comparison"

    if st.button("Search video", type="primary", disabled=(not query if mode == "Description search" else reference is None)):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.name).suffix) as temp:
            temp.write(video.getbuffer())
            temp_path = temp.name
        try:
            with st.spinner("Sampling video frames..."):
                frames, duration = extract_frames(temp_path, interval, max_frames)
            st.info(f"Sampled {len(frames)} frames from a {duration:.1f}-second video.")
            with st.spinner("Comparing frames..."):
                results = semantic_search(frames, query, threshold) if mode == "Description search" else face_search(frames, reference, threshold)
            results.sort(key=lambda row: row[2], reverse=True)
            if not results:
                st.info("No frames passed the selected threshold. Try lowering it or sampling more frames.")
            else:
                st.subheader(f"{len(results)} matching frame(s)")
                for timestamp, image, score in results:
                    left, right = st.columns([1, 2])
                    left.image(image, caption=f"{timestamp:.2f} seconds", use_container_width=True)
                    left.metric("Similarity", f"{score:.3f}")
                    right.write(f"**Timestamp:** {timestamp:.2f}s")
                    right.write(f"**Seek to:** {int(timestamp // 60):02d}:{timestamp % 60:05.2f}")
                    right.progress(min(1.0, max(0.0, score)))
        except Exception as exc:
            st.error(f"Search could not finish: {exc}")
        finally:
            Path(temp_path).unlink(missing_ok=True)
else:
    st.markdown("### Start here\nUpload a short video, choose a search mode, and review the best matching timestamps.")

st.divider()
st.caption("Search Stream · Frames are sampled in memory and temporary video files are removed after each search.")
