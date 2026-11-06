"""Search Stream: local semantic video retrieval and consent-based face similarity demo."""

from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Search Stream", page_icon="🎞️", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#f5f0ff; --muted:#b7abc9; }
.stApp { background:radial-gradient(ellipse at 76% 0%,#482276 0,transparent 33%),radial-gradient(ellipse at 0% 35%,#252043 0,transparent 35%),#100d18; color:var(--ink); font-family:'Manrope',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#1a1525,#120f1b); border-right:1px solid #ffffff15; }
.block-container { max-width:1240px; padding-top:2.2rem; padding-bottom:3rem; }
h1,h2,h3,p,label,span { font-family:'Manrope',sans-serif; }
.hero { position:relative; overflow:hidden; padding:30px 34px 28px; border:1px solid #d5b5ff44; border-radius:25px; background:linear-gradient(112deg,#342044e8,#21182df2 57%,#39204bea); box-shadow:0 24px 80px #05030c88; margin:0 0 22px; }
.hero:after { content:'◉'; position:absolute; right:30px; top:4px; font-size:150px; color:#d8b9ff0b; line-height:1; }
.eyebrow { color:#dcbbff; font:500 11px 'DM Mono',monospace; letter-spacing:.22em; text-transform:uppercase; margin-bottom:14px; }
.hero h1 { margin:0; color:#fff; font-size:clamp(34px,5vw,56px); font-weight:800; letter-spacing:-.055em; line-height:1.05; }
.hero p { color:#c7bad8; margin:12px 0 0; font-size:15px; max-width:640px; }
.hero-tags { display:flex; gap:9px; flex-wrap:wrap; margin-top:22px; }
.tag { border:1px solid #e2caff35; border-radius:999px; padding:7px 11px; color:#e6d7fa; font:11px 'DM Mono',monospace; background:#ffffff08; }
.notice { border:1px solid #d7a9ff33; border-left:3px solid #d88bff; border-radius:12px; padding:12px 15px; background:#a263d212; color:#d8cae7; font-size:12px; margin:0 0 21px; }
.section-kicker { color:#a98bc8; font:11px 'DM Mono',monospace; text-transform:uppercase; letter-spacing:.17em; margin:19px 0 8px; }
.stButton>button { border-radius:12px; border:1px solid #dcb8ff66; background:linear-gradient(100deg,#9856df,#d36ce8); color:white; font-weight:800; min-height:46px; box-shadow:0 9px 24px #9b51d633; }
[data-testid="stFileUploaderDropzone"] { border:1px dashed #b792db77; border-radius:16px; background:#ffffff06; }
[data-testid="stMetric"] { background:#ffffff08; border:1px solid #ffffff14; border-radius:14px; padding:13px 16px; }
[data-testid="stMetricValue"] { color:#efdcff; font-family:'DM Mono',monospace; }
[data-testid="stImage"] img { border-radius:15px; border:1px solid #dec5ff40; }
.stTextInput input { background:#1a1422; color:#fff; border:1px solid #ffffff22; border-radius:12px; }
hr { border-color:#ffffff17; }
</style>
<div class="hero"><div class="eyebrow">LOCAL VIDEO INTELLIGENCE · SEARCH STREAM</div><h1>Find the moment.<br>Skip the scroll.</h1><p>Search sampled video frames by visual description or compare them with one consented reference photo.</p><div class="hero-tags"><span class="tag">CLIP SEMANTIC SEARCH</span><span class="tag">FACENET SIMILARITY</span><span class="tag">TIMESTAMPED RESULTS</span></div></div>
<div class="notice">COURSEWORK PROTOTYPE · Use only authorized footage and consented reference images. Similarity scores can be wrong and are not identity verification.</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading CLIP model (first run downloads model weights)...")
def load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    name="openai/clip-vit-base-patch32"
    return CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name), torch

@st.cache_resource(show_spinner="Loading face detector and embedding model...")
def load_face_models():
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN
    detector=MTCNN(keep_all=True,device="cuda:0" if torch.cuda.is_available() else "cpu")
    embedder=InceptionResnetV1(pretrained="vggface2").eval().to(detector.device)
    return detector,embedder,torch

def extract_frames(path: str, interval: float, max_frames: int):
    import cv2
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): raise ValueError("Could not open this video. Try an MP4 encoded with H.264.")
    fps=cap.get(cv2.CAP_PROP_FPS) or 0
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration=total/fps if fps else 0
    if not duration:
        cap.release()
        raise ValueError("Could not read video duration or frame rate.")
    step=max(1,int(fps*interval)); frames=[]
    for frame_no in range(0,total,step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no); ok,frame=cap.read()
        if ok: frames.append((frame_no/fps,Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))))
        if len(frames)>=max_frames: break
    cap.release()
    return frames,duration

def semantic_search(frames,query,threshold):
    model,processor,torch=load_clip(); images=[img for _,img in frames]
    inputs=processor(text=[query],images=images,return_tensors="pt",padding=True)
    with torch.inference_mode():
        out=model(**inputs); scores=out.logits_per_image[:,0].softmax(dim=0).cpu().numpy()
    return [(frames[i][0],frames[i][1],float(scores[i])) for i in range(len(frames)) if scores[i]>=threshold]

def face_search(frames,reference,threshold):
    detector,embedder,torch=load_face_models(); ref_faces=detector(reference)
    if ref_faces is None: raise ValueError("No face detected in reference image. Use a clear, front-facing photo.")
    if ref_faces.ndim==3: ref_faces=ref_faces.unsqueeze(0)
    with torch.inference_mode():
        ref_vec=torch.nn.functional.normalize(embedder(ref_faces[:1].to(detector.device)),dim=1)
    matches=[]
    for timestamp,image in frames:
        faces=detector(image)
        if faces is None: continue
        if faces.ndim==3: faces=faces.unsqueeze(0)
        with torch.inference_mode():
            vectors=torch.nn.functional.normalize(embedder(faces.to(detector.device)),dim=1)
            similarities=torch.mm(vectors,ref_vec.T).squeeze(1).cpu().numpy()
        best=float(similarities.max())
        if best>=threshold: matches.append((timestamp,image,best))
    return matches

st.markdown('<div class="section-kicker">01 / Configure your search</div>',unsafe_allow_html=True)
with st.sidebar:
    st.header("Analysis settings")
    interval=st.slider("Sample a frame every (seconds)",0.5,10.0,2.0,0.5)
    max_frames=st.slider("Maximum frames to analyze",20,500,120,20)
    st.caption("A larger frame limit may take longer and use more memory.")
st.markdown('<div class="section-kicker">02 / Add source video</div>',unsafe_allow_html=True)
video=st.file_uploader("Drop a clip here or browse your files",type=["mp4","avi","mov","mkv"],help="Your video is processed locally by this app.")
if video:
    st.video(video)
    st.markdown('<div class="section-kicker">03 / Choose a retrieval path</div>',unsafe_allow_html=True)
    mode=st.radio("Search mode",["Description search","Reference face similarity"],horizontal=True)
    if mode=="Description search":
        query=st.text_input("Describe the person, object, or scene",placeholder="e.g., a person carrying a red backpack")
        threshold=st.slider("Minimum relative score",0.0,1.0,0.05,0.01,help="CLIP scores are relative across sampled frames, not calibrated probabilities.")
        reference=None
    else:
        reference_upload=st.file_uploader("Reference face photo (use only with consent)",type=["jpg","jpeg","png"])
        reference=Image.open(reference_upload).convert("RGB") if reference_upload else None
        threshold=st.slider("Minimum cosine similarity",0.0,1.0,0.65,0.01,help="Tune on your own test clips; there is no universally reliable threshold.")
        query="face comparison"
    if st.button("Search video",type="primary",disabled=(not query if mode=="Description search" else reference is None)):
        with tempfile.NamedTemporaryFile(delete=False,suffix=Path(video.name).suffix) as temp:
            temp.write(video.getbuffer()); temp_path=temp.name
        try:
            with st.spinner("Sampling video frames..."):
                frames,duration=extract_frames(temp_path,interval,max_frames)
            st.success(f"Sampled {len(frames)} frames · {duration:.1f} seconds of video")
            with st.spinner("Comparing frames..."):
                results=semantic_search(frames,query,threshold) if mode=="Description search" else face_search(frames,reference,threshold)
            results.sort(key=lambda row:row[2],reverse=True)
            if not results: st.info("No frames passed the selected threshold. Try lowering it or sampling more frames.")
            else:
                st.markdown('<div class="section-kicker">04 / Ranked moments</div>',unsafe_allow_html=True)
                st.subheader(f"{len(results)} moment{'s' if len(results)!=1 else ''} to review")
                for rank,(timestamp,image,score) in enumerate(results,1):
                    left,right=st.columns([1,2])
                    left.image(image,caption=f"MATCH {rank:02d} · {timestamp:.2f} seconds",use_container_width=True)
                    left.metric("Similarity score",f"{score:.3f}")
                    right.markdown(f"### MATCH {rank:02d}")
                    right.write(f"**Jump to moment** · {int(timestamp//60):02d}:{timestamp%60:05.2f}")
                    right.caption(f"Source timestamp: {timestamp:.2f}s")
                    right.progress(min(1.0,max(0.0,score)))
        except Exception as exc:
            st.error(f"Search could not finish: {exc}")
        finally: Path(temp_path).unlink(missing_ok=True)
else:
    st.markdown("### Your clip goes here
Upload a short video to begin. Start with 10-30 seconds while checking model setup and sampling speed.")
st.divider()
st.caption("Search Stream · Frames are sampled in memory and temporary video files are removed after each search.")
