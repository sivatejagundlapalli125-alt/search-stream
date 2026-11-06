# Search Stream: Clip-Enhanced Semantic Video Retrieval and Face Similarity

## Abstract

Search Stream is a local prototype that helps a reviewer locate potentially relevant moments in a video. It provides two search paths: natural-language frame retrieval with CLIP, and comparison of detected faces with one consented reference image using FaceNet embeddings. Results are presented as sampled frame previews with timestamps. This design demonstrates how pretrained visual models can reduce manual browsing while keeping a human reviewer in control.

## Problem statement

Reviewing video frame by frame is slow. Basic keyword or object filters cannot express many visual descriptions. Search Stream explores whether pretrained image-text representations can rank video frames for a natural-language query, alongside a separate face-similarity path.

## Objectives

1. Accept a local video and sample frames at a user-selected interval.
2. Retrieve frames relevant to an open-ended text description.
3. Compare detected faces with one reference photo and return likely visual similarities.
4. Display results with timestamps to support manual review.
5. Keep video analysis local and explain model limitations.

## System design

```text
Video upload
    |
OpenCV frame sampling
    |
Search mode selection
    +-------------------------+
    |                         |
Text description             Reference face photo
    |                         |
CLIP text + image features    MTCNN face detection
    |                         |
Cosine similarity ranking    FaceNet face embeddings
    |                         |
    +-----------+-------------+
                |
       Ranked frame previews
       + timestamps and scores
```

## Methods

### Semantic retrieval

The application encodes a text prompt and each sampled RGB frame with CLIP ViT-B/32. CLIP image-text similarity scores are converted to a softmax distribution across the sampled frames for ranking. The displayed value is relative to this video sample and must not be interpreted as a probability of a match.

### Reference-face similarity

MTCNN detects and aligns faces in the reference image and sampled frames. InceptionResnetV1 pretrained on VGGFace2 maps each face to an embedding. The app compares embeddings using cosine similarity and shows frames above the user-selected threshold. This is an exploratory similarity measure, not reliable identity verification.

### Video sampling

OpenCV seeks through the video at a configurable time interval. Sampling bounds processing time, but a brief event may fall between sampled frames. The displayed timestamp refers to the sampled frame's location in the uploaded video.

## Tools and technologies

- Python, Streamlit
- OpenCV and Pillow
- PyTorch and Transformers
- CLIP ViT-B/32
- facenet-pytorch (MTCNN and InceptionResnetV1)

## Evaluation plan

Use short, consented test clips with manually annotated moments. For semantic retrieval, record whether a relevant frame appears in the top 1, top 5, and top 10 results, and compare precision/recall at several sampling intervals. For face similarity, use a consented reference and labeled same-person/different-person frames; report false-match and missed-match rates across thresholds. Repeat under changes in lighting, distance, pose, blur, and occlusion. Record hardware, model versions, frame interval, thresholds, and processing time. Do not claim accuracy until these experiments have been run.

## Expected outcome

The deliverable is a runnable local demonstration that ranks sampled frames and helps a reviewer navigate the source video. No measured performance results are claimed in this report because no labeled evaluation footage was supplied.

## Limitations and ethical safeguards

Sampling can miss events; CLIP can misread prompts and scenes; face similarity can produce false matches or misses. Video quality and demographic variation can affect performance. This prototype has no cross-camera tracking, person database, or automated decision logic. It is intended for controlled coursework demonstrations using authorized footage and consented reference images. All output requires human review.

## Conclusion

Search Stream demonstrates two complementary retrieval methods within a small local interface. Its main contribution is a clear, inspectable workflow from video sampling to timestamped results. Evaluation with consented, labeled footage is required before drawing claims about usefulness or accuracy.
