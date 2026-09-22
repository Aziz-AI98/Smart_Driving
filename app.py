"""
Smart Front-View Driving Assistant
====================================
A web application for analyzing dashcam videos using CenterNet
(Hourglass backbone) from TensorFlow Hub.

Author: Abdulaziz Alraddai
Converted to Web App for public deployment.
"""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
import tempfile
from io import BytesIO

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Smart Front-View Driving Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS for RTL Support (Arabic) and Aesthetics
# ============================================================
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .subtitle {
            text-align: center;
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 2rem;
        }
        .rtl {
            direction: rtl;
            text-align: right;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Header
# ============================================================
st.markdown('<div class="main-title">🚗 Smart Front-View Driving Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered dashcam analysis using CenterNet + IoU Tracking + Risk Assessment</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Sidebar — Settings
# ============================================================
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Detection Parameters")
    score_threshold = st.slider(
        "Detection Score Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Minimum confidence to consider a detection as valid.",
    )

    st.subheader("Tracking Parameters")
    iou_threshold = st.slider(
        "IoU Tracking Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.30,
        step=0.05,
        help="Minimum IoU to associate a detection with an existing track.",
    )
    max_missing = st.slider(
        "Max Missing Frames",
        min_value=1,
        max_value=30,
        value=12,
        step=1,
        help="Number of frames a track can survive without a detection.",
    )

    st.subheader("Risk Assessment Rules")
    st.markdown("**DANGER** if:")
    st.markdown("- Object in lane center (30%-70% width) **AND**")
    st.markdown("- Box area ratio > 7.5%")
    st.markdown("**CAUTION** if:")
    st.markdown("- In lane center **OR** area ratio > 2.5%")

    st.divider()
    st.caption("Built with ❤️ using TensorFlow, OpenCV, and Streamlit.")

# ============================================================
# COCO Labels Relevant to Driving
# ============================================================
COCO_LABELS = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
    10: "traffic light",
    13: "stop sign",
}

# ============================================================
# Model Loading (Cached)
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """Load CenterNet model from TensorFlow Hub (cached)."""
    model_url = "https://tfhub.dev/tensorflow/centernet/hourglass_512x512_kpts/1"
    detector = hub.load(model_url)
    return detector


# ============================================================
# Detection Function
# ============================================================
def detect_road_objects(frame, detector, score_threshold):
    """Run CenterNet on a single frame and return detections."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)[tf.newaxis, ...]
    output = detector(tensor)

    boxes = output["detection_boxes"][0].numpy()
    classes = output["detection_classes"][0].numpy().astype(int)
    scores = output["detection_scores"][0].numpy()

    h, w = frame.shape[:2]
    detections = []

    for box, class_id, score in zip(boxes, classes, scores):
        if score < score_threshold:
            continue
        if class_id not in COCO_LABELS:
            continue
        ymin, xmin, ymax, xmax = box
        pixel_box = [int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)]
        detections.append(
            {
                "box": pixel_box,
                "label": COCO_LABELS[class_id],
                "confidence": float(score),
            }
        )
    return detections


# ============================================================
# IoU Calculation
# ============================================================
def iou(a, b):
    """Compute Intersection over Union between two boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


# ============================================================
# IoU Tracker
# ============================================================
class IoUTracker:
    """Simple IoU-based multi-object tracker."""

    def __init__(self, threshold=0.30, max_missing=12):
        self.threshold = threshold
        self.max_missing = max_missing
        self.tracks = {}
        self.next_id = 1

    def update(self, detections):
        used = set()

        for track_id, track in list(self.tracks.items()):
            choices = [
                (iou(track["box"], d["box"]), i)
                for i, d in enumerate(detections)
                if i not in used and d["label"] == track["label"]
            ]
            best, idx = max(choices, default=(0, -1))

            if best >= self.threshold:
                self.tracks[track_id].update(box=detections[idx]["box"], missing=0)
                detections[idx]["track_id"] = track_id
                used.add(idx)
            else:
                self.tracks[track_id]["missing"] += 1

        self.tracks = {k: v for k, v in self.tracks.items() if v["missing"] <= self.max_missing}

        for i, d in enumerate(detections):
            if i not in used:
                d["track_id"] = self.next_id
                self.tracks[self.next_id] = {"box": d["box"], "label": d["label"], "missing": 0}
                self.next_id += 1

        return detections


# ============================================================
# Risk Assessment
# ============================================================
def risk_for_object(box, label, frame_w, frame_h):
    """Determine risk level for a given object."""
    x1, y1, x2, y2 = box
    area_ratio = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
    center_x = (x1 + x2) / 2
    in_lane = 0.30 * frame_w < center_x < 0.70 * frame_w
    important = label in {"person", "bicycle", "car", "motorcycle", "bus", "truck"}

    if important and in_lane and area_ratio > 0.075:
        return "DANGER", area_ratio
    if important and (in_lane or area_ratio > 0.025):
        return "CAUTION", area_ratio
    return "SAFE", area_ratio


# ============================================================
# Video Processing
# ============================================================
def process_video(input_path, output_path, detector, score_threshold, iou_threshold, max_missing, progress_bar=None):
    """Process the full video and return events DataFrame + summary stats."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    tracker = IoUTracker(threshold=iou_threshold, max_missing=max_missing)
    events = []
    frame_no = 0

    rank = {"SAFE": 0, "CAUTION": 1, "DANGER": 2}
    colors = {"SAFE": (0, 200, 0), "CAUTION": (0, 190, 255), "DANGER": (0, 0, 255)}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_no += 1
        frame_risk = "SAFE"
        detections = detect_road_objects(frame, detector, score_threshold)
        tracked = tracker.update(detections)

        for d in tracked:
            risk, area = risk_for_object(d["box"], d["label"], w, h)
            if rank[risk] > rank[frame_risk]:
                frame_risk = risk

            x1, y1, x2, y2 = d["box"]
            color = colors[risk]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"#{d['track_id']} {d['label']} {d['confidence']:.2f} {risk}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            if risk != "SAFE":
                events.append(
                    {
                        "frame": frame_no,
                        "time_seconds": round(frame_no / fps, 2),
                        "track_id": d["track_id"],
                        "object": d["label"],
                        "risk": risk,
                        "box_area_ratio": round(area, 4),
                    }
                )

        # Top banner showing overall frame risk
        cv2.rectangle(frame, (0, 0), (w, 60), colors[frame_risk], -1)
        cv2.putText(
            frame,
            f"RISK: {frame_risk}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        writer.write(frame)

        if progress_bar is not None and total_frames > 0:
            progress_bar.progress(min(frame_no / total_frames, 1.0))

    cap.release()
    writer.release()

    events_df = pd.DataFrame(events).drop_duplicates() if events else pd.DataFrame()
    return events_df, frame_no, fps


# ============================================================
# Main UI
# ============================================================
def main():
    st.markdown("---")

    # ---------- Load Model ----------
    with st.spinner("🔄 Loading CenterNet model from TensorFlow Hub... (first time may take ~1 min)"):
        try:
            detector = load_model()
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {e}")
            st.stop()

    # ---------- Upload Video ----------
    st.header("📤 Upload Your Dashcam Video")
    uploaded_file = st.file_uploader(
        "Choose a dashcam video file (MP4, AVI, MOV)",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a front-view driving video to analyze.",
    )

    if uploaded_file is None:
        st.info("👆 Please upload a dashcam video to start the analysis.")
        st.markdown(
            """
            ### 📖 How it works
            1. **Upload** a dashcam video from your device.
            2. The app runs **CenterNet** on each frame to detect vehicles, pedestrians, and traffic signs.
            3. **IoU Tracking** assigns a unique ID to each object across frames.
            4. **Risk Assessment** classifies each object as `SAFE`, `CAUTION`, or `DANGER`.
            5. **Download** the annotated video and the events CSV.
            """
        )
        return

    # Save uploaded file to a temp location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    input_path = tfile.name

    st.video(input_path, caption="📹 Input Video")

    # ---------- Run Analysis ----------
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        status_text.info("⏳ Processing video... This may take a few minutes depending on length.")

        try:
            events_df, total_frames, fps = process_video(
                input_path=input_path,
                output_path=output_path,
                detector=detector,
                score_threshold=score_threshold,
                iou_threshold=iou_threshold,
                max_missing=max_missing,
                progress_bar=progress_bar,
            )
        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
            return

        progress_bar.progress(1.0)
        status_text.success(f"✅ Done! Processed {total_frames} frames at {fps:.1f} FPS.")

        st.markdown("---")
        st.header("📊 Analysis Results")

        # ---------- Summary Metrics ----------
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Frames", f"{total_frames}")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("Total Events", f"{len(events_df)}")
        with col4:
            if not events_df.empty:
                st.metric("Unique Objects", f"{events_df['track_id'].nunique()}")
            else:
                st.metric("Unique Objects", "0")

        # ---------- Event Breakdown ----------
        if not events_df.empty:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Risk Distribution")
                st.bar_chart(events_df["risk"].value_counts())
            with col_b:
                st.subheader("Object Types")
                st.bar_chart(events_df["object"].value_counts())

        # ---------- Annotated Video ----------
        st.subheader("🎬 Annotated Output Video")
        st.video(output_path)

        # ---------- Download Buttons ----------
        col_x, col_y = st.columns(2)

        with col_x:
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Video (MP4)",
                    data=f.read(),
                    file_name="driving_assistant_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )

        with col_y:
            if not events_df.empty:
                csv_buffer = events_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download Events CSV",
                    data=csv_buffer,
                    file_name="driving_events.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        # ---------- Events Table ----------
        st.subheader("📋 Detected Events")
        if not events_df.empty:
            st.dataframe(events_df, use_container_width=True, height=400)
        else:
            st.info("✅ No risky events detected in this video.")

    # Cleanup
    try:
        os.unlink(input_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()