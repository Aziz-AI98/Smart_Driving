---
title: Smart Driving Assistant
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
short_description: AI-powered dashcam analysis using CenterNet
---

# 🚗 Smart Front-View Driving Assistant

An AI-powered web application that analyzes **dashcam videos** to detect road objects (cars, pedestrians, cyclists, traffic signs) and assess **collision risk** in real time.

Built with **TensorFlow Hub's CenterNet (Hourglass)**, **OpenCV**, and **Streamlit** — deployed on **Hugging Face Spaces**.

---

## ✨ Features

- 🎯 **Object Detection** — Uses CenterNet (Hourglass backbone) pre-trained on COCO.
- 🔗 **Multi-Object Tracking** — Custom IoU-based tracker assigns unique IDs to objects.
- ⚠️ **Risk Assessment** — Classifies each object as `SAFE`, `CAUTION`, or `DANGER`.
- 🎥 **Annotated Video Output** — Download the processed video with bounding boxes and labels.
- 📊 **Events CSV** — Download a structured log of all risky events.
- 🌐 **Web Interface** — Upload any dashcam video and get results instantly.
- ⚙️ **Interactive Controls** — Adjust detection, tracking, and risk thresholds from the sidebar.

---

## 🚀 Live Demo

👉 **[Try the app on Hugging Face Spaces](https://huggingface.co/spaces/USERNAME/smart-driving)** *(replace with your deployed URL)*

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Object Detection | CenterNet (Hourglass 512x512) |
| Framework | TensorFlow 2.x + TensorFlow Hub |
| Video Processing | OpenCV |
| Web UI | Streamlit |
| Data Handling | Pandas, NumPy |
| Deployment | Hugging Face Spaces |

---

## 📦 Installation (Local)

```bash
git clone https://github.com/Aziz-AI98/Smart_Driving.git
cd Smart_Driving
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
