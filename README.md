# 🚌 BussPass MREC: Gesture-Controlled Interactive Signage

### A Real-Time Rock–Paper–Scissors and Smart Transit Display  
Powered by **YOLOv11**, **MediaPipe**, and **TensorFlow Lite**

---

## 🚀 Overview

**BussPass MREC** is an interactive digital signage system designed to make public spaces smarter and more engaging.  
It detects nearby pedestrians using **YOLOv11** and recognizes **hand gestures** with **MediaPipe** + a **custom TensorFlow Lite model**.

When a person gestures "game" in ASL, the system invites them to play a real-time **Rock–Paper–Scissors** game using their webcam feed — complete with visual overlays, icons, countdowns, and voice feedback.  

Outside of gameplay, it can also display **traffic and route information** from APIs like HERE, GraphHopper, and TomTom — turning a simple bus stop into an interactive smart hub.

---

## ✨ Features

- 🧠 **Gesture Recognition** — Detects “game”, “rock”, “paper”, and “scissor” gestures in real time  
- 🕹️ **Interactive Gameplay** — Real-time countdown, gesture comparison, and on-screen score tracking  
- 🧍 **Person Detection (YOLOv11)** — Identifies people nearby and labels “PLAYER” above the current participant  
- 🗣️ **Voice Feedback (Speaker Module)** — Announces gameplay prompts and results using text-to-speech  
- 🪟 **Animated Visual Overlays** — Sprites, pulse animations, icons, and round indicators drawn with OpenCV  
- 🛣️ **Traffic Integration (API)** — Displays live traffic routes and accident information  
- 🔁 **Automatic Restart** — Gesture “game” again to replay after three rounds  

---

## 🧩 Tech Stack

| Component | Technology |
|------------|-------------|
| Object Detection | YOLOv11 (Ultralytics) |
| Gesture Tracking | MediaPipe Hands |
| Gesture Classification | TensorFlow Lite (Custom Model) |
| Programming Language | Python 3.10+ |
| Visualization | OpenCV |
| Voice Interaction | pyttsx3 / Speaker.py |
| Data Handling | Pandas, NumPy |
| APIs | GraphHopper / HERE / TomTom |

---

## 🧠 Model Training

Gesture recognition was trained using custom CSV datasets captured via webcam:
`python
label, x0, y0, z0, ... x20, y20, z20
rock, ...
paper, ...
scissor, ...
game, ...

The training pipeline (TensorFlow + Scikit-Learn):

Scales input features using StandardScaler

Encodes labels with LabelEncoder

Trains a dense neural network (256-128-softmax)

Converts to .tflite for real-time inference

Resulting files:

gesture_model_new.tflite
gesture_label_encoder.pkl
gesture_input_scaler.pkl

Installation:
git clone https://github.com/brianad459/BussPass_MREC.git
cd BussPass_MREC

Create & activate a virtual environment:'
python -m venv .venv
.\.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run The Project:
python YOLO11_testing2.py


