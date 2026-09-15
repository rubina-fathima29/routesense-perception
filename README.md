# ROUTESENSE – AI-Powered Road & Traffic Perception System

ROUTESENSE is an AI-powered computer vision perception system designed for intelligent road and traffic monitoring.

The system processes video input to detect, track, count, and analyze vehicles and pedestrians in real time. It combines object detection and multi-object tracking to generate structured traffic perception data.

## 🚦 Key Features

- 🚗 Vehicle detection using YOLOv8
- 🚶 Pedestrian detection
- 🎯 Multi-object tracking using ByteTrack
- 🔢 Vehicle and pedestrian counting
- 🆔 Unique tracking ID assignment
- 📊 Traffic perception and analysis
- 🎥 Video-based real-time processing
- 🧪 Automated test modules for detection, tracking, and counting

## 🧠 System Architecture

```text
                    Input Video
                         │
                         ▼
                ┌─────────────────┐
                │   YOLOv8 Model  │
                │ Object Detection│
                └────────┬────────┘
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
        Vehicle Detection    Pedestrian Detection
              │                     │
              └──────────┬──────────┘
                         ▼
                ┌─────────────────┐
                │    ByteTrack    │
                │ Multi-Object    │
                │    Tracking     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Object Counting │
                └────────┬────────┘
                         │
                         ▼
                 Traffic Analysis


## 🛠️ Technologies Used

- Python
- YOLOv8
- OpenCV
- ByteTrack
- NumPy
- PyTest

## 📁 Project Structure

```text
routesense-perception/
├── src/
│   ├── detection/
│   ├── tracking/
│   └── counting/
├── tests/
├── main.py
├── requirements.txt
├── yolov8n.pt
└── README.md