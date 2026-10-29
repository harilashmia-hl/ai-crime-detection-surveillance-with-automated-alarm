# AI Crime Detection Surveillance with Automated Alarm

An AI-based surveillance system that detects suspicious objects in real time using a camera and triggers an automated alarm.

## Features
- Real-time detection using YOLOv8
- Works with webcam, video file, or RTSP camera
- Automatic alarm when a suspicious object is detected
- Alarm cooldown to avoid repeated alerts

## How It Works
1. The camera captures video frames.
2. The YOLOv8 model detects objects in each frame.
3. If a suspicious object (for example, a knife) appears in several consecutive frames, an alert is raised.
4. The alarm sounds and the alert is printed on screen.

## Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8

## Installation
```bash
git clone https://github.com/harilashmia-hl/ai-crime-detection-surveillance-with-automated-alarm
cd ai-crime-detection-surveillance-with-automated-alarm
pip install -r requirements.txt
```

## Usage
```bash
python src/detect.py
```
Press **q** to quit.

## Future Work
- Train a custom model for fights and weapons
- Send SMS or email alerts
- Save alert logs and video clips

## License
MIT License
