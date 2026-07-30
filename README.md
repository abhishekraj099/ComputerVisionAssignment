# iCloudEMS Smart Campus Computer Vision Pipeline

A real-time computer vision system developed for the iCloudEMS technical assignment. The project uses YOLOv8 and ByteTrack to detect and track people, monitor attendance, analyze movement and posture, detect blurry frames, classify room occupancy, and provide a live Streamlit dashboard.

## Features

- Real-time person detection using YOLOv8
- Multi-object tracking with ByteTrack
- Virtual line-based Entry/Exit detection
- Automatic attendance monitoring
- Motion detection (Moving / Stationary)
- Posture detection (Standing / Seated)
- Blur detection using Variance of Laplacian
- Occupancy status detection
- Live Streamlit dashboard
- Webcam and video upload support

## Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- ByteTrack
- NumPy
- Streamlit

## Project Structure

```
ComputerVisionAssignment/
├── app.py
├── config.py
├── dashboard/
├── models/
├── tracker/
├── events/
├── attendance/
├── motion/
├── posture/
├── quality/
├── occupancy/
├── utils/
├── assets/
├── requirements.txt
└── README.md
```
## Screenshots

<img width="1917" height="1077" alt="Image" src="https://github.com/user-attachments/assets/25fdfb86-3310-4b4e-b984-04eb91d270a3" />
<img width="1917" height="1078" alt="Image" src="https://github.com/user-attachments/assets/b9dd0d30-7cd3-4fc6-8320-c4d3dee5a7cd" />
<img width="1291" height="752" alt="Image" src="https://github.com/user-attachments/assets/614cfa7d-882f-4811-a2a4-0a2c694fc2e9" />
<img width="1917" height="1077" alt="Image" src="https://github.com/user-attachments/assets/94d558e3-6aa6-4f3d-8cbc-a195ecb80a67" />
<img width="215" height="231" alt="Image" src="https://github.com/user-attachments/assets/6f958110-2363-45e6-aa99-e8683728dbe9" />
<img width="1917" height="1078" alt="Image" src="https://github.com/user-attachments/assets/992305d3-3592-429d-93a7-b9f93f8a8611" />
<img width="1917" height="1078" alt="Image" src="https://github.com/user-attachments/assets/92caaf19-0c17-4a7f-81d0-a2fe9fce407a" />
## Installation

```bash
git clone <repository-url>
cd ComputerVisionAssignment
python -m venv venv
```

### Windows

```bash
.\venv\Scripts\Activate.ps1
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Run

### OpenCV Application

```bash
python app.py
```

### Streamlit Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## Dashboard Features

- Webcam support
- Video upload
- Live annotated video
- FPS
- Person Count
- Attendance Statistics
- Motion Summary
- Posture Summary
- Frame Quality
- Occupancy Status
- Recent Entry/Exit Events

## Requirements

- Python 3.10+
- Webcam (optional)
- Windows/Linux/macOS

## Author

**Abhishek Raj**
