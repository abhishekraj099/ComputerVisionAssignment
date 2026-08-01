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

## Accuracy Tuning

All detection/tracking/classification parameters live in `config.py` (plus
`tracker/bytetrack_tuned.yaml` for ByteTrack). Detection values were tuned
directly against the supplied classroom video (`assets/classroom.mp4`,
1280x720, 2150 frames), in which roughly 23-24 people are visible per frame.

Average detected person count over frames sampled across the whole clip:

| conf | NMS IoU | imgsz | avg count | max | FPS |
|---|---|---|---|---|---|
| 0.35 | 0.50 | 832 | 15.6 | 19 | 5.2 |
| 0.25 | 0.60 | 832 | 21.0 | 27 | 8.4 |
| **0.20** | **0.60** | **832** | **23.6** | 29 | 9.4 |
| 0.20 | 0.60 | 1280 | 24.0 | 29 | 4.0 |
| 0.15 | 0.60 | 832 | 26.9 | 32 | 9.3 |

Against ~23-24 visible, `conf=0.20 / iou=0.60 / imgsz=832` tracks the true
count most closely. `conf=0.15` overshoots (invents people); `imgsz=1280`
adds +0.4 count for less than half the throughput.

| Parameter | Value | Why |
|---|---|---|
| `DETECTION_CONFIDENCE_THRESHOLD` | **0.20** | 0.35 undercounts badly (15.6 vs ~23 visible); 0.15 overshoots |
| `DETECTION_IOU_THRESHOLD` (NMS) | **0.6** | Students sit close together; 0.50 suppressed real neighbours and emitted a duplicate box |
| `DETECTION_IMAGE_SIZE` (`imgsz`) | **832** | 1280 gained +0.4 count for >2x the cost |
| `FRAME_WIDTH` / `FRAME_HEIGHT` | **None** | Native resolution; downscaling would destroy small-person recall |
| `new_track_thresh` | **0.20** | Matches the detection floor - person count counts *tracks*, so a higher value silently discards recovered detections |
| `track_high_thresh` | **0.3** | Tracks the detection floor |
| `track_buffer` | **60** | At low CPU FPS, 30 frames covered too short an occlusion |
| `match_thresh` | **0.75** | Tolerant of CCTV box jitter |
| `POSTURE_ASPECT_RATIO_THRESHOLD` | **1.8** | Seated-at-desk boxes measure 1.3-1.7 and were wrongly called Standing |
| `POSTURE_HISTORY_SIZE` | **8** | Damps STANDING/SEATED oscillation near the threshold |
| `MOTION_DISTANCE_THRESHOLD` | **20.0** | At low CPU FPS, box jitter alone exceeded 15.0 for still people |
| `MOTION_HISTORY_SIZE` | **7** | Damps single-frame jitter |
| `BLUR_THRESHOLD` | **100.0** | Unchanged |

`tracker/bytetrack_tuned.yaml` keeps `tracker_type: bytetrack` - the algorithm
is unchanged, only its thresholds are retuned.

**Known miss:** a student at the bottom-left frame edge, heavily cropped and
occluded by her desk, is not detected at any tested setting.

## Requirements

- Python 3.10+
- Webcam (optional)
- Windows/Linux/macOS

## Author

**Abhishek Raj**
