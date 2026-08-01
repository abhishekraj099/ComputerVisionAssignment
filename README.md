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
`tracker/bytetrack_tuned.yaml` for ByteTrack). Values below were tuned for
classroom CCTV footage — mostly-seated, sometimes partially-occluded people
viewed from a fixed angled camera:

| Parameter | Before | After | Why |
|---|---|---|---|
| `DETECTION_CONFIDENCE_THRESHOLD` | 0.5 | **0.35** | 0.5 dropped small/occluded seated people scoring 0.35–0.5 as false negatives |
| `DETECTION_IOU_THRESHOLD` (NMS) | 0.7 (default) | **0.5** | 0.7 let near-duplicate boxes for one person survive NMS, inflating person count |
| `DETECTION_IMAGE_SIZE` (`imgsz`) | 640 (default) | **832** | Wide classroom shots shrink each person; 640 systematically missed small boxes |
| `track_high_thresh` | 0.25 | **0.5** | Restricts first-stage matching to confident boxes → fewer ID switches |
| `new_track_thresh` | 0.25 | **0.4** | Fewer spurious new tracks from noisy detections |
| `track_buffer` | 30 | **60** | At low CPU FPS, 30 frames covered too short an occlusion → new IDs on reappearance |
| `match_thresh` | 0.8 | **0.75** | Slightly more tolerant of CCTV box jitter → fewer rejected valid matches |
| `POSTURE_ASPECT_RATIO_THRESHOLD` | 1.2 | **1.8** | Seated-at-desk boxes measure 1.3–1.7 and were wrongly called Standing |
| `POSTURE_HISTORY_SIZE` | 5 | **8** | Damps STANDING↔SEATED oscillation from ratio jitter near the threshold |
| `MOTION_DISTANCE_THRESHOLD` | 15.0 | **20.0** | At low CPU FPS, box jitter alone exceeded 15.0 for genuinely still people |
| `MOTION_HISTORY_SIZE` | 5 | **7** | Further damps single-frame jitter |
| `BLUR_THRESHOLD` | 100.0 | 100.0 (unchanged) | Verified well-calibrated: sharp frames ≥ 984, blurry ≤ 4.3 |

`tracker/bytetrack_tuned.yaml` keeps `tracker_type: bytetrack` — the algorithm
is unchanged, only its thresholds are retuned.

**Trade-off:** raising `imgsz` 640 → 832 costs roughly 17 → 12.5 FPS on CPU for
the full pipeline; still comfortably real-time for this use case.

## Requirements

- Python 3.10+
- Webcam (optional)
- Windows/Linux/macOS

## Author

**Abhishek Raj**
