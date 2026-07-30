# iCloudEMS — Smart Campus Computer Vision Pipeline

## Project Overview

This project is being built for the iCloudEMS Smart Campus computer vision
technical task. The end goal (across all phases) is a pipeline that runs on
existing CCTV camera streams to support automatic attendance, room
occupancy/crowding alerts, and utility-waste detection (e.g. lights left on
in an empty room).

**Current status: Phase 1 only.** This phase builds the foundation only —
a real-time video capture and display pipeline. No detection, tracking, or
analytics logic has been implemented yet.

## Folder Structure

```
ComputerVisionAssignment/
│── app.py              # Entry point: capture -> display loop, FPS overlay, clean exit
│── config.py           # Central configuration (source selection, display, logging)
│── requirements.txt    # Python dependencies
│── README.md
│
├── assets/             # Sample/test video files go here
├── logs/               # Runtime logs are written here (app.log)
├── models/             # Reserved for future phases (detection model weights)
├── tracker/            # Reserved for future phases (tracking logic)
└── utils/
    ├── logger.py        # Console + file logger setup
    └── fps_counter.py   # Smoothed FPS calculation
```

## Environment Setup

Requires Python 3.11.

```bash
# From inside ComputerVisionAssignment/
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# Windows (cmd):
venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate
```

## Installing Dependencies

```bash
pip install -r requirements.txt
```

Notes:
- `opencv-python` and `numpy` are the only libraries actually used in Phase 1.
- `ultralytics` and `lap` (a ByteTrack dependency) are installed now, ahead of
  future phases, but are **not imported or used anywhere yet**.

## How to Run

1. By default, `config.py` has `SOURCE_TYPE = "webcam"`, which opens the
   default webcam (`WEBCAM_INDEX = 0`).
2. To run against a video file instead, edit `config.py`:
   ```python
   SOURCE_TYPE = "video"
   VIDEO_PATH = "assets/sample_video.mp4"  # or any path to a test clip
   ```
   The video is still streamed and processed frame-by-frame as it plays,
   not loaded and analyzed as a batch.
3. Run the app:
   ```bash
   python app.py
   ```
4. A window opens showing the live stream with the current FPS in the
   top-left corner. Press **Q** (as configured in `config.EXIT_KEY`) to exit
   cleanly.

If the webcam can't be opened or the video path is invalid, the app logs a
clear error (console + `logs/app.log`) and exits with a non-zero status
instead of crashing silently.

## Current Completed Phase

**Phase 1 — Project Foundation**
- Opens webcam by default; switchable to a video file via `config.py`.
- Reads frames continuously in real time.
- Displays the live stream with an FPS overlay.
- Clean exit on keypress.
- Graceful error handling and logging for bad/missing sources.
- Modular, function-based code with no global mutable state.

## Future Phases (not implemented yet)

These are listed for context only — none of this exists in the code yet:

- **Phase 2:** Motion detection and frame quality (blur) detection.
- **Phase 3:** Person detection (YOLO) and multi-object tracking (ByteTrack)
  to assign persistent IDs across frames.
- **Phase 4:** Entry/exit event detection, running attendance count, and
  unique-entry counting based on tracked IDs crossing a boundary/zone.
- **Phase 5:** Posture classification (seated vs. standing/moving).
- **Phase 6:** Room occupancy state (occupied/empty) and the
  lights-on-but-empty bonus check.
- **Phase 7:** A live UI/dashboard (e.g. Streamlit) tying all outputs
  together, and a live demo run against an unseen stream.
