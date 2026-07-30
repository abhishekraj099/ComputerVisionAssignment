"""
Central configuration for the Computer Vision pipeline.

Phase 1 scope: video source selection and display settings only.
Change SOURCE_TYPE to switch between webcam and a video file.
"""

# --- Video source ---
# "webcam" -> use a connected camera (see WEBCAM_INDEX)
# "video"  -> stream a video file frame-by-frame (see VIDEO_PATH)
SOURCE_TYPE = "webcam"

# Index of the webcam to use when SOURCE_TYPE == "webcam"
# 0 is usually the default built-in laptop camera.
WEBCAM_INDEX = 0

# Path to a video file to stream when SOURCE_TYPE == "video"
VIDEO_PATH = "assets/sample_video.mp4"

# --- Display ---
WINDOW_NAME = "iCloudEMS - Live Vision Pipeline (Phase 1)"

# Desired capture resolution. Set to None to keep the source's default.
FRAME_WIDTH = None
FRAME_HEIGHT = None

# Key used to exit the application (lowercase character)
EXIT_KEY = "q"

# --- Logging ---
LOG_DIR = "logs"
LOG_FILE = "app.log"
