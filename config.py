"""
Central configuration for the Computer Vision pipeline.

Purpose:
    Single source of truth for every tunable value used by the pipeline -
    video source selection, display/overlay appearance, logging, person
    detection, and multi-object tracking - so no other module hardcodes a
    literal value that a user or a later phase might need to change.

Responsibilities:
    Hold plain, static configuration values only (strings, numbers, tuples).
    Nothing in this module performs I/O, imports OpenCV/YOLO/torch, or
    contains logic - it is imported by every other module in the project
    (app.py, models/person_detector.py, tracker/byte_tracker.py) purely for
    its constants.

Scope of the current phase (Phase 9):
    Covers Phase 1 (video source/display/logging), Phase 2 (YOLOv8 person
    detection), Phase 3 (ByteTrack tracking), Phase 4 (virtual line
    entry/exit crossing), Phase 5 (attendance panel display), Phase 6
    (per-person motion classification), Phase 7 (per-person posture
    classification), Phase 8 (frame-level blur/quality classification),
    and Phase 9 (room occupancy panel display) settings. Note that
    attendance and occupancy both need almost no configuration - see
    attendance/attendance_manager.py's and
    occupancy/occupancy_detector.py's module docstrings for why.

What this module intentionally does NOT handle:
    - No validation of the values below; each consuming module is expected
      to use them as-is (e.g. an invalid SOURCE_TYPE is only caught when
      app.open_video_source() runs).
    - No validation of the values below; each consuming module is expected
      to use them as-is (e.g. an invalid SOURCE_TYPE is only caught when
      app.open_video_source() runs).

Which future modules will consume this module's output:
    Every current and future pipeline module (app.py, models/, tracker/,
    and later phases' modules) imports this module directly and reads its
    constants by name (e.g. `config.SOURCE_TYPE`).
"""

# --- Video source ---
# Selects where frames come from. Change this one value to switch feeds.
# Valid values: "webcam" or "video". Any other value causes
# app.open_video_source() to raise a RuntimeError at startup.
# "webcam" -> use a connected camera (see WEBCAM_INDEX)
# "video"  -> stream a video file frame-by-frame (see VIDEO_PATH)
SOURCE_TYPE = "webcam"

# Index of the webcam to use when SOURCE_TYPE == "webcam".
# Valid values: a non-negative integer matching an OS-enumerated camera
# device. 0 is usually the default built-in laptop camera; try 1, 2, ...
# if this machine has more than one camera attached.
WEBCAM_INDEX = 0

# Path to a video file to stream when SOURCE_TYPE == "video".
# Valid values: any path OpenCV's VideoCapture can open (relative paths are
# resolved against the process's current working directory). Exists so a
# recorded clip can be used in place of a live camera for testing/demos.
VIDEO_PATH = "assets/sample_video.mp4"

# --- Display ---
# Title of the OpenCV preview window. Purely cosmetic; exists so the
# window is identifiable and so the current phase is visible at a glance.
WINDOW_NAME = "iCloudEMS - Live Vision Pipeline (Phase 3)"

# Desired capture resolution, in pixels. Valid values: a positive integer,
# or None to keep the video source's native/default resolution. Exists to
# let a slow CPU trade resolution for frame rate if needed; unused by
# default since None is usually the right choice.
FRAME_WIDTH = None
FRAME_HEIGHT = None

# Key that exits the application when pressed in the preview window.
# Valid values: a single lowercase character. Exists so the exit key is
# configurable without touching app.py's event loop.
EXIT_KEY = "q"

# --- Logging ---
# Directory the log file lives in. Created automatically if missing.
# Exists so log location is configurable without touching utils/logger.py.
LOG_DIR = "logs"

# Log file name, written inside LOG_DIR. Exists for the same reason as
# LOG_DIR above - keeping the destination configurable in one place.
LOG_FILE = "app.log"

# --- Person detection (Phase 2) ---
# Path to the YOLOv8 weights file. Ultralytics will automatically download
# the weights to this path if the file does not already exist. Valid
# values: any path; "yolov8n.pt" (the smallest/fastest YOLOv8 variant) is
# the default because Phase 2 explicitly calls for the lightweight model.
DETECTION_MODEL_PATH = "models/yolov8n.pt"

# COCO class ID for "person". Valid values: 0-79 (any COCO class), but this
# project only ever detects/tracks class 0. Exists as a named constant
# instead of a bare "0" scattered through the detection/tracking calls.
PERSON_CLASS_ID = 0

# Minimum confidence, in the closed range 0.0-1.0, for a detection to be
# kept. Exists to filter out low-confidence, noisy boxes; 0.5 is a common,
# reasonable default for YOLOv8 person detection.
DETECTION_CONFIDENCE_THRESHOLD = 0.5

# Compute device for YOLO inference. Valid values: "auto", "cpu", or
# "cuda". "auto" (the default) picks GPU (CUDA) automatically if available,
# otherwise falls back to CPU, so the app runs unmodified on either kind
# of machine.
DETECTION_DEVICE = "auto"

# --- Detection/track overlay appearance ---
# BGR color (OpenCV's channel order, not RGB) used for bounding boxes and
# their text labels. Valid values: a (B, G, R) tuple of ints in 0-255.
# Default is green, per the task's example output.
BOX_COLOR = (0, 255, 0)

# Bounding box line thickness, in pixels. Valid values: a positive integer.
BOX_THICKNESS = 2

# Font scale (OpenCV's relative size unit, not points) for the "Person
# <confidence>" / "ID: <n>" labels. Valid values: a positive float.
DETECTION_LABEL_FONT_SCALE = 0.6

# Font stroke thickness, in pixels, for the same labels. Valid values: a
# positive integer.
DETECTION_LABEL_FONT_THICKNESS = 2

# Vertical gap, in pixels, between a box's top edge and its label when the
# label fits above the box, and the minimum distance from the frame's top
# edge required for that placement. When the box is too close to the top,
# the label is drawn this far below the box's top edge instead, so labels
# never get clipped off-screen for people standing near the top of frame.
DETECTION_LABEL_GAP_ABOVE_BOX = 10
DETECTION_LABEL_MIN_Y = 10
DETECTION_LABEL_FALLBACK_OFFSET_BELOW = 20

# --- On-screen status text (FPS / person count) ---
# Font scale, color, and stroke thickness for the FPS and person-count
# overlay lines. Same value ranges/units as the detection label settings
# above; kept as separate constants so overlay and per-box label styling
# can be tuned independently.
OVERLAY_FONT_SCALE = 1.0
OVERLAY_FONT_THICKNESS = 2
OVERLAY_COLOR = (0, 255, 0)

# (x, y) pixel position, from the frame's top-left corner, where each
# status line is drawn. Valid values: a tuple of two non-negative ints.
FPS_TEXT_POSITION = (10, 30)
PERSON_COUNT_TEXT_POSITION = (10, 70)

# --- Multi-object tracking (Phase 3) ---
# Name of the Ultralytics-bundled tracker config. Valid values: any tracker
# config Ultralytics ships or resolves by name (e.g. "bytetrack.yaml",
# "botsort.yaml"). "bytetrack.yaml" selects Ultralytics' built-in ByteTrack
# integration, per the Phase 3 requirement to use ByteTrack and no custom
# tracker code.
TRACKER_CONFIG = "bytetrack.yaml"

# Vertical gap, in pixels, between the "ID: <n>" line and the
# "Person <confidence>" line drawn above/below each tracked box. Valid
# values: a positive integer. Exists so the two-line track label doesn't
# overlap itself.
TRACK_LABEL_LINE_SPACING = 18

# --- Fallback tracker (environment fallback only) ---
# Used only when tracker.byte_tracker (YOLOv8 + ByteTrack, which requires
# torch) fails to import on this machine - see tracker/fallback_tracker.py.
# Not used at all on a machine where torch imports successfully.
FALLBACK_BG_HISTORY = 500
FALLBACK_BG_VAR_THRESHOLD = 16
FALLBACK_MIN_CONTOUR_AREA = 1500
FALLBACK_MAX_MATCH_DISTANCE = 75
FALLBACK_STALE_TRACK_TIMEOUT = 1.0

# --- Entry/Exit virtual line crossing (Phase 4) ---
# Endpoints of the virtual line tracked people are checked against, as
# (x, y) pixel coordinates in the same space as the video frame. Valid
# values: any two distinct points; a horizontal line (as in the default
# below) makes crossings map directly to "top of frame" vs "bottom of
# frame". Exists so the line's position/orientation can be adjusted per
# camera setup without touching detection code.
LINE_START = (100, 250)
LINE_END = (550, 250)

# BGR color and pixel thickness used to draw the virtual line. Valid
# values: a (B, G, R) tuple of ints in 0-255, and a positive integer.
# Default is blue, per the Phase 4 visualization requirement.
LINE_COLOR = (255, 0, 0)
LINE_THICKNESS = 2

# How long, in seconds, a generated ENTRY/EXIT notification stays visible
# on screen after it fires. Valid values: a positive float. Exists so a
# crossing event is noticeable to a human watching the stream without
# permanently cluttering the display.
EVENT_DISPLAY_DURATION = 2.0

# (x, y) pixel position, from the frame's top-left corner, where the first
# visible event notification line is drawn, and the vertical gap between
# subsequent lines. Placed below the FPS/person-count lines
# (see FPS_TEXT_POSITION / PERSON_COUNT_TEXT_POSITION) so they don't
# overlap.
EVENT_NOTIFICATION_POSITION = (10, 110)
EVENT_NOTIFICATION_LINE_SPACING = 30

# Maximum number of recent event notifications drawn at once. Valid
# values: a positive integer. Exists so a burst of crossings (e.g. several
# people at once) cannot stack notification text off the bottom of the
# frame; older events beyond this count are simply not drawn (they are
# still generated and logged, only the display is capped).
EVENT_NOTIFICATION_MAX_LINES = 5

# How long, in seconds, LineCrossingDetector remembers a track ID's last
# known side of the virtual line after that track ID stops appearing in
# the tracked-people list, before forgetting it entirely. Valid values: a
# positive float. Exists to bound memory growth over a long-running
# stream - without this, every distinct track ID ever seen would be
# remembered forever. 300.0 (5 minutes) comfortably outlives a brief
# occlusion/tracker hiccup while still being forgotten well before memory
# usage could become a practical concern.
LINE_CROSSING_STALE_TRACK_TIMEOUT = 300.0

# How long, in seconds, LineCrossingDetector suppresses further ENTRY/EXIT
# events for a track ID after one fires for it. Valid values: a positive
# float. Exists to absorb detector/tracker jitter that makes a centroid
# flicker back and forth right at the line, which would otherwise be
# reported as several rapid, spurious crossings instead of the one real
# one. 1.0 second is short enough not to mask a second, genuine crossing
# by the same person shortly after the first.
LINE_CROSSING_EVENT_COOLDOWN = 1.0

# --- Attendance panel (Phase 5) ---
# (x, y) pixel position, from the frame's top-left corner, where the
# attendance panel's first ("Attendance") line is drawn, and the vertical
# gap between subsequent lines. Placed below the event-notification block
# (see EVENT_NOTIFICATION_POSITION / EVENT_NOTIFICATION_LINE_SPACING /
# EVENT_NOTIFICATION_MAX_LINES) so the two never overlap regardless of how
# many notifications are currently visible.
ATTENDANCE_PANEL_POSITION = (10, 270)
ATTENDANCE_PANEL_LINE_SPACING = 30

# --- Motion detection (Phase 6) ---
# Smoothed (rolling-average) per-frame centroid displacement, in pixels,
# above which a tracked person is classified MOVING rather than
# STATIONARY. Valid values: a positive float. Exists so sensitivity can be
# tuned per camera resolution/distance without touching detection code;
# 15.0 is a reasonable starting point for a typical webcam-resolution feed.
MOTION_DISTANCE_THRESHOLD = 15.0

# Number of recent frame-to-frame displacements averaged together, per
# track, before comparing against MOTION_DISTANCE_THRESHOLD. Valid values:
# a positive integer. Exists to prevent a single noisy/jittery frame from
# flipping the reported motion state; 5 smooths out brief jitter while
# still reacting to genuine movement within about a fifth of a second at
# typical processing rates.
MOTION_HISTORY_SIZE = 5

# How long, in seconds, MotionDetector remembers a track ID's centroid
# history after that track ID stops appearing in the tracked-people list,
# before forgetting it entirely. Valid values: a positive float. Exists to
# bound memory growth over a long-running stream, exactly like
# LINE_CROSSING_STALE_TRACK_TIMEOUT above.
MOTION_STALE_TRACK_TIMEOUT = 300.0

# BGR colors used for the "Moving" / "Stationary" label drawn near each
# tracked person. Valid values: (B, G, R) tuples of ints in 0-255. Moving
# defaults to red (draws attention), Stationary to light gray
# (unobtrusive).
MOTION_MOVING_COLOR = (0, 0, 255)
MOTION_STATIONARY_COLOR = (200, 200, 200)

# Vertical offset, in pixels, from a tracked person's centroid to where
# their "Moving"/"Stationary" label is drawn. Valid values: an integer
# (positive moves the label downward). Exists so the label can be placed
# clear of the centroid point itself.
MOTION_LABEL_OFFSET_Y = 15

# --- Posture detection (Phase 7) ---
# Smoothed (rolling-average) bounding-box height-to-width ratio above
# which a tracked person is classified STANDING rather than SEATED. Valid
# values: a positive float. This is a coarse heuristic, not a calibrated
# measurement - 1.2 is a reasonable starting point (a standing person's
# full-body box is usually noticeably taller than wide; a seated person's
# box tends to be closer to square or wider), but the right value depends
# heavily on camera angle/distance and may need tuning per deployment.
POSTURE_ASPECT_RATIO_THRESHOLD = 1.2

# Number of recent per-frame aspect ratios averaged together, per track,
# before comparing against POSTURE_ASPECT_RATIO_THRESHOLD. Valid values: a
# positive integer. Exists to prevent a single noisy/jittery frame (e.g. a
# momentarily clipped box) from flipping the reported posture; 5 mirrors
# MOTION_HISTORY_SIZE's smoothing/responsiveness trade-off.
POSTURE_HISTORY_SIZE = 5

# How long, in seconds, PostureDetector remembers a track ID's aspect-ratio
# history after that track ID stops appearing in the tracked-people list,
# before forgetting it entirely. Valid values: a positive float. Exists to
# bound memory growth over a long-running stream, exactly like
# MOTION_STALE_TRACK_TIMEOUT above.
POSTURE_STALE_TRACK_TIMEOUT = 300.0

# BGR colors used for the "Standing" / "Seated" label drawn near each
# tracked person. Valid values: (B, G, R) tuples of ints in 0-255. Chosen
# to be visually distinct from the Moving/Stationary motion-label colors.
POSTURE_STANDING_COLOR = (0, 255, 255)
POSTURE_SEATED_COLOR = (255, 0, 255)

# Vertical offset, in pixels, from a tracked person's bounding-box center
# to where their "Standing"/"Seated" label is drawn. Valid values: an
# integer (positive moves the label downward). Set below
# MOTION_LABEL_OFFSET_Y so the motion and posture labels stack without
# overlapping.
POSTURE_LABEL_OFFSET_Y = 35

# --- Blur detection (Phase 8) ---
# Laplacian variance below which a frame is classified BLURRY rather than
# SHARP. Valid values: a positive float. This is a coarse, widely-used
# heuristic threshold, not a calibrated measurement - 100.0 is a common
# starting point for the classic Variance-of-Laplacian method at
# webcam-like resolutions, but the right value depends heavily on
# resolution, lens, and lighting, and may need tuning per deployment.
BLUR_THRESHOLD = 100.0

# BGR colors used for the "Frame Quality" panel depending on
# classification. Valid values: (B, G, R) tuples of ints in 0-255. Sharp
# defaults to green, Blurry to red (a "flag for review" warning color).
BLUR_SHARP_COLOR = (0, 255, 0)
BLUR_BLURRY_COLOR = (0, 0, 255)

# (x, y) pixel position, from the frame's top-left corner, where the
# "Frame Quality" panel's first line is drawn, and the vertical gap
# between subsequent lines. Placed below the attendance panel (see
# ATTENDANCE_PANEL_POSITION / ATTENDANCE_PANEL_LINE_SPACING) so the two
# never overlap.
BLUR_PANEL_POSITION = (10, 420)
BLUR_PANEL_LINE_SPACING = 30

# --- Occupancy detection (Phase 9) ---
# (x, y) pixel position, from the frame's top-left corner, where the
# "Occupancy" panel's first line is drawn, and the vertical gap between
# subsequent lines. Placed below the "Frame Quality" panel (see
# BLUR_PANEL_POSITION / BLUR_PANEL_LINE_SPACING) so the two never overlap.
# No color/font constants are added here - the panel deliberately reuses
# OVERLAY_COLOR / OVERLAY_FONT_SCALE / OVERLAY_FONT_THICKNESS (the same
# style already used for the FPS/person-count/attendance text) rather than
# introducing near-duplicate constants for a panel with no independent
# color-coding requirement.
OCCUPANCY_PANEL_POSITION = (10, 510)
OCCUPANCY_PANEL_LINE_SPACING = 30
