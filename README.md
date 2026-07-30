# iCloudEMS — Smart Campus Computer Vision Pipeline

## Table of Contents

- [Project Overview](#project-overview)
- [Current Architecture](#current-architecture)
- [Phase Completion Status](#phase-completion-status)
- [Directory Structure](#directory-structure)
- [Execution Flow](#execution-flow)
- [Detection Pipeline](#detection-pipeline)
- [Tracking Pipeline](#tracking-pipeline)
- [Entry/Exit Pipeline](#entryexit-pipeline)
- [Attendance System](#attendance-system)
- [Motion Detection](#motion-detection)
- [Posture Detection](#posture-detection)
- [Blur Detection](#blur-detection)
- [Occupancy Detection](#occupancy-detection)
- [Dashboard](#dashboard)
- [Configuration Guide](#configuration-guide)
- [Dependencies](#dependencies)
- [Known Limitations](#known-limitations)
- [Current Windows Torch Policy Issue](#current-windows-torch-policy-issue)
- [Future Roadmap](#future-roadmap)
- [How to Run](#how-to-run)
- [Troubleshooting](#troubleshooting)

## Project Overview

This project is being built for the iCloudEMS Smart Campus computer vision
technical task. The end goal (across all phases) is a pipeline that runs on
existing CCTV camera streams to support automatic attendance, room
occupancy/crowding alerts, and utility-waste detection (e.g. lights left on
in an empty room).

**Current status: Phase 1 through Phase 10 complete.** Phase 1 built the
live capture/display foundation. Phase 2 added real-time person detection
with YOLOv8. Phase 3 added multi-object tracking (ByteTrack) so each
detected person gets a persistent ID. Phase 4 added ENTRY/EXIT event
detection via a configurable virtual line. Phase 5 added an attendance
system (current people inside, total entries/exits, unique visitors) built
on top of those events. Phase 6 added per-person MOVING/STATIONARY
classification from tracked centroids alone. Phase 7 added per-person
STANDING/SEATED classification from bounding-box aspect ratio alone. Phase
8 added frame-level SHARP/BLURRY classification via Variance of Laplacian.
Phase 9 added room-level occupancy classification (EMPTY/OCCUPIED/ACTIVE/
IDLE), derived entirely from Phases 5-8's outputs rather than from video
frames directly. Phase 10 added a Streamlit web dashboard (webcam or video
upload, live annotated video, and every statistic above) built entirely on
top of the same pipeline - no detection logic was added or duplicated. See
[Future Roadmap](#future-roadmap) for what's left.

## Current Architecture

The pipeline is a single-threaded, per-frame loop. Each frame passes
through the same sequence of stages before the next frame is read:

```
Frame
    ↓                              ↘
Video Capture                  BlurDetector.analyze()   (raw frame, BEFORE any drawing - grayscale → Laplacian → variance)
(OpenCV VideoCapture)               ↓
    ↓                          BlurState (SHARP/BLURRY + variance)
PersonTracker.track()    (internally: YOLOv8 detection → ByteTrack association)
    ↓
List[TrackedPerson]      (box + confidence + persistent track_id, per person)
    ↓                                    ↘                          ↘
LineCrossingDetector.update()      MotionDetector.update()    PostureDetector.update()
(centroid vs. virtual line side)  (centroid displacement,     (bbox height/width ratio,
                                    smoothed)                   smoothed)
    ↓                                    ↓                          ↓
List[CrossingEvent]                List[MotionState]         List[PostureState]
(ENTRY/EXIT, per crossing)         (MOVING/STATIONARY)        (STANDING/SEATED)
    ↓                                    ↓                          ↓
AttendanceManager.update()               |                          |                    |
(accepts/ignores events,                 |                          |                    |
 updates running totals)                 |                          |                    |
    ↓                                    ↓                          ↓                    ↓
AttendanceStatistics ─────────────┐  List[MotionState] ─────┐  List[PostureState] ─┐  BlurState
(current people, entries/exits,   │  (MOVING/STATIONARY)    │  (STANDING/SEATED)   │  (from stage 3)
 unique visitors)                 │                          │                      │
                                   ↘                          ↓                      ↙
                                    OccupancyDetector.update()  (occupancy/occupancy_detector.py)
                                    (consumes ONLY these four already-computed outputs -
                                     never a frame, TrackedPerson, YOLO, ByteTrack, or LineCrossingDetector)
                                    ↓
                                    OccupancyState  (EMPTY / OCCUPIED / ACTIVE / IDLE)
    ↓                                    ↓                          ↓                    ↓
Overlay drawing (draw_tracks + draw_line + draw_events + draw_attendance_panel + draw_motion_states + draw_posture_states + draw_blur_panel + draw_occupancy_panel + draw_fps + draw_person_count)
    ↓
cv2.imshow()             (display window)
    ↓
Future Live UI/Dashboard Module         (not yet implemented)
```

**Stage-by-stage explanation:**

1. **Video Capture** (`app.open_video_source()`): opens either the
   configured webcam index or a video file path via `cv2.VideoCapture`,
   based on `config.SOURCE_TYPE`. Produces one BGR frame per loop
   iteration, read live/frame-by-frame — never the whole file at once.
2. **BlurDetector.analyze()** (`quality/blur_detector.py`): runs on the
   *raw* frame from stage 1, before any other stage touches it - grayscale
   conversion → Laplacian → variance → SHARP/BLURRY classification. This
   must happen before any overlay drawing, or the drawn boxes/text would
   artificially inflate the measured sharpness. Has no dependency on
   tracking, motion, posture, line-crossing, or attendance. See
   [Blur Detection](#blur-detection) for the full algorithm.
3. **BlurState**: `(is_blurry, laplacian_variance)` for this frame as a
   whole - not per person.
4. **PersonTracker.track()** (`tracker/byte_tracker.py`): the frame is
   handed to Ultralytics' `model.track()`, which runs a YOLOv8 forward
   pass (person class only) and feeds the resulting boxes into
   Ultralytics' built-in ByteTrack integration in one call. See
   [Detection Pipeline](#detection-pipeline) and
   [Tracking Pipeline](#tracking-pipeline) for why detection and tracking
   are combined at this stage rather than split into two.
5. **List[TrackedPerson]**: the stable, framework-agnostic output type
   every current and future consumer works with — a plain list of
   `(x1, y1, x2, y2, confidence, track_id)` records, decoupled from
   Ultralytics' internal result objects.
6. **LineCrossingDetector.update()** (`events/line_crossing.py`): computes
   each tracked person's bounding-box centroid, checks which side of the
   configured virtual line (`config.LINE_START`/`LINE_END`) it falls on,
   and compares that to the side recorded for that same track ID last
   frame. See [Entry/Exit Pipeline](#entryexit-pipeline) for the full
   algorithm.
7. **List[CrossingEvent]**: zero or more `(track_id, event_type,
   timestamp, centroid)` records for people who crossed the line *this
   frame*; empty on the (overwhelmingly common) frames where nobody
   crosses.
8. **AttendanceManager.update()** (`attendance/attendance_manager.py`):
   consumes that same `List[CrossingEvent]` - never computes a crossing
   itself - and, for each event, either accepts it (updating running
   totals) or rejects it (a duplicate ENTRY/EXIT, logged and ignored). See
   [Attendance System](#attendance-system) for the acceptance rules.
9. **AttendanceStatistics**: an immutable snapshot - current people
   inside, total entries, total exits, unique visitors, and the exact set
   of track IDs currently inside - returned on every call, including
   frames with no new events.
10. **MotionDetector.update()** (`motion/motion_detector.py`): consumes
    the *same* `List[TrackedPerson]` from stage 5 directly - it runs in
    parallel with, and has no dependency on, `LineCrossingDetector` or
    `AttendanceManager`. For each track, it computes the centroid
    displacement since last frame, smooths it over a short rolling
    history, and compares that smoothed value to a threshold. See
    [Motion Detection](#motion-detection) for the full algorithm.
11. **List[MotionState]**: one `(track_id, is_moving, movement_distance,
    current_centroid)` record per tracked person, every frame.
12. **PostureDetector.update()** (`posture/posture_detector.py`): also
    consumes the *same* `List[TrackedPerson]` from stage 5 directly - it
    runs in parallel with, and has no dependency on,
    `LineCrossingDetector`, `AttendanceManager`, or `MotionDetector`. For
    each track, it computes the bounding box's height/width ratio,
    smooths it over a short rolling history, and compares that smoothed
    value to a threshold. See [Posture Detection](#posture-detection) for
    the full algorithm.
13. **List[PostureState]**: one `(track_id, is_standing, aspect_ratio,
    current_bbox)` record per tracked person, every frame.
14. **OccupancyDetector.update()** (`occupancy/occupancy_detector.py`):
    consumes *only* the already-computed `AttendanceStatistics` (stage 9),
    `List[MotionState]` (stage 11), `List[PostureState]` (stage 13), and
    `BlurState` (stage 3) - never a frame, `TrackedPerson`, YOLO,
    ByteTrack, or `LineCrossingDetector`. On a BLURRY (or missing-quality)
    frame, it keeps the previous decision unchanged instead of
    recomputing. See [Occupancy Detection](#occupancy-detection) for the
    full decision flow and dependency diagram.
15. **OccupancyState**: a single room-level snapshot - `occupancy_status`
    (EMPTY/OCCUPIED/ACTIVE/IDLE), `people_inside`, `moving_people`,
    `standing_people`, `seated_people`, `frame_quality_ok`.
16. **Overlay drawing** (`tracker.draw_tracks`, `events.draw_line`,
    `events.draw_events`, `attendance.draw_attendance_panel`,
    `motion.draw_motion_states`, `posture.draw_posture_states`,
    `quality.draw_blur_panel`, `occupancy.draw_occupancy_panel`,
    `app.draw_fps`, `app.draw_person_count`): mutates the frame in place to
    draw each tracked person's box + `ID:`/`Person <confidence>` labels,
    the blue virtual line, any still-visible ENTRY/EXIT notifications, the
    attendance panel, each person's "Moving"/"Stationary" label, each
    person's "Standing"/"Seated" label, the "Frame Quality" panel, the
    "Occupancy" panel, and the FPS/person-count status lines.
17. **`cv2.imshow()`**: displays the annotated frame in the configured
    window (`config.WINDOW_NAME`); `cv2.waitKey()` in the same loop
    iteration checks for the configured exit key.
18. **Display, two ways**: stages 1-16 above (everything except the final
    display step) are packaged into two reusable functions in `app.py` -
    `build_pipeline_components()` (constructs every object in
    `PipelineComponents` once) and `process_frame()` (runs one frame
    through every stage and returns a `PipelineFrameResult` bundling every
    output). `run_pipeline()`'s `cv2.imshow()` loop is one caller of these
    two functions; `dashboard/streamlit_app.py`'s Streamlit UI (see
    [Dashboard](#dashboard)) is the other. Neither caller contains, or
    needs to contain, any detection/tracking/analysis logic - both simply
    call the same two functions and render the same `PipelineFrameResult`.

## Phase Completion Status

**Phase 1 — Project Foundation (complete)**
- Opens webcam by default; switchable to a video file via `config.py`.
- Reads frames continuously in real time.
- Displays the live stream with an FPS overlay.
- Clean exit on keypress.
- Graceful error handling and logging for bad/missing sources.
- Modular, function-based code with no global mutable state.

**Phase 2 — Real-Time Person Detection (complete)**
- Loads YOLOv8 (`yolov8n.pt`) via `models/person_detector.py`, downloading
  the weights automatically if missing.
- Runs inference on every processed frame, filtered to the COCO "person"
  class only (all other classes are ignored).
- Automatically uses GPU (CUDA) if available, otherwise falls back to CPU.
- Draws a bounding box and a `Person <confidence>` label for each detected
  person (this drawing path is no longer used live as of Phase 3 — see
  [Known Limitations](#known-limitations)).
- Displays a live `Persons: <count>` total alongside the existing FPS
  overlay.
- Model loading failures stop the app with a clear error; per-frame
  inference failures are logged and skipped without crashing the stream.
- Webcam/video source switching from Phase 1 is unchanged.

**Phase 3 — Multi-Object Tracking / ByteTrack (complete)**
- Assigns a persistent track ID to each detected person, stable while they
  remain visible.
- Draws `ID: <n>` + `Person <confidence>` for every tracked person.
- Reuses the Phase 2 YOLO model instance rather than loading a second copy.
- Tracker constructed once, reused for the life of the stream.
- See [Tracking Pipeline](#tracking-pipeline) for full detail.

**Phase 4 — Entry/Exit Event Detection (complete)**
- Detects when a tracked person's centroid crosses a configurable virtual
  line, emitting one ENTRY or EXIT event per actual crossing.
- No duplicate events while a person lingers on one side of the line;
  detector/tracker jitter right at the line is absorbed by a configurable
  cooldown, and long-unseen track IDs are evicted after a configurable
  timeout (both added in the Phase 4 robustness pass).
- Draws the virtual line (blue) and briefly displays recent crossing
  events (`ENTRY : ID <n>` / `EXIT : ID <n>`).
- Does not count/accumulate attendance and does not persist events
  anywhere — see [Entry/Exit Pipeline](#entryexit-pipeline).

**Phase 5 — Attendance System (complete)**
- Consumes the `List[CrossingEvent]` produced by `LineCrossingDetector`
  (never computes a crossing itself) to maintain running attendance state.
- Tracks current people inside, total entries, total exits, and unique
  visitors; rejects (logs and ignores) duplicate ENTRY/EXIT events so
  attendance can never go negative or double-count a visitor.
- Draws a small "Attendance" summary panel below the event notifications.
- No persistence (no file/database/CSV) and no UI beyond that one panel
  — see [Attendance System](#attendance-system).

**Phase 6 — Motion Detection (complete)**
- Classifies each tracked person as MOVING or STATIONARY, per track ID,
  using only centroid displacement (no optical flow, no background
  subtraction).
- Smooths displacement over a configurable rolling history so a single
  noisy frame cannot flip the reported state.
- Draws a "Moving"/"Stationary" label near each tracked person.
- Fully independent of attendance/line-crossing: consumes only
  `List[TrackedPerson]` — see [Motion Detection](#motion-detection).

**Phase 7 — Posture Detection (complete)**
- Classifies each tracked person as STANDING or SEATED, per track ID,
  using only their bounding box's height/width aspect ratio (no pose
  estimation, no MediaPipe, no YOLO-Pose, no additional model).
- Smooths the aspect ratio over a configurable rolling history so a
  single noisy frame cannot flip the reported posture.
- Draws a "Standing"/"Seated" label near each tracked person, below the
  motion label.
- Fully independent of attendance/line-crossing/motion: consumes only
  `List[TrackedPerson]` — see [Posture Detection](#posture-detection).

**Phase 8 — Blur Detection (complete)**
- Classifies each processed *frame as a whole* (not any individual
  person) as SHARP or BLURRY, using the classic Variance-of-Laplacian
  method.
- Runs on the raw, unannotated frame, before any overlay drawing, so
  drawn boxes/text cannot artificially inflate the measured sharpness.
- Draws a "Frame Quality" panel showing the classification and raw
  variance value.
- Fully independent of tracking/motion/posture/line-crossing/attendance:
  consumes only the raw frame — see [Blur Detection](#blur-detection).

**Phase 9 — Occupancy Detection (complete)**
- Derives a single room-level `OccupancyState` (EMPTY/OCCUPIED/ACTIVE/
  IDLE), consuming *only* `AttendanceStatistics`, `List[MotionState]`,
  `List[PostureState]`, and `BlurState` - never a frame, `TrackedPerson`,
  YOLO, ByteTrack, or `LineCrossingDetector` directly.
- `AttendanceStatistics.current_people` remains the single source of
  truth for occupancy; never recomputed from tracked people.
- Keeps the previous occupancy decision unchanged on a BLURRY (or
  missing-quality) frame, rather than risking a wrong classification.
- Draws an "Occupancy" summary panel below the "Frame Quality" panel.
- See [Occupancy Detection](#occupancy-detection) for the full decision
  flow and dependency diagram.

**Phase 10 — Streamlit Dashboard & Final Integration (complete)**
- Adds a Streamlit web UI (`dashboard/streamlit_app.py`) with webcam or
  video-upload source selection, Start/Stop/Reset controls, live
  annotated video, and every existing statistic (FPS, person count,
  attendance, motion summary, posture summary, frame quality, occupancy
  status, recent entry/exit events).
- Contains no detection/tracking/analysis logic of its own - it drives
  the pipeline through two functions extracted from `app.py` this phase,
  `build_pipeline_components()` and `process_frame()`, which
  `run_pipeline()` itself now also calls (its OpenCV-window behavior is
  unchanged).
- Handles no webcam, invalid/unsupported uploads, end of stream, and
  model-load failure gracefully - see [Dashboard](#dashboard).

**All planned phases (1-10) are now complete.** See
[Future Roadmap](#future-roadmap) for what remains genuinely out of
scope or listed only as a possible future improvement.

## Directory Structure

```
ComputerVisionAssignment/
│── app.py              # Entry point / composition root: capture -> track -> annotate -> display loop
│                       # Also exposes build_pipeline_components()/process_frame() (Phase 10), reused by dashboard/
│── config.py           # Central configuration (source, display, detection, tracking settings)
│── requirements.txt    # Python dependencies
│── README.md
│
├── assets/             # Sample/test video files go here
├── logs/               # Runtime logs are written here (app.log)
├── models/
│   ├── __init__.py
│   ├── person_detector.py  # PersonDetector (YOLOv8 model load + plain detection) + PersonDetection
│   └── yolov8n.pt           # YOLOv8 weights (auto-downloaded on first run)
├── tracker/
│   ├── __init__.py
│   └── byte_tracker.py      # PersonTracker (ByteTrack-based ID assignment) + TrackedPerson + draw_tracks
├── events/
│   ├── __init__.py
│   └── line_crossing.py     # LineCrossingDetector + CrossingEvent + EventNotificationBoard + draw_line/draw_events
├── attendance/
│   ├── __init__.py
│   └── attendance_manager.py  # AttendanceManager + AttendanceStatistics + draw_attendance_panel
├── motion/
│   ├── __init__.py
│   └── motion_detector.py    # MotionDetector + MotionState + draw_motion_states
├── posture/
│   ├── __init__.py
│   └── posture_detector.py   # PostureDetector + PostureState + draw_posture_states
├── quality/
│   ├── __init__.py
│   └── blur_detector.py      # BlurDetector + BlurState + draw_blur_panel
├── occupancy/
│   ├── __init__.py
│   └── occupancy_detector.py # OccupancyDetector + OccupancyState + draw_occupancy_panel
├── dashboard/
│   ├── __init__.py
│   └── streamlit_app.py      # Streamlit web UI - calls app.build_pipeline_components()/process_frame() only
└── utils/
    ├── __init__.py
    ├── logger.py        # setup_logger(): shared console + file logger setup
    └── fps_counter.py   # FPSCounter: smoothed FPS calculation
```

## Execution Flow

There are now two ways to run this project - `python app.py` (the
original OpenCV window) or `streamlit run dashboard/streamlit_app.py` (the
Phase 10 web dashboard, see [Dashboard](#dashboard)). Both drive the exact
same two functions, `build_pipeline_components()` and `process_frame()`,
in `app.py`.

**Running `python app.py`:**

1. `main()` logs a startup message and calls `run_pipeline()` inside a
   `try/except`.
2. `run_pipeline()`:
   a. `open_video_source()` opens the configured webcam or video file.
   b. `build_pipeline_components()` constructs every subsystem in one
      call - `load_person_detector()` (loading the YOLOv8 model,
      downloading weights if needed, and resolving CPU/CUDA),
      `load_person_tracker(detector)`, `load_line_crossing_detector()` +
      an `EventNotificationBoard`, `load_attendance_manager()`,
      `load_motion_detector()`, `load_posture_detector()`,
      `load_blur_detector()`, `load_occupancy_detector()`, and an
      `FPSCounter` - bundled into one `PipelineComponents`.
   c. The frame loop begins: read a frame → `process_frame(frame,
      components)` (see below for exactly what this does) →
      `cv2.imshow(...)` → check for the exit key or end-of-stream, and
      repeat.
   d. On loop exit (for any reason), the `finally` block releases the
      capture and destroys the display window.
3. Back in `main()`: a `RuntimeError` anywhere in the above (bad source,
   model load failure, tracker init failure on the first frame) is caught,
   logged, and turned into exit code `1`; a clean loop exit returns `0`.

**What `process_frame(frame, components)` does, every single frame, for
both entry points:** `blur_detector.analyze(frame)` (on the raw frame,
before any drawing) → `tracker.track(frame)` → `draw_tracks(...)` →
`motion_detector.update(tracks)` → `draw_motion_states(...)` →
`posture_detector.update(tracks)` → `draw_posture_states(...)` →
`line_crossing_detector.update(tracks)` (log + register any new events) →
`draw_line(...)` → `draw_events(...)` →
`attendance_manager.update(crossing_events)` → `draw_attendance_panel(...)`
→ `draw_blur_panel(...)` → `occupancy_detector.update(attendance_stats,
motion_states, posture_states, blur_state)` → `draw_occupancy_panel(...)`
→ `fps_counter.tick()` → `draw_fps(...)` → `draw_person_count(...)` →
return a `PipelineFrameResult` bundling every one of those outputs. The
frame is mutated in place with every overlay; the caller (either
`run_pipeline()` or the dashboard) decides how to display it and what to
do with the returned statistics.

**Running `streamlit run dashboard/streamlit_app.py`:** see
[Dashboard](#dashboard) for the full flow - in short, the dashboard calls
`build_pipeline_components()` once (on Start) and `process_frame()` once
per frame (in a rerun-driven loop), exactly mirroring steps 2b-2c above,
with no detection logic of its own.

## Detection Pipeline

Owned by `models/person_detector.py`. `PersonDetector` loads
`config.DETECTION_MODEL_PATH` (default `yolov8n.pt`, the lightweight
YOLOv8 variant, auto-downloaded by Ultralytics if missing) exactly once at
startup, resolving the compute device via `config.DETECTION_DEVICE`
(`"auto"` picks CUDA if available, else CPU).

Its `detect(frame)` method calls `model.predict(...)`, filtered to
`config.PERSON_CLASS_ID` (COCO class 0, "person") at or above
`config.DETECTION_CONFIDENCE_THRESHOLD`, and converts Ultralytics' result
boxes into plain `PersonDetection(x1, y1, x2, y2, confidence)` records. A
companion `draw_detections()` function draws a box + `Person <confidence>`
label per detection.

As of Phase 3, **`detect()`/`draw_detections()` are not called by the live
pipeline** — `PersonTracker.track()` performs YOLO detection internally as
part of tracking (see [Tracking Pipeline](#tracking-pipeline)), so running
`detect()` separately in the same loop would mean two YOLO forward passes
per frame instead of one. `PersonDetector` remains the sole owner of the
loaded model (exposed via its `model` property for `PersonTracker` to
reuse), and `detect()`/`draw_detections()` are kept as a standalone,
directly-testable detection-only API for any future use that needs plain
boxes without track IDs.

## Tracking Pipeline

Owned by `tracker/byte_tracker.py`. `PersonTracker` wraps the exact same
YOLO model instance `PersonDetector` already loaded (via its `model`
property) and drives it with Ultralytics' built-in ByteTrack integration:
`model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0], conf=..., device=...)`.

**What ByteTrack is:** a real-time multi-object tracking algorithm that
associates detections across frames into persistent tracks, primarily by
matching boxes between consecutive frames using position/overlap (not
visual appearance), including a second-stage match for low-confidence
boxes so partially-occluded people are less likely to be dropped and
re-assigned a new ID. This project uses Ultralytics' own bundled ByteTrack
integration — no custom tracker was written, per the Phase 3 requirement.

**How it works here:** `persist=True` is what keeps ByteTrack's internal
track state alive between calls, so the same person keeps the same
`track_id` from frame to frame. `PersonTracker` is therefore constructed
exactly once (in `app.load_person_tracker()`) and reused for the entire
stream — it must never be rebuilt per frame, or track continuity would be
lost. Each result box carries an optional `.id`; boxes without a confirmed
ID yet are skipped rather than shown with a placeholder. The result is
converted into `TrackedPerson(x1, y1, x2, y2, confidence, track_id)`
records, and `draw_tracks()` draws a box plus a two-line `ID: <n>` /
`Person <confidence>` label for each.

**Why detection lives inside the tracker call:** Ultralytics' `.track()`
API performs YOLO inference and ByteTrack association together, in one
call, over one model instance. There is no supported public API to hand
it pre-computed detections separately without either running inference
twice per frame (a real performance regression) or reaching into
ByteTrack's private/internal classes (effectively inventing glue code on
top of an undocumented API — the opposite of "use ByteTrack correctly").
Detection is therefore an intentional internal implementation detail of
`PersonTracker`, not something callers need to know about.

**The integration seam for future phases:** every current and future
consumer — `app.py` today, and attendance/entry-exit/occupancy modules
later — depends only on the `List[TrackedPerson]` returned by
`tracker.track(frame)`, never on how it was produced. This is the intended
hand-off point shown in [Current Architecture](#current-architecture):
future phases should add a new call that consumes that same list, not
reach back into detection or tracking internals.

**Error handling:**
- If tracking fails on the very first call, that's treated as a tracker
  initialization failure: a `RuntimeError` propagates up (same pattern as
  a model-load failure), the app logs the error and exits with code 1.
- If tracking fails on any later frame (after having succeeded at least
  once), the failure is logged as a warning and that frame contributes no
  tracks — the stream keeps running.

## Entry/Exit Pipeline

Owned by `events/line_crossing.py`. `LineCrossingDetector` consumes the
same `List[TrackedPerson]` that `draw_tracks()` already receives each
frame — it has no dependency on `PersonTracker` itself, only on its output
type, so it does not add any additional YOLO/ByteTrack calls.

**The virtual line:** one straight line, defined by two endpoints,
`config.LINE_START` and `config.LINE_END` (default `(100, 250)` to
`(550, 250)` — a horizontal line). Crossing this line is what determines
whether a person is considered to have entered or exited. The line's
position/orientation is entirely configuration-driven, so it can be moved
to match a different camera framing without touching any code.

**Centroid calculation:** for each `TrackedPerson`, the centroid is simply
the center of its bounding box: `((x1 + x2) // 2, (y1 + y2) // 2)` (see
`LineCrossingDetector.compute_centroid()`). This is a deliberately simple
choice — the box center, not e.g. a foot position or head position — and
is a reasonable proxy for "where the person is" for a roughly
front-facing/overhead camera angle.

**The algorithm:** for each tracked person, every frame:
1. Compute the centroid (above).
2. Compute which side of the virtual line it falls on, via a 2D
   cross-product sign test (`_signed_side()`): for the default horizontal
   line, a positive result means "below the line" (further down the
   frame) and a negative result means "above the line."
3. Look up the side recorded for that same `track_id` on the previous
   frame it was seen in (`LineCrossingDetector` keeps one dictionary entry
   per track ID for exactly this purpose — this is the "remember previous
   centroid / current centroid" requirement, implemented as "remember
   previous side / current side," which is the only information actually
   needed to detect a crossing).
4. If the side flipped from negative to positive, that is top-to-bottom
   motion → emit an **ENTRY** event. If it flipped from positive to
   negative, that is bottom-to-top motion → emit an **EXIT** event. If the
   side is unchanged (person lingering on the same side), or this is the
   first frame that track ID has been seen (no previous side to compare
   against), or the centroid lands exactly on the line (ambiguous), no
   event is emitted.
5. The recorded side for that track ID is updated to the current side
   either way (except the exactly-on-the-line case, which is left
   unresolved until the centroid moves to a definite side).

This directly guarantees **exactly one event per real crossing**: an event
only fires on a side *change*, so a person standing still or moving around
without crossing never re-triggers one, no matter how many frames they
remain on the same side.

**Display:** `draw_line()` draws the virtual line in
`config.LINE_COLOR` (blue by default). Each emitted `CrossingEvent` is
logged immediately (`ENTRY : ID 7` / `EXIT : ID 4`, matching the existing
logging architecture — console + `logs/app.log`) and handed to an
`EventNotificationBoard`, which keeps it visible for
`config.EVENT_DISPLAY_DURATION` seconds; `draw_events()` renders up to
`config.EVENT_NOTIFICATION_MAX_LINES` currently-visible events as stacked
text below the FPS/person-count lines.

**What this module deliberately does NOT do:** it does not sum ENTRY/EXIT
events into a running "currently present" or "total unique entries" count,
and it does not write events anywhere persistent (file, database, CSV).
Both are explicitly reserved for a future attendance module, which is
expected to consume `LineCrossingDetector.update()`'s
`List[CrossingEvent]` return value directly.

**Error handling:** a per-track failure while computing a centroid/side
(e.g. an unexpected value) is caught, logged as a warning, and that track
is skipped for the frame — it does not stop other tracks in the same
frame from being checked, and never crashes the stream. An empty `tracks`
list (no detections this frame) is a trivial no-op. A track ID that stops
appearing (lost track, or the person left frame) simply stops being
updated; see [Known Limitations](#known-limitations) for what happens if
that same person reappears under a new track ID later.

**Robustness safeguards:** two independent, purely internal safeguards
protect `LineCrossingDetector` over a long-running stream, without
changing its output for a normal, well-separated crossing:
- **Stale-track eviction** (`config.LINE_CROSSING_STALE_TRACK_TIMEOUT`,
  default 300 seconds): a track ID not seen again within this many seconds
  has its remembered side/last-seen/last-event state evicted entirely, so
  the detector's memory usage is bounded by "how many distinct people are
  currently or recently active," not by "every track ID that has ever
  existed in the whole session."
- **Event cooldown** (`config.LINE_CROSSING_EVENT_COOLDOWN`, default 1.0
  second): once an event fires for a track ID, no further event fires for
  that same track ID until the cooldown elapses, even if its centroid
  keeps flipping sides in the meantime (e.g. detector/tracker jitter right
  at the line). Side tracking itself is never suppressed, only the
  emission of a new event - so a genuine second crossing by the same
  person well after the cooldown window still fires normally.

## Attendance System

Owned by `attendance/attendance_manager.py`. `AttendanceManager` consumes
the same `List[CrossingEvent]` that `EventNotificationBoard` already
receives each frame from `LineCrossingDetector.update()` — it has no
dependency on `LineCrossingDetector` itself, only on its output type
(`CrossingEvent`), and **never computes a line crossing itself**.
`LineCrossingDetector` remains the sole source of truth for when a
crossing happened; `AttendanceManager`'s only job is deciding what that
crossing means for running totals.

**Why attendance is a separate module from line-crossing detection:**
`LineCrossingDetector` answers a purely geometric question — did this
track's centroid just cross this line, and which way? `AttendanceManager`
answers a purely bookkeeping question — given that a crossing happened,
what should the running totals now be? Keeping them separate means each
has exactly one reason to change: a future change to how crossings are
detected (e.g. a differently-shaped line, or multiple lines) cannot affect
attendance bookkeeping, and a future change to attendance rules (e.g. a
capacity limit) cannot affect crossing detection.

**AttendanceStatistics** (the return type of every `update()` call) is an
immutable snapshot with:
- `current_people` — always `len(inside_track_ids)`, never tracked as a
  separately-incremented counter, so it cannot drift out of sync with the
  actual inside set.
- `total_entries` / `total_exits` — running totals of *accepted* events
  only (rejected/duplicate events don't count).
- `unique_visitors` — count of distinct track IDs that have ever had an
  accepted ENTRY; does not increase again if the same track ID re-enters.
- `inside_track_ids` — the exact set of track IDs currently inside, as an
  immutable `frozenset` (a defensive copy of internal state).

**AttendanceManager's responsibilities** are attendance state only:
consume `CrossingEvent`s, maintain the `inside_track_ids` set (single
source of truth for "current people") and the `visited_track_ids` set
(single source of truth for "unique visitors"), maintain running
entry/exit counters, and return an `AttendanceStatistics` snapshot. It has
no UI code beyond `draw_attendance_panel()`, no tracking code, and no line
-crossing code.

**Acceptance rules:**
- **ENTRY**, track ID not already inside → accepted: add to
  `inside_track_ids` (and to `visited_track_ids` if new), increment
  `total_entries`.
- **ENTRY**, track ID already inside → ignored (logged): a duplicate.
- **EXIT**, track ID currently inside → accepted: remove from
  `inside_track_ids`, increment `total_exits`.
- **EXIT**, track ID not currently inside → ignored (logged): this covers
  both "duplicate EXIT" and "EXIT with no matching prior ENTRY" — they are
  the same case (the track ID simply isn't in the inside set).

Because a track ID can only ever be removed from `inside_track_ids` if it
was already present, `current_people` can never go negative.

**How attendance consumes CrossingEvents:** `app.py` calls
`attendance_manager.update(crossing_events)` once per frame, immediately
after `line_crossing_detector.update(tracks)` produces that frame's
(usually empty) event list. `update()` is safe to call with an empty list
— it is a no-op that still returns the current snapshot, so the
attendance panel can always be redrawn every frame regardless of whether
anything changed.

**Display:** `draw_attendance_panel()` draws a small panel — "Attendance",
`Current Inside : <n>`, `Entries : <n>`, `Exits : <n>`,
`Unique Visitors : <n>` — at `config.ATTENDANCE_PANEL_POSITION`, placed
below the event-notification block so the two never overlap.

**Error handling:** a malformed event (`None`, or an object missing
`track_id`) is caught, logged as a warning, and skipped individually — it
never aborts processing of the rest of that frame's events, and never
crashes the stream.

**Current limitations:**
- **No persistence**: all statistics live in memory only, for the life of
  the process; nothing is written to a file, database, or CSV.
- **`visited_track_ids` grows for the life of the run**: every distinct
  track ID that ever enters is remembered forever (this is exactly what
  "unique visitors" means, not a bug), so memory usage scales with total
  foot traffic over a session, not just current occupancy.
- **Inherits every upstream limitation unchanged**: if `LineCrossingDetector`
  ever double-fires or misses a crossing (see
  [Entry/Exit Pipeline](#entryexit-pipeline)'s known limitations), or if
  ByteTrack assigns a new track ID to someone re-entering after a long
  absence (see [Tracking Pipeline](#tracking-pipeline)), attendance
  inherits that same imprecision - it has no way to detect or correct for
  it, by design (it only ever sees events, never raw frames or tracks).

## Motion Detection

Owned by `motion/motion_detector.py`. `MotionDetector` consumes only the
same `List[TrackedPerson]` that `draw_tracks()` already receives each
frame — it has **no dependency on `LineCrossingDetector`, `CrossingEvent`,
`AttendanceManager`, or `AttendanceStatistics`**, and knows nothing about
lines, entries, exits, or attendance. It runs entirely in parallel with
those, sharing only the same tracked-person input.

**The motion algorithm:** no optical flow and no background subtraction
are used, per the Phase 6 requirement — only the centroid positions
already produced by tracking. For each tracked person, every frame:
1. Compute the centroid (the same simple box-center formula as
   `LineCrossingDetector.compute_centroid()`, deliberately re-implemented
   locally rather than imported, so this module stays fully decoupled —
   see its docstring's note).
2. Compute the Euclidean distance between this frame's centroid and the
   same track ID's centroid last frame (0.0 on a track's very first
   observed frame, since there is nothing to compare against yet).
3. Push that distance into a per-track rolling history of the last
   `config.MOTION_HISTORY_SIZE` distances, and compute the average of that
   history — this average, not the single latest frame's raw distance, is
   what gets compared against the threshold.
4. Classify **MOVING** if that averaged distance exceeds
   `config.MOTION_DISTANCE_THRESHOLD`, **STATIONARY** otherwise.

**The motion "state machine":** it is deliberately a two-state,
memory-of-recent-history classifier rather than a single-frame threshold
check. Averaging over `MOTION_HISTORY_SIZE` frames is what satisfies "do
not change state from one noisy frame" — a single frame with an
unusually large or small displacement only shifts the average slightly,
it cannot alone flip the classification the way comparing raw per-frame
distance to the threshold would. The two states are simply the current
average vs. threshold comparison; there is no separate hysteresis band or
minimum-duration timer beyond that averaging window.

**Data model:** `MotionState(track_id, is_moving, movement_distance,
current_centroid)` — `movement_distance` is the smoothed (averaged) value
described above, not the raw single-frame distance, so it directly
explains why `is_moving` came out the way it did.

**Display:** `draw_motion_states()` draws "Moving" (in
`config.MOTION_MOVING_COLOR`, red by default) or "Stationary" (in
`config.MOTION_STATIONARY_COLOR`, light gray by default) just below each
tracked person's centroid.

**Error handling:** a per-track failure computing a centroid is caught,
logged as a warning, and that track is skipped for the frame — it never
stops other tracks from being processed and never crashes the stream. An
empty `tracks` list is a trivial no-op. A track ID that stops appearing
(lost track, or the person left frame) simply stops being updated; like
`LineCrossingDetector`, its state is evicted after
`config.MOTION_STALE_TRACK_TIMEOUT` seconds of not being seen, so memory
is bounded rather than growing for every track ID ever seen in the
session.

**Logging:** only a motion state *change* (MOVING → STATIONARY or vice
versa) is logged, at INFO level — never once per frame, and never a
track's very first classification (there is no prior state for it to have
"changed" from).

**Current limitations:**
- **Threshold is in raw pixels, not physical units or resolution-aware**:
  `config.MOTION_DISTANCE_THRESHOLD` is a pixel-displacement cutoff, so
  it may need re-tuning if the capture resolution or the camera's distance
  from the scene changes significantly.
- **No distinction between "walking in place" and genuine stillness**: any
  centroid displacement is treated identically regardless of cause (real
  walking vs. detector/tracker jitter vs. camera shake) - only the
  averaging window softens the effect of jitter, it does not distinguish
  its source.
- **Independent of tracking-loss context**: if a track is briefly lost and
  reappears in a very different position, the first post-reappearance
  centroid comparison can register a large "jump" distance that isn't
  real movement (see the tracking limitations already noted in
  [Tracking Pipeline](#tracking-pipeline)); this module has no way to
  detect that the jump was due to a tracking gap rather than real motion.

**Future use by occupancy detection:** a future occupancy module is a
natural consumer of `List[MotionState]` alongside `AttendanceStatistics`
— e.g. distinguishing "room occupied, people active" from "room occupied,
everyone stationary" (which might indicate seated/idle occupants rather
than active movement) — by reading `MotionDetector.update()`'s output
directly, the same way this phase's design keeps every subsystem's output
a simple, independently-consumable list.

## Posture Detection

Owned by `posture/posture_detector.py`. `PostureDetector` consumes only
the same `List[TrackedPerson]` that `draw_tracks()` already receives each
frame — like `MotionDetector`, it has **no dependency on
`LineCrossingDetector`, `CrossingEvent`, `AttendanceManager`,
`AttendanceStatistics`, or `MotionDetector`/`MotionState`**. It loads no
additional model of any kind - no pose estimation, no MediaPipe, no
YOLO-Pose - only the bounding box geometry already produced by tracking.

**The bounding-box heuristic:** a person's YOLOv8 bounding box is drawn
tightly around their visible extent. A standing person's box is usually
noticeably taller than it is wide (a large height/width ratio); a seated
person's box tends to be closer to square, or even wider than tall,
because sitting compresses their vertical extent relative to their
width. This is a coarse geometric proxy for posture, not a measurement of
actual body pose - it works reasonably well for a roughly front-facing
camera and can be fooled by unusual angles, crops, or partial visibility
(see Current Limitations below).

**The algorithm:** for each tracked person, every frame:
1. Compute `width = x2 - x1` and `height = y2 - y1` from the box.
2. If either is not strictly positive (a zero-width, zero-height, or
   otherwise invalid box), skip that track for this frame entirely (logged
   as a warning) rather than dividing by zero.
3. Otherwise compute `aspect_ratio = height / width`.
4. Push that ratio into a per-track rolling history of the last
   `config.POSTURE_HISTORY_SIZE` frames, and average it - this average,
   not the single latest frame's raw ratio, is what gets compared against
   the threshold.
5. Classify **STANDING** if that averaged ratio exceeds
   `config.POSTURE_ASPECT_RATIO_THRESHOLD`, **SEATED** otherwise.

Unlike `MotionDetector`'s displacement (which requires a *previous* frame
to measure a delta from), a bounding box's aspect ratio is a complete,
meaningful value on the very first frame a track is observed - so a new
track is classified immediately from its first ratio, with no special
"unknown" default needed.

**Jitter smoothing:** averaging over `POSTURE_HISTORY_SIZE` frames is what
prevents a single noisy frame (e.g. a momentarily clipped or
partially-occluded box) from flipping the reported posture - a lone
outlier ratio only nudges the average slightly, exactly mirroring how
`MotionDetector` smooths displacement in Phase 6.

**Data model:** `PostureState(track_id, is_standing, aspect_ratio,
current_bbox)` — `aspect_ratio` is the smoothed (averaged) value described
above, and `current_bbox` is the raw `(x1, y1, x2, y2)` box for this
frame, kept so the drawing helper can position the label without needing
any other module.

**Display:** `draw_posture_states()` draws "Standing" (in
`config.POSTURE_STANDING_COLOR`, yellow by default) or "Seated" (in
`config.POSTURE_SEATED_COLOR`, magenta by default) just below each tracked
person's bounding-box center, offset further down than the motion label
(`config.POSTURE_LABEL_OFFSET_Y > config.MOTION_LABEL_OFFSET_Y`) so the
two stack without overlapping.

**Error handling:** a per-track failure computing the aspect ratio
(invalid/malformed box, zero width, zero height, or negative dimensions)
is caught, logged as a warning, and that track is skipped for the frame —
it never stops other tracks from being processed and never crashes the
stream. An empty `tracks` list is a trivial no-op. A track ID that stops
appearing is simply not updated; like `MotionDetector`, its state is
evicted after `config.POSTURE_STALE_TRACK_TIMEOUT` seconds of not being
seen, so memory is bounded rather than growing for every track ID ever
seen in the session.

**Logging:** only a posture *change* (STANDING → SEATED or vice versa) is
logged, at INFO level — never once per frame, and never a track's very
first classification.

**Current limitations:**
- **Camera-angle sensitive**: the height/width heuristic assumes a
  roughly front-facing or elevated camera looking across or down at
  people. A very low or oblique camera angle, or a person seen mostly
  from directly above, can produce aspect ratios that don't match the
  "tall = standing, square/wide = seated" assumption at all.
- **No true calibration**: `POSTURE_ASPECT_RATIO_THRESHOLD` (default 1.2)
  is a reasonable starting heuristic, not a value derived from measured
  data; it will likely need retuning per camera setup, distance, and
  typical clothing/furniture in view.
- **Partial visibility confuses the heuristic**: if a person is partially
  occluded (e.g. only their upper body is visible behind a desk, or they
  are cut off at the frame edge), the resulting box's aspect ratio
  reflects the *visible* extent, not their actual posture - a standing
  person partially hidden behind furniture can produce a "seated-looking"
  short box, and vice versa.
- **No distinction between sitting on a low chair, crouching, or
  kneeling**: any bounding box in the "short and wide" range is classified
  SEATED, whether or not the person is actually seated.

**Future improvements using pose estimation:** the bounding-box heuristic
here is deliberately simple, per the Phase 7 scope (explicitly excluding
pose estimation, MediaPipe, and YOLO-Pose). A future iteration could
replace or augment `PostureDetector` with a real pose-estimation model
(e.g. keypoint-based hip/knee/shoulder angle analysis) to distinguish
standing/sitting/crouching/kneeling far more reliably, independent of
camera angle and occlusion - while keeping the same `PostureState` output
shape so `app.py` and any future consumer would not need to change.

## Blur Detection

Owned by `quality/blur_detector.py`. `BlurDetector` consumes only a raw
frame (a `numpy.ndarray`) — it has **no dependency whatsoever on
tracking, motion, posture, line-crossing, or attendance**; it doesn't even
know `TrackedPerson` exists. This is a frame-level, not a per-person,
classification.

**Variance of Laplacian:** the Laplacian operator is a second-derivative
edge detector - it responds strongly to areas of rapid intensity change
(edges, fine detail) and weakly to smooth, flat regions. A sharp,
in-focus image has lots of well-defined edges, so its Laplacian has a
wide spread of values (high variance). A blurry image has smeared-out,
weak edges, so its Laplacian values cluster closer together (low
variance). This makes the *variance* of the Laplacian a simple, effective,
long-established proxy for overall image sharpness — used here exactly as
in its classic form, with no additional model.

**The algorithm:**
1. Convert the frame to grayscale (`cv2.cvtColor(..., COLOR_BGR2GRAY)` for
   a standard 3-channel BGR frame; BGRA and already-grayscale frames are
   also handled - see Error handling below).
2. Compute the Laplacian: `cv2.Laplacian(gray, cv2.CV_64F)`.
3. Compute its variance: `.var()`.
4. Compare that variance against `config.BLUR_THRESHOLD`.
5. Classify **BLURRY** if the variance is below the threshold, **SHARP**
   otherwise.

This is a **frame-level** analysis only, run once per frame - it does not
iterate over, or know anything about, individual tracked people, per the
Phase 8 requirement.

**Data model:** `BlurState(is_blurry, laplacian_variance)` — the raw
variance is exposed alongside the boolean classification so the actual
measured value (e.g. for the "Variance : 284.7" panel line) is always
available, not just the threshold comparison result.

**Why analysis happens before any drawing:** `blur_detector.analyze(frame)`
is called in `app.run_pipeline()` immediately after a frame is read,
*before* `tracker.track(frame)`, `draw_tracks()`, or any other overlay
function touches that same frame object. Every other drawing function in
this project mutates the frame in place (`cv2.rectangle`, `cv2.putText`,
`cv2.line`), and drawn boxes/text are themselves sharp, high-contrast
edges - analyzing an already-annotated frame would artificially inflate
its measured sharpness and defeat the whole point of measuring the
camera feed's actual quality.

**Display:** `draw_blur_panel()` draws a three-line "Frame Quality" panel
— the header, "Sharp"/"Blurry" (colored via `config.BLUR_SHARP_COLOR` /
`config.BLUR_BLURRY_COLOR`), and the raw variance value — at
`config.BLUR_PANEL_POSITION`, placed below the attendance panel.

**Error handling:** `None`, an empty array, a non-numpy-array value, or an
unsupported channel count (anything other than grayscale/BGR/BGRA) are
all caught and answered with a conservative fallback:
`BlurState(is_blurry=True, laplacian_variance=0.0)` - i.e. "flag for
review" rather than silently claiming an unmeasurable frame is sharp.
This never raises and never crashes the stream.

**Logging:** only a quality *change* (SHARP → BLURRY or vice versa) is
logged, at INFO level — never once per frame, and never the first
classification (there is no prior state for it to have "changed" from).
Since this is a single, whole-frame classification (not one per track
ID), there is no per-track history to bound and no stale-track eviction
needed — the class holds exactly one piece of state (the last logged
classification), so memory usage is inherently constant.

**Current limitations:**
- **Global, not local**: a frame can be sharp in most of the scene but
  blurry in one region (e.g. a person moving quickly while the background
  is static) - this method measures the *whole frame's* average
  sharpness and cannot localize which region, or which person, is
  actually blurred.
- **Threshold is resolution/content dependent**: `config.BLUR_THRESHOLD`
  (default 100.0) is a widely-cited starting point for the classic
  method, not a value calibrated for this project's specific
  camera/lighting; it may need retuning per deployment, and a busier
  scene (more texture/edges even when in focus) will naturally read a
  higher variance than a plain one.
- **Motion blur vs. focus blur look the same**: this method cannot
  distinguish a frame that's blurry because the lens is out of focus from
  one that's blurry because something (or the whole camera) moved quickly
  during exposure - both simply reduce edge sharpness the same way.

**Future improvements:** a more advanced approach could compute a
per-region (e.g. per-tracked-person) sharpness map instead of one
whole-frame value, or use a learned no-reference image-quality model
instead of a hand-crafted heuristic, while keeping the same `BlurState`-
shaped output so downstream consumers would not need to change.

## Occupancy Detection

Owned by `occupancy/occupancy_detector.py`. `OccupancyDetector` is
architecturally different from every other subsystem so far: it consumes
**no frame, no `TrackedPerson`, and no tracking/detection internals at
all** - only the already-computed outputs of four other subsystems.

**Dependency diagram:**

```
AttendanceStatistics ─┐
List[MotionState]     ├──▶ OccupancyDetector.update() ──▶ OccupancyState
List[PostureState]    │       (occupancy/occupancy_detector.py)
BlurState             ┘

OccupancyDetector has NO import of, and NO dependency on:
  - models.person_detector (PersonDetector, YOLO)
  - tracker.byte_tracker (PersonTracker, TrackedPerson, ByteTrack)
  - events.line_crossing (LineCrossingDetector, CrossingEvent)
  - any video frame (numpy.ndarray) whatsoever
```

This mirrors how Motion/Posture/Blur were each kept independent of each
other in earlier phases, one level higher up: Occupancy depends on their
*outputs*, never on how those outputs were produced.

**`OccupancyState` (the data model):**
- `occupancy_status`: one of `OccupancyStatus.EMPTY` / `.OCCUPIED` /
  `.ACTIVE` / `.IDLE`.
- `people_inside`: copied directly from
  `AttendanceStatistics.current_people` - never recomputed from tracked
  people, motion, or posture data.
- `moving_people`: count of `List[MotionState]` entries with
  `is_moving=True`.
- `standing_people` / `seated_people`: counts of `List[PostureState]`
  entries with `is_standing=True` / `False`.
- `frame_quality_ok`: whether the frame this decision is based on was
  confirmed SHARP.

**Occupancy states and the decision flow:** evaluated in this order, using
`people_inside` from `AttendanceStatistics` and `moving_people` derived
from `List[MotionState]`:
1. `people_inside == 0` → **EMPTY**.
2. `people_inside > 0` and motion data is **unavailable** (`motion_states`
   is `None`) → **OCCUPIED**. This is the generic, honest fallback used
   specifically when we know people are present but cannot say whether
   anyone is moving — it is not a synonym for ACTIVE/IDLE, and is
   deliberately distinct from an *empty* `List[MotionState]` (which means
   "zero people moving," a known value, not an unknown one).
3. `people_inside > 0`, motion data available, and `moving_people > 0` →
   **ACTIVE**.
4. `people_inside > 0`, motion data available, and `moving_people == 0` →
   **IDLE**.

`standing_people`/`seated_people` are always computed and included in the
panel/state regardless of which of the four statuses is chosen - they
describe the room's people, not the occupancy status itself.

**Frame quality gating:** if the current `BlurState` indicates BLURRY (or
is missing/unusable) **and a previous decision already exists**, that
previous `OccupancyState` is returned completely unchanged - not
recomputed, not partially updated - and the skip is logged once (not on
every subsequent blurry frame). On the very first call ever (no previous
decision to fall back on), a decision is still computed from whatever data
is available, since there is nothing to "keep" yet; its
`frame_quality_ok` correctly reflects that the input wasn't confirmed
sharp.

**Relationship with the other subsystems:**
- **Attendance**: the *only* source of `people_inside`. Occupancy never
  counts tracked people, motion states, or posture states to determine how
  many people are present.
- **Motion**: the *only* source of `moving_people`, and the only thing
  that distinguishes ACTIVE from IDLE (or from the OCCUPIED fallback, when
  motion data is unavailable).
- **Posture**: contributes `standing_people`/`seated_people` for display;
  does not itself influence which of the four `occupancy_status` values is
  chosen.
- **Blur**: gates *whether* a new decision is computed at all this frame,
  but never determines the decision's content.

**Error handling:** the entire computation is wrapped so a failure
processing any input (a malformed `MotionState`/`PostureState` entry, an
unexpected type) is caught, logged as a warning, and answered with the
previous decision if one exists, or a safe all-zero EMPTY state if this is
the very first call - never a crash. Missing `attendance_stats` (`None`)
defaults `people_inside` to 0; an empty `motion_states`/`posture_states`
list contributes 0 to the relevant counts (a normal, expected case, not an
error).

**Logging:** only two triggers ever produce a log line - an occupancy
*status change* (e.g. `IDLE -> ACTIVE`) and the *start* of a blur-driven
skip (not its continuation) - never once per ordinary frame.

**Current limitations:**
- **No hysteresis on the ACTIVE/IDLE boundary**: a single frame where
  `moving_people` flips from 0 to 1 (or back) immediately flips ACTIVE/
  IDLE - `MotionDetector` already smooths individual motion
  classifications (see [Motion Detection](#motion-detection)), but
  `OccupancyDetector` does not add a second layer of smoothing on top of
  the aggregated count.
- **OCCUPIED is reachable only via missing motion data**: in normal
  operation (where `app.py` always supplies a `List[MotionState]`, even
  if empty), the OCCUPIED status is rarely, if ever, actually reached -
  it exists specifically to handle a genuinely missing/unavailable motion
  input gracefully, per the Phase 9 error-handling requirement, rather
  than being a commonly-observed state in this pipeline's normal
  operation.
- **No lights-on-but-empty detection**: the original task's bonus
  requirement (flagging a room with lights on but nobody present) would
  need frame brightness analysis, which is out of scope for this phase -
  `OccupancyDetector` reports EMPTY correctly, but does not know anything
  about lighting.
- **Blurry-frame gating is binary, not proportional**: any BLURRY reading
  freezes the decision entirely, regardless of how blurry - there is no
  partial-trust or confidence-weighted blending between the frozen
  decision and fresh (but unreliable) data.

## Dashboard

Owned by `dashboard/streamlit_app.py`. This is a Streamlit web front-end
for the exact same pipeline `app.py` runs via OpenCV - it contains **no
detection, tracking, or analysis logic of its own**. Every frame is
processed by calling `app.process_frame(frame, components)`, the same
function `run_pipeline()` calls; the dashboard only decides *how to
display* the annotated frame and the `PipelineFrameResult` it returns.

**Architecture:** to make this possible without duplicating any pipeline
logic, Phase 10 extracted two functions out of `app.py` (see
[Execution Flow](#execution-flow) for the full detail):
- `build_pipeline_components()` — constructs every subsystem
  (`PersonDetector`, `PersonTracker`, `LineCrossingDetector`,
  `AttendanceManager`, `MotionDetector`, `PostureDetector`,
  `BlurDetector`, `OccupancyDetector`, plus an `EventNotificationBoard`
  and `FPSCounter`) via the existing `load_*()` functions, bundled into
  one `PipelineComponents`.
- `process_frame(frame, components)` — runs one frame through every
  stage of the pipeline (identical to what was previously inlined
  directly in `run_pipeline()`'s loop), annotates the frame in place, and
  returns a `PipelineFrameResult` bundling every statistic produced.

`run_pipeline()` itself was refactored to call these two functions
instead of inlining their bodies - its own behavior (the OpenCV window,
FPS, overlays, exit key) is completely unchanged; only *where* that logic
lives moved. `dashboard/streamlit_app.py` is the second, independent
caller of the same two functions.

**Since Streamlit reruns its whole script on every interaction** (it has
no built-in persistent `while True` video loop the way an OpenCV window
does), the dashboard uses the standard workaround for this: it processes
exactly one frame per script run, then calls `st.rerun()` to trigger
another run if still active. `st.session_state` holds everything that
must survive across those reruns - the open `cv2.VideoCapture`, the built
`PipelineComponents`, the last annotated frame, the last
`PipelineFrameResult`, and a rolling history of recent crossing events -
so nothing is rebuilt or reprocessed on each rerun beyond the one new
frame.

**User interface:**
- **Sidebar** — a "Webcam" / "Upload Video" source choice (with a file
  uploader shown only in Upload mode), and Start / Stop / Reset buttons.
  Start opens the chosen source and builds the pipeline (if not already
  built); Stop pauses processing while keeping the last frame and
  statistics visible; Reset fully clears everything (including deleting
  the temp file backing an uploaded video) back to a blank dashboard.
- **Main page** — the live annotated video (left) and a statistics panel
  (right): current FPS, current person count, Attendance, Motion Summary,
  Posture Summary, Frame Quality, Occupancy Status, and the last 10
  Entry/Exit events (most recent first).

**Display, not recomputation:** every statistic shown is read directly
from the `PipelineFrameResult` fields (`attendance_stats`, `motion_states`,
`posture_states`, `blur_state`, `occupancy_state`, `fps`, `tracks`) -
"Moving: N / Stationary: M" and "Standing: N / Seated: M" are simple
counts over the already-classified `is_moving`/`is_standing` values,
exactly like `OccupancyDetector` itself already does internally; nothing
is detected, tracked, or classified a second time by the dashboard.

**Error handling** (all converted to an `st.error()`/`st.info()` message,
never a crash):
- **No webcam**: `cv2.VideoCapture(...).isOpened()` failing raises a
  caught `RuntimeError` with a clear message.
- **Invalid/unsupported upload**: pressing Start in Upload mode with no
  file chosen, or with a file OpenCV cannot open, is caught the same way.
- **End of stream**: a failed `capture.read()` stops the loop; an
  uploaded video ending is shown as an informational message (`st.info`,
  not an error - reaching the end of a file is expected), while an
  unexpected webcam disconnect is shown as an error.
- **Model loading failure**: `build_pipeline_components()` raising
  `RuntimeError` (the same failure `run_pipeline()` would hit) is caught
  and displayed instead of crashing the dashboard.
- **Unexpected exceptions** while starting are caught by a broad
  `except Exception` in `_handle_start()` as a final safety net.

**How to launch:**
```bash
pip install -r requirements.txt   # includes streamlit as of Phase 10
streamlit run dashboard/streamlit_app.py
```
This opens the dashboard in your browser (Streamlit prints the local URL,
typically `http://localhost:8501`). `python app.py` continues to work
exactly as before, independently.

**Required packages:** everything already in `requirements.txt`
(`opencv-python`, `numpy`, `ultralytics`, `lap`), plus `streamlit` (added
this phase - see [Dependencies](#dependencies)).

**Example screenshots:** *(placeholder - add screenshots here after a
live demo run, e.g. `docs/screenshots/dashboard_webcam.png` and
`docs/screenshots/dashboard_upload.png`; none are committed yet)*.

**Current limitations:**
- **The rerun-driven loop is not true real-time video**: each processed
  frame triggers a full Streamlit script rerun; this is the standard,
  documented way to drive continuous video in Streamlit (which has no
  native persistent frame loop), but it is measurably less smooth than
  the OpenCV window's direct `cv2.imshow()` loop, especially at high FPS.
- **Single session, single source at a time**: no multi-camera support
  (explicitly out of scope for this phase) and no concurrent multi-user
  isolation testing beyond Streamlit's own per-session `session_state`.
- **No authentication, database, or export**: the dashboard is a live
  viewer only, exactly per the Phase 10 scope - nothing is persisted
  beyond the current browser session.
- **Uploaded video temp files** are written to the OS temp directory and
  removed on Reset (or when a new file is uploaded); they are not cleaned
  up if the Streamlit process is killed without a Reset first.

## Configuration Guide

All configuration lives in `config.py`; every value has an inline comment
explaining what it controls, its valid values, and why it exists. Summary
by section:

| Section | Key variables | Purpose |
|---|---|---|
| Video source | `SOURCE_TYPE`, `WEBCAM_INDEX`, `VIDEO_PATH` | Choose webcam vs. video file input |
| Display | `WINDOW_NAME`, `FRAME_WIDTH`, `FRAME_HEIGHT`, `EXIT_KEY` | Window title, optional capture resolution, quit key |
| Logging | `LOG_DIR`, `LOG_FILE` | Where logs are written |
| Person detection | `DETECTION_MODEL_PATH`, `PERSON_CLASS_ID`, `DETECTION_CONFIDENCE_THRESHOLD`, `DETECTION_DEVICE` | YOLOv8 weights, class filter, confidence cutoff, CPU/GPU selection |
| Detection/track overlay | `BOX_COLOR`, `BOX_THICKNESS`, `DETECTION_LABEL_FONT_SCALE`, `DETECTION_LABEL_FONT_THICKNESS`, `DETECTION_LABEL_GAP_ABOVE_BOX`, `DETECTION_LABEL_MIN_Y`, `DETECTION_LABEL_FALLBACK_OFFSET_BELOW` | Box/label appearance and placement |
| Status overlay | `OVERLAY_FONT_SCALE`, `OVERLAY_FONT_THICKNESS`, `OVERLAY_COLOR`, `FPS_TEXT_POSITION`, `PERSON_COUNT_TEXT_POSITION` | FPS/person-count line styling and position |
| Tracking | `TRACKER_CONFIG`, `TRACK_LABEL_LINE_SPACING` | Which Ultralytics tracker config to use; track label line spacing |
| Entry/Exit line | `LINE_START`, `LINE_END`, `LINE_COLOR`, `LINE_THICKNESS` | Virtual line position/appearance |
| Entry/Exit notifications | `EVENT_DISPLAY_DURATION`, `EVENT_NOTIFICATION_POSITION`, `EVENT_NOTIFICATION_LINE_SPACING`, `EVENT_NOTIFICATION_MAX_LINES` | How long/where/how many recent ENTRY/EXIT events are shown |
| Entry/Exit robustness | `LINE_CROSSING_STALE_TRACK_TIMEOUT`, `LINE_CROSSING_EVENT_COOLDOWN` | How long an unseen track ID's state is kept before eviction; how long duplicate events for the same track ID are suppressed |
| Attendance panel | `ATTENDANCE_PANEL_POSITION`, `ATTENDANCE_PANEL_LINE_SPACING` | Where the "Attendance" summary panel is drawn (its text style reuses the status-overlay constants above) |
| Motion detection | `MOTION_DISTANCE_THRESHOLD`, `MOTION_HISTORY_SIZE`, `MOTION_STALE_TRACK_TIMEOUT`, `MOTION_MOVING_COLOR`, `MOTION_STATIONARY_COLOR`, `MOTION_LABEL_OFFSET_Y` | Movement sensitivity/smoothing, memory bound, and label appearance/placement |
| Posture detection | `POSTURE_ASPECT_RATIO_THRESHOLD`, `POSTURE_HISTORY_SIZE`, `POSTURE_STALE_TRACK_TIMEOUT`, `POSTURE_STANDING_COLOR`, `POSTURE_SEATED_COLOR`, `POSTURE_LABEL_OFFSET_Y` | Standing/seated sensitivity/smoothing, memory bound, and label appearance/placement |
| Blur detection | `BLUR_THRESHOLD`, `BLUR_SHARP_COLOR`, `BLUR_BLURRY_COLOR`, `BLUR_PANEL_POSITION`, `BLUR_PANEL_LINE_SPACING` | Sharpness sensitivity and "Frame Quality" panel appearance/placement |
| Occupancy detection | `OCCUPANCY_PANEL_POSITION`, `OCCUPANCY_PANEL_LINE_SPACING` | "Occupancy" panel placement only - no thresholds (occupancy has no tunable logic of its own) and no new colors (reuses `OVERLAY_COLOR`/`OVERLAY_FONT_SCALE`/`OVERLAY_FONT_THICKNESS`) |

To switch from webcam to a video file, edit:
```python
SOURCE_TYPE = "video"
VIDEO_PATH = "assets/sample_video.mp4"  # or any path to a test clip
```

## Dependencies

Declared in `requirements.txt`:

- **`opencv-python`** — video capture (webcam/file), drawing, and the
  display window. Used since Phase 1.
- **`numpy`** — array support that OpenCV/Ultralytics rely on. Used since
  Phase 1.
- **`ultralytics`** — YOLOv8 model loading/inference (Phase 2) and the
  built-in ByteTrack tracking integration (Phase 3). Pulls in `torch`
  (and, transitively, `torchvision`) as a dependency.
- **`lap`** — used internally by Ultralytics' ByteTrack integration for
  frame-to-frame box matching (linear assignment). Used since Phase 3.
- **`streamlit`** — powers the optional web dashboard
  (`dashboard/streamlit_app.py`). Used since Phase 10; not required to run
  `python app.py`.

On first run, Ultralytics automatically downloads the `yolov8n.pt` weights
file into `models/` if it isn't already there — this requires an internet
connection the first time only.

## Known Limitations

These are documented, intentional trade-offs for the current phase, not
oversights:

- **ID swaps on heavy occlusion**: if two people fully cross paths and
  heavily overlap, ByteTrack can occasionally swap their IDs. This is a
  known limitation of IoU-based tracking (no appearance/re-ID model is
  used), not something this project attempts to solve.
- **New ID after a long absence**: if a person leaves the frame for
  longer than ByteTrack's internal track buffer, they are assigned a new
  ID on reappearance rather than their old one — acceptable per the
  Phase 3 scope; re-identification across long absences is not attempted.
- **`PersonDetector.detect()`/`draw_detections()` are unused by the live
  pipeline**: as of Phase 3, `PersonTracker.track()` performs detection
  internally, so these Phase 2 functions are not called from `app.py`.
  They are left in place, unmodified, as a standalone detection-only API
  rather than deleted — see [Detection Pipeline](#detection-pipeline).
- **Single-threaded, synchronous processing**: frame capture, inference,
  tracking, and display all happen sequentially in one thread. This keeps
  the code simple and easy to reason about, at the cost of not overlapping
  I/O with compute; acceptable at single-camera scale.
- **No frame-rate limiting or frame dropping**: the loop processes frames
  as fast as the source and inference allow; there is no logic to skip
  frames to hit a target rate.
- **A crossing under a new track ID after a long absence is invisible to
  entry/exit detection**: since `LineCrossingDetector` keys its per-track
  memory by `track_id`, and ByteTrack assigns a *new* ID to someone who
  reappears after its internal track buffer expires (see the Phase 3
  tracking limitation above), that reappearance is treated as a brand-new,
  never-before-seen track — it takes two observations before a crossing
  can be reported, exactly like any new track ID, even if the person is
  actually re-entering.
- **Line coordinates are absolute pixels, not resolution-independent**:
  `config.LINE_START`/`LINE_END` must be chosen to match the actual
  capture resolution in use; they are not normalized (e.g. as fractions of
  frame width/height), so changing `FRAME_WIDTH`/`FRAME_HEIGHT` or
  switching to a differently-sized video source may require updating the
  line's coordinates too.
- **Event flicker from centroid jitter is bounded, not eliminated**: a
  centroid oscillating right at the line (detector/tracker jitter) is
  absorbed by `LINE_CROSSING_EVENT_COOLDOWN` (default 1.0s) — only the
  first flip in a burst is reported. If jitter happens to persist for
  longer than the cooldown window, or two genuine crossings by the same
  person legitimately happen faster than the cooldown, only the first is
  reported; the cooldown is a fixed timer, not a true hysteresis band on
  distance from the line.
- **Stale-track eviction is time-based, not identity-aware**: a track ID
  unseen for longer than `LINE_CROSSING_STALE_TRACK_TIMEOUT` (default 300s)
  is forgotten. If that exact track ID were somehow seen again after
  eviction (ByteTrack does not currently reuse IDs, so this is only a
  theoretical case), it would be treated as brand-new, exactly like any
  other new track ID.

## Current Windows Torch Policy Issue

On the Windows machine this project was developed and reviewed on, running
the app previously failed at `import torch` with:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading "...\site-packages\torch\lib\shm.dll" or one of its dependencies.
```

This was traced to **Windows Smart App Control** (a machine-level
Application Control/Code Integrity policy, confirmed via
`Microsoft-Windows-CodeIntegrity/Operational` event log entries) blocking
`torch`'s compiled DLLs — **not a bug in this project's code**. On this
machine it has since been resolved at the OS level and `import torch` now
succeeds, so the real YOLOv8 + ByteTrack pipeline runs end-to-end.

**Fallback mode (safety net):** in case this policy (or an equivalent one
on a different machine) blocks torch again, `app.py` wraps its
`models.person_detector`/`tracker.byte_tracker` imports in a try/except. If
that import fails, it transparently switches to
`tracker/fallback_tracker.py` — a torch-free, OpenCV-only tracker
(background subtraction + nearest-centroid ID matching) exposing the exact
same `TrackedPerson`/`draw_tracks()` shape, so every downstream module
(motion, posture, line-crossing, attendance, occupancy) and the dashboard
keep working unmodified, with reduced detection accuracy, instead of the
whole pipeline refusing to start. `app.FALLBACK_MODE` is `False` in normal
operation (as on this machine right now) and only becomes `True` if the
torch import fails again. When it fully blocked
`import torch` and therefore `import ultralytics`, Phase 2 and
Phase 3 (detection and tracking) could not be executed end-to-end on that
specific machine until the policy allowed `shm.dll`, or the app was run on a
machine/account without that restriction. `events/line_crossing.py`,
`motion/motion_detector.py`, and `posture/posture_detector.py` all import
`TrackedPerson` from `tracker/byte_tracker.py`; `attendance/attendance_manager.py`
imports from `events/line_crossing.py`; and `occupancy/occupancy_detector.py`
imports from `attendance/attendance_manager.py`, `motion/motion_detector.py`,
and `posture/posture_detector.py` (though it consumes only their plain
data types, never `TrackedPerson` itself - see
[Occupancy Detection](#occupancy-detection)). So all five modules are
affected transitively even though none of them ever imports
`torch`/`ultralytics` directly. See
[Troubleshooting](#troubleshooting) for what to check if you hit this.
Code-level correctness for the affected modules was instead verified via
static review and, where possible, by reproducing the relevant behavior
with `torch`-free standalone scripts — e.g. the shared logger setup, the
entire `LineCrossingDetector`/`EventNotificationBoard` crossing-direction,
duplicate-avoidance, and expiry logic (Phase 4), the entire
`AttendanceManager` acceptance/rejection logic (Phase 5), the entire
`MotionDetector` classification, smoothing, and stale-eviction logic
(Phase 6), the entire `PostureDetector` classification, smoothing, and
stale-eviction logic (Phase 7), and the entire `OccupancyDetector`
state-decision and blur-gating logic (Phase 9) — all tested against a
lightweight stand-in for `TrackedPerson` with no `torch` dependency.
**Phase 8's `quality/blur_detector.py` is the one exception**: it depends
only on `cv2`/`numpy`, not on `TrackedPerson` or anything else touching
`tracker/byte_tracker.py`, so it was unaffected by this policy and was
tested directly against real frame arrays (checkerboard/flat/Gaussian-
blurred images), not just a stand-in. `dashboard/streamlit_app.py`
(Phase 10) imports `app.py` directly, so it is affected by the same
policy as every module `app.py` imports; its own logic (session-state
handling, source opening, button wiring) was instead verified with the
`tracker`/`models` modules stubbed out in-process (the same technique used
for the other affected modules), confirming `build_pipeline_components()`
and `process_frame()` are called correctly and the dashboard script itself
runs to completion without error in Streamlit's "bare mode."

## Future Roadmap

**All ten planned phases are complete.** What remains is explicitly out of
scope for this project (per Phase 10's own restrictions) or listed only as
a possible future improvement, not a committed next phase:

- A live demo run against a genuinely unseen stream (webcam-on-the-spot or
  a fresh uploaded clip), exercising the dashboard exactly as a real user
  would.
- Replacing/augmenting Phase 7's bounding-box posture heuristic with real
  pose estimation (see [Posture Detection](#posture-detection)'s "Future
  improvements" note).
- Replacing/augmenting Phase 8's whole-frame blur heuristic with a
  per-region or learned image-quality model (see
  [Blur Detection](#blur-detection)'s "Future improvements" note).
- Lights-on-but-empty detection (frame brightness analysis combined with
  Phase 9's EMPTY status) - the original task's bonus requirement, not
  covered by any phase.
- Explicitly out of scope per Phase 10's restrictions, not planned at all
  under this roadmap: a database, REST API, authentication, face
  recognition, cloud deployment, Docker packaging, and multi-camera
  support.

## How to Run

### Environment Setup

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

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running the App

1. By default, `config.py` has `SOURCE_TYPE = "webcam"`, which opens the
   default webcam (`WEBCAM_INDEX = 0`).
2. To run against a video file instead, edit `config.py` as shown in
   [Configuration Guide](#configuration-guide). The video is still
   streamed and processed frame-by-frame as it plays, not loaded and
   analyzed as a batch.
3. Run the app:
   ```bash
   python app.py
   ```
4. A window opens showing the live stream annotated with:
   - A bounding box around every tracked person, labeled with `ID: <n>` and
     `Person <confidence>`.
   - A "Moving" or "Stationary" label just below each tracked person, and
     a "Standing" or "Seated" label just below that.
   - A blue virtual line at `config.LINE_START`/`LINE_END`.
   - Brief `ENTRY : ID <n>` / `EXIT : ID <n>` notifications whenever a
     tracked person's centroid crosses that line.
   - An "Attendance" panel showing Current Inside / Entries / Exits /
     Unique Visitors, below the event notifications.
   - A "Frame Quality" panel showing Sharp/Blurry and the raw Laplacian
     variance, below the attendance panel.
   - An "Occupancy" panel showing Status (EMPTY/OCCUPIED/ACTIVE/IDLE),
     People, Moving, Standing, Seated, and Frame quality, below the Frame
     Quality panel.
   - `FPS: <value>` in the top-left corner.
   - `Persons: <count>` just below the FPS line.
   Press **Q** (as configured in `config.EXIT_KEY`) to exit cleanly.

If the webcam can't be opened, the video path is invalid, the YOLO model
fails to load, or the tracker fails to initialize, the app logs a clear
error (console + `logs/app.log`) and exits with a non-zero status instead
of crashing silently. A single frame that fails inference/tracking is
skipped (logged as a warning) rather than crashing the whole stream; the
same holds for entry/exit checks, motion checks, posture checks (including
zero-width/zero-height boxes), and individual malformed attendance events
on a per-track basis.

### Running the Dashboard

1. Install dependencies as above (`requirements.txt` includes `streamlit`
   as of Phase 10).
2. Launch it:
   ```bash
   streamlit run dashboard/streamlit_app.py
   ```
   Streamlit prints a local URL (typically `http://localhost:8501`) -
   open it in a browser.
3. In the sidebar, choose **Webcam** or **Upload Video** (the latter shows
   a file uploader). Press **Start**.
4. The main page shows the live annotated video (identical overlays to
   the OpenCV window) alongside FPS, person count, Attendance, Motion
   Summary, Posture Summary, Frame Quality, Occupancy Status, and the 10
   most recent Entry/Exit events.
5. **Stop** pauses processing and freezes the last frame/statistics in
   place; **Reset** clears everything back to a blank dashboard (and
   deletes any temp file created for an uploaded video).

See [Dashboard](#dashboard) for the full architecture, error handling, and
current limitations.

## Troubleshooting

- **`import torch` fails with `WinError 4551` / "Application Control
  policy has blocked this file"**: see
  [Current Windows Torch Policy Issue](#current-windows-torch-policy-issue)
  above. This is an OS-level policy blocking a DLL, not a Python or
  project issue — check with whoever manages the machine's Application
  Control / Smart App Control policy about allowing `torch\lib\shm.dll`.
- **Webcam window never opens / "Could not open video source"**: another
  application may already be using the camera, `WEBCAM_INDEX` may not
  match this machine's camera, or (on some systems) camera access needs to
  be granted to the terminal/IDE in the OS privacy settings.
- **"Could not open video source" with `SOURCE_TYPE = "video"`**: check
  `VIDEO_PATH` is correct and resolvable from the directory `python app.py`
  is run from (relative paths are resolved against the current working
  directory, not the project root).
- **First run is very slow / seems to hang**: on first use, Ultralytics
  downloads `yolov8n.pt` over the network into `models/` — this only
  happens once. If there is no internet access and the file is missing,
  model loading will fail with a clear logged error instead.
- **Low FPS / laggy video**: detection+tracking runs on CPU unless a
  CUDA-capable GPU is available (`config.DETECTION_DEVICE = "auto"`
  already picks CUDA automatically when present); on CPU-only machines,
  lower `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py` to trade resolution
  for speed.
- **Window shows video but no boxes/IDs appear**: check
  `DETECTION_CONFIDENCE_THRESHOLD` in `config.py` isn't set too high for
  the current lighting/camera angle, and confirm `PERSON_CLASS_ID = 0` is
  unchanged (only the "person" COCO class is detected/tracked).
- **App exits immediately with exit code 1**: check the console output or
  `logs/app.log` — every startup failure (bad source, model load, tracker
  init) logs a specific, human-readable reason before exiting.
- **No ENTRY/EXIT notifications ever appear**: confirm the virtual line
  (`config.LINE_START`/`LINE_END`, drawn in blue) actually crosses the
  path people take through the frame at the current camera angle/
  resolution — a line placed outside where people actually walk will never
  be crossed. Also confirm `FRAME_WIDTH`/`FRAME_HEIGHT` (if set) match the
  resolution `LINE_START`/`LINE_END` were chosen for.
- **The same person seems to generate an ENTRY and an EXIT in quick
  succession**: if they paused right at the line, minor detection/tracker
  jitter can flip the centroid across it more than once — see the
  "possible event flicker" note in
  [Known Limitations](#known-limitations).
- **`streamlit run dashboard/streamlit_app.py` fails to import `app`**:
  the dashboard adds the project root to `sys.path` automatically based
  on its own file location - make sure `dashboard/streamlit_app.py`
  hasn't been moved out of the `dashboard/` folder relative to `app.py`.
- **Dashboard's Start button does nothing / shows an error immediately**:
  check the message shown via `st.error(...)` - it will name the exact
  cause (no webcam, no file uploaded yet, unsupported/corrupted upload, or
  a YOLO model-load failure), per [Dashboard](#dashboard)'s error-handling
  section; this is the same underlying `RuntimeError` `python app.py`
  would also raise for the same condition.
- **Dashboard video looks noticeably choppier than the OpenCV window**:
  expected — see [Dashboard](#dashboard)'s "Current limitations": each
  displayed frame currently costs one full Streamlit script rerun, which
  has more overhead than `cv2.imshow()`'s direct display loop.
- **Uploaded video temp files pile up in the OS temp directory**: they are
  deleted on Reset or when a new file is uploaded, but not if the
  Streamlit process is killed first — safe to delete manually from the
  temp directory if this accumulates.
