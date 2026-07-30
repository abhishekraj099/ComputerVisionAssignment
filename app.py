"""
Live video pipeline - application entry point and composition root.

Purpose:
    Wire together video capture, person detection/tracking, and on-screen
    display into a single real-time loop, and provide the `python app.py`
    entry point for the whole project.

Responsibilities:
    Phase 1:
        - Open a webcam or a video file, selected via config.SOURCE_TYPE.
        - Read frames continuously, in real time (no batch loading).
        - Display the live stream in a window.
        - Overlay the current FPS in the top-left corner.
        - Exit cleanly when the configured exit key is pressed.
        - Handle source/read errors gracefully, with useful logs.
    Phase 2:
        - Run real-time YOLOv8 person detection on each frame (via
          models.person_detector.PersonDetector).
        - Display the total number of persons currently detected.
    Phase 3:
        - Assign a persistent track ID to each detected person (via
          tracker.byte_tracker.PersonTracker, ByteTrack-based).
        - Draw a bounding box + track ID + confidence label per tracked
          person (via tracker.byte_tracker.draw_tracks).
    Phase 4:
        - Detect ENTRY/EXIT events when a tracked person's centroid
          crosses the configured virtual line (via
          events.line_crossing.LineCrossingDetector).
        - Draw the virtual line and briefly display recent crossing
          events (via events.line_crossing.draw_line/draw_events).
    Phase 5:
        - Feed each frame's crossing events into an AttendanceManager to
          maintain running attendance state (current people inside, total
          entries/exits, unique visitors).
        - Draw an "Attendance" summary panel (via
          attendance.attendance_manager.draw_attendance_panel).
    Phase 6:
        - Classify each tracked person as MOVING or STATIONARY from their
          centroid trajectory alone (via motion.motion_detector.MotionDetector).
        - Draw a "Moving"/"Stationary" label near each tracked person
          (via motion.motion_detector.draw_motion_states).
    Phase 7:
        - Classify each tracked person as STANDING or SEATED from their
          bounding-box aspect ratio alone (via
          posture.posture_detector.PostureDetector).
        - Draw a "Standing"/"Seated" label near each tracked person (via
          posture.posture_detector.draw_posture_states).
    Phase 8:
        - Classify the frame itself (not any individual person) as SHARP
          or BLURRY, via Variance of Laplacian (via
          quality.blur_detector.BlurDetector).
        - Draw a "Frame Quality" panel (via
          quality.blur_detector.draw_blur_panel).
    Phase 9:
        - Derive a room-level occupancy state (EMPTY/OCCUPIED/ACTIVE/IDLE)
          purely from the AttendanceStatistics/MotionState/PostureState/
          BlurState already produced above (via
          occupancy.occupancy_detector.OccupancyDetector) - no frame or
          tracking data is passed to it directly.
        - Draw an "Occupancy" summary panel (via
          occupancy.occupancy_detector.draw_occupancy_panel).
    Phase 10:
        - Expose `build_pipeline_components()` and `process_frame()` (see
          below) as the single, reusable per-frame processing entry point,
          so dashboard.streamlit_app can drive the exact same pipeline
          without duplicating any detection/tracking/analysis logic.
        - `run_pipeline()` itself is otherwise unchanged in behavior - it
          now simply calls these two functions instead of inlining their
          bodies directly in its loop.

Scope of the current phase (Phase 10):
    This module is the orchestrator: it owns the frame loop and delegates
    all detection/tracking/event-detection/attendance/motion/posture/
    quality/occupancy work to models.person_detector, tracker.byte_tracker,
    events.line_crossing, attendance.attendance_manager,
    motion.motion_detector, posture.posture_detector, quality.blur_detector,
    and occupancy.occupancy_detector. It contains no detection, tracking,
    event, attendance, motion, posture, blur-analysis, or occupancy logic
    itself. As of Phase 10, its per-frame logic is also reused by
    dashboard/streamlit_app.py via `build_pipeline_components()` and
    `process_frame()`, so this module is no longer read only by
    `python app.py` - see those two functions' docstrings.

What this module intentionally does NOT handle:
    No Streamlit widget/session-state/UI code belongs in this file - that
    is entirely owned by dashboard/streamlit_app.py, which imports and
    calls this module's functions rather than reimplementing them.

Which future modules will consume this module's output:
    dashboard.streamlit_app calls build_pipeline_components() once and
    process_frame() once per frame, exactly as run_pipeline() does here,
    to get the identical annotated frame and PipelineFrameResult without
    running its own detection logic.
"""

import sys
from dataclasses import dataclass
from typing import List

import cv2

import config
from attendance.attendance_manager import AttendanceManager, AttendanceStatistics, draw_attendance_panel
from events.line_crossing import CrossingEvent, EventNotificationBoard, LineCrossingDetector, draw_events, draw_line
from motion.motion_detector import MotionDetector, MotionState, draw_motion_states
from occupancy.occupancy_detector import OccupancyDetector, OccupancyState, draw_occupancy_panel
from posture.posture_detector import PostureDetector, PostureState, draw_posture_states
from quality.blur_detector import BlurDetector, BlurState, draw_blur_panel
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)

# The real detector/tracker (models.person_detector, tracker.byte_tracker) both
# import torch (via ultralytics). On a machine where an OS-level Application
# Control policy blocks torch's DLLs from loading, that import fails before
# this module can even be imported - not because of anything wrong with this
# project's code. FALLBACK_MODE lets the pipeline keep running (with a
# torch-free, OpenCV-only background-subtraction tracker instead of YOLOv8 +
# ByteTrack) rather than refusing to start at all. On a machine where torch
# imports successfully, this except branch never runs and behavior is
# unchanged from before.
try:
    from models.person_detector import PersonDetector
    from tracker.byte_tracker import PersonTracker, TrackedPerson, draw_tracks

    FALLBACK_MODE = False
except (ImportError, OSError) as exc:
    from tracker.fallback_tracker import FallbackPersonTracker as PersonTracker
    from tracker.fallback_tracker import TrackedPerson, draw_tracks

    PersonDetector = None
    FALLBACK_MODE = True
    logger.warning(
        "Could not import the YOLOv8/ByteTrack detector-tracker (%s). "
        "Falling back to a torch-free OpenCV background-subtraction tracker "
        "with reduced detection accuracy. See README's Troubleshooting "
        "section for how to restore full YOLOv8 detection.",
        exc,
    )


def open_video_source() -> cv2.VideoCapture:
    """
    Open a video capture based on config.SOURCE_TYPE.

    Returns:
        An opened cv2.VideoCapture instance. Ownership transfers to the
        caller, who is responsible for calling `.release()` on it (see
        run_pipeline()'s `finally` block).

    Raises:
        RuntimeError: If SOURCE_TYPE is neither "webcam" nor "video", or if
            the configured source (camera index / file path) cannot be
            opened.

    Side effects:
        Acquires an OS-level hardware/file handle (webcam or video file)
        and logs progress at INFO level.

    Performance considerations:
        One-time setup cost per run; not called per frame.
    """
    if config.SOURCE_TYPE == "webcam":
        logger.info("Opening webcam at index %s ...", config.WEBCAM_INDEX)
        capture = cv2.VideoCapture(config.WEBCAM_INDEX)
    elif config.SOURCE_TYPE == "video":
        logger.info("Opening video file: %s ...", config.VIDEO_PATH)
        capture = cv2.VideoCapture(config.VIDEO_PATH)
    else:
        raise RuntimeError(
            f"Invalid config.SOURCE_TYPE '{config.SOURCE_TYPE}'. Expected 'webcam' or 'video'."
        )

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open video source (SOURCE_TYPE='{config.SOURCE_TYPE}'). "
            "Check that the webcam is connected or the video path is correct."
        )

    if config.FRAME_WIDTH:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    if config.FRAME_HEIGHT:
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    logger.info("Video source opened successfully.")
    return capture


def draw_status_text(frame, text: str, position):
    """
    Draw a single line of status text (FPS, person count, ...) onto the
    frame, using the shared overlay style from config.

    Args:
        frame: The BGR image (numpy array) to annotate.
        text: The line of text to draw.
        position: (x, y) pixel coordinates for the text.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via a single cv2.putText call.
    """
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        config.OVERLAY_FONT_SCALE,
        config.OVERLAY_COLOR,
        config.OVERLAY_FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return frame


def draw_fps(frame, fps: float):
    """
    Draw the current FPS value onto the top-left corner of a frame.

    Args:
        frame: The BGR image (numpy array) to annotate, mutated in place.
        fps: Current smoothed FPS value (see utils.fps_counter.FPSCounter).

    Returns:
        The same frame object, per draw_status_text().
    """
    return draw_status_text(frame, f"FPS: {fps:.1f}", config.FPS_TEXT_POSITION)


def draw_person_count(frame, person_count: int):
    """
    Draw the total number of currently tracked persons onto the frame.

    Args:
        frame: The BGR image (numpy array) to annotate, mutated in place.
        person_count: Number of people currently tracked in this frame
            (i.e. `len(tracks)` from PersonTracker.track()).

    Returns:
        The same frame object, per draw_status_text().
    """
    return draw_status_text(frame, f"Persons: {person_count}", config.PERSON_COUNT_TEXT_POSITION)


def load_person_detector():
    """
    Build the PersonDetector from config.

    Returns:
        A ready-to-use PersonDetector with its YOLOv8 model already loaded,
        or None in FALLBACK_MODE (see the import guard at the top of this
        module) - there is no detector to build in that case, since
        load_person_tracker() below builds a self-contained
        FallbackPersonTracker instead.

    Raises:
        RuntimeError: If the YOLOv8 model fails to load (propagated
            unchanged from PersonDetector.__init__).

    Performance considerations:
        Triggers the one-time, potentially slow model load (and possible
        weights download); call once per run, not per frame.
    """
    if FALLBACK_MODE:
        return None

    return PersonDetector(
        model_path=config.DETECTION_MODEL_PATH,
        confidence_threshold=config.DETECTION_CONFIDENCE_THRESHOLD,
        person_class_id=config.PERSON_CLASS_ID,
        device=config.DETECTION_DEVICE,
    )


def load_person_tracker(detector):
    """
    Build the PersonTracker from config, reusing the detector's already
    loaded YOLO model so the weights are only loaded once.

    Args:
        detector: A constructed PersonDetector (see load_person_detector()),
            whose loaded model and resolved device are reused here - or
            None in FALLBACK_MODE, in which case it is ignored and a
            self-contained FallbackPersonTracker is built instead.

    Returns:
        A ready-to-use tracker exposing the same `.track(frame)` interface:
        a real PersonTracker sharing `detector`'s YOLO model, or a
        FallbackPersonTracker in FALLBACK_MODE.

    Raises:
        Does not raise itself; construction is pure bookkeeping (see
        PersonTracker.__init__ / FallbackPersonTracker.__init__). Tracker
        initialization failures surface later, on the first call to
        `.track()`.
    """
    if FALLBACK_MODE:
        return PersonTracker(
            bg_history=config.FALLBACK_BG_HISTORY,
            bg_var_threshold=config.FALLBACK_BG_VAR_THRESHOLD,
            min_contour_area=config.FALLBACK_MIN_CONTOUR_AREA,
            max_match_distance=config.FALLBACK_MAX_MATCH_DISTANCE,
            stale_track_timeout=config.FALLBACK_STALE_TRACK_TIMEOUT,
        )

    return PersonTracker(
        model=detector.model,
        device=detector.device,
        person_class_id=config.PERSON_CLASS_ID,
        confidence_threshold=config.DETECTION_CONFIDENCE_THRESHOLD,
        tracker_config=config.TRACKER_CONFIG,
    )


def load_line_crossing_detector() -> LineCrossingDetector:
    """
    Build the LineCrossingDetector from config.

    Returns:
        A ready-to-use LineCrossingDetector for the configured virtual
        line (config.LINE_START / config.LINE_END), with its stale-track
        timeout and event cooldown set from config.

    Raises:
        Does not raise; construction is pure bookkeeping (see
        LineCrossingDetector.__init__).
    """
    return LineCrossingDetector(
        line_start=config.LINE_START,
        line_end=config.LINE_END,
        stale_track_timeout=config.LINE_CROSSING_STALE_TRACK_TIMEOUT,
        event_cooldown=config.LINE_CROSSING_EVENT_COOLDOWN,
    )


def load_attendance_manager() -> AttendanceManager:
    """
    Build the AttendanceManager.

    Returns:
        A fresh AttendanceManager with no state (nobody inside, zero
        entries/exits/unique visitors).

    Raises:
        Does not raise; construction is pure bookkeeping (see
        AttendanceManager.__init__).
    """
    return AttendanceManager()


def load_motion_detector() -> MotionDetector:
    """
    Build the MotionDetector from config.

    Returns:
        A ready-to-use MotionDetector with its distance threshold, history
        size, and stale-track timeout set from config.

    Raises:
        Does not raise; construction is pure bookkeeping (see
        MotionDetector.__init__).
    """
    return MotionDetector(
        distance_threshold=config.MOTION_DISTANCE_THRESHOLD,
        history_size=config.MOTION_HISTORY_SIZE,
        stale_track_timeout=config.MOTION_STALE_TRACK_TIMEOUT,
    )


def load_posture_detector() -> PostureDetector:
    """
    Build the PostureDetector from config.

    Returns:
        A ready-to-use PostureDetector with its aspect-ratio threshold,
        history size, and stale-track timeout set from config.

    Raises:
        Does not raise; construction is pure bookkeeping (see
        PostureDetector.__init__).
    """
    return PostureDetector(
        aspect_ratio_threshold=config.POSTURE_ASPECT_RATIO_THRESHOLD,
        history_size=config.POSTURE_HISTORY_SIZE,
        stale_track_timeout=config.POSTURE_STALE_TRACK_TIMEOUT,
    )


def load_blur_detector() -> BlurDetector:
    """
    Build the BlurDetector from config.

    Returns:
        A ready-to-use BlurDetector with its variance threshold set from
        config.

    Raises:
        Does not raise; construction is pure bookkeeping (see
        BlurDetector.__init__).
    """
    return BlurDetector(threshold=config.BLUR_THRESHOLD)


def load_occupancy_detector() -> OccupancyDetector:
    """
    Build the OccupancyDetector.

    Returns:
        A fresh OccupancyDetector with no prior decision.

    Raises:
        Does not raise; construction is pure bookkeeping (see
        OccupancyDetector.__init__).
    """
    return OccupancyDetector()


@dataclass
class PipelineComponents:
    """
    A bundle of every stateful pipeline object, constructed once per run.

    Purpose:
        Give run_pipeline() and dashboard.streamlit_app a single object to
        construct once (via build_pipeline_components()) and pass into
        process_frame() every frame, instead of each caller having to know
        about, and construct, nine separate objects individually.

    Fields:
        detector: A loaded PersonDetector (see load_person_detector()).
        tracker: A PersonTracker sharing `detector`'s model (see
            load_person_tracker()).
        line_crossing_detector: See load_line_crossing_detector().
        event_board: An EventNotificationBoard for transient on-screen
            ENTRY/EXIT display.
        attendance_manager: See load_attendance_manager().
        motion_detector: See load_motion_detector().
        posture_detector: See load_posture_detector().
        blur_detector: See load_blur_detector().
        occupancy_detector: See load_occupancy_detector().
        fps_counter: An FPSCounter for the stream this bundle belongs to.

    Note:
        This is a plain mutable dataclass (not frozen), since several of
        its fields are themselves stateful objects mutated by
        process_frame() every call (e.g. `fps_counter.tick()`); the bundle
        itself is never reassigned once built.
    """

    detector: PersonDetector
    tracker: PersonTracker
    line_crossing_detector: LineCrossingDetector
    event_board: EventNotificationBoard
    attendance_manager: AttendanceManager
    motion_detector: MotionDetector
    posture_detector: PostureDetector
    blur_detector: BlurDetector
    occupancy_detector: OccupancyDetector
    fps_counter: FPSCounter


@dataclass(frozen=True)
class PipelineFrameResult:
    """
    Every per-frame output produced by process_frame(), bundled together.

    Purpose:
        Let a caller (run_pipeline()'s display loop, or a future UI such
        as dashboard.streamlit_app) read this frame's full set of
        statistics without recomputing anything - every field here is a
        direct pass-through of a value some other subsystem already
        produced.

    Fields:
        tracks: This frame's List[TrackedPerson], from PersonTracker.track().
        motion_states: This frame's List[MotionState], from
            MotionDetector.update().
        posture_states: This frame's List[PostureState], from
            PostureDetector.update().
        crossing_events: This frame's List[CrossingEvent] (usually empty),
            from LineCrossingDetector.update().
        attendance_stats: This frame's AttendanceStatistics snapshot, from
            AttendanceManager.update().
        blur_state: This frame's BlurState, from BlurDetector.analyze().
        occupancy_state: This frame's OccupancyState, from
            OccupancyDetector.update().
        fps: The current smoothed FPS value, from FPSCounter.tick().
    """

    tracks: List[TrackedPerson]
    motion_states: List[MotionState]
    posture_states: List[PostureState]
    crossing_events: List[CrossingEvent]
    attendance_stats: AttendanceStatistics
    blur_state: BlurState
    occupancy_state: OccupancyState
    fps: float


def build_pipeline_components() -> PipelineComponents:
    """
    Construct every stateful pipeline object for one run, bundled together.

    This is the single place both run_pipeline() and
    dashboard.streamlit_app build the pipeline from - it does nothing
    beyond calling the existing load_*() functions above and bundling
    their results; no construction logic is duplicated anywhere else.

    Returns:
        A ready-to-use PipelineComponents.

    Raises:
        RuntimeError: Propagated from load_person_detector() (model load
            failure). Tracker/line-crossing/attendance/motion/posture/
            blur/occupancy construction does not raise (see each
            load_*() function's docstring).

    Performance considerations:
        Triggers the one-time, potentially slow YOLOv8 model load (and
        possible weights download) via load_person_detector(); call once
        per run, not per frame.
    """
    detector = load_person_detector()
    return PipelineComponents(
        detector=detector,
        tracker=load_person_tracker(detector),
        line_crossing_detector=load_line_crossing_detector(),
        event_board=EventNotificationBoard(config.EVENT_DISPLAY_DURATION),
        attendance_manager=load_attendance_manager(),
        motion_detector=load_motion_detector(),
        posture_detector=load_posture_detector(),
        blur_detector=load_blur_detector(),
        occupancy_detector=load_occupancy_detector(),
        fps_counter=FPSCounter(),
    )


def process_frame(frame, components: PipelineComponents) -> PipelineFrameResult:
    """
    Run one frame through the entire detection/tracking/analysis pipeline
    and annotate it in place.

    This is the single per-frame processing entry point shared by
    run_pipeline() (displayed via cv2.imshow) and
    dashboard.streamlit_app (displayed via st.image) - neither caller
    contains, or needs to contain, any detection/tracking/analysis logic
    of its own; both simply call this function once per frame.

    Args:
        frame: BGR image (numpy array) as read from OpenCV, for this
            frame. Mutated in place with every overlay this project draws
            (tracks, motion/posture labels, virtual line, event
            notifications, attendance/blur/occupancy panels, FPS, person
            count).
        components: A PipelineComponents built via
            build_pipeline_components(), reused call-to-call so every
            subsystem's internal state (track history, attendance totals,
            ByteTrack's tracker state, etc.) persists correctly across
            frames.

    Returns:
        A PipelineFrameResult bundling every statistic produced this
        frame, for a caller that wants to display them without
        recomputing anything.

    Raises:
        RuntimeError: If tracking fails on the very first frame ever
            processed by `components.tracker` (a tracker initialization
            failure - see PersonTracker.track()). All other per-subsystem
            failures are caught and logged internally by their own
            update()/analyze() methods and do not raise here.

    Side effects:
        Mutates `frame` in place (see Args). Logs one line per accepted
        ENTRY/EXIT event, plus whatever each subsystem's own update() call
        logs (state changes, skipped-frame notices, etc.).

    Performance considerations:
        Dominated by `components.tracker.track(frame)` (one YOLO forward
        pass + ByteTrack association); every other subsystem call here is
        cheap arithmetic/drawing by comparison - see each subsystem's own
        module docstring for its individual cost.
    """
    # Blur analysis must run on the raw, unannotated frame - any overlay
    # drawing below would otherwise artificially inflate its measured
    # sharpness (high-contrast text/box edges).
    blur_state = components.blur_detector.analyze(frame)

    tracks = components.tracker.track(frame)
    draw_tracks(
        frame,
        tracks,
        box_color=config.BOX_COLOR,
        box_thickness=config.BOX_THICKNESS,
        label_font_scale=config.DETECTION_LABEL_FONT_SCALE,
        label_font_thickness=config.DETECTION_LABEL_FONT_THICKNESS,
        line_spacing=config.TRACK_LABEL_LINE_SPACING,
    )

    motion_states = components.motion_detector.update(tracks)
    draw_motion_states(
        frame,
        motion_states,
        moving_color=config.MOTION_MOVING_COLOR,
        stationary_color=config.MOTION_STATIONARY_COLOR,
        font_scale=config.DETECTION_LABEL_FONT_SCALE,
        thickness=config.DETECTION_LABEL_FONT_THICKNESS,
        label_offset_y=config.MOTION_LABEL_OFFSET_Y,
    )

    posture_states = components.posture_detector.update(tracks)
    draw_posture_states(
        frame,
        posture_states,
        standing_color=config.POSTURE_STANDING_COLOR,
        seated_color=config.POSTURE_SEATED_COLOR,
        font_scale=config.DETECTION_LABEL_FONT_SCALE,
        thickness=config.DETECTION_LABEL_FONT_THICKNESS,
        label_offset_y=config.POSTURE_LABEL_OFFSET_Y,
    )

    crossing_events = components.line_crossing_detector.update(tracks)
    for event in crossing_events:
        logger.info("%s : ID %s", event.event_type.value, event.track_id)
    components.event_board.add_events(crossing_events)

    draw_line(frame, config.LINE_START, config.LINE_END, config.LINE_COLOR, config.LINE_THICKNESS)
    draw_events(
        frame,
        components.event_board.get_visible_events(),
        position=config.EVENT_NOTIFICATION_POSITION,
        line_spacing=config.EVENT_NOTIFICATION_LINE_SPACING,
        font_scale=config.OVERLAY_FONT_SCALE,
        color=config.OVERLAY_COLOR,
        thickness=config.OVERLAY_FONT_THICKNESS,
        max_lines=config.EVENT_NOTIFICATION_MAX_LINES,
    )

    attendance_stats = components.attendance_manager.update(crossing_events)
    draw_attendance_panel(
        frame,
        attendance_stats,
        position=config.ATTENDANCE_PANEL_POSITION,
        line_spacing=config.ATTENDANCE_PANEL_LINE_SPACING,
        font_scale=config.OVERLAY_FONT_SCALE,
        color=config.OVERLAY_COLOR,
        thickness=config.OVERLAY_FONT_THICKNESS,
    )

    draw_blur_panel(
        frame,
        blur_state,
        position=config.BLUR_PANEL_POSITION,
        line_spacing=config.BLUR_PANEL_LINE_SPACING,
        font_scale=config.OVERLAY_FONT_SCALE,
        sharp_color=config.BLUR_SHARP_COLOR,
        blurry_color=config.BLUR_BLURRY_COLOR,
        thickness=config.OVERLAY_FONT_THICKNESS,
    )

    occupancy_state = components.occupancy_detector.update(
        attendance_stats, motion_states, posture_states, blur_state
    )
    draw_occupancy_panel(
        frame,
        occupancy_state,
        position=config.OCCUPANCY_PANEL_POSITION,
        line_spacing=config.OCCUPANCY_PANEL_LINE_SPACING,
        font_scale=config.OVERLAY_FONT_SCALE,
        color=config.OVERLAY_COLOR,
        thickness=config.OVERLAY_FONT_THICKNESS,
    )

    fps = components.fps_counter.tick()
    draw_fps(frame, fps)
    draw_person_count(frame, len(tracks))

    return PipelineFrameResult(
        tracks=tracks,
        motion_states=motion_states,
        posture_states=posture_states,
        crossing_events=crossing_events,
        attendance_stats=attendance_stats,
        blur_state=blur_state,
        occupancy_state=occupancy_state,
        fps=fps,
    )


def run_pipeline() -> None:
    """
    Main capture-display loop.

    Opens the configured video source, builds the pipeline via
    build_pipeline_components(), and calls process_frame() once per frame
    to run the entire detection/tracking/analysis pipeline and annotate
    the frame in place, displaying the result via cv2.imshow() and
    exiting cleanly on the configured key or when the source is
    exhausted/disconnected. Contains no detection/tracking/analysis logic
    itself - see process_frame()'s docstring for what actually happens to
    each frame.

    Returns:
        None. Runs until the stream ends, the exit key is pressed, or an
        unrecoverable error is raised (see Raises below); errors and exit
        codes are handled by the caller, main().

    Raises:
        RuntimeError: Propagated from open_video_source() (bad source),
            build_pipeline_components() (model load failure), or
            process_frame() (tracker init failure on the very first
            frame).

    Side effects:
        Opens an OS-level video handle and an OpenCV display window for
        the duration of the call; both are released/closed in the
        `finally` block regardless of how the loop exits.

    Performance considerations:
        This is the application's hot loop - it runs once per processed
        frame for the lifetime of the stream. Per-frame cost is dominated
        by `tracker.track(frame)` (one YOLO forward pass + ByteTrack
        association); the overlay drawing and `cv2.imshow`/`cv2.waitKey`
        calls are comparatively cheap.
    """
    capture = open_video_source()
    components = build_pipeline_components()

    try:
        while True:
            frame_read_ok, frame = capture.read()

            if not frame_read_ok or frame is None:
                logger.warning("Failed to read frame from source. Ending stream.")
                break

            process_frame(frame, components)

            cv2.imshow(config.WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.EXIT_KEY):
                logger.info("Exit key '%s' pressed. Shutting down.", config.EXIT_KEY)
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        logger.info("Video source released and windows closed.")


def main() -> int:
    """
    Application entry point.

    Runs run_pipeline() and converts any failure into a process exit code,
    logging the outcome either way.

    Returns:
        0 if the pipeline ran and exited cleanly (stream ended or exit key
        pressed); 1 if a RuntimeError occurred (source/model/tracker
        failure) or any other unexpected exception was raised.

    Side effects:
        Logs the pipeline's start, outcome, and any error at the
        appropriate level (INFO for normal progress, ERROR/EXCEPTION for
        failures).
    """
    logger.info(
        "Starting live video pipeline with person detection, tracking, entry/exit detection, attendance, "
        "motion classification, posture classification, blur detection, and occupancy detection..."
    )
    try:
        run_pipeline()
    except RuntimeError as exc:
        logger.error("Pipeline failed to start: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level guard for unexpected failures
        logger.exception("Unexpected error while running the pipeline: %s", exc)
        return 1

    logger.info("Pipeline exited cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
