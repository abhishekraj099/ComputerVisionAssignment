"""
Streamlit dashboard for the live vision pipeline.

Purpose:
    Provide a simple web UI (video source selection, Start/Stop/Reset,
    live annotated video, and current statistics) on top of the exact
    same pipeline app.py already runs via OpenCV - without reimplementing
    any detection, tracking, or analysis logic.

Responsibilities:
    - Let the user choose a video source: their webcam, or an uploaded
      video file.
    - Start/stop/reset a live processing loop driven by
      app.build_pipeline_components() and app.process_frame() - the same
      two functions run_pipeline() itself calls.
    - Display the annotated frame (the same OpenCV-drawn overlays already
      produced by process_frame()) inside the page.
    - Display current statistics (FPS, person count, attendance, motion
      summary, posture summary, frame quality, occupancy status, recent
      entry/exit events) by reading the PipelineFrameResult
      process_frame() already returns - never recomputing anything.

Scope of the current phase (Phase 10):
    A Streamlit front-end and final application integration only.

What this module intentionally does NOT handle:
    - No detection, tracking, motion, posture, line-crossing, attendance,
      blur, or occupancy logic of any kind - all of that lives in, and
      stays in, app.py and the subsystem modules it orchestrates. This
      module only calls app.build_pipeline_components() and
      app.process_frame() and renders their results.
    - No database, REST API, authentication, face recognition, cloud
      deployment, Docker, or multi-camera support - all explicitly out of
      scope for this phase.

Which future modules will consume this module's output:
    None - this is a UI leaf, not a library other modules import from.

How to launch:
    From the project root: `streamlit run dashboard/streamlit_app.py`
    (see the README's "Dashboard" section for full details and required
    packages).
"""

import os
import sys
import tempfile
from collections import deque

import cv2
import streamlit as st

# app.py and config.py live at the project root, one directory above this
# file (dashboard/). Streamlit runs this file directly (similar to
# `python dashboard/streamlit_app.py`), so the project root is not
# automatically on sys.path - add it so `import app` resolves. This is
# import-path plumbing only, not detection logic.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import app  # noqa: E402 - see sys.path adjustment above
import config  # noqa: E402

RECENT_EVENTS_MAX = 10
UPLOAD_FILE_TYPES = ["mp4", "avi", "mov", "mkv"]


def _init_session_state() -> None:
    """
    Populate st.session_state with this app's defaults, once per session.

    Side effects:
        Sets any of the keys below that are not already present. Safe to
        call on every script rerun - existing values are left untouched.
    """
    defaults = {
        "running": False,
        "components": None,
        "capture": None,
        "capture_is_webcam": False,
        "uploaded_video_path": None,
        "last_result": None,
        "last_frame_rgb": None,
        "recent_events": deque(maxlen=RECENT_EVENTS_MAX),
        "error_message": None,
        "info_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _release_capture() -> None:
    """Release the current capture (if any) and clear it from session state."""
    if st.session_state.capture is not None:
        st.session_state.capture.release()
        st.session_state.capture = None


def _cleanup_uploaded_file() -> None:
    """Delete the temp file backing an uploaded video (if any), ignoring errors."""
    path = st.session_state.uploaded_video_path
    if path:
        try:
            os.remove(path)
        except OSError:
            pass
        st.session_state.uploaded_video_path = None


def _open_capture(source_mode: str, uploaded_file) -> cv2.VideoCapture:
    """
    Open a cv2.VideoCapture for the user's chosen source.

    Args:
        source_mode: "Webcam" or "Upload Video".
        uploaded_file: The Streamlit UploadedFile from st.file_uploader(),
            or None if nothing has been uploaded yet.

    Returns:
        An opened cv2.VideoCapture.

    Raises:
        RuntimeError: If no file was uploaded (Upload Video mode with
            nothing chosen yet), or if the resulting source (webcam or
            uploaded file) fails to open - covers "no webcam", "invalid
            upload", and "unsupported video".
    """
    if source_mode == "Webcam":
        capture = cv2.VideoCapture(config.WEBCAM_INDEX)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Could not open the webcam. Check that a camera is connected "
                "and not already in use by another application."
            )
        st.session_state.capture_is_webcam = True
        return capture

    if uploaded_file is None:
        raise RuntimeError("Please upload a video file before pressing Start.")

    _cleanup_uploaded_file()
    suffix = os.path.splitext(uploaded_file.name)[1] or ".mp4"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(uploaded_file.getvalue())
    temp_file.close()
    st.session_state.uploaded_video_path = temp_file.name

    capture = cv2.VideoCapture(temp_file.name)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            "Could not open the uploaded file. It may be corrupted or in an "
            "unsupported video format."
        )
    st.session_state.capture_is_webcam = False
    return capture


def _handle_start(source_mode: str, uploaded_file) -> None:
    """
    Start (or resume) processing: open the source, build the pipeline.

    Args:
        source_mode: "Webcam" or "Upload Video".
        uploaded_file: The Streamlit UploadedFile, or None.

    Side effects:
        On success: sets st.session_state.capture/components/running=True
            and clears any prior error/info message.
        On failure (bad source or model load failure): sets
            st.session_state.error_message and leaves running=False -
            never raises out of this function, per the "never crash"
            requirement.
    """
    st.session_state.error_message = None
    st.session_state.info_message = None

    if st.session_state.running:
        return  # already running; Start is a no-op while active

    try:
        if st.session_state.capture is None:
            st.session_state.capture = _open_capture(source_mode, uploaded_file)
        if st.session_state.components is None:
            capture = st.session_state.capture
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            st.session_state.components = app.build_pipeline_components(frame_width, frame_height)
        st.session_state.running = True
    except RuntimeError as exc:
        st.session_state.error_message = str(exc)
        _release_capture()
    except Exception as exc:  # noqa: BLE001 - never let the dashboard crash
        st.session_state.error_message = f"Unexpected error starting the pipeline: {exc}"
        _release_capture()


def _handle_stop() -> None:
    """Pause processing, keeping the last frame/statistics visible."""
    st.session_state.running = False


def _handle_reset() -> None:
    """
    Fully clear all pipeline state back to a blank, unstarted dashboard.

    Side effects:
        Releases the capture, drops the built pipeline components (a
        fresh one is built on the next Start), clears the last frame/
        statistics/recent events/messages, and removes any temp file
        backing an uploaded video.
    """
    st.session_state.running = False
    _release_capture()
    _cleanup_uploaded_file()
    st.session_state.components = None
    st.session_state.last_result = None
    st.session_state.last_frame_rgb = None
    st.session_state.recent_events = deque(maxlen=RECENT_EVENTS_MAX)
    st.session_state.error_message = None
    st.session_state.info_message = None


def _advance_one_frame() -> None:
    """
    Read and process exactly one frame from the current capture, if any.

    Side effects:
        On a successful read: calls app.process_frame() (annotating the
            frame in place), stores the PipelineFrameResult and the
            RGB-converted frame for display, and appends any new
            crossing events to the recent-events history.
        On end of stream: sets running=False and either error_message
            (an unexpected webcam disconnect) or info_message (a normal
            uploaded-video end), per the "end of stream" requirement.
        On a pipeline RuntimeError (tracker init failure): sets
            running=False and error_message, without crashing.
    """
    capture = st.session_state.capture
    components = st.session_state.components
    if capture is None or components is None:
        return

    frame_read_ok, frame = capture.read()
    if not frame_read_ok or frame is None:
        st.session_state.running = False
        if st.session_state.capture_is_webcam:
            st.session_state.error_message = "Webcam feed ended unexpectedly (disconnected?)."
        else:
            st.session_state.info_message = "Reached the end of the video."
        return

    try:
        result = app.process_frame(frame, components)
    except RuntimeError as exc:
        st.session_state.running = False
        st.session_state.error_message = f"Pipeline error: {exc}"
        return

    st.session_state.last_result = result
    st.session_state.recent_events.extend(result.crossing_events)
    st.session_state.last_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _render_sidebar() -> tuple:
    """
    Draw the sidebar (source selection + Start/Stop/Reset) and act on
    whichever button was pressed this run.

    Returns:
        (source_mode, uploaded_file) - the current UI selections, so the
        main body can react to them too (e.g. show the current mode).
    """
    st.sidebar.header("Video Source")
    source_mode = st.sidebar.radio("Choose a source", ["Webcam", "Upload Video"])

    uploaded_file = None
    if source_mode == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Video file", type=UPLOAD_FILE_TYPES)

    st.sidebar.header("Controls")
    start_col, stop_col, reset_col = st.sidebar.columns(3)
    start_clicked = start_col.button("Start", width="stretch")
    stop_clicked = stop_col.button("Stop", width="stretch")
    reset_clicked = reset_col.button("Reset", width="stretch")

    if start_clicked:
        _handle_start(source_mode, uploaded_file)
    if stop_clicked:
        _handle_stop()
    if reset_clicked:
        _handle_reset()

    st.sidebar.caption(f"Status: {'Running' if st.session_state.running else 'Stopped'}")

    return source_mode, uploaded_file


def _render_statistics(stats_container) -> None:
    """
    Render every statistics panel from the last PipelineFrameResult.

    Args:
        stats_container: A Streamlit container (e.g. a column) to render
            into.

    Side effects:
        Writes Streamlit widgets. Reads st.session_state.last_result only
        - never recomputes anything; if no result exists yet (nothing
        processed since the last Reset), shows placeholder zero values.

    Note on the two attendance sections:
        "Live Attendance" (Current Present) and "Visitor Analytics"
        (Current Inside / Entries / Exits / Unique Visitors) deliberately
        measure different things and are expected to differ. Live
        Attendance is simply how many tracked people are visible in this
        frame (`len(result.tracks)`); Visitor Analytics is the unchanged
        line-crossing attendance from AttendanceManager, in which someone
        already seated before the stream started never crosses the virtual
        line and so is correctly not counted. Neither section recomputes
        anything - both read values process_frame() already produced.
    """
    result = st.session_state.last_result

    with stats_container:
        st.subheader("Statistics")

        fps = result.fps if result else 0.0
        person_count = len(result.tracks) if result else 0
        col_a, col_b = st.columns(2)
        col_a.metric("FPS", f"{fps:.1f}")
        col_b.metric("Persons", person_count)

        # "Live Attendance" is a presentation-only view of how many people
        # are visible right now. It reads the exact same
        # `len(result.tracks)` already shown as the "Persons" metric above -
        # the tracked-person list process_frame() has already produced -
        # so it runs no detection, builds no tracker, and computes nothing
        # new. It deliberately does NOT come from AttendanceStatistics:
        # "Visitor Analytics" below remains the untouched line-crossing
        # attendance, and these two sections answer different questions
        # ("who is on screen now" vs "who has crossed the line").
        st.markdown("**Live Attendance**")
        if result:
            st.write(f"Current Present: {person_count}")
        else:
            st.write("No data yet.")

        st.markdown("**Visitor Analytics**")
        if result:
            stats = result.attendance_stats
            st.write(
                f"Current Inside: {stats.current_people} | "
                f"Entries: {stats.total_entries} | "
                f"Exits: {stats.total_exits} | "
                f"Unique Visitors: {stats.unique_visitors}"
            )
        else:
            st.write("No data yet.")

        st.markdown("**Motion Summary**")
        if result:
            moving = sum(1 for state in result.motion_states if state.is_moving)
            st.write(f"Moving: {moving} | Stationary: {len(result.motion_states) - moving}")
        else:
            st.write("No data yet.")

        st.markdown("**Posture Summary**")
        if result:
            standing = sum(1 for state in result.posture_states if state.is_standing)
            st.write(f"Standing: {standing} | Seated: {len(result.posture_states) - standing}")
        else:
            st.write("No data yet.")

        st.markdown("**Frame Quality**")
        if result:
            blur = result.blur_state
            st.write(f"{'Blurry' if blur.is_blurry else 'Sharp'} (variance: {blur.laplacian_variance:.1f})")
        else:
            st.write("No data yet.")

        st.markdown("**Occupancy Status**")
        if result:
            occupancy = result.occupancy_state
            st.write(
                f"Status: {occupancy.occupancy_status.value} | "
                f"People: {occupancy.people_inside} | "
                f"Moving: {occupancy.moving_people} | "
                f"Standing: {occupancy.standing_people} | "
                f"Seated: {occupancy.seated_people}"
            )
        else:
            st.write("No data yet.")

        st.markdown("**Recent Entry / Exit Events**")
        if st.session_state.recent_events:
            for event in reversed(st.session_state.recent_events):
                st.write(f"{event.event_type.value} : ID {event.track_id}")
        else:
            st.write("No events yet.")


def main() -> None:
    """
    Streamlit entry point. Executed top-to-bottom on every rerun (every
    user interaction, and once more per processed frame while running -
    see _advance_one_frame() and the trailing st.rerun() below).
    """
    st.set_page_config(page_title="iCloudEMS Smart Campus Dashboard", layout="wide")
    _init_session_state()

    st.title("iCloudEMS Smart Campus - Live Vision Dashboard")

    _render_sidebar()

    if st.session_state.running:
        _advance_one_frame()

    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    elif st.session_state.info_message:
        st.info(st.session_state.info_message)

    video_col, stats_col = st.columns([2, 1])

    with video_col:
        st.subheader("Live Video")
        if st.session_state.last_frame_rgb is not None:
            st.image(st.session_state.last_frame_rgb, channels="RGB", width="stretch")
        else:
            st.info("Press Start to begin.")

    _render_statistics(stats_col)

    if st.session_state.running:
        st.rerun()


main()
