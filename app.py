"""
Phase 1 - Live video pipeline foundation.

Responsibilities (and ONLY these, per Phase 1 scope):
    - Open a webcam or a video file, selected via config.SOURCE_TYPE.
    - Read frames continuously, in real time (no batch loading).
    - Display the live stream in a window.
    - Overlay the current FPS in the top-left corner.
    - Exit cleanly when the configured exit key is pressed.
    - Handle source/read errors gracefully, with useful logs.

No detection, tracking, or analytics logic belongs in this file yet.
"""

import sys

import cv2

import config
from utils.fps_counter import FPSCounter
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


def open_video_source() -> cv2.VideoCapture:
    """
    Open a video capture based on config.SOURCE_TYPE.

    Returns:
        An opened cv2.VideoCapture instance.

    Raises:
        RuntimeError: If the configured source cannot be opened.
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


def draw_fps(frame, fps: float):
    """
    Draw the current FPS value onto the top-left corner of a frame.

    Args:
        frame: The BGR image (numpy array) to annotate.
        fps: Current FPS value to display.

    Returns:
        The same frame, annotated in place.
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return frame


def run_pipeline() -> None:
    """
    Main capture-display loop.

    Reads frames continuously from the configured source, overlays FPS,
    displays them in a window, and exits cleanly on the configured key
    or when the source is exhausted/disconnected.
    """
    capture = open_video_source()
    fps_counter = FPSCounter()

    try:
        while True:
            frame_read_ok, frame = capture.read()

            if not frame_read_ok or frame is None:
                logger.warning("Failed to read frame from source. Ending stream.")
                break

            fps = fps_counter.tick()
            draw_fps(frame, fps)

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
    """Entry point. Returns a process exit code."""
    logger.info("Starting Phase 1 live video pipeline...")
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
