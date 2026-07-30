"""
Frame-level quality classification (SHARP vs. BLURRY) via Variance of
Laplacian.

Purpose:
    Classify each processed frame, as a whole, as SHARP or BLURRY, using
    the classic Variance-of-Laplacian heuristic - no per-person analysis,
    no additional model.

Responsibilities:
    - Convert the frame to grayscale.
    - Compute the Laplacian and its variance (a measure of how much
      high-frequency edge detail the frame contains).
    - Classify BLURRY if that variance falls below a configurable
      threshold, SHARP otherwise.
    - Draw a small "Frame Quality" panel showing the classification and
      raw variance value.

Scope of the current phase (Phase 8):
    Frame-level quality *classification* and its on-screen panel only.

What this module intentionally does NOT handle:
    - No per-person analysis of any kind - this is a single, whole-frame
      measurement, unrelated to any individual tracked person.
    - No occupancy detection, Streamlit/dashboard UI, database, CSV
      export, API, or face recognition - all reserved for later phases
      (or out of scope entirely).
    - No dependency on tracking, motion, posture, line-crossing, or
      attendance state: this module consumes only a raw frame
      (numpy.ndarray) and knows nothing about
      tracker.byte_tracker.TrackedPerson, motion.motion_detector,
      posture.posture_detector, events.line_crossing, or
      attendance.attendance_manager.

Which future modules will consume this module's output:
    Any future phase that needs to know "is this frame trustworthy enough
    to act on" (e.g. an occupancy module choosing to skip an update on a
    badly blurred frame) is expected to read the BlurState returned by
    BlurDetector.analyze(), rather than re-deriving a sharpness measure
    itself.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

import config
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


@dataclass(frozen=True)
class BlurState:
    """
    A whole-frame quality classification for one frame.

    Neither field is optional - a BlurState is always returned by
    BlurDetector.analyze(), even for an empty/invalid frame (see that
    method's docstring for the conservative fallback used in that case).

    Fields:
        is_blurry: True if the frame is classified BLURRY (its Laplacian
            variance is below config.BLUR_THRESHOLD), False if SHARP.
        laplacian_variance: The raw Laplacian variance measured for this
            frame - a non-negative float. Higher values indicate more
            high-frequency edge detail (a sharper-looking frame); lower
            values indicate less (a blurrier-looking frame). This is the
            exact value compared against the threshold to produce
            `is_blurry`.
    """

    is_blurry: bool
    laplacian_variance: float


class BlurDetector:
    """
    Classifies a whole frame as SHARP or BLURRY using Variance of
    Laplacian.

    Purpose:
        Provide a cheap, model-free, frame-level sharpness measurement so
        the pipeline (or a human watching it) can tell when the camera
        feed itself has degraded (out of focus, heavy motion blur, a
        smudged lens, etc.), independent of anything detection/tracking
        related.

    Lifecycle:
        Construct exactly once per application run (see
        app.load_blur_detector()), before the frame loop starts. Call
        analyze() once per frame, passing that frame. Unlike the
        per-track detectors in this project (Motion/Posture/LineCrossing),
        this class holds only a single piece of state - the last logged
        classification, used solely to avoid repeated logging - so there
        is nothing to evict and no per-track memory growth to bound.

    Thread safety:
        Not thread-safe. analyze() mutates the single `_last_logged_state`
        attribute without locking; it must only be called sequentially
        from the single frame-processing loop.

    Interaction with other classes:
        None. This class has no dependency on, and is not depended on by,
        any tracking/motion/posture/line-crossing/attendance class in this
        project - it consumes only a raw frame. app.py constructs one
        instance per run and calls analyze() on the freshly-captured frame
        *before* any overlay drawing happens (drawing boxes/text onto the
        frame would otherwise artificially inflate its measured
        sharpness).
    """

    def __init__(self, threshold: float):
        """
        Args:
            threshold: Laplacian variance below which a frame is
                classified BLURRY (see config.BLUR_THRESHOLD).

        Side effects:
            None.
        """
        self._threshold = threshold
        self._last_logged_state: Optional[bool] = None

    def analyze(self, frame) -> BlurState:
        """
        Classify a single frame as SHARP or BLURRY.

        Args:
            frame: BGR (or grayscale, or BGRA) image (numpy array) as read
                from OpenCV. May be None, empty, or have an unexpected
                number of channels - all handled gracefully (see Returns).

        Returns:
            A BlurState. On a valid frame, `laplacian_variance` is the
            measured value and `is_blurry` is the threshold comparison
            result. On an empty/invalid/unsupported frame, a conservative
            fallback of `BlurState(is_blurry=True, laplacian_variance=0.0)`
            is returned instead - i.e. "flag for review" rather than
            silently claiming the frame is sharp when it could not
            actually be measured.

        Raises:
            Does not raise - any failure (None/empty frame, wrong type,
            unsupported channel count, an OpenCV error) is caught, logged
            as a warning, and answered with the conservative fallback
            above rather than propagating.

        Side effects:
            Logs an INFO line only when the classification *changes* from
            SHARP to BLURRY or vice versa (never logs on every frame, and
            never logs the very first classification, since there is no
            prior state for it to have "changed" from).

        Performance considerations:
            One grayscale conversion plus one Laplacian convolution over
            the full frame - noticeably more expensive than the O(1)/O(n)
            per-track arithmetic used elsewhere in this project, but still
            a standard, cheap OpenCV operation relative to YOLO inference.
        """
        try:
            gray = self._to_grayscale(frame)
            laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception as exc:
            logger.warning("Skipping blur analysis for this frame: %s", exc)
            return BlurState(is_blurry=True, laplacian_variance=0.0)

        is_blurry = laplacian_variance < self._threshold

        if self._last_logged_state is not None and self._last_logged_state != is_blurry:
            logger.info(
                "Frame quality changed: %s -> %s",
                "BLURRY" if self._last_logged_state else "SHARP",
                "BLURRY" if is_blurry else "SHARP",
            )
        self._last_logged_state = is_blurry

        return BlurState(is_blurry=is_blurry, laplacian_variance=laplacian_variance)

    @staticmethod
    def _to_grayscale(frame):
        """
        Convert a frame to single-channel grayscale, validating it first.

        Args:
            frame: The candidate frame (ideally a numpy array).

        Returns:
            A single-channel grayscale numpy array.

        Raises:
            ValueError: If `frame` is None, empty, not a numpy array, or
                has an unsupported number of channels (anything other than
                grayscale/BGR/BGRA). Callers are expected to catch this
                exactly like any other invalid-frame condition.
        """
        if frame is None:
            raise ValueError("frame is None")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"frame is not a numpy array (got {type(frame)!r})")
        if frame.size == 0:
            raise ValueError("frame is empty")

        if frame.ndim == 2:
            return frame

        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if channels == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            if channels == 1:
                return frame[:, :, 0]
            raise ValueError(f"unsupported channel count: {channels}")

        raise ValueError(f"unsupported frame shape: {frame.shape}")


def draw_blur_panel(
    frame,
    blur_state: BlurState,
    position,
    line_spacing: int,
    font_scale: float,
    sharp_color,
    blurry_color,
    thickness: int,
):
    """
    Draw a small "Frame Quality" panel showing the SHARP/BLURRY
    classification and the raw variance value.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        blur_state: The BlurState to display, as returned by
            BlurDetector.analyze().
        position: (x, y) pixel position for the panel's first ("Frame
            Quality") line.
        line_spacing: Vertical gap, in pixels, between consecutive lines.
        font_scale: Font scale for the panel text.
        sharp_color: BGR color tuple used when the frame is classified
            SHARP.
        blurry_color: BGR color tuple used when the frame is classified
            BLURRY.
        thickness: Font stroke thickness, in pixels.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via repeated cv2.putText calls.

    Performance considerations:
        A fixed, small number of drawing calls per frame; negligible
        compared to the Laplacian computation in analyze() or to
        detection/tracking cost.
    """
    color = blurry_color if blur_state.is_blurry else sharp_color
    lines = [
        "Frame Quality",
        "Blurry" if blur_state.is_blurry else "Sharp",
        f"Variance : {blur_state.laplacian_variance:.1f}",
    ]

    for index, line in enumerate(lines):
        line_y = position[1] + (index * line_spacing)
        cv2.putText(
            frame,
            line,
            (position[0], line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return frame
