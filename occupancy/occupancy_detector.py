"""
Room occupancy classification derived purely from other modules' outputs.

Purpose:
    Determine a single, room-level occupancy state - EMPTY, OCCUPIED,
    ACTIVE, or IDLE - by combining the already-computed outputs of the
    attendance, motion, posture, and blur-detection subsystems. This
    module performs no video/frame analysis and no tracking of its own.

Responsibilities:
    - Read `AttendanceStatistics.current_people` as the single source of
      truth for how many people are present (never recomputed here).
    - Summarize `List[MotionState]` into a moving-people count and
      `List[PostureState]` into standing/seated counts.
    - Classify the room's occupancy status from those summaries.
    - Skip updating the occupancy decision on a BLURRY frame, keeping the
      previous decision instead - poor frame quality should not be allowed
      to change what we believe about the room.
    - Draw a small "Occupancy" panel summarizing the current decision.

Scope of the current phase (Phase 9):
    Occupancy *classification* (from existing outputs) and its on-screen
    panel only.

What this module intentionally does NOT handle:
    - No video frame analysis of any kind - this module never receives a
      frame, a TrackedPerson, or anything from models.person_detector,
      tracker.byte_tracker (PersonTracker/YOLO/ByteTrack), or
      events.line_crossing (LineCrossingDetector). It consumes only the
      four already-computed data types listed in "Interaction with other
      classes" below.
    - No Streamlit/dashboard UI, database, CSV export, REST API, face
      recognition, or person re-identification - all reserved for later
      phases (or out of scope entirely).
    - No lights-on-but-empty detection (a bonus feature mentioned in the
      original task) - that would require analyzing frame brightness,
      which is out of scope for this phase.

Which future modules will consume this module's output:
    A future live UI/dashboard phase is the natural consumer of the
    OccupancyState returned by OccupancyDetector.update() - it is designed
    to be the single, highest-level summary value produced by this
    pipeline, built entirely on top of the other subsystems' outputs
    rather than duplicating their logic.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import cv2

import config
from attendance.attendance_manager import AttendanceStatistics
from motion.motion_detector import MotionState
from posture.posture_detector import PostureState
from quality.blur_detector import BlurState
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


class OccupancyStatus(str, Enum):
    """The four room-level occupancy states this module distinguishes."""

    EMPTY = "EMPTY"
    OCCUPIED = "OCCUPIED"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"


@dataclass(frozen=True)
class OccupancyState:
    """
    A single room-level occupancy snapshot for one frame.

    None of the fields are optional - a fresh, all-zero/EMPTY
    OccupancyState is returned even before any other subsystem has ever
    produced data.

    Fields:
        occupancy_status: One of OccupancyStatus.EMPTY, .OCCUPIED,
            .ACTIVE, or .IDLE - see the module-level "Occupancy states"
            definitions.
        people_inside: The number of people currently inside, copied
            directly from AttendanceStatistics.current_people (never
            recomputed here); a non-negative integer.
        moving_people: Count of List[MotionState] entries with
            `is_moving=True` this frame; a non-negative integer.
        standing_people: Count of List[PostureState] entries with
            `is_standing=True` this frame; a non-negative integer.
        seated_people: Count of List[PostureState] entries with
            `is_standing=False` this frame; a non-negative integer.
        frame_quality_ok: True if the frame this state was computed from
            was classified SHARP (or, on the very first-ever decision,
            whatever quality it happened to be - see
            OccupancyDetector.update()); False if it was BLURRY or the
            BlurState was unavailable. Note this reflects the quality of
            the frame the *current* decision is based on - when a decision
            is kept unchanged due to blur (see Frame Quality below), this
            field is still `False` for that returned snapshot, since the
            frame that triggered the "keep previous" behavior was not
            trustworthy, even though the status itself did not change.
    """

    occupancy_status: OccupancyStatus
    people_inside: int
    moving_people: int
    standing_people: int
    seated_people: int
    frame_quality_ok: bool


class OccupancyDetector:
    """
    Derives a single room-level occupancy state from other subsystems'
    already-computed outputs.

    Purpose:
        Combine AttendanceStatistics, List[MotionState],
        List[PostureState], and BlurState into one OccupancyState per
        frame, without ever touching a video frame, a TrackedPerson, or
        any detection/tracking/line-crossing internals.

    Lifecycle:
        Construct exactly once per application run (see
        app.load_occupancy_detector()), before the frame loop starts. Call
        update() once per frame, passing that frame's four inputs.
        Internal state (the last computed decision, and whether a blur
        skip was already logged) accumulates across calls and is not reset
        between frames - only a fresh instance clears it.

    Thread safety:
        Not thread-safe. update() reads and mutates internal attributes
        without locking; it must only be called sequentially from the
        single frame-processing loop.

    Interaction with other classes:
        Consumes only attendance.attendance_manager.AttendanceStatistics,
        motion.motion_detector.MotionState,
        posture.posture_detector.PostureState, and
        quality.blur_detector.BlurState - plain data types, not the
        classes that produce them. Has no import of, or dependency on,
        models.person_detector, tracker.byte_tracker (TrackedPerson,
        PersonTracker, YOLO, ByteTrack), or events.line_crossing
        (LineCrossingDetector). app.py constructs one instance per run and
        feeds it the same four values it already produced for the
        motion/posture/attendance/blur overlays.
    """

    def __init__(self):
        """
        Side effects:
            None. Starts with no prior decision (the first call to
            update() always computes a fresh decision - see its
            docstring).
        """
        self._last_state: Optional[OccupancyState] = None
        self._last_call_was_blur_skip = False

    def update(
        self,
        attendance_stats: Optional[AttendanceStatistics],
        motion_states: Optional[List[MotionState]],
        posture_states: Optional[List[PostureState]],
        blur_state: Optional[BlurState],
    ) -> OccupancyState:
        """
        Compute (or, on a blurry frame, retain) this frame's occupancy
        decision.

        Args:
            attendance_stats: This frame's AttendanceStatistics, as
                returned by AttendanceManager.update(). If None (missing),
                `people_inside` defaults to 0 for this computation.
            motion_states: This frame's List[MotionState], as returned by
                MotionDetector.update(). May be empty (no tracked people
                had a motion state this frame) or None (motion data
                unavailable this frame) - these are treated differently:
                an empty list means "zero people are moving" (known),
                while None means "movement is unknown" (see the ACTIVE/
                IDLE/OCCUPIED decision logic below).
            posture_states: This frame's List[PostureState], as returned
                by PostureDetector.update(). May be empty or None; both
                simply contribute 0 to the standing/seated counts.
            blur_state: This frame's BlurState, as returned by
                BlurDetector.analyze(). If None (missing) or
                `is_blurry=True`, this frame's data is not trusted to
                update the occupancy decision - see Frame Quality below.

        Returns:
            The current OccupancyState: either a freshly computed one, or
            (on a blurry/missing-quality frame, once a prior decision
            exists) the previous decision unchanged, with
            `frame_quality_ok=False` reflecting this frame's poor quality.

        Raises:
            Does not raise - any failure processing the inputs (malformed
            entries, unexpected types) is caught, logged as a warning, and
            answered with the previous decision if one exists, or a safe
            all-zero EMPTY state if this is the first call.

        Side effects:
            Logs an INFO line only when the occupancy status *changes*
            from one call to the next, and a separate INFO line only when
            a blur-driven skip *begins* (not on every subsequent blurry
            frame while it continues) - never once per ordinary frame.

        Frame Quality:
            If `blur_state` indicates BLURRY (or is missing/unusable) AND
            a previous decision already exists, that previous
            OccupancyState is returned unchanged - a blurry frame must
            never be allowed to flip the occupancy decision. If this is
            the very first call (no previous decision to fall back on),
            a decision is still computed from whatever data is available,
            since there is nothing to "keep" yet; its `frame_quality_ok`
            will correctly read False in that case.

        Attendance as source of truth:
            `people_inside` always comes directly from
            `attendance_stats.current_people` - occupancy is never
            computed by counting tracked people, motion states, or
            posture states.
        """
        try:
            frame_quality_ok = self._is_frame_quality_ok(blur_state)

            if not frame_quality_ok and self._last_state is not None:
                if not self._last_call_was_blur_skip:
                    logger.info("Occupancy update skipped this frame: frame quality is poor (blurry).")
                self._last_call_was_blur_skip = True
                return self._last_state

            self._last_call_was_blur_skip = False

            new_state = self._compute_state(attendance_stats, motion_states, posture_states, frame_quality_ok)
        except Exception as exc:
            logger.warning("Failed to compute occupancy this frame, keeping prior state if any: %s", exc)
            return self._last_state or OccupancyState(
                occupancy_status=OccupancyStatus.EMPTY,
                people_inside=0,
                moving_people=0,
                standing_people=0,
                seated_people=0,
                frame_quality_ok=False,
            )

        if self._last_state is None or self._last_state.occupancy_status != new_state.occupancy_status:
            logger.info(
                "Occupancy changed: %s -> %s",
                self._last_state.occupancy_status.value if self._last_state else "UNKNOWN",
                new_state.occupancy_status.value,
            )

        self._last_state = new_state
        return new_state

    @staticmethod
    def _is_frame_quality_ok(blur_state: Optional[BlurState]) -> bool:
        """
        Args:
            blur_state: This frame's BlurState, or None if unavailable.

        Returns:
            False if `blur_state` is None or `is_blurry` is True (or
            unreadable); True only if it is confirmed present and SHARP.
            Missing quality information is treated the same as BLURRY -
            "when in doubt, don't trust this frame" - consistent with
            BlurDetector's own conservative fallback.
        """
        if blur_state is None:
            return False
        return not bool(getattr(blur_state, "is_blurry", True))

    @staticmethod
    def _compute_state(
        attendance_stats: Optional[AttendanceStatistics],
        motion_states: Optional[List[MotionState]],
        posture_states: Optional[List[PostureState]],
        frame_quality_ok: bool,
    ) -> "OccupancyState":
        """
        Build a fresh OccupancyState from this frame's inputs.

        Args:
            attendance_stats: See update(). None defaults `people_inside`
                to 0.
            motion_states: See update(). None means "movement unknown";
                an empty list means "zero people moving" (known).
            posture_states: See update(). None or empty both contribute 0
                standing/seated.
            frame_quality_ok: Whether this frame was confirmed SHARP,
                copied directly into the returned state.

        Returns:
            A new OccupancyState.

        Occupancy status decision:
            - people_inside == 0                       -> EMPTY
            - people_inside > 0 and motion data unknown -> OCCUPIED
              (people are present, but we cannot say whether anyone is
              moving, so the coarser, honest label is used rather than
              guessing ACTIVE or IDLE)
            - people_inside > 0 and >=1 person moving   -> ACTIVE
            - people_inside > 0 and 0 people moving     -> IDLE
        """
        people_inside = 0
        if attendance_stats is not None:
            people_inside = max(0, getattr(attendance_stats, "current_people", 0))

        motion_known = motion_states is not None
        moving_people = 0
        if motion_states:
            moving_people = sum(1 for state in motion_states if getattr(state, "is_moving", False))

        standing_people = 0
        seated_people = 0
        if posture_states:
            for state in posture_states:
                is_standing = getattr(state, "is_standing", None)
                if is_standing is True:
                    standing_people += 1
                elif is_standing is False:
                    seated_people += 1

        if people_inside == 0:
            status = OccupancyStatus.EMPTY
        elif not motion_known:
            status = OccupancyStatus.OCCUPIED
        elif moving_people > 0:
            status = OccupancyStatus.ACTIVE
        else:
            status = OccupancyStatus.IDLE

        return OccupancyState(
            occupancy_status=status,
            people_inside=people_inside,
            moving_people=moving_people,
            standing_people=standing_people,
            seated_people=seated_people,
            frame_quality_ok=frame_quality_ok,
        )


def draw_occupancy_panel(
    frame,
    occupancy_state: OccupancyState,
    position,
    line_spacing: int,
    font_scale: float,
    color,
    thickness: int,
):
    """
    Draw a small "Occupancy" summary panel onto the frame.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        occupancy_state: The OccupancyState to display, as returned by
            OccupancyDetector.update().
        position: (x, y) pixel position for the panel's first
            ("Occupancy") line.
        line_spacing: Vertical gap, in pixels, between consecutive lines.
        font_scale: Font scale for the panel text.
        color: BGR color tuple for the panel text.
        thickness: Font stroke thickness, in pixels.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via repeated cv2.putText calls.

    Performance considerations:
        A fixed, small number of drawing calls per frame; negligible
        compared to any other subsystem's cost.
    """
    lines = [
        "Occupancy",
        f"Status : {occupancy_state.occupancy_status.value}",
        f"People : {occupancy_state.people_inside}",
        f"Moving : {occupancy_state.moving_people}",
        f"Standing : {occupancy_state.standing_people}",
        f"Seated : {occupancy_state.seated_people}",
        f"Frame : {'Sharp' if occupancy_state.frame_quality_ok else 'Blurry'}",
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
