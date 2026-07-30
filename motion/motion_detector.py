"""
Per-person motion classification (MOVING vs. STATIONARY) via centroids.

Purpose:
    Classify each tracked person as MOVING or STATIONARY, per track ID,
    using only the centroid trajectory already available from tracking -
    no optical flow, no background subtraction, no additional model
    inference.

Responsibilities:
    - Compute a centroid for each tracked person.
    - Remember, per track ID, a short rolling history of recent
      frame-to-frame centroid displacements.
    - Classify a track as MOVING if its *smoothed* (averaged over that
      history) displacement exceeds a configurable threshold, STATIONARY
      otherwise - the averaging is what prevents a single noisy frame
      from flipping the reported state.
    - Forget a track ID's history once it has not been seen for longer
      than a configurable timeout, so memory does not grow without bound.
    - Draw a "Moving"/"Stationary" label near each tracked person.

Scope of the current phase (Phase 6):
    Motion *classification* and its on-screen label only.

What this module intentionally does NOT handle:
    - No optical flow and no background subtraction - only the centroid
      positions already produced by tracking are used, per the Phase 6
      requirement.
    - No blur detection, occupancy detection, seated-vs-standing
      classification, or any UI beyond the one label - all reserved for
      later phases (or out of scope entirely).
    - No dependency on attendance or line-crossing state: this module
      consumes only List[TrackedPerson] and knows nothing about
      events.line_crossing or attendance.attendance_manager.

Which future modules will consume this module's output:
    A future occupancy-detection module is a natural consumer of the
    List[MotionState] returned by MotionDetector.update() (e.g. to
    distinguish "occupied and active" from "occupied but everyone is
    stationary"), rather than re-deriving motion from raw centroids
    itself.
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import cv2

import config
from tracker.byte_tracker import TrackedPerson
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


@dataclass(frozen=True)
class MotionState:
    """
    A single track's motion classification for one frame.

    None of the fields are optional - a MotionState is only ever
    constructed once a tracked person's centroid has been computed.

    Fields:
        track_id: The ByteTrack track ID (see tracker.byte_tracker.
            TrackedPerson.track_id) this classification belongs to.
        is_moving: True if this track is classified as MOVING (its
            smoothed displacement exceeds config.MOTION_DISTANCE_THRESHOLD),
            False if STATIONARY.
        movement_distance: The smoothed (rolling-average, over the last
            up-to-config.MOTION_HISTORY_SIZE frames) frame-to-frame
            centroid displacement, in pixels. This is the exact value
            compared against the threshold to produce `is_moving`; on a
            track's very first observed frame this is 0.0 (no prior
            centroid to measure displacement from).
        current_centroid: (x, y) pixel coordinates of the tracked
            person's bounding-box center this frame, in the same pixel
            space as the source frame.
    """

    track_id: int
    is_moving: bool
    movement_distance: float
    current_centroid: Tuple[int, int]


class MotionDetector:
    """
    Classifies each tracked person as MOVING or STATIONARY using only
    their centroid trajectory.

    Purpose:
        Convert a per-frame stream of TrackedPerson boxes into a per-frame
        stream of MotionState classifications, smoothed over a short
        history so a single noisy frame cannot flip the reported state.

    Lifecycle:
        Construct exactly once per application run (see
        app.load_motion_detector()), before the frame loop starts. Call
        update() once per frame, in frame order, passing that frame's
        tracked people. Internal per-track state (last centroid, distance
        history, last-seen time, last logged state) accumulates across
        calls and is not reset between frames - only a fresh instance
        clears it. A track ID's state is automatically forgotten once it
        has not appeared in `tracks` for longer than `stale_track_timeout`
        seconds, so this accumulation is bounded rather than unbounded
        over a long run.

    Thread safety:
        Not thread-safe. update() reads and mutates internal dicts without
        locking; it must only be called sequentially from the single
        frame-processing loop.

    Interaction with other classes:
        Consumes only tracker.byte_tracker.TrackedPerson objects (does not
        import or depend on PersonTracker itself, only its output type).
        Deliberately has no dependency whatsoever on
        events.line_crossing or attendance.attendance_manager - motion
        classification is independent of entry/exit/attendance state.
        app.py constructs one instance per run and feeds it the same
        `tracks` list already produced for draw_tracks()/person-count
        display.
    """

    def __init__(self, distance_threshold: float, history_size: int, stale_track_timeout: float):
        """
        Args:
            distance_threshold: Smoothed per-frame displacement, in
                pixels, above which a track is classified MOVING (see
                config.MOTION_DISTANCE_THRESHOLD).
            history_size: Number of recent frame-to-frame displacements
                averaged together per track before comparing to the
                threshold (see config.MOTION_HISTORY_SIZE). Larger values
                smooth out more jitter but react more slowly to genuine
                movement starting/stopping.
            stale_track_timeout: How long, in seconds, a track ID's
                history is kept after it stops appearing in `tracks`,
                before being evicted (see
                config.MOTION_STALE_TRACK_TIMEOUT).

        Side effects:
            None.
        """
        self._distance_threshold = distance_threshold
        self._history_size = history_size
        self._stale_track_timeout = stale_track_timeout

        self._last_centroid: Dict[int, Tuple[int, int]] = {}
        self._distance_history: Dict[int, Deque[float]] = {}
        self._last_seen_time: Dict[int, float] = {}
        self._last_logged_state: Dict[int, bool] = {}

    @staticmethod
    def compute_centroid(track: TrackedPerson) -> Tuple[int, int]:
        """
        Compute a tracked person's bounding-box centroid.

        Args:
            track: A TrackedPerson (see tracker.byte_tracker).

        Returns:
            (x, y) pixel coordinates of the box's center, in the same
            pixel space as the source frame.

        Note:
            Deliberately duplicated from
            events.line_crossing.LineCrossingDetector.compute_centroid
            (identical one-line formula) rather than imported from there,
            so this module has zero dependency on events.line_crossing -
            see the module docstring's "Interaction with other classes."
        """
        return ((track.x1 + track.x2) // 2, (track.y1 + track.y2) // 2)

    def update(self, tracks: List[TrackedPerson]) -> List[MotionState]:
        """
        Classify every currently tracked person as MOVING or STATIONARY.

        Args:
            tracks: This frame's tracked people, as returned by
                tracker.byte_tracker.PersonTracker.track(). May be empty
                (e.g. a frame with no detections) - handled as a no-op.

        Returns:
            A list of MotionState objects, one per successfully-processed
            track. Empty if `tracks` is empty or every track failed to
            process (see Raises below - this method does not raise, so
            "failed to process" means logged-and-skipped, not an
            exception surfacing to the caller).

        Raises:
            Does not raise - a failure computing any single track's
            centroid is caught, logged as a warning, and that track is
            skipped for this frame rather than aborting the whole update.

        Side effects:
            Updates the internal per-track centroid/history/last-seen maps,
            evicts any track ID not seen for longer than
            `stale_track_timeout`, and logs an INFO line only when a
            track's classification *changes* from MOVING to STATIONARY or
            vice versa (never logs on every frame, and never logs a
            track's very first classification, since there is no prior
            state for it to have "changed" from).

        Performance considerations:
            O(number of tracks) for the classification, plus O(number of
            remembered track IDs) for stale-track eviction; pure
            arithmetic and dict/deque operations, no I/O or model
            inference - negligible compared to detection/tracking cost.

        Note on missing/lost/new tracks:
            A track ID that stops appearing in `tracks` (person left the
            frame, or the track was lost) simply stops being updated; its
            history remains stored until either it reappears or
            `stale_track_timeout` elapses, at which point it is forgotten
            entirely. A track ID seen for the first time has no prior
            centroid to measure displacement from, so its first
            MotionState always reports `movement_distance=0.0` and
            `is_moving=False` (STATIONARY) - a reasonable default until
            enough frames have accumulated to say otherwise.
        """
        states: List[MotionState] = []
        now = time.time()

        for track in tracks:
            try:
                centroid = self.compute_centroid(track)
            except Exception as exc:
                logger.warning("Skipping motion check for track %s: invalid centroid (%s)", track.track_id, exc)
                continue

            previous_centroid = self._last_centroid.get(track.track_id)
            distance = 0.0 if previous_centroid is None else math.dist(previous_centroid, centroid)

            history = self._distance_history.setdefault(
                track.track_id, deque(maxlen=self._history_size)
            )
            history.append(distance)
            smoothed_distance = sum(history) / len(history)

            is_moving = smoothed_distance > self._distance_threshold

            previous_logged_state = self._last_logged_state.get(track.track_id)
            if previous_logged_state is not None and previous_logged_state != is_moving:
                logger.info(
                    "Track %s motion state changed: %s -> %s",
                    track.track_id,
                    "MOVING" if previous_logged_state else "STATIONARY",
                    "MOVING" if is_moving else "STATIONARY",
                )
            self._last_logged_state[track.track_id] = is_moving

            self._last_centroid[track.track_id] = centroid
            self._last_seen_time[track.track_id] = now

            states.append(
                MotionState(
                    track_id=track.track_id,
                    is_moving=is_moving,
                    movement_distance=smoothed_distance,
                    current_centroid=centroid,
                )
            )

        self._prune_stale_tracks(now)

        return states

    def _prune_stale_tracks(self, now: float) -> None:
        """
        Forget any track ID not seen for longer than `stale_track_timeout`.

        Args:
            now: Current time (time.time()), passed in so every track
                checked within the same update() call is compared against
                one consistent timestamp.

        Side effects:
            Removes stale entries from `_last_centroid`,
            `_distance_history`, `_last_seen_time`, and
            `_last_logged_state`.
        """
        stale_track_ids = [
            track_id
            for track_id, last_seen in self._last_seen_time.items()
            if (now - last_seen) >= self._stale_track_timeout
        ]
        for track_id in stale_track_ids:
            self._last_centroid.pop(track_id, None)
            self._distance_history.pop(track_id, None)
            self._last_seen_time.pop(track_id, None)
            self._last_logged_state.pop(track_id, None)


def draw_motion_states(
    frame,
    motion_states: List[MotionState],
    moving_color,
    stationary_color,
    font_scale: float,
    thickness: int,
    label_offset_y: int,
):
    """
    Draw a "Moving" or "Stationary" label near each tracked person.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        motion_states: Classifications returned by MotionDetector.update().
        moving_color: BGR color tuple used for the "Moving" label.
        stationary_color: BGR color tuple used for the "Stationary" label.
        font_scale: Font scale for the label text.
        thickness: Font stroke thickness, in pixels.
        label_offset_y: Vertical offset, in pixels, from the tracked
            person's centroid to where the label is drawn (positive moves
            the label downward).

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via repeated cv2.putText calls.

    Performance considerations:
        O(number of motion states) simple drawing operations; negligible
        compared to detection/tracking cost.
    """
    for state in motion_states:
        label = "Moving" if state.is_moving else "Stationary"
        color = moving_color if state.is_moving else stationary_color
        position = (state.current_centroid[0], state.current_centroid[1] + label_offset_y)

        cv2.putText(
            frame,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return frame
