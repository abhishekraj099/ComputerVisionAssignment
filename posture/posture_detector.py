"""
Per-person posture classification (STANDING vs. SEATED) via bbox geometry.

Purpose:
    Classify each tracked person as STANDING or SEATED, per track ID,
    using only their bounding box's own geometry (height-to-width aspect
    ratio) - no pose estimation model, no additional AI model of any kind.

Responsibilities:
    - Compute the height/width aspect ratio of each tracked person's
      bounding box.
    - Remember, per track ID, a short rolling history of recent aspect
      ratios.
    - Classify a track as STANDING if its *smoothed* (averaged over that
      history) aspect ratio exceeds a configurable threshold, SEATED
      otherwise - the averaging is what prevents a single noisy frame
      (a momentarily bad box) from flipping the reported posture.
    - Forget a track ID's history once it has not been seen for longer
      than a configurable timeout, so memory does not grow without bound.
    - Draw a "Standing"/"Seated" label near each tracked person.

Scope of the current phase (Phase 7):
    Posture *classification* and its on-screen label only.

What this module intentionally does NOT handle:
    - No pose estimation, no MediaPipe, no YOLO-Pose, and no additional
      model of any kind is loaded - only the existing TrackedPerson
      bounding box geometry is used, per the Phase 7 requirement.
    - No blur detection, occupancy detection, or any UI beyond the one
      label - all reserved for later phases (or out of scope entirely).
    - No dependency on attendance, line-crossing, or motion state: this
      module consumes only List[TrackedPerson] and knows nothing about
      events.line_crossing, attendance.attendance_manager, or
      motion.motion_detector.

Which future modules will consume this module's output:
    A future occupancy-detection module (or a future attendance/seating
    analytics module) is a natural consumer of the List[PostureState]
    returned by PostureDetector.update() (e.g. "how many seated vs.
    standing people are currently present"), rather than re-deriving
    posture from raw bounding boxes itself.
"""

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
class PostureState:
    """
    A single track's posture classification for one frame.

    None of the fields are optional - a PostureState is only ever
    constructed once a tracked person's bounding box has been validated.

    Fields:
        track_id: The ByteTrack track ID (see tracker.byte_tracker.
            TrackedPerson.track_id) this classification belongs to.
        is_standing: True if this track is classified as STANDING (its
            smoothed aspect ratio exceeds
            config.POSTURE_ASPECT_RATIO_THRESHOLD), False if SEATED.
        aspect_ratio: The smoothed (rolling-average, over the last
            up-to-config.POSTURE_HISTORY_SIZE frames) bounding-box
            height-to-width ratio. This is the exact value compared
            against the threshold to produce `is_standing`.
        current_bbox: (x1, y1, x2, y2) pixel coordinates of the tracked
            person's bounding box this frame, in the same pixel space as
            the source frame.
    """

    track_id: int
    is_standing: bool
    aspect_ratio: float
    current_bbox: Tuple[int, int, int, int]


class PostureDetector:
    """
    Classifies each tracked person as STANDING or SEATED using only their
    bounding box's height-to-width aspect ratio.

    Purpose:
        Convert a per-frame stream of TrackedPerson boxes into a per-frame
        stream of PostureState classifications, smoothed over a short
        history so a single noisy frame (e.g. a momentarily clipped box)
        cannot flip the reported posture.

    Lifecycle:
        Construct exactly once per application run (see
        app.load_posture_detector()), before the frame loop starts. Call
        update() once per frame, in frame order, passing that frame's
        tracked people. Internal per-track state (aspect-ratio history,
        last-seen time, last logged state) accumulates across calls and is
        not reset between frames - only a fresh instance clears it. A
        track ID's state is automatically forgotten once it has not
        appeared in `tracks` for longer than `stale_track_timeout`
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
        attendance.attendance_manager, events.line_crossing, or
        motion.motion_detector - posture classification is independent of
        entry/exit/attendance/motion state. app.py constructs one instance
        per run and feeds it the same `tracks` list already produced for
        draw_tracks()/person-count display.
    """

    def __init__(
        self,
        aspect_ratio_threshold: float,
        history_size: int,
        stale_track_timeout: float,
        standing_height_retention: float = 1.0,
    ):
        """
        Args:
            aspect_ratio_threshold: Smoothed height/width ratio above
                which a track is classified STANDING (see
                config.POSTURE_ASPECT_RATIO_THRESHOLD). Standing people
                generally have a taller (larger) ratio; seated people
                generally have a shorter one.
            history_size: Number of recent per-frame aspect ratios
                averaged together per track before comparing to the
                threshold (see config.POSTURE_HISTORY_SIZE). Larger values
                smooth out more jitter but react more slowly to a genuine
                posture change.
            stale_track_timeout: How long, in seconds, a track ID's
                history is kept after it stops appearing in `tracks`,
                before being evicted (see
                config.POSTURE_STALE_TRACK_TIMEOUT).

        Side effects:
            None.
        """
        self._aspect_ratio_threshold = aspect_ratio_threshold
        self._history_size = history_size
        self._stale_track_timeout = stale_track_timeout
        self._standing_height_retention = standing_height_retention

        self._ratio_history: Dict[int, Deque[float]] = {}
        self._last_seen_time: Dict[int, float] = {}
        self._last_logged_state: Dict[int, bool] = {}
        # Per-track evidence used by the height-retention rule (see update()):
        # the tallest box ever seen for this track, and whether this track has
        # ever been confidently classified STANDING by the aspect ratio alone.
        self._max_height: Dict[int, int] = {}
        self._ever_standing: Dict[int, bool] = {}

    @staticmethod
    def compute_aspect_ratio(track: TrackedPerson) -> float:
        """
        Compute a tracked person's bounding-box height-to-width ratio.

        Args:
            track: A TrackedPerson (see tracker.byte_tracker).

        Returns:
            `(y2 - y1) / (x2 - x1)` as a float.

        Raises:
            ValueError: If the box's width or height is not strictly
                positive (a zero or negative width/height box carries no
                meaningful aspect ratio and would otherwise divide by
                zero). Callers are expected to catch this per-track,
                exactly like an invalid/malformed box.
        """
        width = track.x2 - track.x1
        height = track.y2 - track.y1

        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive box dimensions (width={width}, height={height})")

        return height / width

    def update(self, tracks: List[TrackedPerson]) -> List[PostureState]:
        """
        Classify every currently tracked person as STANDING or SEATED.

        Args:
            tracks: This frame's tracked people, as returned by
                tracker.byte_tracker.PersonTracker.track(). May be empty
                (e.g. a frame with no detections) - handled as a no-op.

        Returns:
            A list of PostureState objects, one per successfully-processed
            track. Empty if `tracks` is empty or every track failed to
            process (see Raises below - this method does not raise, so
            "failed to process" means logged-and-skipped, not an
            exception surfacing to the caller).

        Raises:
            Does not raise - a failure computing any single track's aspect
            ratio (including a zero-width or zero-height box) is caught,
            logged as a warning, and that track is skipped for this frame
            rather than aborting the whole update.

        Side effects:
            Updates the internal per-track aspect-ratio-history/last-seen
            maps, evicts any track ID not seen for longer than
            `stale_track_timeout`, and logs an INFO line only when a
            track's classification *changes* from STANDING to SEATED or
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
            aspect-ratio history, so its first ratio is used as-is (the
            average of a single-element history is that element) - unlike
            a displacement measurement, a bounding box's aspect ratio is a
            complete, meaningful value on the very first frame it is
            observed, so no special first-frame default is needed.
        """
        states: List[PostureState] = []
        now = time.time()

        for track in tracks:
            try:
                aspect_ratio = self.compute_aspect_ratio(track)
            except Exception as exc:
                logger.warning("Skipping posture check for track %s: invalid bounding box (%s)", track.track_id, exc)
                continue

            history = self._ratio_history.setdefault(
                track.track_id, deque(maxlen=self._history_size)
            )
            history.append(aspect_ratio)
            smoothed_ratio = sum(history) / len(history)

            # Primary signal, unchanged: a box clearly taller than it is wide
            # is a standing person.
            ratio_says_standing = smoothed_ratio > self._aspect_ratio_threshold

            # Height-retention rule. The aspect ratio alone cannot separate a
            # standing person whose legs are hidden behind a desk (short, wide
            # box) from a genuinely seated one - measured on real footage, a
            # standing person can read 1.23 while a seated person reads 2.38,
            # so the two populations overlap and no threshold splits them.
            #
            # What does distinguish them is the track's own history: someone
            # standing behind a desk was fully visible, and tall, moments
            # earlier; someone who has been seated the whole time never was.
            # So once a track has been confidently called STANDING by the
            # ratio, it stays STANDING while its box remains at least
            # `standing_height_retention` of the tallest box ever seen for
            # that track - and flips to SEATED once the box collapses below
            # that, which is what actually happens when a person sits down.
            #
            # This is still pure bounding-box geometry: no pose estimation, no
            # extra model, only the box dimensions this module already had.
            current_height = track.y2 - track.y1
            max_height = max(self._max_height.get(track.track_id, 0), current_height)
            self._max_height[track.track_id] = max_height

            if ratio_says_standing:
                self._ever_standing[track.track_id] = True
                is_standing = True
            elif self._ever_standing.get(track.track_id) and max_height > 0:
                is_standing = current_height >= (self._standing_height_retention * max_height)
                if not is_standing:
                    # Box has collapsed well below this track's own maximum:
                    # treat as a genuine sit-down and stop retaining STANDING.
                    self._ever_standing[track.track_id] = False
            else:
                is_standing = False

            logger.debug(
                "Track %s aspect_ratio=%.2f smoothed=%.2f threshold=%.2f height=%d max=%d -> %s",
                track.track_id, aspect_ratio, smoothed_ratio, self._aspect_ratio_threshold,
                current_height, max_height, "STANDING" if is_standing else "SEATED",
            )

            previous_logged_state = self._last_logged_state.get(track.track_id)
            if previous_logged_state is not None and previous_logged_state != is_standing:
                logger.info(
                    "Track %s posture changed: %s -> %s",
                    track.track_id,
                    "STANDING" if previous_logged_state else "SEATED",
                    "STANDING" if is_standing else "SEATED",
                )
            self._last_logged_state[track.track_id] = is_standing

            self._last_seen_time[track.track_id] = now

            states.append(
                PostureState(
                    track_id=track.track_id,
                    is_standing=is_standing,
                    aspect_ratio=smoothed_ratio,
                    current_bbox=(track.x1, track.y1, track.x2, track.y2),
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
            Removes stale entries from `_ratio_history`, `_last_seen_time`,
            and `_last_logged_state`.
        """
        stale_track_ids = [
            track_id
            for track_id, last_seen in self._last_seen_time.items()
            if (now - last_seen) >= self._stale_track_timeout
        ]
        for track_id in stale_track_ids:
            self._ratio_history.pop(track_id, None)
            self._last_seen_time.pop(track_id, None)
            self._last_logged_state.pop(track_id, None)
            self._max_height.pop(track_id, None)
            self._ever_standing.pop(track_id, None)


def draw_posture_states(
    frame,
    posture_states: List[PostureState],
    standing_color,
    seated_color,
    font_scale: float,
    thickness: int,
    label_offset_y: int,
):
    """
    Draw a "Standing" or "Seated" label near each tracked person.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        posture_states: Classifications returned by
            PostureDetector.update().
        standing_color: BGR color tuple used for the "Standing" label.
        seated_color: BGR color tuple used for the "Seated" label.
        font_scale: Font scale for the label text.
        thickness: Font stroke thickness, in pixels.
        label_offset_y: Vertical offset, in pixels, from the tracked
            person's bounding-box center to where the label is drawn
            (positive moves the label downward). Chosen, in config, to sit
            below the motion label so the two do not overlap.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via repeated cv2.putText calls.

    Performance considerations:
        O(number of posture states) simple drawing operations; negligible
        compared to detection/tracking cost.
    """
    for state in posture_states:
        x1, y1, x2, y2 = state.current_bbox
        centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

        label = "Standing" if state.is_standing else "Seated"
        color = standing_color if state.is_standing else seated_color
        position = (centroid[0], centroid[1] + label_offset_y)

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
