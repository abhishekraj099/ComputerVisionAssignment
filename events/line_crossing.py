"""
Entry/Exit event detection via virtual line crossing.

Purpose:
    Turn each frame's List[TrackedPerson] (from tracker.byte_tracker) into
    ENTRY/EXIT events, by watching whether a tracked person's bounding-box
    centroid crosses one configured virtual line from one side to the
    other, and briefly display those events on screen.

Responsibilities:
    - Compute a centroid for each tracked person.
    - Remember, per track ID, which side of the virtual line that
      person's centroid was on last frame.
    - When a track's side flips, emit exactly one CrossingEvent
      (ENTRY for top-to-bottom, EXIT for bottom-to-top), suppressing any
      further event for that same track ID within a configurable cooldown
      window (to absorb detector/tracker jitter right at the line).
    - Forget a track ID's remembered state once it has not been seen for
      longer than a configurable timeout, so memory does not grow without
      bound over a long-running stream.
    - Draw the virtual line, and briefly display recent crossing events
      as on-screen text.

Scope of the current phase (Phase 4):
    Event *detection* and transient *display* only.

What this module intentionally does NOT handle:
    - No attendance counting (a running "currently present" / "total
      unique entries" tally) - this module only emits individual events,
      it does not accumulate them into counts.
    - No persistence: events are not written to a file, database, or CSV;
      they exist only as return values and brief on-screen text.
    - No occupancy, motion, blur, or seated/standing logic - all reserved
      for later phases.
    - No re-identification of a person who reappears under a new track ID
      after a long absence (a pre-existing Phase 3 limitation this module
      inherits unchanged).

Which future modules will consume this module's output:
    A future attendance module is expected to consume the
    List[CrossingEvent] returned by LineCrossingDetector.update() each
    frame (summing ENTRY/EXIT events into a running present-count and a
    unique-entries total) rather than re-deriving crossings itself.
"""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import cv2

import config
from tracker.byte_tracker import TrackedPerson
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


class CrossingEventType(str, Enum):
    """The two kinds of virtual line crossing this module detects."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass(frozen=True)
class CrossingEvent:
    """
    A single ENTRY or EXIT event for one tracked person.

    None of the fields are optional - a CrossingEvent is only ever
    constructed at the moment a crossing is detected.

    Fields:
        track_id: The ByteTrack track ID (see tracker.byte_tracker.
            TrackedPerson.track_id) of the person who crossed the line.
        event_type: CrossingEventType.ENTRY (top-to-bottom) or
            CrossingEventType.EXIT (bottom-to-top).
        timestamp: Unix epoch seconds (from time.time()) when the crossing
            was detected - i.e. the frame-processing time it fired, not a
            precise physical-crossing instant.
        centroid: (x, y) pixel coordinates of the tracked person's
            bounding-box center at the moment of crossing, in the same
            pixel space as the source frame.
    """

    track_id: int
    event_type: CrossingEventType
    timestamp: float
    centroid: Tuple[int, int]


def _signed_side(point: Tuple[int, int], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> float:
    """
    Compute which side of a line a point falls on, and how far past it, as
    a signed perpendicular distance in pixels.

    Args:
        point: (x, y) pixel coordinates to test.
        line_start: (x, y) pixel coordinates of the line's first endpoint.
        line_end: (x, y) pixel coordinates of the line's second endpoint.

    Returns:
        A signed float: 0.0 if the point lies exactly on the line; a
        consistent positive value for every point on one side, and a
        consistent negative value for every point on the other. For the
        default horizontal line in config.py, a positive result means
        "below the line" (further down the frame) and a negative result
        means "above the line" - i.e. this is the same cross-product test
        used to detect top-to-bottom vs bottom-to-top motion.

        The magnitude is the point's perpendicular distance from the line,
        **in pixels**. The raw cross product is scaled by the line's own
        length, which would make the magnitude meaningless as a distance
        (and dependent on how long the configured line happens to be), so
        it is divided by that length here. Only the magnitude changes -
        the sign, and therefore every crossing decision derived from it, is
        identical to the unnormalized cross product. This is what lets
        LineCrossingDetector apply a dead zone measured in real pixels
        (see config.LINE_CROSSING_HYSTERESIS_MARGIN).
    """
    line_dx = line_end[0] - line_start[0]
    line_dy = line_end[1] - line_start[1]
    point_dx = point[0] - line_start[0]
    point_dy = point[1] - line_start[1]
    cross_product = (line_dx * point_dy) - (line_dy * point_dx)

    line_length = math.hypot(line_dx, line_dy)
    if line_length == 0:
        # Degenerate "line" (both endpoints identical) - no meaningful
        # side or distance exists. Report "on the line" so the caller's
        # dead-zone check treats every point as ambiguous and no spurious
        # crossing is ever reported, rather than dividing by zero.
        return 0.0

    return cross_product / line_length


class LineCrossingDetector:
    """
    Detects ENTRY/EXIT events when a tracked person's centroid crosses a
    configured virtual line.

    Purpose:
        Convert a per-frame stream of TrackedPerson boxes into a per-frame
        stream of CrossingEvent objects, with exactly one event per actual
        line crossing (no duplicates while a person lingers on one side).

    Lifecycle:
        Construct exactly once per application run (see
        app.load_line_crossing_detector()), before the frame loop starts.
        Call update() once per frame, in frame order, passing that frame's
        tracked people. Internal per-track state accumulates across calls
        and is not reset between frames - only a fresh instance clears it.
        A track ID's state is automatically forgotten once it has not
        appeared in `tracks` for longer than `stale_track_timeout` seconds,
        so this accumulation is bounded rather than unbounded over a long
        run (see Robustness note below).

    Thread safety:
        Not thread-safe. update() reads and mutates internal dicts without
        locking; it must only be called sequentially from the single
        frame-processing loop.

    Interaction with other classes:
        Consumes tracker.byte_tracker.TrackedPerson objects (does not
        import or depend on PersonTracker itself, only its output type).
        app.py constructs one instance per run and feeds it the same
        `tracks` list already produced for draw_tracks()/person-count
        display.

    Robustness note (memory growth and event jitter):
        Three independent safeguards protect this class from different
        failure modes, all purely internal to this class - none changes the
        crossing algorithm or its output for a normal, well-separated
        crossing:
        - `stale_track_timeout` bounds memory: a track ID not seen for
          longer than this many seconds has its remembered side, last-seen
          time, and last-event time evicted entirely, so the internal
          dicts cannot grow forever as new, distinct people pass through
          over a long session.
        - `hysteresis_margin` prevents bouncing (the primary safeguard):
          a centroid within this many pixels of the line is treated as
          ambiguous - its confirmed side is left unchanged and no event
          can fire. Because a track's stored side only ever updates once
          the centroid is decisively past the line, a person loitering or
          seated near the line cannot produce the repeated
          ENTRY/EXIT/ENTRY/EXIT sequence that a raw sign test produces
          from a few pixels of detector jitter. Producing two events now
          requires genuinely traversing the full 2x margin band.
        - `event_cooldown` bounds duplicate events (a secondary backstop):
          once an event fires for a track ID, no further event fires for
          that same track ID until `event_cooldown` seconds have passed.
          Side tracking itself is never suppressed - only the emission of
          a *new* CrossingEvent is - so a genuine crossing that happens
          well after the cooldown window is unaffected.
    """

    def __init__(
        self,
        line_start: Tuple[int, int],
        line_end: Tuple[int, int],
        stale_track_timeout: float,
        event_cooldown: float,
        hysteresis_margin: float = 0.0,
    ):
        """
        Args:
            line_start: (x, y) pixel coordinates of the virtual line's
                first endpoint (see config.LINE_START).
            line_end: (x, y) pixel coordinates of the virtual line's
                second endpoint (see config.LINE_END).
            stale_track_timeout: How long, in seconds, a track ID's
                remembered state is kept after it stops appearing in
                `tracks`, before being evicted (see
                config.LINE_CROSSING_STALE_TRACK_TIMEOUT).
            event_cooldown: How long, in seconds, to suppress further
                events for a track ID after one fires for it (see
                config.LINE_CROSSING_EVENT_COOLDOWN).
            hysteresis_margin: Perpendicular distance from the line, in
                pixels, a centroid must be past before its side counts as
                confirmed (see config.LINE_CROSSING_HYSTERESIS_MARGIN).
                Defaults to 0.0, which disables the dead zone and restores
                pure raw-sign behavior.

        Side effects:
            None.
        """
        self._line_start = line_start
        self._line_end = line_end
        self._stale_track_timeout = stale_track_timeout
        self._event_cooldown = event_cooldown
        self._hysteresis_margin = hysteresis_margin
        self._last_side: Dict[int, float] = {}
        self._last_seen_time: Dict[int, float] = {}
        self._last_event_time: Dict[int, float] = {}

    @property
    def line_start(self) -> Tuple[int, int]:
        """The virtual line's first endpoint actually in use, in pixels (see __init__)."""
        return self._line_start

    @property
    def line_end(self) -> Tuple[int, int]:
        """The virtual line's second endpoint actually in use, in pixels (see __init__)."""
        return self._line_end

    @staticmethod
    def compute_centroid(track: TrackedPerson) -> Tuple[int, int]:
        """
        Compute a tracked person's bounding-box centroid.

        Args:
            track: A TrackedPerson (see tracker.byte_tracker).

        Returns:
            (x, y) pixel coordinates of the box's center, in the same
            pixel space as the source frame.
        """
        return ((track.x1 + track.x2) // 2, (track.y1 + track.y2) // 2)

    def update(self, tracks: List[TrackedPerson]) -> List[CrossingEvent]:
        """
        Check every currently tracked person against the virtual line and
        emit an event for each one that just crossed it.

        Args:
            tracks: This frame's tracked people, as returned by
                tracker.byte_tracker.PersonTracker.track(). May be empty
                (e.g. a frame with no detections) - handled as a no-op.

        Returns:
            A list of CrossingEvent objects, one per track whose centroid
            moved from one side of the line to the other since the last
            call in which that track ID was seen, excluding any such
            crossing suppressed by the event cooldown (see class
            docstring's Robustness note). Empty if no (unsuppressed)
            crossings occurred this frame (the common case).

        Raises:
            Does not raise - a failure computing any single track's
            centroid/side is caught, logged as a warning, and that track
            is skipped for this frame rather than aborting the whole
            update.

        Side effects:
            Updates the internal per-track side/last-seen/last-event maps,
            evicts any track ID not seen for longer than
            `stale_track_timeout`, and logs a warning for any track that
            could not be processed.

        Performance considerations:
            O(number of tracks) for the crossing check, plus O(number of
            remembered track IDs) for stale-track eviction; pure
            arithmetic and dict operations, no I/O or model inference -
            negligible compared to detection/tracking cost.

        Note on missing/lost tracks:
            A track ID that stops appearing in `tracks` (person left the
            frame, or the track was lost) simply stops being updated; its
            last known side remains stored until either it reappears or
            `stale_track_timeout` elapses, at which point it is forgotten
            entirely. If ByteTrack later assigns that same physical person
            a *new* track ID (see the Phase 3 known limitation on long
            absences), this detector treats it as a brand-new track and
            requires two observations before it can report a crossing,
            exactly as it does for any new track ID.
        """
        events: List[CrossingEvent] = []
        now = time.time()

        for track in tracks:
            try:
                centroid = self.compute_centroid(track)
                signed_distance = _signed_side(centroid, self._line_start, self._line_end)
            except Exception as exc:
                logger.warning("Skipping line-crossing check for track %s: invalid centroid (%s)", track.track_id, exc)
                continue

            logger.debug(
                "Track %s centroid=%s signed_distance=%.1fpx margin=%.1fpx (line %s -> %s)",
                track.track_id, centroid, signed_distance, self._hysteresis_margin,
                self._line_start, self._line_end,
            )

            if abs(signed_distance) <= self._hysteresis_margin:
                # Inside the dead zone (or exactly on the line): too close
                # to the line to tell a real crossing apart from ordinary
                # centroid jitter. Deliberately leave this track's last
                # *confirmed* side untouched and emit nothing, so a person
                # loitering near the line cannot produce the repeated
                # ENTRY/EXIT/ENTRY bouncing that a raw sign test does.
                # Still counts as "seen" for staleness purposes.
                self._last_seen_time[track.track_id] = now
                continue

            # Outside the dead zone: the side is now unambiguous. Collapse
            # to +/-1 so what is stored is the confirmed side itself, not a
            # distance that later comparisons would have to re-interpret.
            side = 1.0 if signed_distance > 0 else -1.0

            previous_side = self._last_side.get(track.track_id)
            crossing_event_type = None

            if previous_side is not None and previous_side != 0:
                if previous_side < 0 < side:
                    crossing_event_type = CrossingEventType.ENTRY
                elif previous_side > 0 > side:
                    crossing_event_type = CrossingEventType.EXIT

            if crossing_event_type is not None:
                last_event_time = self._last_event_time.get(track.track_id)
                if last_event_time is None or (now - last_event_time) >= self._event_cooldown:
                    events.append(
                        CrossingEvent(
                            track_id=track.track_id,
                            event_type=crossing_event_type,
                            timestamp=now,
                            centroid=centroid,
                        )
                    )
                    self._last_event_time[track.track_id] = now
                else:
                    logger.debug(
                        "Suppressing %s for track %s: within %.2fs cooldown of the last event.",
                        crossing_event_type.value, track.track_id, self._event_cooldown,
                    )

            self._last_side[track.track_id] = side
            self._last_seen_time[track.track_id] = now

        self._prune_stale_tracks(now)

        return events

    def _prune_stale_tracks(self, now: float) -> None:
        """
        Forget any track ID not seen for longer than `stale_track_timeout`.

        Args:
            now: Current time (time.time()), passed in so every track
                checked within the same update() call is compared against
                one consistent timestamp.

        Side effects:
            Removes stale entries from `_last_side`, `_last_seen_time`,
            and `_last_event_time`.
        """
        stale_track_ids = [
            track_id
            for track_id, last_seen in self._last_seen_time.items()
            if (now - last_seen) >= self._stale_track_timeout
        ]
        for track_id in stale_track_ids:
            self._last_side.pop(track_id, None)
            self._last_seen_time.pop(track_id, None)
            self._last_event_time.pop(track_id, None)


class EventNotificationBoard:
    """
    Keeps recently generated crossing events visible for a configured
    duration, so a human watching the stream can see them fire.

    Purpose:
        Decouple "an event happened" (LineCrossingDetector.update(), fired
        for exactly one frame) from "showing it on screen for a moment"
        (which must persist across several frames to be readable).

    Lifecycle:
        Construct exactly once per application run, with the configured
        display duration. Call add_events() with each frame's newly
        generated events (if any), then get_visible_events() once per
        frame to retrieve what should currently be drawn. Both methods are
        safe to call every frame, including frames with no new events.

    Thread safety:
        Not thread-safe. Mutates an internal list without locking; must
        only be driven from the single frame-processing loop.

    Interaction with other classes:
        Consumes CrossingEvent objects produced by LineCrossingDetector.
        app.py owns one instance alongside its LineCrossingDetector and
        passes get_visible_events()'s result to draw_events().
    """

    def __init__(self, display_duration: float):
        """
        Args:
            display_duration: How long, in seconds, an event remains
                visible after it is added (see config.EVENT_DISPLAY_DURATION).
        """
        self._display_duration = display_duration
        self._active_events: List[CrossingEvent] = []

    def add_events(self, events: List[CrossingEvent]) -> None:
        """
        Register newly fired events so they become visible.

        Args:
            events: Events produced this frame by LineCrossingDetector.update()
                (may be empty).

        Side effects:
            Appends to the internal active-events list.
        """
        self._active_events.extend(events)

    def get_visible_events(self) -> List[CrossingEvent]:
        """
        Return the events that should currently be displayed, discarding
        any that have aged past the configured display duration.

        Returns:
            The still-visible CrossingEvent objects, oldest first.

        Side effects:
            Prunes expired events from the internal active-events list, so
            this list cannot grow without bound even if update() is called
            every frame for a long-running stream.

        Performance considerations:
            O(number of currently-active events); bounded by
            EVENT_DISPLAY_DURATION and typical crossing frequency, not by
            total stream length.
        """
        now = time.time()
        self._active_events = [
            event for event in self._active_events
            if (now - event.timestamp) < self._display_duration
        ]
        return self._active_events


def draw_line(frame, line_start: Tuple[int, int], line_end: Tuple[int, int], color, thickness: int):
    """
    Draw the configured virtual crossing line onto the frame.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        line_start: (x, y) pixel coordinates of the line's first endpoint.
        line_end: (x, y) pixel coordinates of the line's second endpoint.
        color: BGR color tuple for the line.
        thickness: Line thickness, in pixels.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via a single cv2.line call.
    """
    cv2.line(frame, line_start, line_end, color, thickness)
    return frame


def draw_events(
    frame,
    events: List[CrossingEvent],
    position: Tuple[int, int],
    line_spacing: int,
    font_scale: float,
    color,
    thickness: int,
    max_lines: int,
):
    """
    Draw up to `max_lines` recent crossing events as stacked text lines,
    formatted as "ENTRY : ID <n>" / "EXIT : ID <n>".

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        events: Currently-visible events, as returned by
            EventNotificationBoard.get_visible_events().
        position: (x, y) pixel position for the first line.
        line_spacing: Vertical gap, in pixels, between consecutive lines.
        font_scale: Font scale for the notification text.
        color: BGR color tuple for the notification text.
        thickness: Font stroke thickness, in pixels.
        max_lines: Maximum number of events to draw, oldest-first; any
            beyond this count are not drawn (they were still generated,
            logged, and returned to the caller - only the display is
            capped, to avoid stacking text off the bottom of the frame
            during a burst of crossings).

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via repeated cv2.putText calls.

    Performance considerations:
        O(min(len(events), max_lines)) simple drawing operations;
        negligible compared to detection/tracking cost.
    """
    for index, event in enumerate(events[:max_lines]):
        text = f"{event.event_type.value} : ID {event.track_id}"
        line_y = position[1] + (index * line_spacing)
        cv2.putText(
            frame,
            text,
            (position[0], line_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return frame
