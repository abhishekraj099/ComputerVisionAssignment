"""
Attendance state derived from virtual-line crossing events.

Purpose:
    Turn a stream of individual ENTRY/EXIT CrossingEvent objects (from
    events.line_crossing.LineCrossingDetector) into running attendance
    statistics: how many people are currently inside, how many entries and
    exits have happened in total, and how many unique people have ever
    entered.

Responsibilities:
    - Consume CrossingEvent objects, one batch per frame.
    - Maintain the set of track IDs currently "inside" (i.e. have entered
      and not yet exited) - the single source of truth for "current
      people".
    - Maintain the set of track IDs that have ever entered - the single
      source of truth for "unique visitors".
    - Maintain running totals of accepted ENTRY and EXIT events.
    - Reject (log and ignore, never raise) any event that would make
      attendance state inconsistent: a duplicate ENTRY for someone already
      inside, or an EXIT for someone not currently inside.
    - Return an immutable AttendanceStatistics snapshot after processing.
    - Draw a small "Attendance" panel summarizing that snapshot.

Scope of the current phase (Phase 5):
    Attendance bookkeeping and its on-screen summary panel only.

What this module intentionally does NOT handle:
    - No line-crossing detection of any kind. AttendanceManager never
      computes a centroid, a line side, or a crossing itself - it only
      consumes CrossingEvent objects already produced by
      events.line_crossing.LineCrossingDetector, which remains the sole
      source of truth for when a crossing occurred.
    - No motion detection, blur detection, occupancy state, seated/
      standing classification, face recognition, or person
      re-identification - all reserved for later phases (or out of scope
      entirely).
    - No persistence: statistics are held in memory only and are not
      written to a file, database, or CSV.
    - No UI beyond the one summary panel described below; no dashboard.

Which future modules will consume this module's output:
    Any future phase that needs a point-in-time attendance summary (e.g.
    an occupancy-state module deciding "occupied" vs "empty") is expected
    to read the AttendanceStatistics returned by
    AttendanceManager.update(), rather than re-deriving counts from raw
    CrossingEvent or TrackedPerson data itself.

Why attendance is a separate module from line-crossing detection:
    LineCrossingDetector answers a purely geometric question - "did this
    track's centroid just cross this line, and in which direction?" - and
    knows nothing about what an ENTRY or EXIT *means* for a running count.
    AttendanceManager answers a purely bookkeeping question - "given that
    a crossing happened, what should the running totals now be?" - and
    knows nothing about centroids, lines, or frames. Keeping them separate
    means each has one reason to change: a future change to how crossings
    are detected (e.g. a different line shape) cannot affect attendance
    bookkeeping, and a future change to attendance rules (e.g. capacity
    limits) cannot affect crossing detection.
"""

from dataclasses import dataclass
from typing import FrozenSet, List, Optional

import cv2

import config
from events.line_crossing import CrossingEvent, CrossingEventType
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


@dataclass(frozen=True)
class AttendanceStatistics:
    """
    An immutable snapshot of attendance state at one point in time.

    None of the fields are optional - a fresh AttendanceStatistics (all
    zeros/empty) is returned even before any CrossingEvent has ever been
    processed.

    Fields:
        current_people: Number of people currently inside (entered but not
            yet exited). Always equal to len(inside_track_ids); never
            tracked independently, so it can never drift out of sync with
            that set. Non-negative integer.
        total_entries: Total number of ENTRY events accepted so far (i.e.
            excluding duplicate/ignored ENTRYs). Monotonically
            non-decreasing for the life of the AttendanceManager instance.
        total_exits: Total number of EXIT events accepted so far (i.e.
            excluding duplicate/ignored EXITs). Monotonically
            non-decreasing for the life of the AttendanceManager instance.
        unique_visitors: Total number of distinct track IDs that have ever
            had an accepted ENTRY. Always equal to len(visited_track_ids);
            monotonically non-decreasing (a person who exits and
            re-enters under the same track ID does not increase this
            again).
        inside_track_ids: The exact set of track IDs currently inside, as
            an immutable frozenset (a defensive copy - mutating it has no
            effect on the AttendanceManager's internal state). Its length
            is always `current_people`.
    """

    current_people: int
    total_entries: int
    total_exits: int
    unique_visitors: int
    inside_track_ids: FrozenSet[int]


class AttendanceManager:
    """
    Maintains running attendance state from a stream of CrossingEvents.

    Purpose:
        Be the single place that decides what an ENTRY or EXIT event means
        for "how many people are inside" and "how many unique people have
        ever entered," so no other module needs its own copy of this
        bookkeeping.

    Lifecycle:
        Construct exactly once per application run (see
        app.load_attendance_manager()), before the frame loop starts. Call
        update() once per frame, passing that frame's CrossingEvent list
        (which is very often empty - see Note below). Internal state
        accumulates across calls for the life of the instance; there is no
        reset method, since attendance for a run is meant to persist for
        that entire run.

    Thread safety:
        Not thread-safe. update() mutates internal sets/counters without
        locking; it must only be called sequentially from the single
        frame-processing loop.

    Interaction with other classes:
        Consumes events.line_crossing.CrossingEvent objects (does not
        import or depend on LineCrossingDetector itself, only its output
        type - see the module docstring's "Why attendance is a separate
        module" note). Never computes a crossing itself. app.py constructs
        one instance per run and feeds it the same `crossing_events` list
        already produced by LineCrossingDetector.update() each frame.

    Note on calling update() every frame:
        Most frames produce no CrossingEvents at all (nobody crossed the
        line that frame). Calling update() with an empty list is a
        deliberate, safe no-op: it still returns the current
        AttendanceStatistics snapshot, so the caller can always draw an
        up-to-date attendance panel regardless of whether anything changed
        this frame.
    """

    def __init__(self):
        """
        Side effects:
            None. Starts with zero entries/exits/visitors and nobody
            inside.
        """
        self._inside_track_ids = set()
        self._visited_track_ids = set()
        self._total_entries = 0
        self._total_exits = 0

    def update(self, events: List[CrossingEvent]) -> AttendanceStatistics:
        """
        Process this frame's crossing events and return the current
        attendance snapshot.

        Args:
            events: This frame's CrossingEvent list, as returned by
                events.line_crossing.LineCrossingDetector.update(). May be
                empty (the common case) - handled as a no-op. May contain
                malformed entries (e.g. None, or an object missing
                `track_id`) - each such entry is skipped individually.

        Returns:
            The current AttendanceStatistics, reflecting every previously
            accepted event plus any accepted in this call. Returned even
            when `events` is empty or every event in it was rejected.

        Raises:
            Does not raise - processing any single event that fails
            unexpectedly is caught, logged as a warning, and that event is
            skipped rather than aborting the whole update.

        Side effects:
            For each accepted ENTRY: adds the track ID to the "inside" set
            (and to the "ever visited" set if new), increments
            total_entries. For each accepted EXIT: removes the track ID
            from the "inside" set, increments total_exits. For each
            rejected event (duplicate ENTRY, EXIT without a matching
            ENTRY, or malformed event): no state change. Logs one line per
            accepted/ignored event, plus one summary line if at least one
            event was accepted this call.

        Performance considerations:
            O(number of events); simple set/counter operations, no I/O or
            model inference - negligible compared to detection/tracking/
            line-crossing cost.
        """
        any_accepted = False

        for event in events:
            try:
                track_id = self._extract_track_id(event)
            except Exception as exc:
                logger.warning("Skipping malformed crossing event (%s): %s", event, exc)
                continue

            if track_id is None:
                logger.warning("Skipping crossing event with a missing track_id: %s", event)
                continue

            if event.event_type == CrossingEventType.ENTRY:
                if self._accept_entry(track_id):
                    any_accepted = True
            elif event.event_type == CrossingEventType.EXIT:
                if self._accept_exit(track_id):
                    any_accepted = True
            else:
                logger.warning(
                    "Skipping crossing event for track %s with unrecognized event_type: %s",
                    track_id, event.event_type,
                )

        if any_accepted:
            logger.info(
                "Attendance updated: current=%d, entries=%d, exits=%d, unique=%d",
                len(self._inside_track_ids), self._total_entries,
                self._total_exits, len(self._visited_track_ids),
            )

        return self._snapshot()

    @staticmethod
    def _extract_track_id(event: Optional[CrossingEvent]) -> Optional[int]:
        """
        Defensively read `track_id` off a possibly-malformed event.

        Args:
            event: A CrossingEvent, or possibly None/malformed input.

        Returns:
            The event's track_id, or None if `event` is None or has no
            `track_id` attribute.
        """
        if event is None:
            return None
        return getattr(event, "track_id", None)

    def _accept_entry(self, track_id: int) -> bool:
        """
        Apply an ENTRY event for one track ID, if valid.

        Args:
            track_id: The track ID that generated the ENTRY event.

        Returns:
            True if the entry was accepted (state changed); False if it
            was ignored because that track ID is already inside.

        Side effects:
            On acceptance: adds `track_id` to the inside set (and to the
            visited set, if new), increments total_entries. Logs the
            outcome either way.
        """
        if track_id in self._inside_track_ids:
            logger.info("ENTRY ignored for track %s: already inside.", track_id)
            return False

        self._inside_track_ids.add(track_id)
        self._total_entries += 1
        is_new_visitor = track_id not in self._visited_track_ids
        self._visited_track_ids.add(track_id)

        logger.info("ENTRY accepted for track %s (new unique visitor: %s).", track_id, is_new_visitor)
        return True

    def _accept_exit(self, track_id: int) -> bool:
        """
        Apply an EXIT event for one track ID, if valid.

        Args:
            track_id: The track ID that generated the EXIT event.

        Returns:
            True if the exit was accepted (state changed); False if it
            was ignored because that track ID is not currently inside
            (covers both "duplicate EXIT" and "EXIT without a prior
            ENTRY" - both are the same case: the track ID is not in the
            inside set).

        Side effects:
            On acceptance: removes `track_id` from the inside set,
            increments total_exits. Logs the outcome either way. Never
            allows `current_people` (len(inside_track_ids)) to go
            negative, since a track ID can only be removed if it was
            already present.
        """
        if track_id not in self._inside_track_ids:
            logger.info("EXIT ignored for track %s: not currently inside.", track_id)
            return False

        self._inside_track_ids.discard(track_id)
        self._total_exits += 1

        logger.info("EXIT accepted for track %s.", track_id)
        return True

    def _snapshot(self) -> AttendanceStatistics:
        """
        Build an immutable AttendanceStatistics snapshot of current state.

        Returns:
            A new AttendanceStatistics with `current_people` and
            `inside_track_ids` both derived from the same internal set at
            this instant, so they cannot disagree with each other.
        """
        return AttendanceStatistics(
            current_people=len(self._inside_track_ids),
            total_entries=self._total_entries,
            total_exits=self._total_exits,
            unique_visitors=len(self._visited_track_ids),
            inside_track_ids=frozenset(self._inside_track_ids),
        )


def draw_attendance_panel(
    frame,
    statistics: AttendanceStatistics,
    position,
    line_spacing: int,
    font_scale: float,
    color,
    thickness: int,
):
    """
    Draw a small "Attendance" summary panel onto the frame.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        statistics: The AttendanceStatistics to display, as returned by
            AttendanceManager.update().
        position: (x, y) pixel position for the panel's first ("Attendance")
            line.
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
        compared to detection/tracking/line-crossing cost.
    """
    lines = [
        "Attendance",
        f"Current Inside : {statistics.current_people}",
        f"Entries : {statistics.total_entries}",
        f"Exits : {statistics.total_exits}",
        f"Unique Visitors : {statistics.unique_visitors}",
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
