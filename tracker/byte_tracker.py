"""
Multi-object tracking for detected persons using ByteTrack.

Purpose:
    Assign a persistent track ID to each detected person so the same
    physical person keeps the same ID from frame to frame while they
    remain visible, using Ultralytics' built-in ByteTrack integration
    (Ultralytics YOLO's `.track()` API with `tracker="bytetrack.yaml"`).
    No custom tracking algorithm is implemented here - ByteTrack's
    association logic is used as-is.

Responsibilities:
    - Drive a single, shared YOLO model with `.track(..., persist=True)` so
      detection and ByteTrack association happen together, frame by frame.
    - Convert Ultralytics' internal result objects into a small, stable,
      framework-agnostic data type (TrackedPerson) the rest of the project
      can depend on.
    - Draw bounding boxes plus a two-line "ID: <n>" / "Person <confidence>"
      label for each tracked person.

Scope of the current phase (Phase 3):
    Detection + persistent-ID tracking only, for the "person" class.

What this module intentionally does NOT handle:
    - Attendance counting, entry/exit events, motion detection, blur
      detection, or occupancy state - all reserved for later phases.
    - Re-identifying a person who was gone long enough for ByteTrack's
      internal track buffer to expire; they will simply get a new ID.
    - Any UI/window management (app.py owns the display loop).

Which future modules will consume this module's output:
    Every later phase that needs "who is currently in frame, with a
    stable identity" - attendance, entry/exit, occupancy, seated/standing
    classification - is expected to consume the List[TrackedPerson]
    returned by PersonTracker.track(frame), exactly as app.py already
    does, rather than re-deriving detections or IDs themselves.

Architecture note: PersonTracker.track() takes a raw frame, not a list of
pre-computed detections, because Ultralytics' `.track()` performs YOLO
inference and ByteTrack association in one call over the same model
instance - there is no supported public API to hand it detections computed
separately without either running inference twice per frame or reaching
into ByteTrack's private/internal classes. Detection is therefore an
internal implementation detail of this module, not something callers need
to know about. Downstream code (this and future phases) should depend only
on the List[TrackedPerson] this module returns, not on how it was produced.
"""

from dataclasses import dataclass
from typing import List

import cv2
from ultralytics import YOLO

import config
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


@dataclass(frozen=True)
class TrackedPerson:
    """
    A single tracked person in one frame, with a persistent ID.

    All coordinate fields are in the source frame's pixel space (the same
    width/height as the frame passed to PersonTracker.track()), using
    OpenCV's convention of (0, 0) at the top-left corner. None of the
    fields are optional - a TrackedPerson is only ever constructed once
    ByteTrack has confirmed a stable ID for the underlying detection.

    Fields:
        x1: Left edge of the bounding box, in pixels (int, >= 0).
        y1: Top edge of the bounding box, in pixels (int, >= 0).
        x2: Right edge of the bounding box, in pixels (int, > x1).
        y2: Bottom edge of the bounding box, in pixels (int, > y1).
        confidence: Model confidence score for this detection, in the
            closed range 0.0-1.0.
        track_id: Positive integer assigned by ByteTrack. Stable across
            consecutive frames while the same person remains visible (or
            briefly occluded, within ByteTrack's internal track buffer);
            not reused within a run once a track ends, but a person who
            disappears for longer than that buffer will be assigned a new,
            different track_id on reappearance rather than their old one.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int


class PersonTracker:
    """
    Assigns persistent IDs to detected persons across frames.

    Purpose:
        Turn a stream of independent per-frame detections into a stream of
        identity-stable tracks, by driving Ultralytics' bundled ByteTrack
        integration on a shared YOLO model.

    Lifecycle:
        Construct exactly once per application run, after the YOLO model
        has already been loaded (typically via `detector.model` from an
        existing PersonDetector - see app.load_person_tracker()). Call
        track() once per frame, in frame order, for the life of the
        stream. The model instance must be reused call-to-call (with
        `persist=True`) so ByteTrack's internal track state carries over
        between frames - this class must therefore be created once and
        reused, never rebuilt per frame or per stream restart. There is no
        explicit close/shutdown method; tracker state is released along
        with the underlying model at process exit.

    Thread safety:
        Not thread-safe. track() mutates internal state (`_initialized`)
        and drives a shared, stateful YOLO/ByteTrack model instance; it
        must only ever be called sequentially from the single frame-
        processing loop, never concurrently from multiple threads.

    Interaction with other classes:
        Wraps the YOLO model owned by models.person_detector.PersonDetector
        (passed in via its `model` property) rather than loading its own
        copy, so the weights are only loaded once for the whole
        application. app.py constructs one PersonTracker per run and feeds
        its track() output to draw_tracks() and to the FPS/person-count
        overlay.
    """

    def __init__(
        self,
        model: YOLO,
        device: str,
        person_class_id: int,
        confidence_threshold: float,
        tracker_config: str,
    ):
        """
        Store the shared model and tracking configuration.

        Args:
            model: An already-loaded Ultralytics YOLO model (e.g. from
                PersonDetector.model), reused here so the weights are only
                loaded once for the whole application.
            device: "cpu" or "cuda", as resolved by the detector.
            person_class_id: COCO class ID to filter on (person = 0).
            confidence_threshold: Minimum confidence, in 0.0-1.0, to keep a
                tracked box.
            tracker_config: Name of the Ultralytics tracker config to use
                (e.g. "bytetrack.yaml").

        Side effects:
            None. Construction is pure bookkeeping - it does not touch the
            model or perform any inference; Ultralytics lazily initializes
            its internal ByteTrack tracker on the first call to track().

        Performance considerations:
            Cheap and instantaneous; safe to call once during application
            startup.
        """
        self._model = model
        self._device = device
        self._person_class_id = person_class_id
        self._confidence_threshold = confidence_threshold
        self._tracker_config = tracker_config
        self._initialized = False

    def track(self, frame) -> List[TrackedPerson]:
        """
        Run detection + ByteTrack association on a single frame.

        Args:
            frame: BGR image (numpy array) as read from OpenCV.

        Returns:
            A list of TrackedPerson objects, one per person with a
            confirmed track ID. Empty if none are found.

        Raises:
            RuntimeError: If tracking fails on the very first call, which
                indicates the tracker itself failed to initialize (e.g. a
                bad tracker_config) rather than a transient per-frame
                issue. Once track() has succeeded at least once, later
                failures are treated as recoverable (see Side effects).

        Side effects:
            If tracking fails on a frame *after* having succeeded at least
            once before, the failure is logged at WARNING level and an
            empty list is returned, so a single bad frame does not stop the
            live stream. On success, sets the internal `_initialized` flag.

        Performance considerations:
            Runs one full YOLOv8 forward pass plus ByteTrack association
            per call - the dominant per-frame cost of Phase 3. `persist=True`
            reuses ByteTrack's internal state rather than reinitializing it,
            which is required both for correct ID persistence and to avoid
            the overhead of rebuilding tracker state every frame.
        """
        try:
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=self._tracker_config,
                classes=[self._person_class_id],
                conf=self._confidence_threshold,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            if not self._initialized:
                raise RuntimeError(
                    f"Failed to initialize ByteTrack tracker (tracker_config='{self._tracker_config}'): {exc}"
                ) from exc
            logger.warning("Tracking failed on this frame: %s", exc)
            return []

        self._initialized = True

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        tracks: List[TrackedPerson] = []
        for box in boxes:
            if box.id is None:
                # ByteTrack has not confirmed a stable ID for this detection yet.
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            track_id = int(box.id[0])
            tracks.append(
                TrackedPerson(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=confidence, track_id=track_id,
                )
            )

        return tracks


def draw_tracks(
    frame,
    tracks: List[TrackedPerson],
    box_color,
    box_thickness: int,
    label_font_scale: float,
    label_font_thickness: int,
    line_spacing: int,
):
    """
    Draw a bounding box, track ID line, and confidence line for each
    tracked person onto the frame.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        tracks: Tracks returned by PersonTracker.track().
        box_color: BGR color tuple for the box and label text.
        box_thickness: Bounding box line thickness.
        label_font_scale: Font scale for the label text.
        label_font_thickness: Font thickness for the label text.
        line_spacing: Vertical gap, in pixels, between the ID line and the
            confidence line.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via OpenCV drawing calls (cv2.rectangle,
        cv2.putText). Does not read or write anything outside `frame`.

    Performance considerations:
        O(number of tracks) simple drawing operations; negligible compared
        to the tracking/inference cost in PersonTracker.track().
    """
    for track in tracks:
        cv2.rectangle(frame, (track.x1, track.y1), (track.x2, track.y2), box_color, box_thickness)

        id_label = f"ID: {track.track_id}"
        confidence_label = f"Person {track.confidence:.2f}"

        top_line_y = track.y1 - config.DETECTION_LABEL_GAP_ABOVE_BOX
        if top_line_y > config.DETECTION_LABEL_MIN_Y:
            # Enough room above the box: stack both lines above its top edge.
            confidence_line_y = top_line_y
            id_line_y = confidence_line_y - line_spacing
        else:
            # Box too close to the frame edge: stack both lines below its top edge instead.
            id_line_y = track.y1 + config.DETECTION_LABEL_FALLBACK_OFFSET_BELOW
            confidence_line_y = id_line_y + line_spacing

        cv2.putText(
            frame, id_label, (track.x1, id_line_y),
            cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, box_color, label_font_thickness, cv2.LINE_AA,
        )
        cv2.putText(
            frame, confidence_label, (track.x1, confidence_line_y),
            cv2.FONT_HERSHEY_SIMPLEX, label_font_scale, box_color, label_font_thickness, cv2.LINE_AA,
        )

    return frame
