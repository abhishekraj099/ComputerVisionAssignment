"""
Fallback person tracker used only when tracker.byte_tracker cannot be
imported (i.e. torch fails to load on this machine - see app.py's import
guard).

Purpose:
    Provide a torch-free, OpenCV-only stand-in for PersonTracker so the
    rest of the pipeline (motion, posture, line-crossing, attendance,
    occupancy, and the dashboard) keeps working end-to-end - with degraded
    detection quality - on a machine where the real YOLOv8 + ByteTrack path
    is blocked by an OS-level Application Control policy, instead of the
    whole pipeline refusing to start.

    This is NOT a replacement for YOLOv8 + ByteTrack and is not meant to be
    accurate: it uses background subtraction (MOG2) to find moving-blob
    contours as "person" boxes, and a simple nearest-centroid match to keep
    IDs stable across frames. It has no notion of what a person looks like
    (a moving pet, chair being dragged, or shadow can be picked up as a
    "person"), and a person who stops moving long enough to be absorbed
    into the background model will stop being tracked until they move
    again.

Responsibilities:
    - Detect foreground blobs above a minimum area via
      cv2.createBackgroundSubtractorMOG2.
    - Assign persistent-ish integer IDs via nearest-centroid matching
      against the previous frame's tracks, within a maximum match distance.
    - Evict stale IDs (no match for FALLBACK_STALE_TRACK_TIMEOUT seconds) so
      memory does not grow unbounded across a long stream.
    - Expose the exact same TrackedPerson shape and draw_tracks() signature
      as tracker.byte_tracker, so app.py's load_person_tracker() can return
      this class instead with no other code changes required anywhere else
      in the pipeline.

What this module intentionally does NOT handle:
    - No YOLO, no torch, no ByteTrack, no re-identification after an
      occlusion - a person who leaves and re-enters frame gets a new ID.
    - No confidence score in any meaningful sense; `confidence` is always
      1.0 (there is no model score to report).

Which future modules will consume this module's output:
    Only app.py's load_person_tracker(), and only when
    tracker.byte_tracker's import fails there. Every other module (motion,
    posture, line-crossing, attendance, occupancy) already depends only on
    the List[TrackedPerson] shape, not on how it was produced, so none of
    them need to change.
"""

import time
from dataclasses import dataclass
from typing import Dict, List

import cv2

import config
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


@dataclass(frozen=True)
class TrackedPerson:
    """
    Same shape as tracker.byte_tracker.TrackedPerson (see that module for
    field docs) - duplicated here, rather than imported, because importing
    tracker.byte_tracker at all would re-trigger the blocked torch import
    this module exists to avoid.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    track_id: int


class FallbackPersonTracker:
    """
    OpenCV-only stand-in for tracker.byte_tracker.PersonTracker.

    Lifecycle:
        Construct exactly once per run (mirrors PersonTracker). Call
        track() once per frame, in frame order, for the life of the
        stream.

    Thread safety:
        Not thread-safe - identical constraint to PersonTracker, for the
        same reason (mutates internal state per call).
    """

    def __init__(
        self,
        bg_history: int,
        bg_var_threshold: int,
        min_contour_area: int,
        max_match_distance: float,
        stale_track_timeout: float,
    ):
        """
        Args:
            bg_history: Number of frames MOG2 uses to build its background
                model. Valid values: a positive integer.
            bg_var_threshold: MOG2's pixel-variance threshold for
                foreground/background classification. Valid values: a
                positive number; lower is more sensitive to small changes.
            min_contour_area: Minimum foreground contour area, in pixels,
                to be treated as a person rather than noise.
            max_match_distance: Maximum centroid distance, in pixels,
                allowed when matching a detection to a previous frame's
                track. Beyond this, a new ID is assigned instead.
            stale_track_timeout: Seconds a track ID may go unmatched before
                being evicted from internal state.
        """
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=bg_history, varThreshold=bg_var_threshold, detectShadows=True
        )
        self._min_contour_area = min_contour_area
        self._max_match_distance = max_match_distance
        self._stale_track_timeout = stale_track_timeout
        self._next_track_id = 1
        # track_id -> (centroid_x, centroid_y, last_seen_monotonic_time)
        self._active_tracks: Dict[int, tuple] = {}

    def track(self, frame) -> List[TrackedPerson]:
        """
        Detect foreground blobs and assign persistent-ish IDs.

        Args:
            frame: BGR image (numpy array) as read from OpenCV.

        Returns:
            A list of TrackedPerson objects, one per foreground blob above
            the configured minimum area. Empty if none are found.

        Side effects:
            Updates the internal background model and active-track state.
            Never raises - a bad frame simply yields an empty result,
            logged at WARNING level, consistent with PersonTracker's
            per-frame failure handling.
        """
        try:
            foreground_mask = self._subtractor.apply(frame)
            _, foreground_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception as exc:
            logger.warning("Fallback tracking failed on this frame: %s", exc)
            return []

        now = time.monotonic()
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w / 2.0, y + h / 2.0)
            detections.append((x, y, x + w, y + h, centroid))

        tracks: List[TrackedPerson] = []
        matched_ids = set()
        for x1, y1, x2, y2, centroid in detections:
            best_id = None
            best_distance = self._max_match_distance
            for track_id, (prev_x, prev_y, _) in self._active_tracks.items():
                if track_id in matched_ids:
                    continue
                distance = ((centroid[0] - prev_x) ** 2 + (centroid[1] - prev_y) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_id = track_id

            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1

            matched_ids.add(best_id)
            self._active_tracks[best_id] = (centroid[0], centroid[1], now)
            tracks.append(
                TrackedPerson(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), confidence=1.0, track_id=best_id)
            )

        stale_ids = [
            track_id
            for track_id, (_, _, last_seen) in self._active_tracks.items()
            if now - last_seen > self._stale_track_timeout
        ]
        for track_id in stale_ids:
            del self._active_tracks[track_id]

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
    """Identical drawing behavior to tracker.byte_tracker.draw_tracks() - see that module for docs."""
    for track in tracks:
        cv2.rectangle(frame, (track.x1, track.y1), (track.x2, track.y2), box_color, box_thickness)

        id_label = f"ID: {track.track_id}"
        confidence_label = "Person (fallback)"

        top_line_y = track.y1 - config.DETECTION_LABEL_GAP_ABOVE_BOX
        if top_line_y > config.DETECTION_LABEL_MIN_Y:
            confidence_line_y = top_line_y
            id_line_y = confidence_line_y - line_spacing
        else:
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
