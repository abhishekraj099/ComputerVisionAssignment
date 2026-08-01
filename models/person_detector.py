"""
Real-time person detection built on Ultralytics YOLOv8.

Purpose:
    Load a YOLOv8 model exactly once and run person-only object detection
    on individual video frames, converting Ultralytics' internal result
    objects into a small, stable, framework-agnostic data type
    (PersonDetection) the rest of the project can depend on.

Responsibilities:
    - Load YOLOv8 weights (auto-downloading them if missing) and pick a
      compute device (CPU or CUDA).
    - Run inference on a single BGR frame, filtered to the COCO "person"
      class only, above a configurable confidence threshold.
    - Draw plain bounding boxes + confidence labels for those detections.

Scope of the current phase (Phase 2):
    Detection only. This module has no concept of an object's identity
    across frames (that is Phase 3's job, see tracker/byte_tracker.py).

What this module intentionally does NOT handle:
    - Multi-object tracking / persistent IDs (tracker/byte_tracker.py).
    - Attendance, entry/exit, motion, blur, or occupancy logic - all
      reserved for later phases.
    - Any UI/window management (app.py owns the display loop).

Which future modules will consume this module's output:
    As of Phase 3, tracker/byte_tracker.py's PersonTracker consumes this
    module's *loaded model* (via the `PersonDetector.model` property) so
    detection and tracking share one YOLO instance instead of loading the
    weights twice. PersonDetector.detect()/draw_detections() themselves are
    not currently called by the live pipeline (app.py), since
    PersonTracker.track() performs detection internally as part of
    tracking - they remain here as a standalone, directly-testable
    detection-only API for any future use that needs boxes without IDs.
"""

from dataclasses import dataclass
from typing import List

import cv2
import torch
from ultralytics import YOLO

import config
from utils.logger import setup_logger

logger = setup_logger(__name__, config.LOG_DIR, config.LOG_FILE)


@dataclass(frozen=True)
class PersonDetection:
    """
    A single detected person in one frame.

    All coordinate fields are in the source frame's pixel space (the same
    width/height as the frame passed to PersonDetector.detect()), using
    OpenCV's convention of (0, 0) at the top-left corner. None of the
    fields are optional - every PersonDetection is fully populated.

    Fields:
        x1: Left edge of the bounding box, in pixels (int, >= 0).
        y1: Top edge of the bounding box, in pixels (int, >= 0).
        x2: Right edge of the bounding box, in pixels (int, > x1).
        y2: Bottom edge of the bounding box, in pixels (int, > y1).
        confidence: Model confidence score for this detection, in the
            closed range 0.0-1.0 (already filtered to be >= the configured
            confidence threshold).
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PersonDetector:
    """
    Loads a YOLOv8 model once and runs person-only inference on frames.

    Purpose:
        Own the one expensive resource in the detection path - the loaded
        YOLOv8 model - and expose a simple, repeatable per-frame detect()
        call on top of it.

    Lifecycle:
        Construct exactly once per application run (model loading is slow
        and, on first use, may download weights from the network). Call
        detect() as many times as needed afterwards, once per frame. There
        is no explicit close/shutdown method: the underlying model and any
        device memory it holds are released when this object is garbage
        collected at process exit.

    Thread safety:
        Not thread-safe. This class assumes a single caller (the app's one
        frame-processing loop) drives detect() sequentially, one frame at a
        time. Calling detect() concurrently from multiple threads on the
        same instance is unsupported and untested.

    Interaction with other classes:
        tracker.byte_tracker.PersonTracker wraps this class's loaded model
        (via the `model` property) so detection and ByteTrack association
        run through a single shared YOLO instance rather than two separate
        models. app.py constructs one PersonDetector per run and passes it
        into load_person_tracker() for that purpose.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float,
        person_class_id: int,
        device: str = "auto",
        iou_threshold: float = 0.7,
        image_size: int = 640,
    ):
        """
        Load the YOLOv8 model and resolve the compute device.

        Args:
            model_path: Path to YOLOv8 weights (.pt). Auto-downloaded by
                Ultralytics if the file does not exist yet.
            confidence_threshold: Minimum confidence, in 0.0-1.0, to keep a
                detection.
            person_class_id: COCO class ID to filter on (person = 0).
            device: "auto", "cpu", or "cuda". "auto" uses CUDA if available.
            iou_threshold: NMS IoU threshold for duplicate-box suppression,
                in 0.0-1.0 (see config.DETECTION_IOU_THRESHOLD). Defaults to
                Ultralytics' own default (0.7) if not given.
            image_size: Inference resolution, in pixels (must be a multiple
                of 32; see config.DETECTION_IMAGE_SIZE). Defaults to
                Ultralytics' own default (640) if not given.

        Raises:
            RuntimeError: If the model weights cannot be loaded (missing
                file with no network access, corrupted weights, unsupported
                file format, etc.). The original exception is chained via
                `from exc` for full traceback context.

        Side effects:
            Logs progress at INFO level before and after loading. May
            perform a one-time network download of the weights file if
            `model_path` does not already exist on disk.

        Performance considerations:
            This is the slow, one-time setup cost of the whole pipeline
            (model deserialization plus, optionally, a network download).
            It must only be called once per run - never per frame.
        """
        self._confidence_threshold = confidence_threshold
        self._person_class_id = person_class_id
        self._device = self._resolve_device(device)
        self._iou_threshold = iou_threshold
        self._image_size = image_size

        logger.info("Loading YOLOv8 model from '%s' on device '%s' ...", model_path, self._device)
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLOv8 model from '{model_path}': {exc}") from exc

        logger.info("YOLOv8 model loaded successfully.")

    @staticmethod
    def _resolve_device(device: str) -> str:
        """
        Resolve the "auto" device setting to a concrete "cpu" or "cuda".

        Args:
            device: "auto", "cpu", or "cuda", as passed to __init__.

        Returns:
            The unchanged value if it was already "cpu"/"cuda"; otherwise
            "cuda" if a CUDA-capable GPU is available, else "cpu".
        """
        if device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def device(self) -> str:
        """The resolved compute device ("cpu" or "cuda") inference runs on."""
        return self._device

    @property
    def model(self) -> YOLO:
        """The underlying loaded YOLO model, for reuse by other components (e.g. the tracker)."""
        return self._model

    def detect(self, frame) -> List[PersonDetection]:
        """
        Run person detection on a single frame.

        Args:
            frame: BGR image (numpy array) as read from OpenCV.

        Returns:
            A list of PersonDetection objects, one per detected person
            above the configured confidence threshold. Empty if none are
            found, or if inference fails on this frame (the failure is
            logged, not raised, so a single bad frame does not stop the
            live stream).

        Raises:
            Does not raise - inference errors are caught internally and
            reported as an empty result plus a logged warning.

        Side effects:
            Logs a warning if inference fails on this frame.

        Performance considerations:
            Runs one full YOLOv8 forward pass per call. This is the
            dominant per-frame cost of Phase 2; keep the model on the same
            device across calls (already guaranteed, since the device is
            fixed at construction time) to avoid repeated CPU/GPU transfer
            overhead.
        """
        try:
            results = self._model.predict(
                source=frame,
                classes=[self._person_class_id],
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            logger.warning("YOLO inference failed on this frame: %s", exc)
            return []

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None:
            return []

        detections: List[PersonDetection] = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            detections.append(
                PersonDetection(x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2), confidence=confidence)
            )

        return detections


def draw_detections(
    frame,
    detections: List[PersonDetection],
    box_color,
    box_thickness: int,
    label_font_scale: float,
    label_font_thickness: int,
):
    """
    Draw a green bounding box and a "Person <confidence>" label for each
    detection onto the frame.

    Args:
        frame: BGR image (numpy array) to annotate, modified in place.
        detections: Detections returned by PersonDetector.detect().
        box_color: BGR color tuple for the box and label.
        box_thickness: Bounding box line thickness.
        label_font_scale: Font scale for the label text.
        label_font_thickness: Font thickness for the label text.

    Returns:
        The same frame object passed in, for convenient chaining/inline
        use at the call site - no copy is made.

    Side effects:
        Mutates `frame` in place via OpenCV drawing calls (cv2.rectangle,
        cv2.putText). Does not read or write anything outside `frame`.

    Performance considerations:
        O(number of detections) simple drawing operations; negligible
        compared to the detection inference cost.
    """
    for detection in detections:
        cv2.rectangle(
            frame,
            (detection.x1, detection.y1),
            (detection.x2, detection.y2),
            box_color,
            box_thickness,
        )

        label = f"Person {detection.confidence:.2f}"
        label_y_above = detection.y1 - config.DETECTION_LABEL_GAP_ABOVE_BOX
        label_y = (
            label_y_above
            if label_y_above > config.DETECTION_LABEL_MIN_Y
            else detection.y1 + config.DETECTION_LABEL_FALLBACK_OFFSET_BELOW
        )
        cv2.putText(
            frame,
            label,
            (detection.x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            label_font_scale,
            box_color,
            label_font_thickness,
            cv2.LINE_AA,
        )

    return frame
