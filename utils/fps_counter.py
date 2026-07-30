"""
Simple frame-time based FPS counter.

Purpose:
    Measure and smooth the pipeline's actual processing rate so it can be
    displayed on screen (config.FPS_TEXT_POSITION), independent of the
    video source's own nominal frame rate.

Responsibilities:
    Track wall-clock time between successive tick() calls and turn that
    into an exponentially-smoothed frames-per-second estimate.

Scope of the current phase:
    Display-only measurement; unchanged since Phase 1. Used once per frame
    by app.run_pipeline().

What this module intentionally does NOT handle:
    Any notion of "real-time deadline" enforcement, frame dropping, or
    frame-rate limiting - it only measures and reports, it never
    influences pipeline timing.

Which future modules will consume this module's output:
    None beyond app.py; this is a leaf utility, not expected to be
    consumed by future detection/tracking/attendance modules.

Kept as a small class instead of module-level globals so the app can
create one instance per stream and query it each frame.
"""

import time


class FPSCounter:
    """
    Tracks elapsed time between frames and reports a smoothed FPS value.

    Lifecycle:
        Construct exactly once per stream (see app.run_pipeline()), before
        the frame loop starts. Call tick() exactly once per processed
        frame, in order, for the life of that stream. No explicit
        shutdown/reset method is provided; construct a new instance if a
        fresh measurement window is needed.

    Thread safety:
        Not thread-safe. `tick()` mutates internal state
        (`_last_time`, `_fps`) without locking; it must only be called
        sequentially from a single thread (the main frame loop).

    Interaction with other classes:
        Standalone; has no dependency on and is not depended on by any
        detection/tracking class. app.py holds one instance and reads
        `tick()`'s return value each frame for display via draw_fps().
    """

    def __init__(self, smoothing: float = 0.9):
        """
        Args:
            smoothing: Exponential moving average factor, in the range
                (0, 1). Higher values make the reported FPS more stable
                but slower to react to change; lower values react faster
                but are noisier. 0.9 is a reasonable default for a
                human-readable on-screen counter.

        Side effects:
            None.
        """
        self._smoothing = smoothing
        self._last_time = None
        self._fps = 0.0

    def tick(self) -> float:
        """
        Record that one frame was just processed and update the estimate.

        Call once per processed frame, immediately after the frame's work
        is done (or at a consistent point in the loop), so the interval
        measured reflects true per-frame processing time.

        Returns:
            The current smoothed FPS estimate. Returns 0.0 on the very
            first call (no prior timestamp to measure an interval from).

        Raises:
            Does not raise.

        Side effects:
            Updates internal timestamp and smoothed FPS state
            (`_last_time`, `_fps`) used by the next call and by the
            `fps` property.

        Performance considerations:
            O(1); a single `time.perf_counter()` call and a few
            arithmetic operations. Negligible compared to detection/
            tracking cost.
        """
        now = time.perf_counter()

        if self._last_time is None:
            self._last_time = now
            return self._fps

        elapsed = now - self._last_time
        self._last_time = now

        if elapsed > 0:
            instantaneous_fps = 1.0 / elapsed
            self._fps = (self._smoothing * self._fps) + ((1.0 - self._smoothing) * instantaneous_fps)

        return self._fps

    @property
    def fps(self) -> float:
        """The most recently computed smoothed FPS value (0.0 before the first tick())."""
        return self._fps
