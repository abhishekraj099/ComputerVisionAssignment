"""
Simple frame-time based FPS counter.

Kept as a small class instead of module-level globals so the app can
create one instance per stream and query it each frame.
"""

import time


class FPSCounter:
    """Tracks elapsed time between frames and reports a smoothed FPS value."""

    def __init__(self, smoothing: float = 0.9):
        """
        Args:
            smoothing: Exponential moving average factor (0-1). Higher values
                make the reported FPS more stable but slower to react to change.
        """
        self._smoothing = smoothing
        self._last_time = None
        self._fps = 0.0

    def tick(self) -> float:
        """
        Call once per processed frame.

        Returns:
            The current smoothed FPS estimate.
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
        return self._fps
