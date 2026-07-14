"""Screen clip capture: dxcam ring buffer -> mp4 segments for notable moments.

Optional feature. Requires Windows plus the `clips` extra:

    pip install kovadapt[clips]     # dxcam + opencv-python

Everything is import-guarded so the rest of kovadapt works without these
dependencies (and on non-Windows platforms for analysis/tests).

Design: a background thread grabs frames at `fps` into a fixed-size ring
buffer covering `buffer_seconds` of wall-clock time. When the analysis
pass flags notable moments, `save_clip(t0, t1, path)` slices the buffer by
epoch timestamps and encodes the segment. Memory: 1080p BGRA at 30 fps for
90 s is ~2.2 GB, so frames are stored downscaled (`scale`, default 0.5) and
as BGR — ~420 MB at defaults. Capture overhead is one desktop duplication
copy per frame; dxcam is the fastest Python option (Desktop Duplication API).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

try:  # pragma: no cover - Windows-only deps
    import dxcam  # type: ignore
    import cv2  # type: ignore

    CLIPS_AVAILABLE = True
except Exception:  # ImportError or dxcam's own platform errors
    dxcam = None
    cv2 = None
    CLIPS_AVAILABLE = False


class ClipRecorder:
    """Ring-buffer desktop recorder.

        rec = ClipRecorder(fps=30, buffer_seconds=90)
        rec.start()
        ... run happens ...
        rec.save_clip(t0, t1, "overshoot_1.mp4")   # epoch seconds
        rec.stop()

    Thread-safe: capture thread is the only writer; `save_clip` snapshots
    the deque under the lock, then encodes without holding it.
    """

    def __init__(
        self,
        fps: int = 30,
        buffer_seconds: float = 90.0,
        scale: float = 0.5,
        monitor: int = 0,
    ) -> None:
        self.fps = fps
        self.buffer_seconds = buffer_seconds
        self.scale = scale
        self.monitor = monitor
        maxlen = int(fps * buffer_seconds) + 8
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._camera = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not CLIPS_AVAILABLE:
            raise RuntimeError(
                "Clip capture requires Windows with the 'clips' extra: "
                "pip install kovadapt[clips]"
            )
        if self._thread is not None:
            return
        self._camera = dxcam.create(output_idx=self.monitor, output_color="BGR")
        if self._camera is None:
            raise RuntimeError("dxcam could not open the display")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kovadapt-clips", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover - Windows-only thread
        period = 1.0 / self.fps
        cam = self._camera
        while not self._stop.is_set():
            t0 = time.time()
            frame = cam.grab()  # None when the screen hasn't changed
            if frame is not None:
                if self.scale != 1.0:
                    frame = cv2.resize(
                        frame, None, fx=self.scale, fy=self.scale,
                        interpolation=cv2.INTER_AREA,
                    )
                with self._lock:
                    self._frames.append((t0, frame))
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    # ------------------------------------------------------------------
    def save_clip(self, t0: float, t1: float, path: Path | str) -> Path | None:
        """Encode buffered frames with timestamps in [t0, t1] to mp4.
        Returns the path, or None if the window isn't in the buffer."""
        if cv2 is None:
            return None
        with self._lock:
            frames = [(t, f) for t, f in self._frames if t0 <= t <= t1]
        if len(frames) < 2:
            return None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames[0][1].shape[:2]
        # actual achieved rate, so playback speed matches wall clock
        span = frames[-1][0] - frames[0][0]
        rate = max((len(frames) - 1) / span, 1.0) if span > 0 else float(self.fps)
        vw = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, (w, h)
        )
        try:
            for _, f in frames:
                vw.write(f)
        finally:
            vw.release()
        return path

    def coverage(self) -> tuple[float, float] | None:
        """Epoch window currently held in the buffer."""
        with self._lock:
            if not self._frames:
                return None
            return self._frames[0][0], self._frames[-1][0]
