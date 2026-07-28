"""Clip ring-buffer tests: the ring stores JPEG-encoded frames, never raw BGR.

A raw ring at default settings (30 fps x 90 s, 0.5-scale) holds multiple GB;
the encode-at-capture / decode-at-save contract keeps it ~20x smaller. The
real dxcam/opencv stack is optional, so a stub cv2 pins the contract on any
OS, and a second test does a real JPEG round-trip when opencv is installed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from kovadapt.capture import clips
from kovadapt.capture.clips import ClipRecorder


class _StubCV2:
    """Just enough of cv2 for the encode/decode/save_clip path."""

    IMWRITE_JPEG_QUALITY = 1
    IMREAD_COLOR = 1

    def __init__(self) -> None:
        self.store: dict[int, np.ndarray] = {}
        self.written: list[tuple[int, ...]] = []

    def imencode(self, ext, frame, params=None):
        assert ext == ".jpg"
        token = len(self.store)
        self.store[token] = frame.copy()
        return True, np.full(8, token, dtype=np.uint8)  # tiny "compressed" blob

    def imdecode(self, enc, flags):
        return self.store[int(enc[0])]

    def VideoWriter_fourcc(self, *chars):
        return 0

    def VideoWriter(self, path, fourcc, rate, size):
        stub = self

        class _W:
            def write(self, frame):
                stub.written.append(frame.shape)

            def release(self):
                pass

        return _W()


def test_ring_stores_encoded_frames_and_save_clip_decodes(monkeypatch, tmp_path):
    stub = _StubCV2()
    monkeypatch.setattr(clips, "cv2", stub)
    rec = ClipRecorder(fps=30, buffer_seconds=2)
    t0 = time.time()
    raw = np.zeros((54, 96, 3), dtype=np.uint8)
    for i in range(10):
        enc = clips._encode_frame(raw)          # exactly what the capture thread stores
        assert enc is not None
        assert enc.ndim == 1 and enc.nbytes < raw.nbytes   # encoded blob, not raw BGR
        rec._frames.append((t0 + i / 30, enc))
    out = rec.save_clip(t0 - 0.1, t0 + 1.0, tmp_path / "clip.mp4")
    assert out is not None
    # every buffered frame was decoded back to a full BGR frame before writing
    assert stub.written == [(54, 96, 3)] * 10
    cov = rec.coverage()
    assert cov is not None and cov[0] == pytest.approx(t0) and cov[1] > cov[0]


def test_jpeg_roundtrip_real_opencv(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    # clips.cv2 may be None when dxcam is missing even though opencv works;
    # encode/decode/save_clip only need opencv.
    monkeypatch.setattr(clips, "cv2", cv2)
    y = np.linspace(0, 255, 72, dtype=np.uint8)[:, None, None]
    x = np.linspace(0, 255, 128, dtype=np.uint8)[None, :, None]
    frame = np.broadcast_to((y // 2 + x // 2), (72, 128, 3)).astype(np.uint8).copy()

    enc = clips._encode_frame(frame)
    assert enc is not None and enc.ndim == 1
    assert enc.nbytes < frame.nbytes / 5        # actually compressed
    dec = clips._decode_frame(enc)
    assert dec is not None and dec.shape == frame.shape

    rec = ClipRecorder(fps=30, buffer_seconds=2)
    t0 = time.time()
    for i in range(12):
        rec._frames.append((t0 + i / 30, clips._encode_frame(frame)))
    out = rec.save_clip(t0, t0 + 1.0, tmp_path / "clip.mp4")
    assert out is not None and out.stat().st_size > 0
