"""Two failure modes of the optional feature modules.

1. ``ClipRecorder.save_clip`` must not hand back a path when the mp4
   encoder never opened — the watcher records whatever path it gets into
   ``RunReport.clip_files`` and the Analysis tab then enables a Play
   button for a file that has no video in it.
2. ``FlickScorer`` must not divide by the target std that ``train.py``
   clamps to 1e-6: a head that was constant across the train split would
   otherwise turn every flick into a ~1e5 residual and a saturated -3
   quality, and that lands in the report JSON.

Both modules are optional-dependency guarded, so the clip tests drive a
stub cv2 (opencv is not needed, and a real one would rarely fail to open
on demand) and the scorer tests skip without torch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kovadapt.ml as ml_pkg
from kovadapt.capture import clips
from kovadapt.capture.clips import ClipRecorder
from kovadapt.ml.dataset import N_SAMPLES, RATE, TRAIN_HEADS
from kovadapt.ml.infer import FlickScorer, summarize

from test_telemetry import TraceBuilder

needs_torch = pytest.mark.skipif(not ml_pkg.ML_AVAILABLE, reason="torch not installed")


# ------------------------------------------------------------------- clips
class _StubCV2:
    """Enough of cv2 for save_clip, with a switchable encoder outcome.

    ``opened=False`` reproduces a build without an mp4v encoder: cv2 leaves
    the writer closed, every write() is a no-op, and some backends leave a
    0-byte file behind.
    """

    IMWRITE_JPEG_QUALITY = 1
    IMREAD_COLOR = 1

    def __init__(self, opened: bool = True, leaves_stub_file: bool = False) -> None:
        self.opened = opened
        self.leaves_stub_file = leaves_stub_file
        self.store: dict[int, np.ndarray] = {}
        self.written: list[tuple[int, ...]] = []
        self.released = 0

    def imencode(self, ext, frame, params=None):
        token = len(self.store)
        self.store[token] = frame.copy()
        return True, np.full(8, token, dtype=np.uint8)

    def imdecode(self, enc, flags):
        return self.store[int(enc[0])]

    def VideoWriter_fourcc(self, *chars):
        return 0

    def VideoWriter(self, path, fourcc, rate, size):
        stub = self
        if not stub.opened and stub.leaves_stub_file:
            Path(path).write_bytes(b"")

        class _W:
            def isOpened(self):
                return stub.opened

            def write(self, frame):
                if stub.opened:            # closed writers no-op, like cv2's
                    stub.written.append(frame.shape)

            def release(self):
                stub.released += 1

        return _W()


def _fill(rec: ClipRecorder, t0: float, n: int = 6) -> None:
    raw = np.zeros((36, 64, 3), dtype=np.uint8)
    for i in range(n):
        rec._frames.append((t0 + i / 30, clips._encode_frame(raw)))


def test_save_clip_returns_none_when_encoder_never_opened(monkeypatch, tmp_path):
    stub = _StubCV2(opened=False, leaves_stub_file=True)
    monkeypatch.setattr(clips, "cv2", stub)
    rec = ClipRecorder(fps=30, buffer_seconds=2)
    t0 = 1000.0
    _fill(rec, t0)

    path = tmp_path / "clip.mp4"
    assert rec.save_clip(t0, t0 + 1.0, path) is None
    assert stub.written == []               # nothing was ever encoded
    assert stub.released == 1               # the writer is still released
    assert not path.exists()                # and the 0-byte stub is cleaned up


def test_save_clip_leaves_a_non_empty_file_alone(monkeypatch, tmp_path):
    """Cleanup may only remove the empty stub this call created — never an
    existing clip at the same path."""
    stub = _StubCV2(opened=False)
    monkeypatch.setattr(clips, "cv2", stub)
    rec = ClipRecorder(fps=30, buffer_seconds=2)
    t0 = 1000.0
    _fill(rec, t0)

    path = tmp_path / "clip.mp4"
    path.write_bytes(b"previous clip")
    assert rec.save_clip(t0, t0 + 1.0, path) is None
    assert path.read_bytes() == b"previous clip"


def test_save_clip_returns_path_when_encoder_opens(monkeypatch, tmp_path):
    stub = _StubCV2(opened=True)
    monkeypatch.setattr(clips, "cv2", stub)
    rec = ClipRecorder(fps=30, buffer_seconds=2)
    t0 = 1000.0
    _fill(rec, t0)

    out = rec.save_clip(t0, t0 + 1.0, tmp_path / "clip.mp4")
    assert out is not None
    assert stub.written == [(36, 64, 3)] * 6


# ------------------------------------------------------------------- infer
class _ZeroModel:
    """Predicts the train mean for every head (standardized output 0) — what
    a head fitted on a constant target converges to."""

    def eval(self):
        return self

    def __call__(self, x):
        import torch

        n = int(x.shape[0])
        return torch.zeros((n, 3)), torch.zeros((n, 64))


#: Plausible train-split means for (overshoot, corrections, log_duration);
#: with _ZeroModel the prediction is exactly this, so residual = actual - MEAN.
MEAN = np.array([0.0, 0.0, float(np.log(0.15))])


def _scorer(std: list[float]) -> FlickScorer:
    return FlickScorer(
        model=_ZeroModel(),
        mean=MEAN,
        std=np.asarray(std, dtype=np.float64),
        heads=list(TRAIN_HEADS),
        n_samples=N_SAMPLES,
        rate=RATE,
    )


def _overshooting_trace():
    tb = TraceBuilder()
    for i in range(6):
        tb.flick(300 + 20 * i, 40, dur=0.14, overshoot=0.12)
    return tb.build()


@needs_torch
def test_degenerate_head_is_dropped_instead_of_exploding():
    """overshoot/corrections were constant in training (std at the 1e-6
    clamp); only log_duration carries a usable scale."""
    trace = _overshooting_trace()
    scores = _scorer([1e-6, 1e-6, 0.4]).score(trace)
    assert scores

    for s in scores:
        assert s.residual["overshoot"] == 0.0
        assert s.residual["corrections"] == 0.0
        assert abs(s.quality) < 3.0                     # not pinned at the clip
        # quality is the surviving head alone, not an average dragged to -3
        assert s.quality == pytest.approx(-s.residual["log_duration"], abs=1e-9)
        assert abs(s.residual["log_duration"]) < 10.0

    digest = summarize(scores)
    assert digest["mean_residual"]["overshoot"] == 0.0
    assert abs(digest["mean_quality"]) < 3.0


@needs_torch
def test_all_heads_degenerate_scores_as_no_signal():
    scores = _scorer([1e-6, 1e-6, 1e-6]).score(_overshooting_trace())
    assert scores
    assert all(s.quality == 0.0 for s in scores)
    assert all(v == 0.0 for s in scores for v in s.residual.values())


@needs_torch
def test_healthy_stds_still_score_in_z_units():
    """The masking must not disturb the normal path: with a zero-prediction
    model every residual is just actual / std."""
    std = np.array([0.05, 0.8, 0.3])
    scores = _scorer(list(std)).score(_overshooting_trace())
    assert scores

    for s in scores:
        z = np.array([(s.actual[h] - MEAN[j]) / std[j] for j, h in enumerate(TRAIN_HEADS)])
        for j, h in enumerate(TRAIN_HEADS):
            assert s.residual[h] == pytest.approx(z[j], rel=1e-9, abs=1e-9)
        assert s.quality == pytest.approx(float(np.clip(-z.mean(), -3.0, 3.0)), abs=1e-9)
    assert any(s.residual["overshoot"] > 0 for s in scores)   # the flicks do overshoot
