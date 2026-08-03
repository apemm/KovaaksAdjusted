"""Neural workstream tests: import guards, dataset builder, micro-train.

The synthetic trace builder below mirrors tests/test_telemetry.py's
TraceBuilder but is duplicated locally so this file stays self-contained.
Torch-dependent tests skip cleanly on a core (numpy-only) install; the
guard tests simulate a missing torch via sys.modules even when it IS
installed.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import kovadapt
import kovadapt.ml as ml_pkg
from kovadapt.ml.dataset import (
    N_SAMPLES,
    TARGET_NAMES,
    build_dataset,
    extract_samples,
    is_val_sample,
    iter_trace_files,
)
from kovadapt.analysis.report import RunReport, run_time_window
from kovadapt.ml.infer import FlickScore, summarize
from kovadapt.ml.shadow import DifficultyShadowPolicy, ShadowSuggestion
from kovadapt.ml.train import MIN_FLICKS, TrainResult
from kovadapt.stats.parser import parse_stats_csv
from kovadapt.telemetry.trace import MouseTrace
from kovadapt.watcher import SessionWatcher

from test_watcher import BASE, make_kovaaks_tree, make_settings, write_stats_csv

needs_torch = pytest.mark.skipif(not ml_pkg.ML_AVAILABLE, reason="torch not installed")

RATE = 1000.0  # synthetic packet rate (Hz)


def _deltas(total: float, n: int) -> np.ndarray:
    """n integer deltas summing to ~total, bell-weighted (sin^2)."""
    w = np.sin(np.linspace(0.0, np.pi, n)) ** 2
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
    cum = np.round(np.cumsum(w * total))
    return np.diff(np.concatenate([[0.0], cum])).astype(np.int32)


class _Synth:
    """Minimal packet-stream builder: rest gaps, aimed moves, clicks."""

    def __init__(self, t0: float = 1000.0) -> None:
        self.t: list[float] = []
        self.dx: list[int] = []
        self.dy: list[int] = []
        self.clicks: list[float] = []
        self.clicks_up: list[float] = []
        self.now = t0

    def move(self, dx_total: float, dy_total: float, dur: float = 0.15) -> "_Synth":
        n = max(int(dur * RATE), 6)
        ts = self.now + np.arange(1, n + 1) / RATE
        self.t.extend(ts.tolist())
        self.dx.extend(_deltas(dx_total, n).tolist())
        self.dy.extend(_deltas(dy_total, n).tolist())
        self.now = float(ts[-1])
        return self

    def flick(self, dx: float, dy: float, dur: float = 0.15,
              overshoot: float = 0.0) -> "_Synth":
        self.now += 0.5  # rest gap separates aimed movements
        if overshoot > 0:
            self.move(dx * (1 + overshoot), dy * (1 + overshoot), dur * 0.8)
            self.move(-dx * overshoot, -dy * overshoot, dur * 0.2)
        else:
            self.move(dx, dy, dur)
        self.now += 0.005
        self.clicks.append(self.now)
        self.clicks_up.append(self.now + 0.06)
        return self

    def build(self) -> MouseTrace:
        return MouseTrace(
            t=np.asarray(self.t, dtype=np.float64),
            dx=np.asarray(self.dx, dtype=np.int32),
            dy=np.asarray(self.dy, dtype=np.int32),
            clicks=np.asarray(self.clicks, dtype=np.float64),
            clicks_up=np.asarray(self.clicks_up, dtype=np.float64),
        )


def make_trace(n_flicks: int, seed: int = 0, t0: float = 1000.0) -> MouseTrace:
    rng = np.random.default_rng(seed)
    b = _Synth(t0)
    for _ in range(n_flicks):
        ang = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(80.0, 400.0)
        over = float(rng.choice([0.0, 0.15, 0.3]))
        b.flick(amp * np.cos(ang), amp * np.sin(ang),
                dur=float(rng.uniform(0.10, 0.25)), overshoot=over)
    return b.build()


def write_traces(root: Path, n_traces: int = 2, flicks_per: int = 30) -> None:
    """Trace library shaped like <profile_dir>/traces/<slug>/<ts>.npz."""
    for i in range(n_traces):
        tr = make_trace(flicks_per, seed=i)
        tr.save(root / f"Scenario_{i}_Adaptive_" / f"2026-07-28T10-0{i}-00.npz")


# ------------------------------------------------------------- import guards
def test_ml_package_reports_unavailable_without_torch(monkeypatch):
    assert isinstance(ml_pkg.ML_AVAILABLE, bool)
    monkeypatch.setattr(kovadapt, "ml", ml_pkg)   # restore package attr after
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` -> ImportError
    monkeypatch.delitem(sys.modules, "kovadapt.ml", raising=False)
    reloaded = importlib.import_module("kovadapt.ml")
    assert reloaded.ML_AVAILABLE is False


def test_entry_points_degrade_without_torch(tmp_path, monkeypatch):
    from kovadapt.ml import dataset as dataset_mod
    from kovadapt.ml import infer as infer_mod
    from kovadapt.ml import train as train_mod

    monkeypatch.setitem(sys.modules, "torch", None)
    assert dataset_mod.build_dataset(tmp_path) is None
    assert train_mod.train(tmp_path, tmp_path / "ml") is None
    assert infer_mod.load_scorer(tmp_path) is None


def test_shadow_scaffold_is_torch_free_and_untrained(tmp_path):
    policy = DifficultyShadowPolicy(tmp_path)
    assert policy.propose({"ewma_accuracy": 0.9, "target_scale": 1.0}) is None
    assert policy.propose({}) is None  # tolerates missing keys
    p = policy.log_transition({"ts": "2026-07-28T00:00:00", "suggestion": None})
    assert p is not None and p.is_file()
    rec = json.loads(p.read_text().splitlines()[0])
    assert rec["schema"] == "shadow-v2"
    assert DifficultyShadowPolicy().log_transition({}) is None
    s = ShadowSuggestion(target_scale=1.0, movement=0.5, confidence=0.0, reason="stub")
    assert s.target_scale == 1.0


# ------------------------------------------------------------ dataset (numpy)
def test_extract_samples_shapes_and_targets():
    tr = make_trace(8, seed=1)
    x, y, kept = extract_samples(tr)
    assert len(kept) >= 6  # nearly all synthetic flicks survive segmentation
    assert x.shape == (len(kept), 3, N_SAMPLES) and x.dtype == np.float32
    assert y.shape == (len(kept), len(TARGET_NAMES)) and y.dtype == np.float32
    peaks = x[:, 0].max(axis=1)  # speed channel normalized by peak speed
    assert np.all(peaks <= 1.0 + 1e-5) and np.all(peaks > 0.5)
    assert np.all(np.abs(x[:, 1:]) <= 1.0 + 1e-5)  # unit-direction channels
    assert np.allclose(y[:, 0], [f.overshoot for f in kept], atol=1e-6)
    assert np.allclose(y[:, 1], [f.corrections for f in kept], atol=1e-6)
    assert np.allclose(y[:, 2], [np.log(max(f.duration, 1e-3)) for f in kept], atol=1e-5)
    assert np.allclose(y[:, 3], [np.log(max(f.amplitude, 1.0)) for f in kept], atol=1e-5)


def test_extract_samples_empty_trace():
    x, y, kept = extract_samples(MouseTrace())
    assert x.shape == (0, 3, N_SAMPLES) and y.shape == (0, len(TARGET_NAMES))
    assert kept == []


def test_val_split_is_deterministic_and_reasonable():
    flags = [is_val_sample("slug/2026-07-28T10-00-00", i) for i in range(500)]
    assert flags == [is_val_sample("slug/2026-07-28T10-00-00", i) for i in range(500)]
    frac = float(np.mean(flags))
    assert 0.10 < frac < 0.32  # ~20% by construction
    assert any(flags) and not all(flags)


def test_iter_trace_files_walks_and_sorts(tmp_path):
    write_traces(tmp_path / "traces", n_traces=2, flicks_per=2)
    files = iter_trace_files(tmp_path / "traces")
    assert len(files) == 2
    assert files == sorted(files, key=lambda p: str(p).lower())
    assert iter_trace_files(tmp_path / "nowhere") == []


# ------------------------------------------------------------- torch-only path
@needs_torch
def test_build_dataset_tensors_and_deterministic_split(tmp_path):
    import torch

    root = tmp_path / "traces"
    write_traces(root, n_traces=2, flicks_per=30)
    ds = build_dataset(root)
    assert ds is not None
    assert ds.n_traces == 2
    n_train, n_val = int(ds.x_train.shape[0]), int(ds.x_val.shape[0])
    assert n_train + n_val == ds.n_flicks and ds.n_flicks >= 50
    assert n_train > 0 and n_val > 0
    assert tuple(ds.x_train.shape[1:]) == (3, N_SAMPLES)
    assert tuple(ds.y_val.shape[1:]) == (len(TARGET_NAMES),)
    assert ds.x_train.dtype == torch.float32 and ds.y_train.dtype == torch.float32
    ds2 = build_dataset(root)
    assert torch.equal(ds.x_train, ds2.x_train) and torch.equal(ds.x_val, ds2.x_val)
    assert torch.equal(ds.y_train, ds2.y_train) and torch.equal(ds.y_val, ds2.y_val)


@needs_torch
def test_model_size_and_forward_shapes():
    import torch

    from kovadapt.ml.model import FlickEncoder, count_parameters

    m = FlickEncoder()
    n = count_parameters(m)
    assert 1_000_000 <= n <= 2_500_000  # the "bigger model": ~1-2M params
    m.eval()
    with torch.no_grad():
        pred, emb = m(torch.randn(5, 3, N_SAMPLES))
    assert tuple(pred.shape) == (5, 3) and tuple(emb.shape) == (5, 64)


@needs_torch
def test_micro_train_loss_decreases_and_checkpoint_lands(tmp_path):
    from kovadapt.ml.infer import load_scorer
    from kovadapt.ml.train import train

    root = tmp_path / "traces"
    write_traces(root, n_traces=2, flicks_per=40)  # ~80 flicks >= MIN_FLICKS
    state = tmp_path / "state"
    res = train(root, state / "ml", epochs=3, batch_size=16, lr=1e-3,
                seed=0, device="cpu")
    assert isinstance(res, TrainResult)
    assert res.checkpoint.is_file() and res.metadata.is_file()
    assert res.train_size + res.val_size >= MIN_FLICKS
    assert res.history["train_loss"][-1] < res.history["train_loss"][0]
    meta = json.loads(res.metadata.read_text())
    assert meta["n_flicks"] == res.train_size + res.val_size
    assert meta["params"] == res.params
    assert meta["device"] == "cpu" and meta["best_epoch"] == res.best_epoch
    assert set(meta["val_loss_per_head"]) == {"overshoot", "corrections", "log_duration"}

    scorer = load_scorer(state)
    assert scorer is not None
    scores = scorer.score(make_trace(6, seed=99))
    assert scores and all(len(s.embedding) == 64 for s in scores)
    assert all(np.isfinite(s.quality) and -3.0 <= s.quality <= 3.0 for s in scores)
    assert all(set(s.residual) == {"overshoot", "corrections", "log_duration"}
               for s in scores)
    assert load_scorer(tmp_path / "nowhere") is None  # no checkpoint -> None


@needs_torch
def test_train_raises_on_insufficient_data(tmp_path):
    from kovadapt.ml.train import train

    root = tmp_path / "traces"
    write_traces(root, n_traces=1, flicks_per=4)
    with pytest.raises(RuntimeError, match="not enough flick data"):
        train(root, tmp_path / "ml", epochs=1)


# ------------------------------------------------------------------------ CLI
def test_cli_train_without_torch_exits_with_install_hint(monkeypatch):
    from kovadapt import cli

    monkeypatch.setattr(ml_pkg, "ML_AVAILABLE", False)
    with pytest.raises(SystemExit) as ei:
        cli.main(["train"])
    assert "pip install kovadapt[ml]" in str(ei.value)


def test_cli_train_prints_summary(tmp_path, monkeypatch, capsys):
    import kovadapt.ml.train as train_mod
    from kovadapt import cli
    from kovadapt.config import Settings

    fake = TrainResult(
        checkpoint=tmp_path / "ml" / "flick_encoder.pt",
        metadata=tmp_path / "ml" / "flick_encoder.json",
        device="cpu", params=1_411_523, n_traces=2,
        train_size=64, val_size=16, epochs_run=3, best_epoch=2,
        train_loss=0.5, val_loss=0.61,
        val_loss_per_head={"overshoot": 0.7, "corrections": 0.6, "log_duration": 0.5},
    )
    calls: dict = {}

    def fake_train(traces_root, out_dir, **kw):
        calls["traces_root"], calls["out_dir"] = Path(traces_root), Path(out_dir)
        calls["kw"] = kw
        return fake

    monkeypatch.setattr(ml_pkg, "ML_AVAILABLE", True)
    monkeypatch.setattr(train_mod, "train", fake_train)
    s = Settings(profile_dir=str(tmp_path))
    monkeypatch.setattr(Settings, "load", classmethod(lambda cls, path=None: s))
    cli.main(["train", "--epochs", "5", "--seed", "7"])
    out = capsys.readouterr().out
    assert "80 flicks from 2 traces" in out
    assert "device:      cpu" in out
    assert "1,411,523 parameters" in out
    assert "checkpoint:" in out and "flick_encoder.pt" in out
    assert calls["traces_root"] == tmp_path / "traces"
    assert calls["out_dir"] == tmp_path / "ml"
    assert calls["kw"] == {"epochs": 5, "seed": 7}


# ------------------------------------------------------------ loop integration
def test_summarize_flick_scores():
    scores = [
        FlickScore(t_click=1.0, quality=0.5, residual={"overshoot": -0.5}),
        FlickScore(t_click=2.0, quality=-1.0, residual={"overshoot": 1.0}),
        FlickScore(t_click=3.0, quality=0.1, residual={"overshoot": 0.1}),
    ]
    d = summarize(scores)
    assert d["n_scored"] == 3
    assert d["worst_t_click"] == 2.0 and d["worst_quality"] == -1.0
    assert d["best_t_click"] == 1.0 and d["best_quality"] == 0.5
    assert d["mean_quality"] == pytest.approx((0.5 - 1.0 + 0.1) / 3, abs=1e-4)
    assert d["mean_residual"]["overshoot"] == pytest.approx(0.2, abs=1e-4)
    json.dumps(d)  # digest must be JSON-ready (it lands in RunReport.ml)
    assert summarize([]) == {}


def test_runreport_ml_field_roundtrips(tmp_path):
    rep = RunReport(scenario="s", started_iso="t", score=1.0, accuracy=0.5,
                    avg_ttk=0.1, kills=3, kps=1.0,
                    ml={"n_scored": 2, "mean_quality": 0.1})
    p = rep.save(tmp_path / "r.json")
    assert RunReport.load(p).ml == {"n_scored": 2, "mean_quality": 0.1}
    # pre-ml report JSON (no "ml" key) still loads — defaults to {} (the
    # JSON dataclass round-trip contract: new fields need defaults)
    d = json.loads(p.read_text())
    d.pop("ml")
    old = tmp_path / "old.json"
    old.write_text(json.dumps(d))
    assert RunReport.load(old).ml == {}


def test_watcher_stamps_ml_digest_and_shadow_log(tmp_path, monkeypatch):
    import kovadapt.ml.infer as infer_mod

    root = tmp_path / "kovaaks"
    make_kovaaks_tree(root, BASE)
    s = make_settings(root, tmp_path / "state")
    csv = write_stats_csv(s.stats_dir, BASE)
    t0, _t1 = run_time_window(parse_stats_csv(csv))
    trace = make_trace(6, seed=3, t0=t0 + 0.3)  # inside the run's window

    class FakeScorer:
        def score(self, tr, flicks):
            return [FlickScore(t_click=f.t_click, quality=0.25,
                               residual={"overshoot": -0.25}) for f in flicks]

    monkeypatch.setattr(infer_mod, "load_scorer", lambda profile_dir: FakeScorer())
    monkeypatch.setattr(SessionWatcher, "_run_trace", lambda self, run: trace)
    w = SessionWatcher(s, BASE, on_update=lambda m: None)
    w.process_run(csv)

    rep = w.last_report
    assert rep is not None and rep.n_flicks >= 1
    assert rep.ml["n_scored"] >= 1
    assert rep.ml["mean_quality"] == pytest.approx(0.25, abs=1e-6)
    # digest persisted in the saved report JSON
    saved = RunReport.load(
        s.profile_path / "reports" / "mini_test" / "2026-05-27T20-25-38.json"
    )
    assert saved.ml == rep.ml
    # one shadow transition per processed run, schema-tagged, plan captured,
    # profile state snapshotted BEFORE observe() (fresh profile -> 0 runs)
    log = s.profile_path / "ml" / "shadow_log.jsonl"
    assert log.is_file()
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["schema"] == "shadow-v2"
    assert rec["ts"] == "2026-05-27T20:25:38"
    assert rec["suggestion"] is None, "an untrained policy proposed something"
    # The REWARD. v1 specified an `outcome` key as "next-run outcome when
    # known" and hard-coded it to None, so every transition ever logged was a
    # state and an action with nothing to learn from. record[i]'s plan is
    # rewarded by record[i+1]["run_outcome"] — pair forward.
    assert "outcome" not in rec, "the always-null v1 field is back"
    got = rec["run_outcome"]
    assert set(got) == {"accuracy", "score", "kps"}, got
    assert all(isinstance(v, float) for v in got.values()), got
    assert got["accuracy"] == pytest.approx(parse_stats_csv(csv).accuracy, abs=1e-6), (
        "the logged reward is not this run's measured accuracy")
    assert rec["profile_state"]["run_count"] == 0
    assert rec["plan"]  # non-empty dict of the emitted AdaptationPlan


def test_watcher_without_checkpoint_leaves_ml_empty(tmp_path, monkeypatch):
    root = tmp_path / "kovaaks"
    make_kovaaks_tree(root, BASE)
    s = make_settings(root, tmp_path / "state")
    csv = write_stats_csv(s.stats_dir, BASE)
    t0, _t1 = run_time_window(parse_stats_csv(csv))
    trace = make_trace(4, seed=5, t0=t0 + 0.3)
    monkeypatch.setattr(SessionWatcher, "_run_trace", lambda self, run: trace)
    w = SessionWatcher(s, BASE, on_update=lambda m: None)
    w.process_run(csv)  # no checkpoint under the tmp profile dir
    assert w.last_report is not None
    assert w.last_report.ml == {}  # scorer None -> nothing stamped, no crash


def test_a_broken_shadow_log_says_so_instead_of_vanishing(tmp_path, monkeypatch):
    """`_log_shadow` swallows every exception on purpose — a logging failure
    must never touch the adaptation loop. But a bare `pass` is how a broken
    training set becomes invisible: `run.kills_per_second` is a method, not a
    property, so `float(run.kills_per_second or 0)` raised and every
    transition silently stopped being written, with the suite still green
    because the assertion on the old always-null field passed either way.
    """
    from kovadapt.ml import shadow as shadow_mod
    from kovadapt.watcher import SessionWatcher

    said: list[str] = []
    root = tmp_path / "kovaaks"
    make_kovaaks_tree(root, BASE)
    s = make_settings(root, tmp_path / "state")
    csv = write_stats_csv(s.stats_dir, BASE)

    def boom(self, record):
        raise RuntimeError("disk full")

    monkeypatch.setattr(shadow_mod.DifficultyShadowPolicy, "log_transition", boom)
    w = SessionWatcher(s, BASE, on_update=said.append)
    w.process_run(csv)

    assert any("shadow transition not logged" in m for m in said), (
        f"a failed shadow write left no trace at all: {said}")
    assert any("adaptation is unaffected" in m for m in said)
    # ...and the run itself still completed
    assert w.last_report is not None
