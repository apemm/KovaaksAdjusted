import numpy as np

from kovadapt.analysis.fatigue import SessionFatigueTracker


def _feed(tracker, badness_series, flick_ms=120.0):
    """badness enters via overshoot_rate; flick duration held constant."""
    state = tracker.state
    for b in badness_series:
        state = tracker.add_run(n_flicks=20, overshoot_rate=b, mean_flick_ms=flick_ms)
    return state


def test_stable_session_stays_fresh():
    t = SessionFatigueTracker(min_runs=5, sensitivity=1.0)
    rng = np.random.default_rng(0)
    state = _feed(t, 0.20 + rng.normal(0, 0.01, size=12))
    assert state.level == "fresh"
    assert state.score < 0.4


def test_declining_session_flags_fatigue():
    t = SessionFatigueTracker(min_runs=5, sensitivity=1.0)
    # overshoot rate doubles over the session — a strong, steady decline
    state = _feed(t, np.linspace(0.15, 0.45, 12))
    assert state.level == "fatigued"
    assert state.score >= 0.8
    assert "break" in state.message


def test_needs_min_runs_before_judging():
    t = SessionFatigueTracker(min_runs=5, sensitivity=1.0)
    state = _feed(t, [0.1, 0.5, 0.9])   # wild but too few runs
    assert state.level == "fresh"
    assert state.runs == 3


def test_low_telemetry_runs_ignored():
    t = SessionFatigueTracker(min_runs=3, sensitivity=1.0)
    for _ in range(10):
        state = t.add_run(n_flicks=2, overshoot_rate=0.9, mean_flick_ms=500.0)
    assert state.runs == 0
    assert state.level == "fresh"


def test_improving_session_never_fatigued():
    t = SessionFatigueTracker(min_runs=5, sensitivity=1.0)
    state = _feed(t, np.linspace(0.45, 0.15, 12))   # warming up
    assert state.level == "fresh"
    assert state.score == 0.0
