"""Sensitivity input + the both-sided case: cm/360 math pinned, every
argument line carries live numbers and a kb citation, and the no-directive
discipline holds — evidence for both directions, or nothing (None)."""

from __future__ import annotations

import pytest

from kovadapt.analysis.report import RunReport
from kovadapt.analysis.sens import cm_per_360, sens_case
from kovadapt.config import Settings
from kovadapt.profile.player import PlayerProfile


def make_rep(**kw) -> RunReport:
    base = dict(
        scenario="X [Adaptive]", started_iso="2026-07-28T12:00:00",
        score=800.0, accuracy=0.90, avg_ttk=0.9, kills=40, kps=0.8,
        n_flicks=40, mean_flick_ms=180.0, overshoot_rate=0.1,
        mean_corrections=0.8,
        input_health={"polling_hz_est": 1000.0, "jitter_ms": 0.4},
    )
    base.update(kw)
    return RunReport(**base)


def make_prof(archetype: str = "clicking") -> PlayerProfile:
    p = PlayerProfile(scenario="X [Adaptive]")
    p.run_count = 12
    p.archetype = archetype
    return p


def settings(**kw) -> Settings:
    return Settings(kovaaks_root=".", profile_dir=".", **kw)


# ------------------------------------------------------------------- math
def test_cm_per_360_pinned():
    # 800 dpi x 1.0 sens: 360 / 0.022 = 16363.6 counts, / 800 dpi, x 2.54
    assert cm_per_360(800.0, 1.0) == pytest.approx(51.9545, abs=1e-3)
    # dpi and sens are interchangeable in the denominator
    assert cm_per_360(1600.0, 1.0) == pytest.approx(cm_per_360(800.0, 2.0))
    # halving sens doubles cm/360
    assert cm_per_360(800.0, 0.5) == pytest.approx(2 * 51.9545, abs=1e-2)


def test_cm_per_360_rejects_nonpositive():
    for dpi, sens in ((0.0, 1.0), (800.0, 0.0), (-800.0, 1.0)):
        with pytest.raises(ValueError):
            cm_per_360(dpi, sens)


# --------------------------------------------------------------- the case
def test_overshoot_chain_argues_lower():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    case = sens_case(rep, make_prof(), settings())
    assert case is not None and case.for_lower
    line = case.for_lower[0]
    assert "45%" in line and "kb:" in line and "Casiez" in line
    assert not case.for_higher       # nothing on this run argues for higher
    assert case.cm360 == pytest.approx(51.95, abs=0.01)


def test_fitts_slow_argues_higher():
    rep = make_rep(mean_flick_ms=310.0, fitts_slope_ms=180.0, overshoot_rate=0.05)
    case = sens_case(rep, make_prof(), settings())
    assert case is not None and case.for_higher and not case.for_lower
    line = " ".join(case.for_higher)
    assert "310" in line and "180" in line and "kb:" in line


def test_travel_clutching_argues_higher():
    # 200000 counts / 800 dpi * 2.54 = 635 cm over 40 kills = 15.9 cm/kill
    rep = make_rep(total_travel_counts=200000.0)
    case = sens_case(rep, make_prof(), settings())
    assert case is not None
    assert any("cm per kill" in ln and "Casiez" in ln for ln in case.for_higher)


def test_style_range_is_aimer7s_and_matches_archetype():
    # tracking: 51.95 cm/360 sits above Aimer7's 20-25 -> a HIGHER argument
    case = sens_case(make_rep(), make_prof("tracking"), settings())
    assert case is not None and case.style_range == ("tracking", 20.0, 25.0)
    assert any("Aimer7" in ln for ln in case.for_higher)
    # clicking at ~26 cm/360 (sens 2.0) is under his 30+ -> a LOWER argument
    case2 = sens_case(make_rep(), make_prof("clicking"), settings(game_sens=2.0))
    assert case2 is not None and any("Aimer7" in ln for ln in case2.for_lower)


def test_both_sides_at_once():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5,
                   total_travel_counts=200000.0)
    case = sens_case(rep, make_prof(), settings())
    assert case is not None and case.for_lower and case.for_higher


def test_cross_session_overshoot_strengthens_lower():
    from kovadapt.analysis.skill import fit_skill

    ents = [dict(scenario="X", started_iso=f"2026-07-01T{i:02d}:00:00",
                 score=800.0, accuracy=0.90, kps=0.8,
                 overshoot_rate=0.15 + 0.01 * i, mean_flick_ms=180.0,
                 mean_corrections=1.0, fitts_slope_ms=120.0)
            for i in range(20)]
    case = sens_case(make_rep(), make_prof(), settings(), trends=fit_skill(ents))
    assert case is not None
    assert any("saved runs" in ln for ln in case.for_lower)


# ------------------------------------------------------------- discipline
def test_every_line_carries_numbers_and_citation():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5,
                   total_travel_counts=200000.0)
    case = sens_case(rep, make_prof("tracking"), settings())
    assert case is not None and case.sources
    for ln in case.for_lower + case.for_higher + [case.neutral]:
        assert any(ch.isdigit() for ch in ln), ln
        assert "kb:" in ln, ln
    assert "contested" in case.neutral   # the stability rule stays contested


def test_thin_evidence_returns_none():
    # clean run, clicking archetype inside his 30+ range: no case at all
    assert sens_case(make_rep(), make_prof(), settings()) is None
    # too few flicks -> no microstructure claims -> still None
    rep = make_rep(n_flicks=3, overshoot_rate=0.6, mean_corrections=3.0)
    assert sens_case(rep, make_prof(), settings()) is None


def test_unconfigured_dpi_or_sens_returns_none():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    assert sens_case(rep, make_prof(), settings(mouse_dpi=0.0)) is None
    assert sens_case(rep, make_prof(), settings(game_sens=0.0)) is None


def test_bad_input_health_suppresses_microstructure_arguments():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5,
                   input_health={"polling_hz_est": 1000.0, "jitter_ms": 5.0})
    assert sens_case(rep, make_prof(), settings()) is None


def test_deterministic():
    rep = make_rep(overshoot_rate=0.45, mean_corrections=2.5)
    assert sens_case(rep, make_prof(), settings()) == \
        sens_case(rep, make_prof(), settings())


# ------------------------------------------------- the flick amplitude floor
def test_the_flick_floor_is_an_angle_and_follows_sens_not_dpi():
    """The floor means "big enough that the overshoot ratio measures aim" —
    which is an ANGLE. The angle a mouse count is worth is
    YAW_DEG_PER_COUNT * sens.

    NOT DPI. A count is a count: DPI decides how many counts a centimetre of
    desk produces, not how far the view turns for one of them. Getting that
    backwards would give two players at the same sensitivity different floors
    for no reason.
    """
    from types import SimpleNamespace

    from kovadapt.analysis.movement import MIN_FLICK_COUNTS, MIN_FLICK_DEG
    from kovadapt.analysis.sens import YAW_DEG_PER_COUNT, min_flick_counts

    for sens in (0.4, 1.0, 2.5):
        counts = min_flick_counts(SimpleNamespace(game_sens=sens, mouse_dpi=800))
        assert counts * YAW_DEG_PER_COUNT * sens == pytest.approx(MIN_FLICK_DEG), (
            f"sens {sens} does not resolve to {MIN_FLICK_DEG} degrees")

    # DPI must not move it
    at_800 = min_flick_counts(SimpleNamespace(game_sens=1.0, mouse_dpi=800))
    at_1600 = min_flick_counts(SimpleNamespace(game_sens=1.0, mouse_dpi=1600))
    assert at_800 == at_1600, "the floor moved with DPI, which cannot affect an angle"

    # unconfigured falls back to the sens-1.0 reference rather than no floor
    assert min_flick_counts(SimpleNamespace(game_sens=0)) == MIN_FLICK_COUNTS


def test_sub_degree_segmentation_artefacts_do_not_become_flicks():
    """Overshoot is a FRACTION of amplitude, so a mis-segmented onset — which
    underestimates amplitude — produces an enormous ratio from a tiny
    movement. On the real trace library at the old 15-count floor (0.33 deg),
    37 "flicks" between 30 and 60 counts carried a MEAN overshoot of 1.737 and
    a maximum of 17.4, against 0.030 for the 596 flicks above 120 counts.

    `directional_bias` takes a plain mean, so fifty of those outvoted six
    hundred real flicks and INVERTED the verdict — +0.041 (left weaker) at the
    old floor, -0.216 (right weaker) at 2 degrees. That verdict is what writes
    Left/RightStrafeTimeMult into the generated .sce.
    """
    import numpy as np

    from kovadapt.analysis.movement import MIN_FLICK_COUNTS, segment_flicks
    from test_telemetry import TraceBuilder

    # a clean 400-count flick, then a 40-count twitch that segments badly
    b = TraceBuilder(t0=1000.0)
    b.rest(0.4).move(0.16, 400.0, 0.0).click(0.02)
    b.rest(0.4).move(0.05, 40.0, 0.0).move(0.10, 600.0, 0.0).click(0.02)
    trace = b.build()

    kept = segment_flicks(trace)
    assert kept, "the real flick was thrown away too"
    amps = np.array([f.amplitude for f in kept])
    # an ABSOLUTE bound, not MIN_FLICK_COUNTS: asserting against the constant
    # makes the test move with it, so lowering the floor back to 15 passes.
    # 60 counts is 1.3 degrees at sens 1.0 — above the twitch, below the floor.
    assert (amps >= 60).all(), (
        f"a sub-degree movement was counted as a flick: {amps.round(1)}")
    assert MIN_FLICK_COUNTS >= 60, (
        f"the floor itself has dropped to {MIN_FLICK_COUNTS:.0f} counts, back "
        "into the band where overshoot measures segmentation error")


def test_the_page_and_the_profile_agree_about_what_a_flick_is(tmp_path):
    """The report writes the profile and the .sce; the Analysis page
    re-segments the same trace for its replay overlay. Given different floors
    they would mark flicks the page's own numbers never counted."""
    import inspect

    from kovadapt.analysis import report as report_mod
    from kovadapt.gui import analysis_view

    assert "min_amplitude" in inspect.signature(report_mod.build_report).parameters, (
        "build_report cannot be told the floor, so the caller cannot keep the "
        "two in step")
    src = inspect.getsource(analysis_view.AnalysisView.show_report)
    assert "min_flick_counts" in src, (
        "the Analysis page segments with the default floor while the report "
        "uses the sens-derived one")
