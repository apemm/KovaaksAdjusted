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
