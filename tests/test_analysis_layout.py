"""Analysis page layout + takeaways (offscreen QPA).

Pins the reading order the page was rebuilt around — headline numbers, then
charts, then a folded Coach — the reads those numbers carry, and the rule
that every chart title is derived from the data on screen and falls back to
its neutral descriptor the moment that data stops supporting a claim.
Skipped wholesale without PySide6/pyqtgraph.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no system font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from test_telemetry import TraceBuilder  # noqa: E402

from kovadapt.analysis.insights import generate_insights  # noqa: E402
from kovadapt.analysis.report import RunReport  # noqa: E402
from kovadapt.config import Settings  # noqa: E402
from kovadapt.gui import theme, viz  # noqa: E402
from kovadapt.gui.analysis_view import (  # noqa: E402
    _BIAS_TITLE,
    _COACH_FOLD,
    _DEFICIT_TITLE,
    _TRAVEL_TITLE,
    AnalysisView,
    _bias_title,
    _deficit_title,
    _travel_title,
    _trend_title,
)
from kovadapt.profile.player import PlayerProfile, RegionPosterior  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never touch the developer's real ~/.kovadapt: profile loading and any
    settings save under this view resolve through Path.home()."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def settings(tmp_path):
    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    (root / "Saved" / "SaveGames" / "Scenarios").mkdir(parents=True)
    return Settings(
        kovaaks_root=str(root),
        profile_dir=str(tmp_path / "prof"),
        telemetry_enabled=False,
        onboarding_done=True,
    )


def _report(**over) -> RunReport:
    base = dict(
        scenario="Beta 1wall Click", started_iso="2026-07-28T10:00:00",
        score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
        summary_text="30 kills at 61% accuracy.")
    base.update(over)
    return RunReport(**base)


def _profile(**over) -> PlayerProfile:
    prof = PlayerProfile(scenario="Beta 1wall Click [Adaptive]")
    prof.run_count = 6
    prof.ewma_kps = 1.0
    prof.history = [{"accuracy": 0.61, "kps": 1.0, "score": 400.0} for _ in range(6)]
    for key, val in over.items():
        setattr(prof, key, val)
    return prof


# ------------------------------------------------------------ reading order
def test_numbers_lead_charts_follow_coach_lands_last(qapp, settings):
    """The complaint this page was rebuilt for: four dense Coach cards filled
    the first screen and every chart sat below the fold."""
    view = AnalysisView(settings)
    lay = view.layout()
    assert lay.indexOf(view.kpi_strip) < lay.indexOf(view.charts)
    assert lay.indexOf(view.charts) < lay.indexOf(view.detail)
    assert lay.indexOf(view.detail) < lay.indexOf(view.coach_box)
    # the 1400px column is spent sideways, not stacked in a narrow rail
    assert view.charts.orientation() == Qt.Horizontal
    assert view.charts.count() == 2
    assert view.detail.orientation() == Qt.Horizontal
    assert len(view.kpis) == 4
    view.deleteLater()


def test_kpi_strip_reads_the_run_in_mono(qapp, settings):
    view = AnalysisView(settings)
    view.show_report(
        _report(n_flicks=24, mean_flick_ms=180.0, overshoot_rate=0.40,
                mean_corrections=2.4),
        profile=_profile())

    assert view.kpis["accuracy"].value.text() == "61%"
    assert view.kpis["accuracy"].unit.text() == "hit rate"
    assert view.kpis["accuracy"].read.text() == "below-band"   # band is 85-95%
    assert view.kpis["kills"].value.text() == "30"
    assert view.kpis["kills"].read.text() == "24 flicks"       # evidence behind the charts
    assert view.kpis["pace"].value.text() == "1.40"
    assert view.kpis["pace"].unit.text() == "kills/s"
    assert view.kpis["pace"].read.text() == "faster"           # 1.40 vs the 1.00 EWMA
    assert view.kpis["flick"].value.text() == "180"
    assert view.kpis["flick"].read.text() == "repaired"        # 40% overshoot, 2.4 fixes

    for tile in view.kpis.values():
        assert tile.toolTip()                                  # cite everything
        assert tile.value.font().family() == theme.mono_family()
        assert tile.value.font().pixelSize() in theme.CELL_SIZES
        # the sheet, not setFont, is what survives theme.py's app-wide
        # `* { font-family: "Segoe UI"; font-size: 13px }`
        assert theme.mono_family() in tile.value.styleSheet()
        assert "font-size: 24px" in tile.value.styleSheet()
    view.deleteLater()


def test_kpi_reads_state_their_baseline_when_they_have_none(qapp, settings):
    """No profile history: the pace tile must say so, not invent a comparison."""
    view = AnalysisView(settings)
    view.show_report(_report(), profile=PlayerProfile(scenario="cold"))
    assert view.kpis["pace"].read.text() == "no-baseline"
    # observe_run seeds every EWMA to the first run's own value, so a
    # baseline only exists from run 2 — before that the tile must say why
    # rather than compare the run against itself and call it "steady".
    tip = view.kpis["pace"].toolTip()
    assert "baseline" in tip and "seeded from the first run" in tip
    assert view.kpis["kills"].read.text() == "no-telemetry"
    assert view.kpis["flick"].value.text() == "—"
    assert view.kpis["flick"].read.text() == "thin-data"
    view.deleteLater()


@pytest.mark.parametrize("accuracy,read", [(0.98, "above-band"),
                                           (0.90, "in-band"),
                                           (0.50, "below-band")])
def test_accuracy_read_follows_the_archetype_band(qapp, settings, accuracy, read):
    view = AnalysisView(settings)
    view.show_report(_report(accuracy=accuracy), profile=_profile())
    assert view.kpis["accuracy"].read.text() == read
    view.deleteLater()


# ------------------------------------------------------------------- coach
def _loud_run() -> tuple[RunReport, PlayerProfile]:
    """A run that trips five insights across both severities."""
    prof = _profile(ewma_bias=0.4)
    prof.run_count = 8
    prof.history = [{"accuracy": 0.99, "kps": 1.0, "score": 400.0} for _ in range(3)]
    prof.regions["r0c0"] = RegionPosterior(mean=0.5, var=0.2, n=4)
    rep = _report(accuracy=0.99,
                  input_health={"jitter_ms": 5.0, "polling_hz_est": 1000.0},
                  fatigue={"level": "tired", "runs": 6, "score": 0.4})
    return rep, prof


def test_coach_folds_to_the_two_worst_and_keeps_every_card(qapp, settings):
    view = AnalysisView(settings)
    rep, prof = _loud_run()
    view.show_report(rep, profile=prof)

    cards = view._coach_cards
    assert len(cards) == len(generate_insights(rep, prof, settings)) >= 3
    # severity order: warnings first, generate_insights' own order within one
    ranks = [{"warning": 0, "attention": 1, "info": 2}[c.insight.severity] for c in cards]
    assert ranks == sorted(ranks)
    assert [c.isHidden() for c in cards[:_COACH_FOLD]] == [False] * _COACH_FOLD
    assert all(c.isHidden() for c in cards[_COACH_FOLD:])
    assert view.coach_more is not None
    assert view.coach_more.text() == f"show all ({len(cards)})"

    view.coach_more.click()
    assert not any(c.isHidden() for c in cards)     # nothing was ever deleted
    assert view.coach_more.text() == "show fewer"
    view.coach_more.click()
    assert all(c.isHidden() for c in cards[_COACH_FOLD:])
    view.deleteLater()


def test_a_new_report_opens_folded_again(qapp, settings):
    view = AnalysisView(settings)
    rep, prof = _loud_run()
    view.show_report(rep, profile=prof)
    view.coach_more.click()
    assert view._coach_open
    view.show_report(rep, profile=prof)
    assert not view._coach_open
    assert view.coach_more.text().startswith("show all")
    view.deleteLater()


def test_a_short_coach_needs_no_disclosure(qapp, settings):
    view = AnalysisView(settings)
    prof = _profile()                       # only the below-band card fires
    view.show_report(_report(), profile=prof)
    assert 0 < len(view._coach_cards) <= _COACH_FOLD
    assert view.coach_more is None
    assert not any(c.isHidden() for c in view._coach_cards)
    view.deleteLater()


# --------------------------------------------------------------- takeaways
def test_bias_title_names_the_weaker_side_only_with_the_flicks_to_say_so():
    # [left, vertical, right] costs and flick counts, as plotted
    hot = _bias_title([0.40, 0.05, 0.20], [12, 4, 10])
    assert "left" in hot and "2.0x" in hot and "0.40 vs 0.20" in hot
    assert _bias_title([0.30, 0.05, 0.28], [12, 4, 10]).startswith(
        "left and right flicks are even")
    assert "too few" in _bias_title([0.40, 0.0, 0.10], [12, 0, 1])
    assert _bias_title([0.0, 0.0, 0.0], [0, 0, 0]) == _BIAS_TITLE
    assert "vertical" in _bias_title([0.20, 0.60, 0.18], [10, 6, 10])
    assert "no overshoot" in _bias_title([0.0, 0.0, 0.0], [10, 4, 10])


def test_deficit_title_will_not_name_a_zone_the_map_draws_flat(settings):
    loud = _deficit_title({"r0c0": 1.40, "r2c2": -0.50}, settings)
    # "this run" is load-bearing: the Coach's own weakest-region card reads
    # the PROFILE's cross-run posteriors and can legitimately name a different
    # zone on the same screen.
    assert loud == "weakest zone this run: lower left, +1.40 SD above average"
    quiet = {"r0c0": 0.20, "r2c2": -0.20}           # under viz.NOISE_FLOOR
    assert max(quiet.values()) < viz.NOISE_FLOOR
    assert _deficit_title(quiet, settings).startswith("no zone stands out")
    assert _deficit_title({}, settings) == _DEFICIT_TITLE


def test_deficit_title_tests_the_flat_claim_on_the_absolute_deviation():
    """"All within N SD" must be judged on what the MAP colours.

    The map's magnitude is |z|, so testing only the maximum let a strongly
    negative zone — drawn cool and fully saturated — sit under a headline
    claiming nothing deviated at all.
    """
    # nothing weak, but one zone is a genuine strength the map draws loudly
    strong = {"r0c0": 0.10, "r2c2": -1.80}
    assert max(strong.values()) < viz.NOISE_FLOOR          # old check passed
    title = _deficit_title(strong, None)
    assert not title.startswith("no zone stands out"), (
        "claimed a flat map while a -1.80 SD zone is fully coloured")
    assert "no zone is weaker" in title and "-1.80" in title
    # genuinely flat in both directions still reads flat
    flat = {"r0c0": 0.10, "r2c2": -0.20}
    assert _deficit_title(flat, None).startswith("no zone stands out")


def test_travel_title_leans_only_when_the_occupancy_map_is_lopsided():
    heat = np.ones((8, 8))
    heat[:4] = 3.0                                   # 75% of samples left of the shot
    lean = _travel_title(heat)
    assert "leans left" in lean and "75%" in lean
    assert _travel_title(np.ones((8, 8))) == "aim travel is balanced left/right of your shots"
    assert _travel_title(np.zeros((8, 8))) == _TRAVEL_TITLE


def test_trend_title_calls_a_direction_only_over_enough_runs():
    assert "too few" in _trend_title([0.60, 0.62, 0.64])
    assert _trend_title([0.60] * 6).startswith("accuracy flat near 60%")
    rising = _trend_title([0.55, 0.56, 0.57, 0.65, 0.66, 0.67])
    assert "up" in rising and "10 points" in rising and "56% → 66%" in rising
    falling = _trend_title([0.70, 0.70, 0.70, 0.60, 0.60, 0.60])
    assert "down" in falling


def test_takeaway_titles_reach_the_widgets(qapp, settings):
    """The derived sentence has to land on the chart, not just compute."""
    view = AnalysisView(settings)
    bias = {"left": {"n": 12, "overshoot": 0.40, "corrections": 0.0},
            "vertical": {"n": 4, "overshoot": 0.05, "corrections": 0.0},
            "right": {"n": 10, "overshoot": 0.20, "corrections": 0.0},
            "bias_score": 0.33}
    prof = _profile()
    view.show_report(_report(bias=bias, region_deficits={"r0c0": 1.4, "r2c2": -0.5},
                             n_flicks=26),
                     profile=prof)
    assert "2.0x" in view.bias_bars._title
    assert view.heat_map._title.startswith("weakest zone")
    assert view.trend_spark._title != "accuracy over runs"    # 6 runs of history
    # a bare report claims nothing
    view.show_report(_report(), profile=prof)
    assert view.bias_bars._title == _BIAS_TITLE
    assert view.heat_map._title == _TRAVEL_TITLE
    view.deleteLater()


# ------------------------------------------------- moments / replay wiring
def test_selected_moment_and_replay_describe_the_same_segment(qapp, settings):
    """Row 0 is selected on load while the replay separately loaded the full
    run, so the highlight and the replay disagreed about what you were
    looking at."""
    trace = TraceBuilder().flick(200, 0, overshoot=0.3).flick(-150, 30).build()
    rep = _report(notable=[{"kind": "overshoot", "text": "worst overshoot",
                            "t_start": float(trace.t[0]), "t_end": float(trace.t[-1])},
                           {"kind": "clean_flick", "text": "cleanest flick",
                            "t_start": float(trace.t[0]), "t_end": float(trace.t[-1])}])
    view = AnalysisView(settings)
    view.show_report(rep, trace=trace)

    assert view.moments.currentRow() == 0
    assert view.replay.info.text() == "overshoot"      # not "full run"
    assert view.full_btn.isEnabled()

    view.moments.setCurrentRow(1)
    assert view.replay.info.text() == "clean flick"

    view.full_btn.click()                              # back out to everything
    assert view.moments.currentRow() == -1             # no row claims to be showing
    assert view.replay.info.text() == "full run"

    # the clip story survived the reorder: the dead button explains itself and
    # the same one-liner is the inline hint
    assert not view.clip_btn.isEnabled()
    assert "Capture video clips" in view.clip_btn.toolTip()
    assert view.clip_hint.text() == view.clip_btn.toolTip()
    view.deleteLater()


def test_a_run_without_telemetry_disables_the_whole_run_button(qapp, settings):
    view = AnalysisView(settings)
    view.show_report(_report(notable=[{"kind": "overshoot", "text": "x",
                                       "t_start": 1.0, "t_end": 2.0}]))
    assert not view.full_btn.isEnabled()
    assert view.replay.info.text() == "no trace for this run"
    view.deleteLater()


def test_captions_stay_short_now_that_titles_carry_the_meaning(qapp, settings):
    view = AnalysisView(settings)
    view.show_report(_report(region_deficits={"r0c0": 1.4, "r2c2": -0.5}),
                     profile=_profile())
    caps = (view.bias_caption, view.heat_caption, view.trend_caption)
    for cap in caps:
        assert cap.text()                       # never an unexplained chart
        assert cap.wordWrap()
        assert cap.property("dim") is True
    assert sum(len(cap.text()) for cap in caps) < 450    # was ~700
    view.deleteLater()


def test_kpi_flick_tile_honours_the_coachs_input_health_gate(qapp, settings):
    """One rule, two surfaces.

    insights.generate_insights suppresses every microstructure diagnosis when
    the run's input timing is noisy. The KPI strip re-derived its own
    overshoot verdict from the same fields WITHOUT that gate, so a run could
    be labelled "repaired" on the very screen where the Coach reported that
    microstructure diagnoses were suppressed for it.
    """
    from kovadapt.analysis.insights import input_degraded

    # numbers that would otherwise read as a clear "repaired" verdict
    micro = dict(n_flicks=40, overshoot_rate=0.55, mean_corrections=3.0,
                 mean_flick_ms=180.0)
    clean = _report(**micro, input_health={"jitter_ms": 0.4,
                                           "polling_hz_est": 1000.0})
    noisy = _report(**micro, input_health={"jitter_ms": 6.0,
                                           "polling_hz_est": 125.0})
    assert not input_degraded(clean) and input_degraded(noisy)

    view = AnalysisView(settings)
    view.show_report(clean, profile=_profile())
    assert view.kpis["flick"].read.text() == "repaired"

    view.show_report(noisy, profile=_profile())
    assert view.kpis["flick"].read.text() == "noisy-input", (
        "the tile gave an overshoot verdict the Coach refuses to give")
    assert "too noisy" in view.kpis["flick"].toolTip()
    # and the Coach really is suppressing on the same report
    ids = {i.id for i in generate_insights(noisy, _profile(), settings)}
    assert "dx-input-health" in ids
    assert "dx-overshoot-control" not in ids
    view.deleteLater()


def test_report_loads_the_replay_exactly_once(qapp, settings):
    """show_report used to load the full trace AND then a moment window.

    That threw away a whole path() + decimate + setData pass on every run
    that had notable moments.
    """
    trace = TraceBuilder().flick(400, 0, 0.12).rest(0.3).flick(-300, 80, 0.10).build()
    view = AnalysisView(settings)
    calls: list = []
    real_load = view.replay.load
    view.replay.load = lambda *a, **k: (calls.append(k.get("label", "window")),
                                        real_load(*a, **k))[1]

    rep = _report(notable=[{"kind": "overshoot", "text": "worst overshoot",
                            "t_start": float(trace.t[0]),
                            "t_end": float(trace.t[0]) + 0.2}])
    view.show_report(rep, trace=trace, profile=_profile())
    assert len(calls) == 1, f"replay loaded {len(calls)} times: {calls}"
    assert view.moments.currentRow() == 0        # and the selection still wins

    # with no notable moments the full run is what loads — still exactly once
    calls.clear()
    view.show_report(_report(), trace=trace, profile=_profile())
    assert calls == ["full run"]
    view.deleteLater()


def test_the_two_weakest_region_claims_name_their_own_scope(settings):
    """The chart reads THIS run; the Coach reads cross-run posteriors.

    They can legitimately disagree, and unlabelled they read as one claim
    contradicting itself on a single screen.
    """
    prof = _profile()
    prof.regions = {"r4c4": RegionPosterior(mean=0.9, var=0.05, n=6)}
    rep = _report(region_deficits={"r0c0": 1.4})

    chart = _deficit_title(rep.region_deficits, settings)
    coach = [i for i in generate_insights(rep, prof, settings)
             if i.id == "dx-region-deficit"]
    assert chart.startswith("weakest zone this run")
    assert coach and "across runs" in coach[0].title


def test_no_surface_gives_a_microstructure_verdict_on_a_noisy_run(qapp, settings):
    """The whole-page contract, not one widget's.

    Gating only the KPI tile was not enough: the header summary sat directly
    above it printing "40% of flicks overshot - consider a slight sens
    decrease", the bias chart headlined "your left flicks cost 3.2x more than
    your right", and the moments list narrated per-flick overshoot - all on a
    run the Coach had already declared too noisy to diagnose. Every one of
    those reads the same microstructure, so every one answers to the same gate.
    """
    from kovadapt.analysis.report import input_degraded

    bias = {
        "bias_score": 0.42,
        "left": {"overshoot": 0.70, "corrections": 2.4, "n": 14},
        "right": {"overshoot": 0.20, "corrections": 0.6, "n": 13},
        "vertical": {"overshoot": 0.25, "corrections": 0.8, "n": 9},
    }
    noisy = _report(
        n_flicks=40, overshoot_rate=0.55, mean_corrections=3.0,
        mean_flick_ms=180.0, bias=bias,
        input_health={"jitter_ms": 6.0, "polling_hz_est": 1000.0},
        notable=[{"kind": "overshoot", "text": "kill 7: overshot 38%",
                  "t_start": 0.0, "t_end": 0.2}])
    assert input_degraded(noisy)

    view = AnalysisView(settings)
    view.show_report(noisy, profile=_profile())

    assert view.kpis["flick"].read.text() == "noisy-input"
    assert "too noisy" in view.bias_bars._title.lower()
    # the summary the watcher writes into the report must not prescribe either
    from kovadapt.analysis.report import _summary_text
    summary = _summary_text(noisy, flicks_exist=True)
    for banned in ("overshot", "sens decrease", "measurably weaker", "balanced"):
        assert banned not in summary, f"summary still claims: {banned!r}"
    assert "too noisy" in summary
    # the moments stay listed and replayable, but carry the caveat
    texts = [view.moments.item(r).text() for r in range(view.moments.count())]
    assert any("noisy" in t for t in texts)
    assert any("kill 7" in t for t in texts)
    # and the caption row must not steal the selection or shift the mapping
    row = view.moments.currentRow()
    assert view._moment_index(row) == 0
    view.deleteLater()


def test_the_input_health_gate_has_exactly_one_definition():
    """sens.py used to re-derive it from its own copy of the thresholds, one
    call away from insights.py's — so the docstring claiming a single
    definition was false until both deferred to report.py."""
    import inspect
    import re

    from kovadapt.analysis import insights, report, sens

    assert insights.input_degraded is report.input_degraded
    assert sens.input_degraded is report.input_degraded
    # and neither may bind a threshold to its own numeric literal again
    literal = re.compile(r"^_?(JITTER_BAD_MS|POLLING_LOW_HZ)\s*=\s*[-\d.]", re.M)
    for mod in (insights, sens):
        hit = literal.search(inspect.getsource(mod))
        assert hit is None, f"{mod.__name__} re-declares {hit.group(1)}"
    assert literal.search(inspect.getsource(report)) is not None
