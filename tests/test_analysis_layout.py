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

from pathlib import Path  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402
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


def _degraded_two_moments(tmp_path, clips: dict) -> RunReport:
    """A noisy run with two notable moments, so _fill_moments prepends the
    caption row and every list row sits one below its moment."""
    return _report(
        n_flicks=40, overshoot_rate=0.55, mean_corrections=3.0,
        input_health={"jitter_ms": 99.0, "polling_hz_est": 1000.0},
        notable=[{"kind": "clean", "text": "kill 3: your benchmark flick",
                  "t_start": 0.0, "t_end": 0.2},
                 {"kind": "overshoot", "text": "kill 7: overshot 38%",
                  "t_start": 0.4, "t_end": 0.6}],
        clip_files=clips)


def test_the_clip_button_plays_the_moment_that_is_selected(qapp, settings,
                                                           tmp_path,
                                                           monkeypatch):
    """clip_files is keyed by MOMENT (watcher.py enumerates rep.notable), but
    three sites read the list ROW. On a degraded run the caption row shifts
    every row one past its moment, so Play opened the NEXT moment's video
    while the highlighted row and the trajectory replay both showed the one
    before it. `input_degraded` trips on a 500 Hz mouse, so this is not an
    exotic state."""
    from kovadapt.gui import analysis_view as av

    p0 = tmp_path / "m0.mp4"
    p1 = tmp_path / "m1.mp4"
    for p in (p0, p1):
        p.write_bytes(b"x")

    opened = []
    monkeypatch.setattr(av.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toLocalFile()))

    view = AnalysisView(settings)
    view.show_report(_degraded_two_moments(tmp_path,
                                           {"0": str(p0), "1": str(p1)}),
                     profile=_profile())
    assert view._moment_index(view.moments.currentRow()) == 0
    view._play_clip()
    assert opened and Path(opened[-1]) == p0, (
        f"played {opened} — the row, not the selected moment")

    # ...and the LAST moment, whose row index is never a clip key at all
    view.moments.setCurrentRow(view.moments.count() - 1)
    assert view._moment_index(view.moments.currentRow()) == 1
    view._play_clip()
    assert Path(opened[-1]) == p1
    view.deleteLater()


def test_the_clip_button_is_not_greyed_out_for_a_moment_that_has_a_clip(
        qapp, settings, tmp_path):
    """show_report's trailing _update_clip_state passed a ROW, and it runs
    last — so it overwrote the correct state computed by _select_moment and
    claimed 'No clip was captured for this moment' about a moment with one."""
    p0 = tmp_path / "m0.mp4"
    p0.write_bytes(b"x")
    view = AnalysisView(settings)
    view.show_report(_degraded_two_moments(tmp_path, {"0": str(p0)}),
                     profile=_profile())
    assert view._moment_index(view.moments.currentRow()) == 0
    assert view.clip_btn.isEnabled(), view.clip_btn.toolTip()
    assert view.clip_btn.toolTip() == ""
    view.deleteLater()


def test_a_theme_switch_keeps_every_moment_its_own_colour(qapp, settings,
                                                          tmp_path):
    """restyle indexed report.notable by ROW. The list is severity-sorted and
    the clean reference flick scores 1.0, so on a degraded run a theme change
    painted 'your benchmark flick' in the BAD colour and dropped the
    caption's dim grey. On this page the colour IS the evidence.

    Needs no clips extra and no checkbox — a degraded run plus a theme switch
    is the whole reproduction.
    """
    from kovadapt.gui import theme
    from kovadapt.gui.analysis_view import _kind_color

    view = AnalysisView(settings)
    view.show_report(_degraded_two_moments(tmp_path, {}), profile=_profile())
    assert view.moments.count() == 3            # caption + two moments
    view.restyle(theme.current())

    pal = theme.current()
    assert view.moments.item(0).foreground().color().name() == pal.fg_dim
    for row, kind in ((1, "clean"), (2, "overshoot")):
        got = view.moments.item(row).foreground().color().name()
        assert got == QColor(_kind_color(kind)).name(), (
            f"row {row} took another moment's colour: {got}")
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


def test_pace_is_not_measurable_on_a_tracking_run(qapp, settings):
    """Tracking scenarios use INVINCIBLE targets, so KovaaK's reports
    Kills: 0 by design and kills-per-second is structurally undefined — not
    slow. On a real 95-scenario library that is 49 scenarios, and PACE showed
    them a permanent "0.00 kills/s".

    The because-clause was the worse half: it fell through to the no-baseline
    branch and explained the zero with "the EWMA is seeded from the first run,
    so it needs a second" — a specific, checkable, wrong reason on a profile
    with fifty runs.
    """
    view = AnalysisView(settings)
    prof = _profile(archetype="tracking")
    prof.run_count = 50
    prof.ewma_kps = 0.0                       # never had a kill to average
    view.show_report(_report(kills=0, kps=0.0, accuracy=0.91), profile=prof)

    tile = view.kpis["pace"]
    assert tile.value.text() == "—", "a fake zero, not a measurement"
    assert tile.read.text() == "not-measurable"
    tip = tile.toolTip().lower()
    assert "invincible" in tip, tip
    assert "seeded from the first run" not in tip, "the false reason came back"
    assert "50" not in tip, "run count is irrelevant to why this is unmeasurable"
    view.deleteLater()


def test_pace_still_reads_normally_when_there_are_kills(qapp, settings):
    """The guard must not silence a scenario that does have a pace."""
    view = AnalysisView(settings)
    prof = _profile()
    prof.run_count = 10
    prof.ewma_kps = 1.0
    view.show_report(_report(kills=30, kps=1.4), profile=prof)
    tile = view.kpis["pace"]
    assert tile.value.text() == "1.40"
    assert tile.read.text() == "faster"
    assert "1.00 EWMA" in tile.toolTip()
    view.deleteLater()


# ------------------------------------------------- a damaged recording file
def _truncated_trace(tmp_path) -> Path:
    """A real .npz, cut in half — what a crash mid-write or a half-synced
    cloud folder leaves behind. np.load raises BadZipFile on it."""
    tr = (TraceBuilder(t0=1000.0)
          .move(0.30, 400.0, 0.0).click(0.02).move(0.30, -400.0, 0.0).build())
    p = tmp_path / "traces" / "damaged.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    tr.save(p)
    raw = p.read_bytes()
    assert len(raw) > 200, "need a file big enough for half of it to be broken"
    p.write_bytes(raw[: len(raw) // 2])
    return p


def test_a_damaged_recording_does_not_take_the_report_down(qapp, settings, tmp_path):
    """A truncated .npz raised zipfile.BadZipFile straight out of
    show_report, so opening one saved report killed the whole page —
    including the stats half, which never touches the trace.

    The run's score, accuracy and summary are all in the report JSON. Losing
    the recording costs the replay and the flick overlays; it does not cost
    the reader anything the CSV already paid for.
    """
    bad = _truncated_trace(tmp_path)
    rep = _report(trace_file=str(bad))
    dest = tmp_path / "reports" / "run.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    rep.save(dest)

    view = AnalysisView(settings)
    view.load_report_file(dest)          # exactly what the Open report… button calls

    assert view.trace is None
    assert view._trace_unreadable is True
    assert view.summary.text() == "30 kills at 61% accuracy.", "the stats half rendered"
    assert view.kpis["accuracy"].value.text() == "61%", "stats survived"
    assert not view.full_btn.isEnabled(), "nothing to replay"
    view.deleteLater()


def test_a_damaged_recording_says_it_is_damaged(qapp, settings, tmp_path):
    """'no trace for this run' would be a lie: there IS a recording, and the
    reason it is not on screen is that the file is broken — which is the one
    thing that tells the user to go look at the disk."""
    view = AnalysisView(settings)
    view.show_report(_report(trace_file=str(_truncated_trace(tmp_path))))
    said = view.replay.info.text().lower()
    assert "damaged" in said or "unreadable" in said, said
    assert said != "no trace for this run"

    # ...and a run that genuinely has no telemetry still says so
    view.show_report(_report())
    assert view._trace_unreadable is False
    assert view.replay.info.text() == "no trace for this run"
    view.deleteLater()


def test_trace_store_treats_a_damaged_file_as_no_trace(tmp_path):
    """TraceStore.load already returns None for a missing file; a file that
    cannot be read is not a different answer to the caller. It must NOT
    delete the file — a recording cannot be regenerated from anything."""
    from kovadapt.telemetry.trace import TraceStore

    store = TraceStore(tmp_path / "traces")
    p = store.path_for("Beta 1wall Click", "2026-07-28T10:00:00")
    p.parent.mkdir(parents=True, exist_ok=True)
    tr = TraceBuilder(t0=1000.0).move(0.30, 400.0, 0.0).build()
    tr.save(p)
    p.write_bytes(p.read_bytes()[:120])

    assert store.load("Beta 1wall Click", "2026-07-28T10:00:00") is None
    assert p.is_file(), "the damaged recording was deleted"


# ---------------------------------- the bias panel agrees with itself or shuts up
def _bias(**over):
    """A bias dict at the given per-direction cost, with healthy input."""
    d = {k: {"n": 40, "overshoot": v, "corrections": 0.0}
         for k, v in over.items()}
    return d


def test_a_run_with_no_flicks_does_not_paint_three_zero_bars(qapp, settings):
    """directional_bias([]) returns a POPULATED all-zero dict, so vals was
    [0.0, 0.0, 0.0] — truthy — and walked past AsciiBars' own empty state to
    draw three empty tracks with 0.00 beside each. The heatmap on that same
    run says "no movement data" and the KPI tile shows an em-dash; only this
    panel pretended it had measured something."""
    view = AnalysisView(settings)
    view.show_report(_report(n_flicks=0, bias={
        "left": {"n": 0, "overshoot": 0.0, "corrections": 0.0},
        "vertical": {"n": 0, "overshoot": 0.0, "corrections": 0.0},
        "right": {"n": 0, "overshoot": 0.0, "corrections": 0.0}}))

    assert view.bias_bars._values == [], "the bars still hold data to draw"
    assert "red bar" not in view.bias_caption.text(), (
        "the caption points at a colour on a panel with no bars")
    view.deleteLater()


def test_the_headline_does_not_compute_a_ratio_between_two_zeros(qapp, settings):
    """The title read "your left flicks cost 1.3x more than your right — 0.00
    vs 0.00": a finding carrying its own refutation, sitting over a footer
    that already said there was no cost to compare."""
    view = AnalysisView(settings)
    view.show_report(_report(n_flicks=119, bias=_bias(
        left=0.004, vertical=0.002, right=0.003)))

    title = view.bias_bars._title.lower()
    assert "x more" not in title, f"a ratio off noise: {title!r}"
    assert "0.00 vs 0.00" not in title
    assert "no measurable cost" in title, title
    assert "red bar" not in view.bias_caption.text(), (
        "nothing is red on this panel — the highlight is gated too")
    view.deleteLater()


def test_a_real_bias_still_gets_its_headline_and_its_red_bar(qapp, settings):
    """The guards must not mute a run that genuinely has a weak side."""
    view = AnalysisView(settings)
    view.show_report(_report(n_flicks=119, bias=_bias(
        left=0.42, vertical=0.10, right=0.16)))

    assert "x more" in view.bias_bars._title.lower(), view.bias_bars._title
    assert "red bar is this run's worst" in view.bias_caption.text()
    view.deleteLater()


def test_the_cold_start_caption_promises_no_ink(qapp, settings):
    """Before any run is loaded the panel reads "waiting for flick data", and
    the caption underneath pointed at a red bar no run has produced yet."""
    view = AnalysisView(settings)
    assert "red bar" not in view.bias_caption.text(), view.bias_caption.text()
    view.deleteLater()


# --------------------------------------------- severity is an ordinal ramp
def test_the_coach_dot_never_runs_backwards_against_its_own_sort_order():
    """The map was `{"warning": warn, "attention": bad}.get(sev, good)` while
    _SEVERITY_RANK puts warning at 0 — sorted first, always left unfolded. So
    the most severe card wore amber and the less severe one wore red, and with
    two warnings the default folded Coach showed two amber dots and no red at
    all: the only red card was behind "show all".
    """
    from kovadapt.gui.analysis_view import _SEVERITY_RANK, _severity_color

    pal = theme.build_palette(dark=True, accent="indigo")
    alarm = {pal.bad: 0, pal.warn: 1, pal.fg_dim: 2}
    ranked = sorted(_SEVERITY_RANK, key=lambda s: _SEVERITY_RANK[s])
    got = [_severity_color(s, pal) for s in ranked]

    assert all(c in alarm for c in got), f"a dot colour outside the ramp: {got}"
    levels = [alarm[c] for c in got]
    assert levels == sorted(levels), (
        f"severity {ranked} maps to alarm order {levels} — a card ranked worse "
        "than another is carrying the calmer colour")
    assert len(set(got)) == len(got), f"two severities share a dot: {got}"


def test_an_informational_card_does_not_wear_the_all_clear():
    """`info` fell through to pal.good, so a card that merely states something
    — sensitivity doctrine, say — wore the same green as a passed check."""
    from kovadapt.gui.analysis_view import _severity_color

    pal = theme.build_palette(dark=True, accent="indigo")
    assert _severity_color("info", pal) != pal.good
    assert _severity_color("info", pal) == pal.fg_dim


@pytest.mark.parametrize("accent", ["indigo", "ocean", "mint", "rose", "ember"])
def test_no_coach_dot_is_ever_the_users_accent(accent):
    """The accent is a preference. Under rose it is red and under mint it is
    green, so any severity painted with it reports the user's taste as a
    verdict."""
    from kovadapt.gui.analysis_view import _SEVERITY_RANK, _severity_color

    pal = theme.build_palette(dark=True, accent=accent)
    for sev in _SEVERITY_RANK:
        assert _severity_color(sev, pal) != pal.accent, (
            f"{accent}: severity {sev!r} is painted in the accent")


# ------------------------------------------------------------- cold start
def test_the_page_explains_itself_before_any_run_is_loaded(qapp, settings):
    """On every launch the two biggest canvases on this page — a 246,448px
    moments list and a 531,912px replay — rendered as 100.000% one flat
    colour with zero ink, while the two chart panels beside them explained
    themselves. The replay's sentence already existed (clear() takes one); it
    was simply unreachable except through show_report.
    """
    view = AnalysisView(settings)

    assert view.moments.count() == 1, "the moments list is blank on launch"
    placeholder = view.moments.item(0)
    assert placeholder.text().strip(), "the placeholder row has no text"
    assert not (placeholder.flags() & Qt.ItemIsSelectable), (
        "the placeholder can be selected, so it reads as a moment")

    assert view.replay.info.text(), "the replay canvas says nothing"
    assert "no run loaded" in view.replay.info.text().lower()
    view.deleteLater()


def test_the_replay_transport_is_dead_until_there_is_something_to_replay(
        qapp, settings):
    """The panel rendered a full LIVE control strip over an empty canvas: an
    enabled Replay button, a speed cycle, three checked layer boxes and a
    scrub slider. Clicking Replay did nothing, which is the worst answer a
    control can give."""
    view = AnalysisView(settings)
    for name in ("btn", "speed_btn", "scrub", "toggle_path", "toggle_flicks",
                 "toggle_shots"):
        assert not getattr(view.replay, name).isEnabled(), (
            f"replay.{name} is live with no trace loaded")

    # ...and a real run brings the whole transport back
    tr = (TraceBuilder(t0=1000.0)
          .move(0.30, 400.0, 0.0).click(0.02).move(0.30, -400.0, 0.0)
          .click(0.02).build())
    view.show_report(_report(n_flicks=2), trace=tr)
    for name in ("btn", "speed_btn", "scrub", "toggle_path"):
        assert getattr(view.replay, name).isEnabled(), (
            f"replay.{name} stayed dead on a run that has telemetry")
    # A run with NO notable moments legitimately keeps the placeholder — the
    # panel still has to say what it is for; a run that HAS them replaces it.
    from kovadapt.gui.analysis_view import _MOMENTS_EMPTY
    assert view.moments.item(0).text() == _MOMENTS_EMPTY, (
        "a run with no notable moments left the panel blank again")
    view.show_report(_report(n_flicks=2, notable=[
        {"t_start": 1000.0, "t_end": 1000.4, "kind": "overshoot",
         "text": "Overshot a right flick by 36%."}]), trace=tr)
    assert view.moments.item(0).text() != _MOMENTS_EMPTY, (
        "the placeholder survived a run that produced moments")
    view.deleteLater()


def test_no_view_widens_a_splitter_handle_past_what_the_theme_asked_for(qapp, settings):
    """`setHandleWidth(14)` for "room to breathe" did not add space — the theme
    FILLS a splitter handle with `pal.border`, so widening it produced a 14px
    column of border colour between panels that carry their own frames, and
    ran it 26px above the panels, up beside the titles where it divides
    nothing.

    Compared against a control splitter of the SAME ORIENTATION under the same
    sheet, not against `QApplication.style().pixelMetric(PM_SplitterWidth)` —
    that returns Qt's raw 4 with no widget context and would pass a view that
    overrides the theme by one.
    """
    from PySide6.QtWidgets import QSplitter

    view = AnalysisView(settings)
    control = QSplitter(Qt.Horizontal)
    want = control.handleWidth()
    for name in ("charts", "detail"):
        got = getattr(view, name).handleWidth()
        assert got == want, (
            f"{name} sets its handle to {got}px against the theme's {want}px — "
            "a view is overriding the sheet")
    control.deleteLater()
    view.deleteLater()
    qapp.processEvents()



def _red_bar_row(bars, n_rows: int) -> int | None:
    """Index of the bar actually painted in pal.bad, off the pixels."""
    from PySide6.QtGui import QImage, QPixmap

    pal = theme.current()
    w, h = 720, 300
    pm = QPixmap(w, h)
    pm.fill(QColor(pal.bg))
    bars.resize(w, h)
    bars.render(pm)
    img = pm.toImage().convertToFormat(QImage.Format_RGB32)
    buf = np.frombuffer(img.constBits(), dtype=np.uint8)
    arr = buf.reshape(img.height(), img.bytesPerLine() // 4, 4)[:, :w, :3][:, :, ::-1]
    want = np.array([int(pal.bad[i:i + 2], 16) for i in (1, 3, 5)])
    hit = np.all(arr == want, axis=-1)
    rows = np.nonzero(hit.any(axis=1))[0]
    if not rows.size:
        return None
    top = 26.0
    body = h - top - viz._RULER_H - viz._FOOTER_H - 4
    centre = float(rows.mean())
    return min(range(n_rows),
               key=lambda i: abs(centre - (top + (i + 0.5) * body / n_rows)))

def test_the_bias_panel_never_names_two_different_worst_directions(qapp, settings):
    """The real report that exposed this: left 0.104, vertical 0.115, right
    0.042, n = 58/24/57. The headline said "your left flicks cost 2.5x more
    than your right", the red bar and the red numeral sat on VERTICAL, the
    footer cited "vertical 0.12 / left 0.10 = 1.11x" under that 2.5x headline,
    and the caption told the reader the red bar was this run's worst.

    Cause: `_bias_title` deliberately ignores a vertical bar unless it
    dominates by 1.25x (0.115/0.104 = 1.105, under the gate), while viz
    derived the red bar as the global argmax. Two independent rules for one
    verdict. The caller owns the claim now and viz renders it.
    """
    view = AnalysisView(settings)
    view.show_report(_report(n_flicks=139, bias={
        "left": {"n": 58, "overshoot": 0.104, "corrections": 0.0},
        "vertical": {"n": 24, "overshoot": 0.115, "corrections": 0.0},
        "right": {"n": 57, "overshoot": 0.042, "corrections": 0.0}}))

    dirs = ["left", "vertical", "right"]
    title = view.bias_bars._title.lower()
    named = [d for d in dirs if d in title]
    assert named, f"the headline names no direction at all: {title!r}"

    # WHICH BAR IS ACTUALLY RED — read off the render, not off the stored
    # claim. Asserting on state cannot see the defect: the bug was that viz
    # painted something other than what the caller claimed.
    marked = _red_bar_row(view.bias_bars, len(dirs))
    assert marked is not None, "a headline naming a worst side marked no bar"
    assert dirs[marked] in named, (
        f"the red bar is on {dirs[marked]!r} while the headline names "
        f"{named} — the panel is giving two answers")

    footer = view.bias_bars.ratio_footer().lower()
    if footer:
        # the footer reads "top two: <worse> X / <other> Y = Nx", so the
        # FIRST direction it names is the one it calls worse. Membership is
        # not enough: citing the right pair in the wrong order still puts a
        # different direction at the top of the sentence than under the red
        # bar, which is the whole defect.
        first = min((d for d in dirs if d in footer),
                    key=lambda d: footer.index(d), default=None)
        assert first == dirs[marked], (
            f"the footer leads with {first!r} while the red bar is on "
            f"{dirs[marked]!r}: {footer!r}")
    view.deleteLater()


def test_the_region_map_answers_to_the_same_gate_as_the_bias_panel(qapp, settings):
    """Three of the five real reports on this machine rendered "WEAKEST ZONE
    THIS RUN: CENTER, +3.60 SD ABOVE AVERAGE" on the same baseline, 700px to
    the right of "INPUT TIMING TOO NOISY TO COMPARE DIRECTIONS THIS RUN".

    A region deficit is overshoot + 0.15*corrections + 0.25*slowness, z-scored
    — pure flick microstructure — so it answers to the same input-health gate
    as everything else on the page. It was the one surface that never asked.
    """
    noisy = _report(
        n_flicks=119,
        input_health={"jitter_ms": 9.0, "polling_hz_est": 125.0, "gaps": 4},
        region_deficits={"r2c2": 3.60, "r0c0": -0.4, "r1c1": 0.2})
    view = AnalysisView(settings)
    view.show_report(noisy)

    heat = view.heat_map._title.lower()
    assert "weakest zone" not in heat, (
        f"the map names a zone on a run the page calls too noisy: {heat!r}")
    assert "noisy" in heat, heat
    # ...and a clean run still gets its verdict
    view.show_report(_report(
        n_flicks=119,
        input_health={"jitter_ms": 0.4, "polling_hz_est": 1000.0},
        region_deficits={"r2c2": 3.60, "r0c0": -0.4, "r1c1": 0.2}))
    assert "weakest zone" in view.heat_map._title.lower()
    view.deleteLater()


def test_a_run_with_no_flicks_does_not_claim_nothing_cost_enough(qapp, settings):
    """"Nothing this run cost enough to rank" is a MEASUREMENT, and a run
    with no flicks has not made one — the chart beside this caption reads
    "waiting for flick data" on exactly that run. Two sentences about the
    same absence, one of them claiming more than it can."""
    from kovadapt.gui.analysis_view import (_BIAS_CAPTION_NO_COST,
                                            _BIAS_CAPTION_NO_FLICKS)

    view = AnalysisView(settings)
    view.show_report(_report(n_flicks=0, bias={
        d: {"n": 0, "overshoot": 0.0, "corrections": 0.0}
        for d in ("left", "vertical", "right")}))
    assert view.bias_caption.text() == _BIAS_CAPTION_NO_FLICKS
    assert "cost enough to rank" not in view.bias_caption.text()

    # a run that DID measure flicks, all of which rounded to nothing, still
    # gets the measured sentence — the two states are not the same state
    view.show_report(_report(n_flicks=119, bias=_bias(
        left=0.004, vertical=0.002, right=0.003)))
    assert view.bias_caption.text() == _BIAS_CAPTION_NO_COST
    view.deleteLater()


def test_the_moments_placeholder_survives_a_run_that_produced_none(qapp, settings):
    """`_fill_moments` clears the list, which destroyed the placeholder put
    there at construction — so the cold-start fix held only until the first
    run, and any run with no notable moments went back to a blank panel."""
    from kovadapt.gui.analysis_view import _MOMENTS_EMPTY

    view = AnalysisView(settings)
    assert view.moments.item(0).text() == _MOMENTS_EMPTY

    view.show_report(_report(n_flicks=40))          # no notable moments
    assert view.moments.count() == 1
    assert view.moments.item(0).text() == _MOMENTS_EMPTY, (
        "a run with no notable moments left the panel blank")
    assert not (view.moments.item(0).flags() & Qt.ItemIsSelectable)
    view.deleteLater()
