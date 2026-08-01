"""Dashboard shape tests (offscreen QPA): HERO + ONE TREND + PLAY.

Pins the contract the rebuild exists to enforce — three hero numerals that can
never render without a "because …" clause, a cold start that dashes instead of
faking zeros, the accuracy trend as gui/viz.py character art rather than
pyqtgraph, the Play/overlay lockups intact, and the log tucked behind a
disclosure that takes none of the column's vertical stretch.

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no system font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import ADAPTIVE_SUFFIX, Settings  # noqa: E402
from kovadapt.profile.player import PlayerProfile  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Settings.save() defaults to Path.home()/.kovadapt/settings.json — the
    dashboard's overlay/hint paths save, and must never write the developer's
    real settings file (that actually happened once)."""
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


def _profile(scenario: str, accs: list[float], *, regions: int = 0,
             ewma: float | None = None, when: datetime | None = None,
             bias_obs: int = 0) -> PlayerProfile:
    """A profile with `accs` worth of history, stamped `when` (default now).

    `bias_obs` is explicit because readiness counts directional-bias
    MEASUREMENTS, not runs: a profile can log a hundred runs and never
    produce one (the watcher needs 8+ flicks with 3+ per side, and `replay`
    supplies none at all).
    """
    from kovadapt.profile.player import RegionPosterior

    ts = (when or datetime.now()).isoformat()
    prof = PlayerProfile(scenario=scenario + ADAPTIVE_SUFFIX)
    prof.run_count = len(accs)
    prof.ewma_accuracy = sum(accs) / len(accs) if ewma is None else ewma
    prof.history = [{"ts": ts, "accuracy": round(a, 4)} for a in accs]
    for i in range(regions):
        prof.regions[f"r{i // 5}c{i % 5}"] = RegionPosterior(mean=0.1, var=0.2, n=4)
    if bias_obs:
        prof.bias_obs = bias_obs
        prof.ewma_bias = 0.2
    return prof


def _dashboard(settings, scenario: str = "", prof: PlayerProfile | None = None):
    """A Dashboard whose picker is on `scenario` (its .sce and profile
    written first, so __init__'s own refresh_profile sees them)."""
    from kovadapt.gui.dashboard import Dashboard

    if scenario:
        (settings.scenarios_dir / f"{scenario}.sce").write_text("[Scenario]\n")
    if prof is not None:
        prof.save(settings.profile_path)
    dash = Dashboard(settings)
    if scenario:
        dash.scenario.setCurrentText(scenario)
    return dash


# ------------------------------------------------------------- cold start
def test_cold_start_dashes_every_hero_and_still_cites(qapp, settings):
    """No profile: dashes, never fake zeros — and each dash explains itself."""
    dash = _dashboard(settings, "Alpha Track")
    assert set(dash.heroes) == {"readiness", "form", "load"}
    for key, card in dash.heroes.items():
        assert card.value.text() == "—", key
        assert card.word.text(), key                    # a state word regardless
        assert card.because.text().startswith("because "), key
        assert card.toolTip().strip(), key
    assert "0%" not in " ".join(c.value.text() for c in dash.heroes.values())
    dash.shutdown()
    dash.deleteLater()


def test_hero_cards_never_render_a_value_without_a_because(qapp, settings):
    """The whole point of the rebuild: a numeral always carries its evidence."""
    prof = _profile("Beta Click", [0.60, 0.62, 0.65, 0.68, 0.70, 0.72], regions=25)
    dash = _dashboard(settings, "Beta Click", prof)
    for key, card in dash.heroes.items():
        assert card.value.text().strip(), key
        assert card.value.text() != "—", key            # this profile has data
        assert card.because.text().startswith("because "), key
        assert len(card.because.text()) > len("because ") + 10, key
    dash.shutdown()
    dash.deleteLater()


# ----------------------------------------------------------------- heroes
def test_readiness_hero_is_the_profiles_own_readiness(qapp, settings):
    from kovadapt.gui.dashboard import readiness_hero

    prof = _profile("Gamma", [0.6] * 12, regions=9)
    n = settings.region_cols * settings.region_rows
    ready = prof.readiness(n)
    hero = readiness_hero(prof, n)
    assert hero.value == f"{ready['score']:.0%}"        # not re-derived
    assert hero.word == ready["stage"]
    for part in ready["detail"]:                        # the evidence, verbatim
        assert part in hero.because
    assert hero.tip == ready["message"]


def test_readiness_hero_dashes_with_no_runs(qapp, settings):
    from kovadapt.gui.dashboard import readiness_hero

    hero = readiness_hero(PlayerProfile(scenario="Cold"), 25)
    assert hero.value == "—"
    assert "no history" in hero.because
    assert "first run" in hero.because


def test_form_hero_cites_recent_window_against_the_ewma_baseline(qapp, settings):
    from kovadapt.gui.dashboard import FORM_WINDOW, form_hero

    # last five runs sit well above a deliberately low baseline EWMA
    prof = _profile("Delta", [0.50, 0.51, 0.70, 0.71, 0.72, 0.73, 0.74],
                    ewma=0.60)
    hero = form_hero(prof)
    recent = sum([0.70, 0.71, 0.72, 0.73, 0.74]) / FORM_WINDOW
    assert hero.value == f"{(recent - 0.60) * 100.0:+.1f}pp"
    assert hero.value.startswith("+")
    assert hero.word == "climbing"
    assert f"{recent:.1%}" in hero.because and "60.0%" in hero.because
    assert f"last {FORM_WINDOW} runs" in hero.because


def test_form_hero_flags_a_dip_and_a_hold(qapp, settings):
    from kovadapt.gui.dashboard import form_hero

    assert form_hero(_profile("D1", [0.7] * 6, ewma=0.80)).word == "dipping"
    assert form_hero(_profile("D2", [0.7] * 6, ewma=0.70)).word == "holding"


def test_form_hero_dashes_under_the_minimum_and_says_so(qapp, settings):
    from kovadapt.gui.dashboard import FORM_MIN_RUNS, form_hero

    hero = form_hero(_profile("Eps", [0.6, 0.7]))
    assert hero.value == "—"
    assert str(FORM_MIN_RUNS) in hero.because
    assert "2" in hero.because                    # how many it actually has


def test_load_hero_uses_session_fatigue_once_it_is_trusted(qapp, settings):
    from kovadapt.gui.dashboard import load_hero

    prof = _profile("Zeta", [0.6] * 8)
    fat = {"level": "declining", "score": 0.55, "trend": 0.04, "runs": 6,
           "message": "Flick quality is trending down."}
    hero = load_hero(prof, fat, settings.fatigue_min_runs)
    assert hero.value == "55%"
    assert hero.word == "declining"
    assert hero.tone == "warn"
    assert "6 telemetry runs" in hero.because
    # The slope is a BADNESS slope (analysis/fatigue.py): positive means
    # overshoot and flick duration are rising, i.e. quality falling. The
    # clause used to print it as "flick quality trends +4.0%", which states
    # the opposite of the evidence — the direction word carries the sign now.
    assert "4.0%" in hero.because                 # the Theil-Sen slope, named
    assert "degrading" in hero.because
    assert "+4.0%" not in hero.because
    assert hero.tip == fat["message"]


def test_load_hero_falls_back_to_volume_below_the_fatigue_minimum(qapp, settings):
    """runs < fatigue_min_runs means "no evidence yet", not "you are fresh" —
    presenting the tracker's placeholder score would invent signal."""
    from kovadapt.gui.dashboard import load_hero

    prof = _profile("Eta", [0.6] * 4)             # all stamped today
    hero = load_hero(prof, {"level": "fresh", "score": 0.0, "runs": 2},
                     settings.fatigue_min_runs)
    assert hero.value == "4 runs"
    assert hero.word == "light"
    assert "4 runs logged today" in hero.because
    assert f"{settings.fatigue_min_runs} runs with telemetry" in hero.because
    assert "(2 so far)" in hero.because


def test_load_hero_counts_only_todays_runs(qapp, settings):
    from kovadapt.gui.dashboard import load_hero

    prof = _profile("Theta", [0.6] * 5, when=datetime.now() - timedelta(days=2))
    hero = load_hero(prof, {}, settings.fatigue_min_runs)
    assert hero.value == "0 runs"
    assert hero.word == "idle"


def test_load_hero_dashes_with_no_history_at_all(qapp, settings):
    from kovadapt.gui.dashboard import load_hero

    hero = load_hero(PlayerProfile(scenario="Iota"), {},
                     settings.fatigue_min_runs)
    assert hero.value == "—"
    assert "neither yet" in hero.because


def test_a_report_drives_load_and_a_scenario_switch_clears_it(qapp, settings):
    """The fatigue reading belongs to the session that produced it."""
    from kovadapt.analysis.report import RunReport

    prof = _profile("Kappa", [0.6] * 8)
    dash = _dashboard(settings, "Kappa", prof)
    (settings.scenarios_dir / "Lambda.sce").write_text("[Scenario]\n")
    rep = RunReport(
        scenario="Kappa" + ADAPTIVE_SUFFIX, started_iso="2026-07-28T10:00:00",
        score=420.0, accuracy=0.61, avg_ttk=0.9, kills=30, kps=1.4,
        fatigue={"level": "fatigued", "score": 0.9, "trend": 0.06, "runs": 7},
        summary_text="30 kills at 61% accuracy.")
    dash._on_report(rep)
    assert dash.heroes["load"].value.text() == "90%"
    assert dash.heroes["load"].word.text() == "fatigued"

    dash.scenario.setCurrentText("Lambda")        # different scenario entirely
    assert dash._fatigue == {}
    assert dash.heroes["load"].value.text() != "90%"
    dash.shutdown()
    dash.deleteLater()


# ------------------------------------------------------------- the trend
def test_trend_is_ascii_art_and_pyqtgraph_is_gone(qapp, settings):
    import pyqtgraph as pg

    from kovadapt.gui.viz import AsciiTrend

    prof = _profile("Mu", [0.55, 0.6, 0.65, 0.7])
    dash = _dashboard(settings, "Mu", prof)
    assert isinstance(dash.trend, AsciiTrend)
    assert dash.findChildren(pg.PlotWidget) == []
    assert dash.trend._values == [0.55, 0.6, 0.65, 0.7]
    assert dash.trend._tag == "70%"
    assert dash.trend_caption.wordWrap()
    assert dash.trend_caption.property("dim") is True
    dash.shutdown()
    dash.deleteLater()


def test_trend_stays_empty_below_two_runs(qapp, settings):
    dash = _dashboard(settings, "Nu", _profile("Nu", [0.6]))
    assert dash.trend._values == []               # AsciiTrend paints its own
    dash.shutdown()                               # "not enough runs yet"
    dash.deleteLater()


# ---------------------------------------------------------------- the log
def test_log_is_collapsed_capped_and_takes_no_stretch(qapp, settings):
    dash = _dashboard(settings, "Xi")
    lay = dash.layout()
    assert dash.log.isHidden()
    assert dash.log_btn.text() == "[ log ]"
    assert not dash.log_btn.isChecked()
    assert dash.log.maximumHeight() <= 150
    # NO panel may carry the column's vertical stretch — every one of them has
    # a natural height, and handing the slack to a panel is what made the
    # trend render 1400px tall on a tall window (and, earlier, a hint bar fill
    # a whole section). The slack lives in a trailing spacer instead.
    stretches = {lay.itemAt(i).widget(): lay.stretch(i)
                 for i in range(lay.count()) if lay.itemAt(i).widget()}
    assert sum(stretches.values()) == 0, "a panel is absorbing the slack"
    assert lay.itemAt(lay.count() - 1).spacerItem() is not None
    assert dash.trend.maximumHeight() <= 220
    dash.shutdown()
    dash.deleteLater()


def test_log_toggle_reveals_and_clears_the_unread_mark(qapp, settings):
    dash = _dashboard(settings, "Omicron")
    dash.append_log("scenario file not found: nope.sce")
    assert dash.log_btn.text() == "[ log • ]"     # collapsed failures still show
    dash.log_btn.setChecked(True)
    assert not dash.log.isHidden()
    assert dash.log_btn.text() == "[ log ]"
    assert "nope.sce" in dash.log.toPlainText()
    dash.append_log("stopped")                    # open: no mark
    assert dash.log_btn.text() == "[ log ]"
    dash.log_btn.setChecked(False)
    assert dash.log.isHidden()
    dash.shutdown()
    dash.deleteLater()


# ------------------------------------------------------------ play + theme
def test_play_lockup_and_overlay_row_survive(qapp, settings):
    dash = _dashboard(settings, "Pi")
    for w in (dash.scenario, dash.refresh_btn, dash.start_btn, dash.play_btn,
              dash.launch_btn, dash.install_lbl, dash.rec_lbl,
              dash.ov_toggle, dash.ov_unlock, dash.ov_opacity, dash.ov_auto):
        assert w is not None
    assert dash.scenario.currentText() == "Pi"
    assert dash.start_btn.text() == "Start adapting"
    assert dash.worker is None                    # nothing launched by building
    dash.shutdown()
    dash.deleteLater()


def test_scenario_change_refreshes_the_heroes(qapp, settings):
    """refresh_profile() runs in __init__ AND on every pick change."""
    prof = _profile("Rho", [0.6] * 30, regions=25)
    dash = _dashboard(settings, "Sigma", prof)    # picker starts on Sigma
    (settings.scenarios_dir / "Rho.sce").write_text("[Scenario]\n")
    assert dash.heroes["readiness"].value.text() == "—"      # Sigma is cold
    dash.scenario.setCurrentText("Rho")
    assert dash.heroes["readiness"].value.text() != "—"
    assert dash.last_profile is not None
    assert dash.last_profile.run_count == 30
    dash.shutdown()
    dash.deleteLater()


def test_hero_numerals_render_at_hero_scale_in_the_mono_face(qapp, settings):
    """theme.py's app QSS opens with `* { font-family: "Segoe UI"; font-size:
    13px }`, and a QSS font property beats QWidget.setFont() — so the numeral
    size has to be restated in a widget-level sheet. Without it every hero
    rendered at body size and the lockup collapsed into a stat list.

    The ThemeManager here is load-bearing: it is what applies the app sheet,
    and without it this test would pass on setFont() alone.
    """
    from kovadapt.gui import theme
    from kovadapt.gui.dashboard import HeroStat
    from kovadapt.gui.theme import ThemeManager

    ThemeManager(qapp, settings)
    dash = _dashboard(settings, "Phi")
    dash.show()
    qapp.processEvents()                          # polish resolves the QSS font
    assert HeroStat.NUMERAL_PX == theme.CELL_SIZES[-1] * 2   # stays grid-snapped
    for key, card in dash.heroes.items():
        assert card.value.font().pixelSize() == HeroStat.NUMERAL_PX, key
        assert card.value.font().family() == theme.mono_family(), key
        assert card.name.font().family() == theme.mono_family(), key
    assert dash.log_btn.font().family() == theme.mono_family()
    dash.hide()
    dash.shutdown()
    dash.deleteLater()


def test_restyle_repaints_heroes_across_theme_switches(qapp, settings):
    from kovadapt.gui import theme
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    prof = _profile("Tau", [0.6, 0.65, 0.7, 0.75], regions=25)
    dash = _dashboard(settings, "Tau", prof)
    for mode in ("light", "dark", "rgb"):
        themes.set_mode(mode)
        dash.restyle(theme.current())
        pal = theme.current()
        # colors are resolved at render time, never cached from construction
        assert pal.accent in dash.heroes["readiness"].value.styleSheet()
        assert not dash.grab().isNull()
    dash.shutdown()
    dash.deleteLater()


def test_profile_json_round_trips_into_the_heroes(qapp, settings):
    """The dashboard reads the on-disk profile, not an in-memory shortcut."""
    prof = _profile("Upsilon", [0.6] * 24, regions=25, bias_obs=8)
    path = prof.save(settings.profile_path)
    assert json.loads(path.read_text())["run_count"] == 24
    dash = _dashboard(settings, "Upsilon")
    assert dash.heroes["readiness"].value.text() == "100%"
    assert dash.heroes["readiness"].word.text() == "dialed in"
    dash.shutdown()
    dash.deleteLater()


def test_readiness_will_not_say_dialed_in_without_bias_evidence(qapp, settings):
    """This test previously asserted the OPPOSITE, by accident.

    A profile with 24 runs and all 25 regions mapped but no directional-bias
    measurement used to render 100% / "dialed in", because the bias term read
    `1.0 if (run_count >= BIAS_RUNS and ewma_bias) else min(run_count/8, 1)` —
    an expression identically equal to `min(run_count/8, 1)`, since the else
    branch is already 1.0 once run_count reaches 8. It was a run counter
    labelled as evidence, and `kovadapt replay` reaches this state on real
    data because it never supplies a bias measurement at all.
    """
    prof = _profile("Nu", [0.6] * 24, regions=25)      # bias_obs == 0
    assert prof.bias_obs == 0 and prof.ewma_bias == 0.0
    prof.save(settings.profile_path)
    dash = _dashboard(settings, "Nu")
    hero = dash.heroes["readiness"]
    assert hero.value.text() == "85%", "the missing 15% is the bias component"
    assert hero.word.text() != "dialed in"
    because = hero.because.text().lower()
    assert "bias evidence 0/8 measurements" in because, because
    assert "run" not in because.split("bias evidence")[1][:24], (
        "the bias clause must count measurements, not runs")
    dash.shutdown()
    dash.deleteLater()


def test_the_state_word_sits_on_the_numerals_baseline(qapp, settings):
    """Qt.AlignBottom aligns bottom EDGES, and a label's bottom edge is one
    DESCENT below its baseline — so the 13px state word hung 8px below the
    48px hero numeral it qualifies, and a 12px KPI unit hung 3px below its
    24px value.

    The fonts are passed to theme.align_baselines explicitly rather than read
    off the widgets, and that is load-bearing: theme.py's app-wide
    `* { font-size: 13px }` outranks setFont(), so widget.font() reports 13px
    for BOTH labels and the correction computes to zero.
    """
    from PySide6.QtGui import QFontMetricsF

    from kovadapt.gui import theme

    prof = _profile("Baseline", [0.6] * 6, regions=4, bias_obs=2)
    prof.save(settings.profile_path)
    dash = _dashboard(settings, "Baseline")
    try:
        for key in dash.heroes:
            h = dash.heroes[key]
            numeral = theme.mono(24)
            numeral.setPixelSize(h.NUMERAL_PX)
            drop = round(QFontMetricsF(numeral).descent()
                         - QFontMetricsF(theme.body_font()).descent())
            assert drop > 0, "the fonts stopped differing; this test is moot"
            assert h.word.contentsMargins().bottom() == drop, (
                f"{key}: word margin {h.word.contentsMargins().bottom()} "
                f"!= the {drop}px descent difference")
    finally:
        dash.shutdown()
        dash.deleteLater()
