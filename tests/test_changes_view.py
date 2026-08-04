"""Per-task adaptation ledger (gui/changes_view.py), offscreen QPA.

Pins the things that make this page honest rather than merely populated: that
a criterion with no evidence behind it says so instead of claiming a change,
that the two speed paths are never confused (writing the absolute ramp onto an
authored-speed bot destroys the scenario), that the before/after is READ from
the two .sce files rather than reconstructed, and that the page never touches
the model it is reporting on.

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    # the offscreen platform has no font database of its own
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import ADAPTIVE_SUFFIX, Settings  # noqa: E402
from kovadapt.analysis.movement import MIN_FLICK_DEG  # noqa: E402
from kovadapt.gui import motion  # noqa: E402
from kovadapt.gui.changes_view import (  # noqa: E402
    ChangesView,
    _dodge_knob,
    Knob,
    build_knobs,
    planned_weights,
    read_report_evidence,
    read_sce_facts,
    scan_profiles,
    size_check,
    takeaway,
)
from kovadapt.profile.player import PlayerProfile, RegionPosterior  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never touch the developer's real ~/.kovadapt. Settings.profile_dir is a
    CLASS-LEVEL default evaluated at import, so every Settings below also
    passes profile_dir explicitly — patching Path.home alone does not move it."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


# --------------------------------------------------------------- .sce fixture
_HEAD = """Name={name}
AddedBots=bot1.bot
PlayerTeam=1
Timelimit=60.0

[Character Profile]
Name=Player
MaxHealth=100.0
MainBBRadius=1.0

[Character Profile]
Name=char1
MaxHealth=1.0
MaxSpeed={speed}
MainBBRadius=0.5
MainBBHeight=2.0
ProjBBRadius=0.5

[Bot Profile]
Name=bot1
CharacterProfile=char1
DodgeProfileNames=Dodge1

[Dodge Profile]
Name=Dodge1
MinLRTimeChange=0.4
MaxLRTimeChange=0.9
LeftStrafeTimeMult=1.0
RightStrafeTimeMult=1.0
JumpFrequency=0.1

[Map Data]
reflex map version 8
global
\tentity
\t\ttype WorldSpawn
\tentity
\t\ttype PlayerSpawn
\t\tVector3 position 0.000000 0.000000 -960.000000
\t\tBool8 teamB 0
"""


def _sce_text(name: str, speed: float = 1300.0, xs=6, ys=5) -> str:
    body = _HEAD.format(name=name, speed=speed)
    pts = [
        f"\tentity\n\t\ttype PlayerSpawn\n"
        f"\t\tVector3 position {x * 200 - 600}.000000 {y * 250 + 200}.000000 960.000000\n"
        f"\t\tBool8 teamA 0\n"
        for x in range(xs) for y in range(ys)
    ]
    return body + "".join(pts)


def _install(tmp_path: Path, name="Wall Task", speed=1300.0, xs=6, ys=5) -> Settings:
    root = tmp_path / "game"
    (root / "stats").mkdir(parents=True, exist_ok=True)
    scen = root / "Saved" / "SaveGames" / "Scenarios"
    scen.mkdir(parents=True, exist_ok=True)
    (scen / f"{name}.sce").write_text(_sce_text(name, speed, xs, ys),
                                     encoding="utf-8")
    return Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof"),
                    telemetry_enabled=False, onboarding_done=True)


def _variant(settings: Settings, profile: PlayerProfile, name="Wall Task"):
    """Write a real variant the way the app does, so the ledger reads a real
    before/after rather than a fixture pretending to be one."""
    from kovadapt.adapt.engine import AdaptationEngine
    from kovadapt.scenario.generator import generate_adaptive_variant

    plan = AdaptationEngine(settings).plan(profile, None)
    out = generate_adaptive_variant(
        settings.scenarios_dir / f"{name}.sce", plan, settings,
        settings.scenarios_dir / f"{name}{ADAPTIVE_SUFFIX}.sce")
    profile.save(settings.profile_path)      # plan() mutates: persist, as watcher does
    return plan, out


def _trained(name="Wall Task", runs=14) -> PlayerProfile:
    prof = PlayerProfile(scenario=name + ADAPTIVE_SUFFIX)
    prof.archetype = "clicking"
    prof.run_count = runs
    prof.ewma_accuracy = 0.97
    prof.ewma_kps = 1.5
    for _ in range(6):          # observe_bias, not hand-set: it is what stamps
        prof.observe_bias(0.30)  # the flick floor the EWMA was earned under
    prof.target_scale = 0.80
    prof.movement = 0.55
    prof.ou_state = 0.2
    prof.last_focus = "r2c1"
    prof.last_run_ts = "2026-07-28T21:00:00"
    prof.regions = {"r2c1": RegionPosterior(mean=0.4, var=0.08, n=6)}
    prof.history = [{"ts": f"2026-07-28T20:{i:02d}:00", "accuracy": 0.97,
                     "kps": 1.5, "score": 700.0, "target_scale": 0.9,
                     "movement": 0.4, "focus": "r2c1"} for i in range(runs)]
    return prof


# ------------------------------------------------------------ evidence rule
def test_a_knob_cannot_exist_without_its_evidence():
    """The structural half of the cite-everything rule, same trick as
    dashboard.Hero's required because-clause."""
    with pytest.raises(ValueError):
        Knob(key="k", name="k", lo=0.0, hi=1.0, baseline=0.0, now=0.5)
    ok = Knob(key="k", name="k", lo=0.0, hi=1.0, baseline=0.0, now=0.5,
              evidence="because 3 runs")
    assert ok.moved and not ok.at_bound
    # a move under 1% of the knob's own range is not a move
    assert not Knob(key="k", name="k", lo=0.0, hi=100.0, baseline=50.0,
                    now=50.2, evidence="x").moved


def test_cold_profile_claims_nothing(tmp_path):
    s = _install(tmp_path)
    prof = PlayerProfile(scenario="Wall Task" + ADAPTIVE_SUFFIX)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    knobs = build_knobs(prof, s, facts, read_report_evidence(s.profile_path,
                                                             "Wall Task"))
    assert [k.key for k in knobs] == ["target_scale", "target_speed",
                                      "spawn_focus", "dodge_bias", "movement"]
    assert not any(k.measured for k in knobs), \
        "a profile with no runs must not back a single criterion"
    for knob in knobs:
        assert knob.evidence
    head = takeaway(knobs, prof, facts)
    assert "no completed runs" in head and "cold-start default" in head


def test_takeaway_counts_only_evidenced_moves(tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    knobs = build_knobs(prof, s, facts, read_report_evidence(s.profile_path,
                                                            "Wall Task"))
    moved = [k for k in knobs if k.measured and k.moved]
    head = takeaway(knobs, prof, facts)
    assert head.startswith(f"{len(moved)} of {len(knobs)} criteria")
    assert "14 runs" in head
    # no variant was written, and the headline has to admit it
    assert "no [Adaptive] file has been written yet" in head


# ------------------------------------------------------- the two speed paths
def test_authored_speed_is_modulated_and_the_ramp_is_not_offered(tmp_path):
    """ARCHITECTURE.md's loudest scenario contract: an authored MaxSpeed is scaled
    0.65-1.35x, and the absolute 0-170 ramp applies ONLY to base-speed-0
    walls. The page must never present one as the other."""
    s = _install(tmp_path, speed=1300.0)
    prof = _trained()
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert facts.speed_path == "multiplier"
    assert facts.authored == {"char1": 1300.0}
    knob = build_knobs(prof, s, facts, read_report_evidence(s.profile_path,
                                                            "Wall Task"))[1]
    assert (knob.lo, knob.hi, knob.baseline) == (0.65, 1.35, 1.0)
    assert "MaxSpeed=1300" in knob.evidence
    assert "never writes the absolute 0-170" in knob.evidence


def test_static_wall_gets_the_absolute_ramp(tmp_path):
    s = _install(tmp_path, speed=0.0)
    prof = _trained()
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert facts.speed_path == "ramp"
    knob = build_knobs(prof, s, facts, read_report_evidence(s.profile_path,
                                                            "Wall Task"))[1]
    assert (knob.lo, knob.hi, knob.baseline) == (0.0, 170.0, 0.0)
    assert "MaxSpeed=0" in knob.evidence and "static wall" in knob.evidence


def test_without_the_base_file_the_speed_path_refuses_to_guess(tmp_path):
    s = Settings(kovaaks_root=str(tmp_path / "nogame"),
                 profile_dir=str(tmp_path / "prof"), telemetry_enabled=False)
    facts = read_sce_facts(s, "Wall Task", None)
    assert facts.speed_path == "unknown" and not facts.have_base
    assert "Wall Task.sce" in facts.error
    knob = build_knobs(_trained(), s, facts,
                       read_report_evidence(s.profile_path, "Wall Task"))[1]
    assert not knob.measured
    assert "cannot say which speed path applies" in knob.evidence


# --------------------------------------------------------- the file ledger
def test_the_before_after_is_read_from_the_two_files(tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    plan, out = _variant(s, prof)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert facts.have_base and facts.have_variant
    rows = {r.label: r for r in facts.rows}
    radius = rows["char1 - MainBBRadius"]
    assert radius.base == "0.5"
    assert float(radius.adaptive) == pytest.approx(0.5 * plan.target_scale, rel=1e-3)
    speed = rows["char1 - MaxSpeed"]
    # authored speed MODULATED, never replaced by the 0-170 ramp
    assert float(speed.adaptive) == pytest.approx(1300.0 * plan.target_speed_mult,
                                                  rel=1e-3)
    assert float(speed.adaptive) > 170.0
    assert facts.description.startswith("kovadapt auto-generated")
    assert facts.written


def test_a_missing_variant_is_a_state_not_a_crash(tmp_path):
    s = _install(tmp_path)
    facts = read_sce_facts(s, "Wall Task", None)
    assert facts.have_base and not facts.have_variant
    assert facts.rows and all(r.adaptive == "—" for r in facts.rows)
    assert all(r.delta == "" for r in facts.rows)


def test_size_check_reconciles_the_file_with_the_model(tmp_path):
    """The file's size ratio is the model's scale TIMES the movement coupling,
    so the two numbers legitimately differ and the arithmetic has to be shown
    — and when they do not reconcile, that is the finding."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert "reconcile" in size_check(prof, s, facts)
    prof.target_scale = 2.4              # a model state the file predates
    assert "does not match" in size_check(prof, s, facts)


# ------------------------------------------------------------- spawn mapping
def test_spawn_shares_are_measured_from_the_variant(tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    sm = facts.spawns
    assert sm is not None and sm.total_base == 30 and not sm.reason
    assert sm.focus == prof.last_focus
    # every region key is on the r{row}c{col} contract and inside the grid
    for key in list(sm.base) + list(sm.adaptive):
        row, col = key[1:].split("c")
        assert 0 <= int(row) < s.region_rows and 0 <= int(col) < s.region_cols
    knob = build_knobs(prof, s, facts,
                       read_report_evidence(s.profile_path, "Wall Task"))[2]
    assert knob.now == pytest.approx(sm.share(sm.focus))
    assert "Verified in the file" in knob.evidence


def test_a_layout_too_small_to_reweight_says_so(tmp_path):
    """resample_spawns leaves the layout alone below one candidate per cell, so
    no focus can be applied at all — claiming spawn emphasis there would be a
    change that never happened."""
    s = _install(tmp_path, xs=3, ys=3)          # 9 spawns for a 5x5 grid
    prof = _trained()
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert facts.spawns is not None and facts.spawns.reason
    assert "leaves the layout untouched" in facts.spawns.reason
    knob = build_knobs(prof, s, facts,
                       read_report_evidence(s.profile_path, "Wall Task"))[2]
    assert not knob.measured and knob.note == facts.spawns.reason


def test_an_unobserved_focus_arm_is_exploration_not_a_finding(tmp_path):
    """A Thompson draw from an untouched prior moves real spawns, but calling
    that evidence would dress the bandit's exploration up as a weakness."""
    s = _install(tmp_path)
    prof = _trained()
    prof.regions = {}                            # focus arm has no observations
    _variant(s, prof)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    knob = build_knobs(prof, s, facts,
                       read_report_evidence(s.profile_path, "Wall Task"))[2]
    assert not knob.measured and knob.flag == "exploration"
    assert "no observations at all" in knob.evidence
    assert "exploration, not a finding" in knob.note


def test_planned_weights_never_touch_the_profile(tmp_path):
    """profile.region() creates arms as a side effect; this page must not."""
    s = _install(tmp_path)
    prof = _trained()
    before = dict(prof.regions)
    weights = planned_weights(prof, s)
    assert weights and weights[prof.last_focus] == pytest.approx(s.focus_weight)
    assert prof.regions.keys() == before.keys()
    assert planned_weights(PlayerProfile(scenario="x"), s) == {}


# --------------------------------------------------------- report evidence
def _write_report(s: Settings, slug: str, ts: str, **over) -> None:
    d = s.profile_path / "reports" / slug
    d.mkdir(parents=True, exist_ok=True)
    body = {"scenario": "Wall Task", "started_iso": ts, "score": 1.0,
            "accuracy": 0.9, "avg_ttk": 0.5, "kills": 10, "kps": 1.0,
            "n_flicks": 20,
            # A report without this is a report from before the floor moved,
            # and the page is right to exclude it. Stamped here so the default
            # fixture models a run the CURRENT app produced; the exclusion
            # itself is tested by overriding it.
            "flick_floor_deg": MIN_FLICK_DEG,
            "bias": {"left": {"n": 9}, "right": {"n": 8}, "bias_score": 0.3},
            "region_deficits": {"r2c1": 0.8},
            "input_health": {"jitter_ms": 0.3, "polling_hz_est": 1000.0}}
    body.update(over)
    (d / f"{ts}.json").write_text(json.dumps(body), encoding="utf-8")


def test_report_evidence_reads_both_slug_directories(tmp_path):
    """Runs of the base scenario and of the [Adaptive] variant feed one
    profile, and their reports land in two different slug directories."""
    s = _install(tmp_path)
    _write_report(s, "Wall_Task", "2026-07-28T20-00-00")
    _write_report(s, "Wall_Task_Adaptive_", "2026-07-28T20-05-00")
    ev = read_report_evidence(s.profile_path, "Wall Task")
    assert ev.files == 2
    assert set(ev.dirs) == {"Wall_Task", "Wall_Task_Adaptive_"}
    assert ev.region_n["r2c1"] == 2
    assert ev.region_mean["r2c1"] == pytest.approx(0.8)
    assert ev.bias_runs == 2 and ev.bias_mean == pytest.approx(0.3)
    assert ev.degraded == 0


def test_bias_evidence_applies_the_watchers_own_gate(tmp_path):
    """directional_bias returns a flat 0.0 below 3 flicks per side, so a
    bias_score that merely EXISTS is not an observation."""
    s = _install(tmp_path)
    _write_report(s, "Wall_Task", "2026-07-28T20-00-00",
                  bias={"left": {"n": 2}, "right": {"n": 9}, "bias_score": 0.0})
    _write_report(s, "Wall_Task", "2026-07-28T20-01-00", n_flicks=4)
    _write_report(s, "Wall_Task", "2026-07-28T20-02-00")      # the only usable one
    ev = read_report_evidence(s.profile_path, "Wall Task")
    assert ev.files == 3 and ev.bias_runs == 1


def test_reports_from_an_older_flick_floor_are_excluded_and_named(tmp_path):
    """A bias score is only comparable to another one measured the same way.

    These reports clear the flick gate — they are not noisy, not short, not
    one-sided. They were segmented at 0.33 degrees, where the overshoot ratio
    measures segmentation error rather than aim, and pooling them into the
    mean is how a wrong verdict outlives the rule that produced it.

    Excluding them silently would be its own defect: the page would read "0
    contributing" over a directory holding five reports, which looks like the
    recording broke. So the count and the floor are both named.
    """
    s = _install(tmp_path)
    _write_report(s, "Wall_Task", "2026-07-28T20-00-00", flick_floor_deg=0.33)
    _write_report(s, "Wall_Task", "2026-07-28T20-01-00")          # current
    _write_report(s, "Wall_Task", "2026-07-28T20-02-00", flick_floor_deg=None)
    ev = read_report_evidence(s.profile_path, "Wall Task")
    assert ev.files == 3
    assert ev.bias_runs == 1, "only the report on the current floor counts"
    assert ev.bias_stale == 2, "a missing floor is an OLD floor, not a pass"
    assert ev.stale_floors == (0.33,)

    prof = PlayerProfile(scenario="Wall Task" + ADAPTIVE_SUFFIX)
    prof.observe_bias(0.3)
    knob = _dodge_knob(prof, s, ev)
    assert "2 more clear it" in knob.evidence
    assert "0.33-degree flick floor" in knob.evidence
    assert "instead of today's 2 degrees" in knob.evidence


def test_dropped_bias_does_not_read_as_a_recording_failure(tmp_path):
    """The first launch after the floor moved shows a profile with no bias and
    a directory full of reports that used to supply one. "No run has produced
    a usable measurement yet" is true of the profile and false about the
    runs — on its own it sends the user to the Optimizer for a fault that is
    not there."""
    s = _install(tmp_path)
    _write_report(s, "Wall_Task", "2026-07-28T20-00-00", flick_floor_deg=0.33)
    ev = read_report_evidence(s.profile_path, "Wall Task")
    fresh = PlayerProfile(scenario="Wall Task" + ADAPTIVE_SUFFIX)
    why = _dodge_knob(fresh, s, ev).evidence
    assert "no run has produced a usable directional-bias measurement" in why
    assert "dropped rather than lost to noise" in why
    assert "a 0.33-degree flick floor instead of today's 2 degrees" in why
    assert "the next run re-earns one" in why


def test_noisy_runs_are_counted_with_the_one_shared_gate(tmp_path):
    """analysis/report.py:input_degraded is the only definition of "too noisy
    to read microstructure", and this page defers to it rather than restating
    the thresholds."""
    s = _install(tmp_path)
    _write_report(s, "Wall_Task", "2026-07-28T20-00-00",
                  input_health={"jitter_ms": 9.0, "polling_hz_est": 1000.0})
    _write_report(s, "Wall_Task", "2026-07-28T20-01-00")
    ev = read_report_evidence(s.profile_path, "Wall Task")
    assert ev.bias_runs == 2 and ev.degraded == 1
    prof = _trained()
    knob = build_knobs(prof, s, read_sce_facts(s, "Wall Task", prof.last_focus),
                       ev)[3]
    assert "too noisy" in knob.note and "input_degraded" in knob.note


def test_report_scan_survives_a_corrupt_file(tmp_path):
    s = _install(tmp_path)
    _write_report(s, "Wall_Task", "2026-07-28T20-00-00")
    (s.profile_path / "reports" / "Wall_Task" / "bad.json").write_text("{oops")
    assert read_report_evidence(s.profile_path, "Wall Task").files == 1


# --------------------------------------------------------------- the picker
def test_scan_profiles_uses_each_files_own_scenario_name(tmp_path):
    """The slug is lossy, so a name can only come from inside the JSON."""
    s = _install(tmp_path)
    _trained("Wall Task", runs=14).save(s.profile_path)
    older = PlayerProfile(scenario="Other Task" + ADAPTIVE_SUFFIX)
    older.run_count = 3
    older.last_run_ts = "2026-01-01T00:00:00"
    older.save(s.profile_path)
    (s.profile_path / "profiles" / "junk.json").write_text("not json")

    entries = scan_profiles(s.profile_path)
    assert [e.base for e in entries] == ["Wall Task", "Other Task"]   # recent first
    assert entries[0].stored == "Wall Task" + ADAPTIVE_SUFFIX
    assert entries[0].runs == 14
    assert scan_profiles(tmp_path / "nothing") == []


def test_both_spellings_of_one_task_collapse_to_one_entry(tmp_path):
    s = _install(tmp_path)
    _trained("Wall Task", runs=14).save(s.profile_path)
    bare = PlayerProfile(scenario="Wall Task")     # older/manual profile
    bare.run_count = 2
    bare.save(s.profile_path)
    entries = scan_profiles(s.profile_path)
    assert [(e.base, e.runs) for e in entries] == [("Wall Task", 14)]


# ---------------------------------------------------------------- the widget
def test_constructs_with_nothing_on_disk(qapp, tmp_path):
    s = _install(tmp_path)
    view = ChangesView(s)
    assert not view.picker.isEnabled()
    assert view.picker.currentText() == "no player models yet"
    assert view.scenario == ""
    assert "player model yet" in view.subhead.text()
    view.restyle()                       # safe with no data at all
    view.deleteLater()


def test_shows_a_task_and_strips_the_adaptive_suffix(qapp, tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    view = ChangesView(s)
    assert view.scenario == "Wall Task"          # auto-selected the only task
    view.show_scenario("Wall Task" + ADAPTIVE_SUFFIX)
    assert view.scenario == "Wall Task"          # compounding guard
    assert len(view._knobs) == 5
    assert view.ladder._knobs == view._knobs
    assert view.grid._map is not None
    assert view.ledger._rows == view._facts.rows
    assert "criteria have moved" in view.headline.text()
    view.restyle()
    view.deleteLater()


def test_rendering_never_writes_to_the_model(qapp, tmp_path):
    """The report must not be able to change what it reports on."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    path = PlayerProfile.path_for(prof.scenario, s.profile_path)
    before = path.read_bytes()
    view = ChangesView(s)
    view.show_scenario("Wall Task")
    view.refresh()
    view.restyle()
    assert path.read_bytes() == before
    view.deleteLater()


def test_refresh_keeps_the_selected_task(qapp, tmp_path):
    s = _install(tmp_path)
    _trained("Wall Task").save(s.profile_path)
    other = _trained("Other Task")
    other.last_run_ts = "2020-01-01T00:00:00"
    other.save(s.profile_path)
    view = ChangesView(s)
    view.show_scenario("Other Task")
    view.refresh()
    assert view.scenario == "Other Task"
    assert view.picker.currentData() == "Other Task"
    view.deleteLater()


# --------------------------------------------------------------- motion rules
def test_motion_off_jumps_to_the_end_state_without_a_timer(qapp, tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    s.motion = motion.OFF
    view = ChangesView(s)
    view.show()
    view.show_scenario("Wall Task")
    assert not view._clock.isActive()
    for art in view._art:
        assert art._reveal == art.DONE
        assert art.reveal_total(s) == 0
    view.deleteLater()


def test_the_clock_never_runs_while_the_page_is_hidden(qapp, tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    view = ChangesView(s)
    # a reveal requested before the page is ever shown must not start a timer
    view.show_scenario("Wall Task")
    assert not view._clock.isActive() and view._pending
    view.show()
    assert view._clock.isActive()
    view.hide()
    assert not view._clock.isActive()
    assert view._pending, "the interrupted reveal must replay when it returns"
    view.show()
    assert view._clock.isActive()
    view.deleteLater()


def test_the_reveal_is_glyph_rate_and_resolves_inside_one_ceremony(qapp, tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    view = ChangesView(s)
    view.resize(1400, 1400)
    view.show()
    view.show_scenario("Wall Task")
    assert view._clock.interval() == motion.GLYPH_MS
    total = max(a.reveal_total(s) for a in view._art)
    assert 0 < total <= motion.CEREMONY, "a per-panel reveal outran one beat"
    # nothing ambient: once the reveal is over the clock stops for good
    view._t0 -= 10.0                     # pretend the reveal already finished
    view._tick()
    assert not view._clock.isActive()
    for art in view._art:
        assert art._reveal == art.DONE
    view.deleteLater()


def test_reduced_motion_shortens_the_reveal_but_keeps_it(qapp, tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    view = ChangesView(s)
    view.resize(1400, 1400)
    full = max(a.reveal_total(s) for a in view._art)
    s.motion = motion.REDUCED
    reduced = max(a.reveal_total(s) for a in view._art)
    assert 0 < reduced < full
    view.deleteLater()


def test_the_page_owns_no_second_charting_stack(qapp, tmp_path):
    """House rule: pyqtgraph exists only in gui/replay.py."""
    import inspect

    from kovadapt.gui import changes_view

    src = inspect.getsource(changes_view)
    assert "pyqtgraph" not in src
    assert "import pyqtgraph" not in src
