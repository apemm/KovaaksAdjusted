"""gui/changes_view.py: the ladder, the ledger and the provenance line have to
agree about TARGET SPEED, and none of them may present a number the two .sce
files do not carry.

Every case here was first found by RENDERING the page and reading it back:

  * on a fatigue-eased variant the ladder read "TARGET SPEED (X AUTHORED)
    1.00 -> 1.03x  +3%" while the file ledger in the same section read
    "char1 - MaxSpeed  1300 -> 1182  x0.909". Both are computed from the same
    variant; the ladder was reading the un-eased MODEL and the ledger the eased
    FILE. On one render the two disagreed about the direction of the change
    (+17% against x1.00), and the ladder's number was counted in the headline's
    "4 of 5 criteria have moved on evidence".
  * the provenance line echoed the variant's Description header verbatim, which
    always carries `speed=<absolute 0-170 ramp>` — a figure never written on a
    scenario whose targets author a MaxSpeed of their own, because the generator
    took the multiplier path per character. Echoing a stored string is not
    evidence that its numbers landed, and CLAUDE.md is emphatic that writing the
    ramp onto a strafe bot collapses the scenario.
  * the cold-start note on that same row read "…but the file already carries
    this number" one line under a headline saying "there is no [Adaptive] file
    on disk yet" and over a ledger whose every VARIANT cell was a dash.
  * read_sce_facts gives up on a missing base .sce before it ever looks at the
    variant, so a task whose base file had been renamed printed "no [Adaptive]
    .sce on disk carries this value" — four notes and one headline — about a
    file sitting right beside the one it could not find.

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.adapt.engine import AdaptationEngine  # noqa: E402
from kovadapt.adapt.stochastic import movement_speed, speed_multiplier  # noqa: E402
from kovadapt.config import ADAPTIVE_SUFFIX, Settings  # noqa: E402
from kovadapt.gui import motion  # noqa: E402
from kovadapt.gui.changes_view import (  # noqa: E402
    _LADDER_CAPTION,
    ChangesView,
    KnobLadder,
    _plan_fields,
    _plan_summary,
    build_knobs,
    read_report_evidence,
    read_sce_facts,
    takeaway,
)
from kovadapt.profile.player import PlayerProfile, RegionPosterior  # noqa: E402
from kovadapt.scenario.generator import generate_adaptive_variant  # noqa: E402
from kovadapt.scenario.sce import SceFile  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Never touch the developer's real ~/.kovadapt. Settings.profile_dir is a
    CLASS-LEVEL default evaluated at import, so every Settings below also passes
    profile_dir explicitly — patching Path.home alone does not move it."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


# ------------------------------------------------------------- .sce fixtures
_HEAD = """Name={name}
AddedBots={bots}
PlayerTeam=1
Timelimit=60.0

[Character Profile]
Name=Player
MaxHealth=100.0
MainBBRadius=1.0

{chars}
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

_CHAR = """[Character Profile]
Name={char}
MaxHealth=1.0
MaxSpeed={speed}
MainBBRadius=0.5
MainBBHeight=2.0
ProjBBRadius=0.5

[Bot Profile]
Name={bot}
CharacterProfile={char}
DodgeProfileNames=Dodge{i}

[Dodge Profile]
Name=Dodge{i}
MinLRTimeChange=0.4
MaxLRTimeChange=0.9
LeftStrafeTimeMult=1.0
RightStrafeTimeMult=1.0
JumpFrequency=0.1

"""

NAME = "Wall Task"


def _sce_text(name: str, speeds=(1300.0,), xs=6, ys=5) -> str:
    chars = "".join(_CHAR.format(char=f"char{i}", bot=f"bot{i}", i=i, speed=sp)
                    for i, sp in enumerate(speeds, start=1))
    bots = ";".join(f"bot{i}.bot" for i in range(1, len(speeds) + 1))
    pts = [
        f"\tentity\n\t\ttype PlayerSpawn\n"
        f"\t\tVector3 position {x * 200 - 600}.000000 {y * 250 + 200}.000000 960.000000\n"
        f"\t\tBool8 teamA 0\n"
        for x in range(xs) for y in range(ys)
    ]
    return _HEAD.format(name=name, bots=bots, chars=chars) + "".join(pts)


def _install(tmp_path: Path, speeds=(1300.0,), xs=6, ys=5) -> Settings:
    root = tmp_path / "game"
    (root / "stats").mkdir(parents=True, exist_ok=True)
    scen = root / "Saved" / "SaveGames" / "Scenarios"
    scen.mkdir(parents=True, exist_ok=True)
    (scen / f"{NAME}.sce").write_text(_sce_text(NAME, speeds, xs, ys),
                                      encoding="utf-8")
    return Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof"),
                    telemetry_enabled=False, onboarding_done=True,
                    motion=motion.OFF)


def _trained(runs=14) -> PlayerProfile:
    prof = PlayerProfile(scenario=NAME + ADAPTIVE_SUFFIX)
    prof.archetype = "clicking"
    prof.run_count = runs
    prof.ewma_accuracy = 0.97
    prof.ewma_kps = 1.5
    prof.ewma_bias = 0.30
    prof.bias_obs = 6
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


def _variant(settings: Settings, profile: PlayerProfile, fatigue: float = 0.0,
             seed: int = 7):
    """Write a real variant the way the watcher does, on a SEEDED rng so every
    assertion below is about two real files and not about a lucky draw."""
    engine = AdaptationEngine(settings, rng=np.random.default_rng(seed))
    plan = engine.plan(profile, None, fatigue=fatigue)
    generate_adaptive_variant(settings.scenarios_dir / f"{NAME}.sce", plan,
                              settings,
                              settings.scenarios_dir / f"{NAME}{ADAPTIVE_SUFFIX}.sce")
    profile.save(settings.profile_path)   # plan() mutates: persist, as watcher does
    return plan


def _knobs(settings: Settings, profile: PlayerProfile):
    facts = read_sce_facts(settings, NAME, profile.last_focus)
    ev = read_report_evidence(settings.profile_path, NAME)
    return build_knobs(profile, settings, facts, ev), facts


def _speed_row(facts, char="char1"):
    return next(r for r in facts.rows if r.label == f"{char} - MaxSpeed")


def _variant_path(settings: Settings) -> Path:
    return settings.scenarios_dir / f"{NAME}{ADAPTIVE_SUFFIX}.sce"


def _rewrite_variant(settings: Settings, sub) -> None:
    path = _variant_path(settings)
    path.write_text(sub(path.read_text(encoding="utf-8")), encoding="utf-8")


def _page(settings: Settings) -> ChangesView:
    view = ChangesView(settings)
    view.resize(1400, 2400)
    view.show_scenario(NAME)
    view.show()
    return view


def _plain(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html.replace("<br>", " "))


# ---------------------------------------- 1. the two surfaces, one variant
def test_the_ladder_plots_the_speed_the_eased_variant_actually_carries(tmp_path):
    """THE reported contradiction: ladder "1.00 -> 1.03x +3%" over a ledger row
    "1300 -> 1182 x0.909", both computed from the same file. `plan(fatigue=...)`
    eases the EMITTED plan while persisted state stays un-eased by contract, so
    the model's number and the file's are two different claims — and the ladder
    makes the file's, because the file is what the game loads."""
    s = _install(tmp_path)
    prof = _trained()
    plan = _variant(s, prof, fatigue=0.8)
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert speed.key == "target_speed"

    row = _speed_row(facts)
    written = facts.written_speeds["char1"] / facts.authored["char1"]
    model = speed_multiplier(prof.movement)
    # fixture premise: the easing really did move the emitted value away from
    # the persisted model, and in the direction the report described
    assert plan.fatigue == pytest.approx(0.8)
    assert abs(written - model) > 0.05 and written < model

    assert speed.now == pytest.approx(written, abs=1e-9), \
        "the ladder must plot the multiplier the variant carries"
    assert row.delta == f"x{written:.3g}", \
        "fixture premise: the ledger reports that same ratio for the same key"
    assert speed.delta_text.startswith("-"), \
        f"the file went DOWN by x{written:.3f}; the delta may not read as a rise"
    assert f"x{written:.3g}" in speed.evidence, \
        "the row has to show the ratio it plots, per character"
    assert "read out of the [Adaptive] file on disk" in speed.evidence
    assert "eased=0.80" in speed.evidence and "un-eased" in speed.evidence, \
        "and it has to say why the model implies something else"
    assert f"{model:.2f}x" in speed.evidence, "the model's own number is not hidden"


def test_the_ramp_row_prints_the_same_characters_as_its_ledger_row(tmp_path, qapp):
    """Same defect on the other speed path, where the numbers are absolute: the
    ladder said "0 -> 108u/s" over a ledger row reading "0 -> 54.8"."""
    s = _install(tmp_path, speeds=(0.0,))
    prof = _trained()
    _variant(s, prof, fatigue=0.8)
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert facts.speed_path == "ramp"

    row = _speed_row(facts)
    assert speed.now == pytest.approx(float(row.adaptive), abs=1e-9)
    assert speed.now < movement_speed(prof.movement), "fixture premise: eased"
    lad = KnobLadder(s)
    lad.set_knobs(knobs)
    lad.resize(1400, lad.height())
    assert lad._read_of(speed)[1] == row.adaptive + speed.unit, \
        "the NOW column and the ledger cell must print the same characters"
    assert speed.delta_text == row.delta, \
        f"two CHANGE columns for one key: {speed.delta_text!r} vs {row.delta!r}"
    lad.deleteLater()


def test_an_uneased_variant_still_reconciles_with_the_model(tmp_path):
    """The fix must not invent a disagreement where there is none: with no
    easing the file and the model are the same number."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof, fatigue=0.0)
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    model = speed_multiplier(prof.movement)
    assert speed.now == pytest.approx(model, abs=1e-3)
    assert speed.now == pytest.approx(
        facts.written_speeds["char1"] / facts.authored["char1"])
    assert "written from an earlier model state" not in speed.evidence
    assert "eased=" not in speed.evidence


def test_with_no_variant_the_row_is_the_model_and_labels_itself_planned(
        tmp_path, qapp):
    """With nothing written there is no file to read, so the row falls back to
    the model — and every surface has to say so rather than implying the game
    has seen it."""
    s = _install(tmp_path)
    prof = _trained()
    prof.save(s.profile_path)
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert not facts.have_variant and not facts.variant_on_disk
    assert speed.now == pytest.approx(speed_multiplier(prof.movement))
    assert "Movement stands at" in speed.evidence
    assert "read out of the [Adaptive] file" not in speed.evidence
    assert speed.pending and "not written yet" in speed.note
    lad = KnobLadder(s)
    lad.set_knobs(knobs)
    assert lad.delta_header() == "PLANNED"
    lad.deleteLater()


def test_the_caption_names_where_the_marker_comes_from(tmp_path):
    """The rail's caption is a claim too: it promised "@ is where the MODEL
    stands now" for two rows that are measured out of the written file."""
    assert "[Adaptive]" in _LADDER_CAPTION
    assert "PLANNED" in _LADDER_CAPTION


# ------------------------------------------- 2. a ramp figure never written
def test_the_provenance_never_quotes_a_ramp_the_multiplier_path_did_not_write(
        tmp_path, qapp):
    """describe() ALWAYS records `speed=<absolute 0-170 ramp>`. On a scenario
    whose targets author a speed of their own that number is never written — the
    generator modulates instead — and echoing the header verbatim put it on the
    page as though it had landed."""
    s = _install(tmp_path)
    prof = _trained()
    plan = _variant(s, prof)
    view = _page(s)
    text = _plain(view.provenance.text())
    facts = view._facts
    assert facts.speed_path == "multiplier"
    ramp = f"{plan.target_max_speed:.0f}"
    assert ramp == _plan_fields(facts.description)["speed"], "fixture premise"

    assert facts.description not in view.provenance.text(), \
        "the header may not be echoed verbatim — not every number in it landed"
    assert "NOT written here" in text and f"speed={ramp}" in text, \
        "the ramp figure has to be named as the thing that did NOT apply"
    assert "multiplier path" in text
    # …and it appears nowhere else on the page, in particular not as an applied
    # field of the plan record
    applied = text.split("its record also carries")[0]
    assert f"speed={ramp}" not in applied
    assert "u/s" not in applied
    # the raw header stays reachable, as raw text, in the tooltip
    assert facts.description in view.provenance.toolTip()
    assert "verbatim" in view.provenance.toolTip()
    view.deleteLater()


def test_the_provenance_names_the_ramp_path_where_the_ramp_applied(tmp_path, qapp):
    """The fix may not simply delete the field: on a base-speed-0 wall the ramp
    IS the path that ran. Its FIGURE still stays off the page — describe()
    records it to whole units, so a variant carrying 65.9 records speed=66, and
    printing that would put a third rounding of one quantity beside the ledger's
    exact one. The path is named; the number lives in the ledger."""
    s = _install(tmp_path, speeds=(0.0,))
    prof = _trained()
    plan = _variant(s, prof)
    view = _page(s)
    text = _plain(view.provenance.text())
    assert view._facts.speed_path == "ramp"
    assert "the absolute MaxSpeed ramp" in text
    assert "MaxSpeed rows above carry what was written" in text
    assert "NOT written here" not in text
    assert f"{plan.target_max_speed:.0f} u/s" not in text and "speed=" not in text
    assert float(_speed_row(view._facts).adaptive) == pytest.approx(
        plan.target_max_speed), "and the ledger carries the exact written number"
    view.deleteLater()


def test_a_mixed_scenario_says_which_characters_the_ramp_reached(tmp_path, qapp):
    """One strafe bot and one static wall: the ramp applied to exactly one of
    them, and a page-wide claim either way is false."""
    s = _install(tmp_path, speeds=(1300.0, 0.0))
    prof = _trained()
    _variant(s, prof)
    view = _page(s)
    text = _plain(view.provenance.text())
    assert view._facts.speed_path == "mixed"
    assert "the absolute MaxSpeed ramp" in text
    assert "only to the base-speed-0 targets (char2)" in text
    assert "char1" not in text.split("only to the base-speed-0")[1]
    # the ledger is the surface that reports per character, and it still does
    assert float(_speed_row(view._facts, "char1").adaptive) > 170.0
    assert 0.0 < float(_speed_row(view._facts, "char2").adaptive) <= 170.0
    view.deleteLater()


def test_a_hand_edited_description_is_never_read_as_a_plan_record(tmp_path, qapp):
    """No record, no quoting: the page may not attribute fields to a header it
    cannot prove kovadapt wrote."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    _rewrite_variant(s, lambda t: t.replace("kovadapt auto-generated", "by hand"))
    facts = read_sce_facts(s, NAME, prof.last_focus)
    summary, tail = _plan_summary(facts)
    assert "not a kovadapt plan record" in summary and not tail
    assert "target size x" not in summary and "speed=" not in summary


def test_a_variant_with_no_description_header_claims_no_plan(tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    _rewrite_variant(s, lambda t: "\n".join(
        ln for ln in t.split("\n") if not ln.startswith("Description=")))
    facts = read_sce_facts(s, NAME, prof.last_focus)
    assert facts.have_variant and not facts.description
    assert "no Description header" in _plan_summary(facts)[0]


def test_the_plan_parser_ignores_the_base_name_segment():
    """A scenario name holding an "=" would otherwise become a plan field that
    never existed."""
    desc = ("kovadapt auto-generated | scale=1.10 movement=0.37 focus=r0c0 "
            "speed=63 eased=0.80 | base: a=b thing")
    assert _plan_fields(desc) == {"scale": "1.10", "movement": "0.37",
                                 "focus": "r0c0", "speed": "63",
                                 "eased": "0.80"}
    assert _plan_fields("hand written | scale=1.00 |") == {}


# ------------------------------------ 3. planned-not-written, all five rows
def test_the_pending_clause_never_claims_a_file_it_did_not_look_for(
        tmp_path, qapp):
    """read_sce_facts returns early on a missing base .sce WITHOUT looking at the
    variant, so the page asserted "no [Adaptive] .sce on disk carries this
    value" — four notes plus the headline — about a file that was on disk."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    (s.scenarios_dir / f"{NAME}.sce").unlink()          # base gone, variant stays
    assert _variant_path(s).is_file(), "fixture premise: the variant IS there"

    view = _page(s)
    facts = view._facts
    assert not facts.have_base and facts.variant_on_disk
    surfaces = [view.headline.text(), _plain(view.provenance.text())] + [
        k.note for k in view._knobs]
    for text in surfaces:
        assert "no [Adaptive] file has been written yet" not in text
        assert "not written yet" not in text
    assert "IS on disk" in " ".join(k.note for k in view._knobs)
    assert "is on disk" in _plain(view.provenance.text())
    assert "base .sce is missing" in view.headline.text()
    view.deleteLater()


def test_an_unreadable_variant_is_not_reported_as_an_absent_one(
        tmp_path, qapp, monkeypatch):
    """A file that will not open (the game holds it, a scanner blocks it) and a
    file that was never written are different absences — and the failure belongs
    to the variant, not to a base file that parsed perfectly. One try block
    around both reads reported "could not read <base>.sce" and discarded every
    fact the base had already given up."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    real = SceFile.read

    def locked(path):
        if ADAPTIVE_SUFFIX in str(path):
            raise OSError("locked by the game")
        return real(path)

    monkeypatch.setattr(SceFile, "read", staticmethod(locked))
    view = _page(s)
    facts = view._facts
    assert facts.variant_on_disk and not facts.have_variant
    assert facts.have_base and facts.rows, \
        "the base file's own facts survive a variant that will not open"
    assert NAME not in facts.error, f"the base is not what failed: {facts.error!r}"
    text = " ".join([view.headline.text(), _plain(view.provenance.text()),
                     _plain(view.spawn_note.text())]
                    + [k.note for k in view._knobs])
    assert "could not be read" in text
    assert "has not written anything" not in text
    assert "no [Adaptive] file has been written yet" not in text
    assert "no variant has been written yet" not in text
    view.deleteLater()


def test_a_missing_variant_still_reads_as_a_missing_variant(tmp_path, qapp):
    """The three-way split must not blur the ordinary case."""
    s = _install(tmp_path)
    prof = _trained()
    prof.save(s.profile_path)
    view = _page(s)
    assert "no [Adaptive] file has been written yet" in view.headline.text()
    assert all("not written yet" in k.note
               for k in view._knobs if k.measured)
    view.deleteLater()


def test_the_cold_start_note_claims_a_file_only_when_there_is_one(tmp_path):
    """"…but the file already carries this number" printed under a headline
    saying there is no [Adaptive] file on disk yet."""
    s = _install(tmp_path)
    cold = PlayerProfile(scenario=NAME + ADAPTIVE_SUFFIX)
    cold.save(s.profile_path)
    knobs, facts = _knobs(s, cold)
    assert not facts.have_variant
    assert "the file already carries this number" not in knobs[1].note
    assert "cold-start default" in knobs[1].note

    _variant(s, PlayerProfile(scenario=NAME + ADAPTIVE_SUFFIX))
    cold = PlayerProfile(scenario=NAME + ADAPTIVE_SUFFIX)
    warm_knobs, warm_facts = _knobs(s, cold)
    assert warm_facts.have_variant
    assert "the file already carries this number" in warm_knobs[1].note


@pytest.mark.parametrize("kwargs", [
    {},                              # ordinary authored-speed wall
    {"speeds": (0.0,)},              # static wall: the absolute ramp
    {"speeds": (1300.0, 0.0)},       # both paths in one scenario
    {"xs": 1, "ys": 5},              # fewer spawns than cells: layout untouched
])
@pytest.mark.parametrize("fatigue", [0.0, 0.8])
def test_the_headline_count_only_ever_names_measured_criteria(
        tmp_path, kwargs, fatigue):
    """"N criteria have moved" may not be driven by a value the page itself
    marks unmeasured, and the criterion it calls largest must be one of them."""
    s = _install(tmp_path, **kwargs)
    prof = _trained()
    _variant(s, prof, fatigue=fatigue)
    knobs, facts = _knobs(s, prof)
    head = takeaway(knobs, prof, facts)
    moved = [k for k in knobs if k.measured and k.moved]
    assert head.startswith(f"{len(moved)} of {len(knobs)} criteria")
    for knob in knobs:
        if not knob.measured:
            assert knob.name not in head.split("largest:")[-1], \
                f"{knob.key} is unmeasured and may not be the headline's largest"
    if moved:
        assert "largest:" in head
        assert max(moved, key=lambda k: abs(k.now - k.baseline)
                   / max(k.hi - k.lo, 1e-9)).name in head


def test_every_criterion_carries_the_same_pending_state(tmp_path):
    """The clause lived on the spawn knob alone once; it has to be all five or
    none, on both sides of the switch."""
    s = _install(tmp_path)
    prof = _trained()
    prof.save(s.profile_path)
    knobs, facts = _knobs(s, prof)
    assert not facts.have_variant and all(k.pending for k in knobs)
    _variant(s, prof)
    knobs, facts = _knobs(s, prof)
    assert facts.have_variant and not any(k.pending for k in knobs)
    assert "not written yet" not in " ".join(k.note for k in knobs)


# ---------------------------------------------- files that cannot be trusted
def test_a_speed_outside_the_controllers_range_widens_the_rail(tmp_path):
    """A hand-edited or foreign variant can carry a multiplier the controller
    could never emit. The rail must hold what it plots — Knob refuses to be
    built otherwise, which would take the whole page down."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    _rewrite_variant(s, lambda t: re.sub(r"MaxSpeed=1[0-9]{3}\.?[0-9]*",
                                         "MaxSpeed=3900.0", t))
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert facts.written_speeds == {"char1": 3900.0}
    assert speed.now == pytest.approx(3.0)
    assert speed.lo <= speed.now <= speed.hi
    assert "outside the controller's own range" in speed.note


def test_targets_written_different_multipliers_are_not_one_reading(tmp_path):
    """The generator writes ONE multiplier for every target, so a variant that
    does not is not a reading this rail can hold — the same refusal the mixed
    speed path makes."""
    s = _install(tmp_path, speeds=(1300.0, 100.0))
    prof = _trained()
    _variant(s, prof)
    _rewrite_variant(s, lambda t: re.sub(r"MaxSpeed=[0-9]{2,3}\.[0-9]\n",
                                         "MaxSpeed=150.0\n", t))
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert facts.speed_path == "multiplier"
    ratios = {c: facts.written_speeds[c] / facts.authored[c]
              for c in facts.authored}
    assert max(ratios.values()) - min(ratios.values()) > 0.2, "fixture premise"
    assert speed.now == pytest.approx(speed_multiplier(prof.movement)), \
        "it falls back to the model rather than picking one target's ratio"
    assert "not one reading" in speed.note
    assert "plots what the model implies" in speed.note


def test_the_whole_page_renders_with_an_eased_variant(tmp_path, qapp):
    """A green suite is not proof a screen looks right: draw it and check every
    panel put ink down."""
    from kovadapt.gui import theme
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof, fatigue=0.8)
    view = _page(s)
    for art in view._art:
        art.setStyleSheet(f"background:{theme.current().bg};")
        img = art.grab().toImage()
        bg = img.pixelColor(1, 1)
        ink = sum(1 for y in range(0, img.height(), 3)
                  for x in range(0, img.width(), 3)
                  if img.pixelColor(x, y) != bg)
        assert ink > 0, f"{type(art).__name__} rendered blank"
    view.deleteLater()
