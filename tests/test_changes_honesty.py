"""gui/changes_view.py: the page may not show a number, verdict or claim its
own files do not support, and two surfaces may not contradict each other about
the same thing.

Every case here was first found by RENDERING the page and reading the picture,
so each test names the lie it pins shut:

  * a fabricated speed baseline (1.0x / 0.0 hard-coded) driving the headline
  * `speed_path` collapsing a mixed scenario, so the ladder claimed the
    absolute ramp is "never written here" above a ledger row showing it written
  * "@" living at index 0 of the run ramp, the same glyph as the marker, so a
    long run rendered "|@@@@#####...@" with four @s of which one was the value
  * the marker test running before the anchor test, so an unmoved criterion drew
    a lone "@" and no baseline tick at all
  * a movement rail starting at min_movement (0.35) while plotting the 0.15
    dataclass default, printing one number and painting another
  * size_check ignoring `plan.fatigue`, calling every eased variant stale
  * the spawn grid painting maximum density for a layout the generator provably
    left alone, one line above the knob saying so
  * the Fitts clause appended on the observation count alone
  * "N criteria have moved" over a ledger whose every VARIANT cell was a dash
  * a rail below ~800px rendering a real -20% move as "@|"

Skipped wholesale without PySide6.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
if sys.platform == "win32":
    os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.adapt.engine import AdaptationEngine  # noqa: E402
from kovadapt.adapt.stochastic import movement_speed, speed_multiplier  # noqa: E402
from kovadapt.config import ADAPTIVE_SUFFIX, Settings  # noqa: E402
from kovadapt.gui import motion, theme  # noqa: E402
from kovadapt.gui.changes_view import (  # noqa: E402
    MIN_RAIL_CELLS,
    MIN_RUN_CELLS,
    _ANCHOR_GLYPH,
    _MARKER_GLYPH,
    _ON_BASELINE_GLYPH,
    _RUN_RAMP,
    ChangesView,
    Knob,
    KnobLadder,
    SpawnGrid,
    SpawnMap,
    build_knobs,
    read_report_evidence,
    read_sce_facts,
    size_check,
    takeaway,
)
from kovadapt.profile.player import PlayerProfile, RegionPosterior  # noqa: E402
from kovadapt.scenario.generator import generate_adaptive_variant  # noqa: E402


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
{speed_line}MainBBRadius=0.5
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


def _sce_text(name: str, speeds=(1300.0,), xs=6, ys=5) -> str:
    """`None` in `speeds` writes a character with NO MaxSpeed line at all —
    which is not an authored 0: set_in_section only rewrites a key that already
    exists, so the generator never gives that character a speed."""
    chars = "".join(
        _CHAR.format(char=f"char{i}", bot=f"bot{i}", i=i,
                     speed_line="" if sp is None else f"MaxSpeed={sp}\n")
        for i, sp in enumerate(speeds, start=1))
    bots = ";".join(f"bot{i}.bot" for i in range(1, len(speeds) + 1))
    pts = [
        f"\tentity\n\t\ttype PlayerSpawn\n"
        f"\t\tVector3 position {x * 200 - 600}.000000 {y * 250 + 200}.000000 960.000000\n"
        f"\t\tBool8 teamA 0\n"
        for x in range(xs) for y in range(ys)
    ]
    return _HEAD.format(name=name, bots=bots, chars=chars) + "".join(pts)


def _install(tmp_path: Path, name="Wall Task", speeds=(1300.0,), xs=6,
             ys=5) -> Settings:
    root = tmp_path / "game"
    (root / "stats").mkdir(parents=True, exist_ok=True)
    scen = root / "Saved" / "SaveGames" / "Scenarios"
    scen.mkdir(parents=True, exist_ok=True)
    (scen / f"{name}.sce").write_text(_sce_text(name, speeds, xs, ys),
                                      encoding="utf-8")
    return Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof"),
                    telemetry_enabled=False, onboarding_done=True,
                    motion=motion.OFF)


def _trained(name="Wall Task", runs=14, archetype="clicking") -> PlayerProfile:
    prof = PlayerProfile(scenario=name + ADAPTIVE_SUFFIX)
    prof.archetype = archetype
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


def _variant(settings: Settings, profile: PlayerProfile, name="Wall Task",
             fatigue: float = 0.0):
    """Write a real variant the way the watcher does, so every before/after in
    these tests is measured off two real files."""
    plan = AdaptationEngine(settings).plan(profile, None, fatigue=fatigue)
    generate_adaptive_variant(
        settings.scenarios_dir / f"{name}.sce", plan, settings,
        settings.scenarios_dir / f"{name}{ADAPTIVE_SUFFIX}.sce")
    profile.save(settings.profile_path)      # plan() mutates: persist, as watcher does
    return plan


def _knobs(settings: Settings, profile: PlayerProfile, name="Wall Task"):
    facts = read_sce_facts(settings, name, profile.last_focus)
    ev = read_report_evidence(settings.profile_path, name)
    return build_knobs(profile, settings, facts, ev), facts


def _ladder(settings: Settings, knobs, width=1400) -> KnobLadder:
    lad = KnobLadder(settings)
    lad.set_knobs(knobs)
    lad.resize(width, lad.height())
    return lad


# ---------------------------------------------------- 1. the speed baseline
def test_the_multiplier_baseline_is_the_authors_own_number_from_the_file(tmp_path):
    """1.00x may only stand for the author's MaxSpeed, and the page has to say
    that is where it came from — it used to be a bare literal."""
    s = _install(tmp_path, speeds=(1300.0,))
    prof = _trained()
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert facts.authored == {"char1": 1300.0}
    assert facts.speed_paths == {"char1": "multiplier"}
    assert (speed.lo, speed.hi, speed.baseline) == (0.65, 1.35, 1.0)
    assert "MaxSpeed=1300" in speed.evidence
    assert "1.00x below IS that authored value, read out of the base .sce" \
        in speed.evidence
    assert speed.now == pytest.approx(speed_multiplier(prof.movement))


def test_the_ramp_baseline_is_the_zero_the_file_actually_carries(tmp_path):
    s = _install(tmp_path, speeds=(0.0,))
    prof = _trained()
    knobs, facts = _knobs(s, prof)
    speed = knobs[1]
    assert facts.speed_path == "ramp"
    assert (speed.lo, speed.hi, speed.baseline) == (0.0, 170.0, 0.0)
    assert "MaxSpeed line in the base file reads 0" in speed.evidence
    assert "the author's own 0 is the baseline" in speed.evidence
    assert speed.now == pytest.approx(movement_speed(prof.movement))


def test_a_missing_maxspeed_line_is_not_an_authored_zero(tmp_path):
    """The generator's set_in_section only rewrites a key that is there, so a
    character with no MaxSpeed line never receives a speed. Reading that as 0
    put the scenario on the static-wall ramp here while its file never moved."""
    s = _install(tmp_path, speeds=(None,))
    prof = _trained()
    _variant(s, prof)
    knobs, facts = _knobs(s, prof)
    assert facts.no_speed_key == ("char1",) and facts.authored == {}
    assert facts.speed_path == "unknown"
    speed = knobs[1]
    assert not speed.measured and not speed.rail
    assert "no MaxSpeed line in the base .sce at all" in speed.evidence
    # and the file agrees: no MaxSpeed row exists to compare
    assert not [r for r in facts.rows if r.label.endswith("MaxSpeed")]


# ----------------------------------------------- 2. the mixed-path collapse
def test_a_scenario_that_mixes_both_speed_paths_refuses_to_collapse_them(tmp_path):
    """generate_adaptive_variant decides PER CHARACTER. With one strafe bot and
    one static wall, `any(authored > 0)` made the page print "never writes the
    absolute 0-170 static-wall ramp here" directly above a ledger row showing
    that exact ramp written to the other character."""
    s = _install(tmp_path, speeds=(1300.0, 0.0))
    prof = _trained()
    _variant(s, prof)
    knobs, facts = _knobs(s, prof)
    assert facts.speed_paths == {"char1": "multiplier", "char2": "ramp"}
    assert facts.speed_path == "mixed"

    speed = knobs[1]
    assert not speed.measured, "a mixed scenario is not one speed reading"
    assert not speed.rail, "no single rail can hold a multiplier and a ramp"
    assert speed.flag == "mixed paths"
    for char in ("char1", "char2"):
        assert char in speed.evidence
    assert "never writes the absolute" not in speed.evidence

    rows = {r.label: r for r in facts.rows}
    # the LEDGER is the surface that reports per character, and it must show
    # exactly the two different paths the knob now refuses to merge
    assert float(rows["char1 - MaxSpeed"].adaptive) > 170.0      # modulated
    assert 0.0 < float(rows["char2 - MaxSpeed"].adaptive) <= 170.0   # ramp

    # and the mixed reading may not reach the headline count
    head = takeaway(knobs, prof, facts)
    assert speed.name not in head.split("largest:")[-1]


# --------------------------------------------------- 3./4. the rail glyphs
def test_the_run_ramp_never_reuses_a_mark_glyph():
    for mark in (_MARKER_GLYPH, _ANCHOR_GLYPH, _ON_BASELINE_GLYPH):
        assert mark not in _RUN_RAMP, \
            f"{mark!r} is both a mark and a ramp glyph — the row loses its marker"
    assert len({_MARKER_GLYPH, _ANCHOR_GLYPH, _ON_BASELINE_GLYPH}) == 3


def test_a_long_run_carries_exactly_one_marker_and_one_baseline(tmp_path, qapp):
    """The reported symptom, in characters: "|@@@@#####*****...@" had four ramp
    @s ahead of the value."""
    s = _install(tmp_path)
    knob = Knob(key="k", name="k", lo=0.0, hi=1.0, baseline=0.05, now=0.95,
                evidence="because a long move")
    lad = _ladder(s, [knob])
    row = lad.row_glyphs(knob, 40)
    assert row.count(_MARKER_GLYPH) == 1
    assert row.count(_ANCHOR_GLYPH) == 1
    assert row.count(_ON_BASELINE_GLYPH) == 0
    assert row.index(_ANCHOR_GLYPH) < row.index(_MARKER_GLYPH)
    body = row[row.index(_ANCHOR_GLYPH) + 1:row.index(_MARKER_GLYPH)]
    assert body and set(body) <= set(_RUN_RAMP), "the run between the marks"
    lad.deleteLater()


def test_an_unmoved_criterion_shows_both_marks_and_never_a_lone_marker(tmp_path, qapp):
    """A cold profile drew a lone "@" and no "|", against a caption promising
    both, because `col == marker` was tested before `col == anchor`."""
    s = _install(tmp_path)
    prof = PlayerProfile(scenario="Wall Task" + ADAPTIVE_SUFFIX)
    knobs, _ = _knobs(s, prof)
    lad = _ladder(s, knobs)
    still = [k for k in knobs if k.rail and not k.moved]
    assert still, "a cold profile has criteria that have not moved"
    for knob in still:
        row = lad.row_glyphs(knob, 40)
        assert row.count(_ON_BASELINE_GLYPH) == 1, knob.key
        assert _MARKER_GLYPH not in row, \
            f"{knob.key} drew a marker with no baseline tick beside it"
        assert _ANCHOR_GLYPH not in row
    lad.deleteLater()


# -------------------------------------------- 5. the rail holds its values
@pytest.mark.parametrize("archetype", ["clicking", "tracking", "switching"])
def test_every_rail_contains_the_values_it_plots(tmp_path, archetype):
    """A tracking profile had lo=min_movement 0.35 with baseline 0.15, so the
    row printed 0.15 and painted "|" on the 0.35 tick."""
    s = _install(tmp_path)
    for prof in (PlayerProfile(scenario="Wall Task" + ADAPTIVE_SUFFIX),
                 _trained(archetype=archetype)):
        prof.archetype = archetype
        knobs, _ = _knobs(s, prof)
        for k in knobs:
            assert k.lo <= k.baseline <= k.hi, f"{archetype}/{k.key} baseline"
            assert k.lo <= k.now <= k.hi, f"{archetype}/{k.key} now"


def test_the_tracking_movement_baseline_is_the_floor_it_is_drawn_on(tmp_path):
    s = _install(tmp_path)
    eff = s.for_archetype("tracking")
    assert eff.min_movement > PlayerProfile(scenario="").movement, \
        "fixture premise: the tracking floor is above the dataclass default"
    knobs, _ = _knobs(s, _trained(archetype="tracking"))
    mv = knobs[4]
    assert mv.key == "movement"
    assert mv.lo == pytest.approx(eff.min_movement)
    assert mv.baseline == pytest.approx(eff.min_movement)
    assert "outside this archetype's" in mv.note


def test_a_knob_cannot_be_built_outside_its_own_rail():
    with pytest.raises(ValueError, match="outside its own rail"):
        Knob(key="k", name="k", lo=0.35, hi=1.0, baseline=0.15, now=0.5,
             evidence="because")
    with pytest.raises(ValueError, match="outside its own rail"):
        Knob(key="k", name="k", lo=0.0, hi=1.0, baseline=0.5, now=1.4,
             evidence="because")


# ------------------------------------------------ 6. fatigue vs staleness
def test_size_check_reads_the_easing_the_variant_records_about_itself(tmp_path):
    """`plan(fatigue=...)` eases the emitted plan while the persisted profile
    stays un-eased by contract, and the tracker is never written to disk — so
    the variant's own Description is the only record. Ignoring it made this
    accuse a freshly written variant of being stale, in amber, directly below
    that Description reading `eased=0.80`."""
    s = _install(tmp_path)
    prof = _trained()
    plan = _variant(s, prof, fatigue=0.8)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert plan.fatigue == pytest.approx(0.8)
    assert "eased=0.80" in facts.description
    assert plan.target_scale > prof.target_scale, "fixture premise: eased bigger"
    check = size_check(prof, s, facts)
    assert "does not match" not in check, check
    assert "reconcile" in check and "eased=0.80" in check


def test_size_check_still_catches_a_genuinely_stale_variant(tmp_path):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    facts = read_sce_facts(s, "Wall Task", prof.last_focus)
    assert "reconcile" in size_check(prof, s, facts)
    prof.target_scale = 2.4                  # a model state the file predates
    assert "does not match" in size_check(prof, s, facts)


def test_without_a_plan_record_no_staleness_verdict_is_given(tmp_path):
    """A variant whose Description is not a kovadapt plan record cannot say
    whether it was eased, so neither may this page."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    var = s.scenarios_dir / f"Wall Task{ADAPTIVE_SUFFIX}.sce"
    var.write_text(var.read_text(encoding="utf-8").replace(
        "kovadapt auto-generated", "hand-edited by someone"), encoding="utf-8")
    check = size_check(prof, s, read_sce_facts(s, "Wall Task", prof.last_focus))
    assert "no kovadapt plan record" in check
    assert "does not match" not in check and "reconcile" not in check


# --------------------------------------- 7. the grid agrees with the knob
def test_an_untouched_layout_paints_no_density_and_no_focus_ring(tmp_path, qapp):
    """5 target spawns against a 5x5 grid: resample_spawns returns an empty set
    and touches nothing, yet each occupied cell held five times an even share,
    saturated the absolute 3x anchor, and painted solid "@" blocks."""
    s = _install(tmp_path, xs=1, ys=5)
    prof = _trained()
    # a focus the layout DOES hold a point in, so nothing else can explain the
    # panel staying quiet
    prof.last_focus = "r2c0"
    prof.regions = {"r2c0": RegionPosterior(mean=0.4, var=0.08, n=6)}
    knobs, facts = _knobs(s, prof)
    sm = facts.spawns
    assert sm is not None and sm.total_base == 5 and sm.untouched
    assert not knobs[2].measured, "the knob reports unmeasured"

    grid = SpawnGrid(s)
    grid.set_map(sm)
    grid.resize(1200, grid.height())
    assert "untouched" in grid.title_text()
    assert all(grid.cell_density(k) == 0.0 for k in sm.base), \
        "an untouched layout has no emphasis to paint"
    assert not grid.shows_focus_ring(sm.focus or "")
    assert "NOT applied" in (grid.zone_info(*_centre_of(grid, sm.focus)) or "")

    # …and the pixels agree: the very same counts on a layout the generator DOES
    # reweight paint solid glyph blocks, and the untouched one must not.
    touched = SpawnGrid(s)
    touched.set_map(SpawnMap(cols=sm.cols, rows=sm.rows, base=dict(sm.base),
                             adaptive=dict(sm.base), focus=sm.focus))
    touched.resize(1200, touched.height())
    # 360, not 240: the threshold has to sit ABOVE the occupied-cell hairline
    # so this measures the DENSITY RAMP, which is the emphasis an untouched
    # layout must not paint. 240 was calibrated when that hairline was drawn
    # in `border` and later `border_control` at 231 summed delta — i.e. below
    # "real glyph ink", which was the invisibility later fixed. Now it reads
    # 258 on dark and 327 on light and the old cutoff counted it as emphasis.
    # The hairline is deliberate; the ramp is the claim.
    quiet, loud = _ink(grid, 360), _ink(touched, 360)
    assert loud > 8 * max(quiet, 1), \
        f"the untouched layout renders as loud as an adapted one ({quiet} vs {loud})"
    grid.deleteLater()
    touched.deleteLater()


def _centre_of(grid: SpawnGrid, key: str | None) -> tuple[float, float]:
    """Pixel centre of one zone cell, for the tooltip path."""
    geom = grid._geom()
    x0, y0, cw, ch, gap, rows, cols = geom
    row, col = (int(v) for v in (key or "r0c0")[1:].split("c"))
    disp = rows - 1 - row
    return x0 + col * (cw + gap) + cw / 2, y0 + disp * (ch + gap) + ch / 2


def _ink(widget, thresh: int = 24) -> int:
    """Pixels that differ from the panel's own background by more than `thresh`
    (summed over the three channels) — a blunt "how loud is this render"
    measure, which is exactly what the reported violation was about. A high
    threshold counts only real glyph ink, not a faint cell fill.

    The panel is given the page background first: an unparented QWidget grabs
    against an uninitialised (white) surface, where the theme's light foreground
    is invisible and every measurement comes out the same.
    """
    widget.setStyleSheet(f"background:{theme.current().bg};")
    img = widget.grab().toImage()
    bg = img.pixelColor(1, 1)
    count = 0
    for y in range(0, img.height(), 2):
        for x in range(0, img.width(), 2):
            px = img.pixelColor(x, y)
            if (abs(px.red() - bg.red()) + abs(px.green() - bg.green())
                    + abs(px.blue() - bg.blue())) > thresh:
                count += 1
    return count


def test_a_reweightable_layout_still_shows_its_density(tmp_path, qapp):
    """The fix must not flatten the panel's real job."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    _, facts = _knobs(s, prof)
    sm = facts.spawns
    assert sm is not None and not sm.untouched and sm.adaptive
    grid = SpawnGrid(s)
    grid.set_map(sm)
    grid.resize(1200, grid.height())
    assert "untouched" not in grid.title_text()
    assert max(grid.cell_density(k) for k in sm.base) > 0.0
    assert grid.shows_focus_ring(sm.focus or "")
    grid.deleteLater()


# ------------------------------------------------- 8. the Fitts mechanism
def test_the_fitts_clause_needs_a_run_that_was_actually_in_band(tmp_path):
    """The sub-controller lives in the deadband's `elif`, so it only fires on a
    run INSIDE the band. The clause was appended on the observation count, and
    the same sentence read "0 sat inside it" and then claimed the shrink step."""
    s = _install(tmp_path)
    prof = _trained()
    prof.fitts_obs, prof.ewma_fitts_ms, prof.slow_fitts_ms = 7, 400.0, 300.0
    for h in prof.history:
        h["accuracy"] = 0.99                 # every run ABOVE the band
    never, _ = _knobs(s, prof)
    assert "ms-per-bit has also stalled" in never[0].evidence
    assert "it has never fired here" in never[0].evidence
    assert "added one extra shrink step" not in never[0].evidence

    for h in prof.history[:4]:
        h["accuracy"] = 0.90                 # four runs inside the band
    fired, _ = _knobs(s, prof)
    assert "added one extra shrink step on each of those 4 in-band runs" \
        in fired[0].evidence


# ------------------------------------------- 9. planned is not written
def test_nothing_written_reads_that_way_on_every_surface(tmp_path, qapp):
    """The headline said "5 of 5 criteria have moved" while every VARIANT cell
    in the ledger below read "—", and only the spawn knob said otherwise."""
    s = _install(tmp_path)
    prof = _trained()
    prof.save(s.profile_path)                     # no variant on disk
    knobs, facts = _knobs(s, prof)
    assert facts.have_base and not facts.have_variant
    assert all(r.adaptive == "—" for r in facts.rows)

    head = takeaway(knobs, prof, facts)
    assert "have moved in the model" in head
    assert "no [Adaptive] file has been written yet" in head
    assert "have moved on evidence" not in head

    assert all(k.pending for k in knobs)
    for k in knobs:
        if k.measured:
            assert "not written yet" in k.note, k.key
    lad = _ladder(s, knobs)
    assert lad.delta_header() == "PLANNED"
    lad.deleteLater()


def test_a_written_variant_reads_as_written(tmp_path, qapp):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    knobs, facts = _knobs(s, prof)
    assert facts.have_variant
    assert not any(k.pending for k in knobs)
    assert "not written yet" not in " ".join(k.note for k in knobs)
    lad = _ladder(s, knobs)
    assert lad.delta_header() == "CHANGE"
    lad.deleteLater()


# ------------------------------------------------ 10. the narrow rail
def test_a_real_move_always_renders_as_a_move(tmp_path, qapp):
    """MOVE_EPS_FRAC calls 1% of a range a move, which is a fifth of a cell even
    at full width — so a move could land in the anchor's own cell and draw the
    no-move glyph while the delta column said -20%."""
    s = _install(tmp_path)
    small = Knob(key="k", name="k", lo=0.0, hi=1.0, baseline=0.50, now=0.485,
                 evidence="because a small but real move")
    assert small.moved, "fixture premise: just past the epsilon"
    lad = _ladder(s, [small])
    for cells in (MIN_RAIL_CELLS, 30, 55):
        row = lad.row_glyphs(small, cells)
        assert row.count(_MARKER_GLYPH) == 1, cells
        assert row.count(_ANCHOR_GLYPH) == 1, cells
        assert _ON_BASELINE_GLYPH not in row, cells
        gap = abs(row.index(_MARKER_GLYPH) - row.index(_ANCHOR_GLYPH))
        assert gap >= MIN_RUN_CELLS, f"{cells} cells rendered a move as no move"
        assert row.index(_MARKER_GLYPH) < row.index(_ANCHOR_GLYPH), \
            "the marker must stay on the side the move went"
    lad.deleteLater()


def test_a_move_pushed_against_the_rails_end_still_reads_as_a_move(tmp_path, qapp):
    s = _install(tmp_path)
    lad = _ladder(s, [Knob(key="k", name="k", lo=0.0, hi=1.0, baseline=0.98,
                           now=1.0, evidence="because")])
    at_top = lad._knobs[0]
    row = lad.row_glyphs(at_top, MIN_RAIL_CELLS)
    assert row.count(_MARKER_GLYPH) == 1 and row.count(_ANCHOR_GLYPH) == 1
    assert row.index(_MARKER_GLYPH) == len(row) - 1
    assert row.index(_ANCHOR_GLYPH) <= len(row) - 1 - MIN_RUN_CELLS
    lad.deleteLater()


def test_a_rail_too_narrow_to_be_honest_is_dropped_not_squeezed(tmp_path, qapp):
    """At 8 cells a -20% size move rendered "@|" and a -6% speed move rendered
    as a lone "@": the row said "no move" while its delta column disagreed."""
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    knobs, _ = _knobs(s, prof)
    wide = _ladder(s, knobs, width=1400)
    assert wide._layout().cells >= MIN_RAIL_CELLS
    assert all(wide.row_glyphs(k) for k in knobs if k.rail)

    narrow = _ladder(s, knobs, width=620)
    assert narrow._layout().cells == 0, "a rail this cramped cannot be honest"
    assert all(narrow.row_glyphs(k) == "" for k in knobs)
    # dropping it must not drop the reading: the numbers are still there
    for knob in knobs:
        if knob.rail:
            assert narrow._read_of(knob) == (
                knob.text(knob.baseline), knob.text(knob.now) + knob.unit)
    assert _ink(narrow) > 0, "the row still renders its numbers"
    wide.deleteLater()
    narrow.deleteLater()


def test_an_unreadable_criterion_prints_dashes_rather_than_its_rail_floor(
        tmp_path, qapp):
    s = _install(tmp_path, speeds=(1300.0, 0.0))       # mixed -> no reading
    prof = _trained()
    knobs, _ = _knobs(s, prof)
    lad = _ladder(s, knobs)
    speed = knobs[1]
    assert lad._read_of(speed) == ("—", "—")
    assert lad.row_glyphs(speed) == ""
    lad.deleteLater()


# ------------------------------------------------------- render-level smoke
@pytest.mark.parametrize("width", [1400, 780, 620])
def test_the_whole_page_renders_at_every_width_without_a_blank_panel(
        tmp_path, qapp, width):
    s = _install(tmp_path)
    prof = _trained()
    _variant(s, prof)
    view = ChangesView(s)
    view.show_scenario("Wall Task")
    view.resize(width, 2000)
    view.show()
    for art in view._art:
        assert _ink(art) > 0, f"{type(art).__name__} rendered blank at {width}"
    view.deleteLater()


def test_a_base_file_that_cannot_be_read_is_not_reported_as_left_untouched(tmp_path):
    """`untouched` means the generator LOOKED at this layout and chose not to
    change it — a positive finding, and the panel says so: "the author's own,
    left untouched". `reason` carried that meaning and was also being used as
    the I/O-error channel, so a base .sce that is simply not on disk came out
    the same way — a claim about a file that was never opened, printed
    directly above the panel's own band reading "Wall Task.sce is not in the
    game's Scenarios folder".

    It also fires for a base that exists and will not parse.
    """
    missing = SpawnMap(cols=5, rows=5, reason="no scenario file to read",
                       error="no scenario file to read")
    assert not missing.untouched, (
        "an unreadable file is reported as one the generator left alone")

    grid = SpawnGrid(_install(tmp_path, xs=1, ys=5))
    grid.set_map(missing)
    title = grid.title_text().lower()
    assert "untouched" not in title, title
    assert "unreadable" in title, title

    # the legitimate case still reads the way it did
    left_alone = SpawnMap(cols=5, rows=5, base={"r2c0": 5},
                          reason="5 spawns on a 5x5 grid: nothing to resample")
    assert left_alone.untouched
    grid.set_map(left_alone)
    assert "untouched" in grid.title_text().lower()


def test_the_page_itself_does_not_claim_untouched_for_a_scenario_it_cannot_find(
        tmp_path, qapp):
    """End to end through ChangesView, because the field that carries this is
    set at the construction site and asserting on a hand-built SpawnMap cannot
    see that site at all — dropping `error=` there passes a direct test.

    A profile exists for a scenario whose .sce has been removed from the
    game's Scenarios folder: exactly what happens when someone reinstalls
    KovaaK's, or renames a scenario they had been training.
    """
    from kovadapt.gui.changes_view import ChangesView

    s = _install(tmp_path, xs=1, ys=5)
    prof = _trained()
    prof.save(s.profile_path)
    scen = Path(s.kovaaks_root) / "Saved" / "SaveGames" / "Scenarios"
    for f in scen.glob("*.sce"):
        f.unlink()                       # the base is gone; the profile is not

    view = ChangesView(s)
    view.show_scenario("Wall Task")
    title = view.grid.title_text().lower()
    assert "untouched" not in title, (
        f"the page claims a layout it could not read was left untouched: {title!r}")
    assert "unreadable" in title, title
    view.deleteLater()


def test_a_knob_at_its_clamp_says_so_in_words_not_only_in_amber(tmp_path):
    """Amber is `pal.warn`, and every accent is fitted to the SAME lightness as
    the semantic roles — so only hue separates them. Measured in OKLab, the
    ember accent sits 0.065 from warn and mint 0.028, which is below a
    just-noticeable difference. On those two accents an at-clamp value had no
    carrier at all: the colour was the only thing saying it, and the colour was
    the same colour.

    Every other knob in this file already writes its bound in prose. These two
    were the exceptions, which is why they were the two the audit named as
    "colour is the sole carrier".
    """
    s = _install(tmp_path)
    prof = _trained()

    # dodge: a bias far past what the generator will ever write
    prof.ewma_bias = 5.0
    prof.bias_obs = 12
    knobs, _facts = _knobs(s, prof)
    dodge = next(k for k in knobs if "strafe" in k.name or "dodge" in k.key)
    assert "clamp" in dodge.note.lower(), (
        f"a dodge skew pinned at its bound says nothing: {dodge.note!r}")

    # movement: pinned at the archetype ceiling
    prof2 = _trained()
    prof2.movement = s.max_movement
    knobs2, _f2 = _knobs(s, prof2)
    move = next(k for k in knobs2 if k.key == "movement")
    assert "ceiling" in move.note.lower() or "cannot push" in move.note.lower(), (
        f"movement pinned at its ceiling says nothing: {move.note!r}")

    # ...and a knob that is NOT at its bound must not claim to be
    prof3 = _trained()
    prof3.movement = (s.min_movement + s.max_movement) / 2
    knobs3, _f3 = _knobs(s, prof3)
    mid = next(k for k in knobs3 if k.key == "movement")
    assert "ceiling" not in mid.note.lower() and "floor" not in mid.note.lower(), (
        f"a mid-range movement claims to be at a bound: {mid.note!r}")
