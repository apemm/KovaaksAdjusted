from pathlib import Path

import numpy as np
import pytest

from kovadapt.adapt.engine import AdaptationEngine
from kovadapt.config import Settings
from kovadapt.profile.player import PlayerProfile
from kovadapt.scenario.sce import SceFile, SpawnPoint
from kovadapt.scenario.generator import (
    _region_grid,
    _region_of,
    _spawn_team,
    _target_profiles,
    generate_adaptive_variant,
)

# Mirrors the real 1wall layout: the wall spans x (horizontal) and y
# (vertical) at depth z=960; the player stands on the opposite side at
# (0, 0, -960). PlayerTeam=1 (odd) -> the player's spawn carries the teamB
# flag, targets carry teamA. The bot links to its character via the real
# [Bot Profile] CharacterProfile= chain (bot name != character name).
MINI_SCE = """Name=mini test
AddedBots=target.bot;target.bot
PlayerTeam=1
Timelimit=60.0

[Character Profile]
Name=Player
MaxHealth=100.0
MainBBRadius=1.0

[Character Profile]
Name=target_char
MaxHealth=1.0
MaxSpeed=0.0
MainBBRadius=0.5
MainBBHeight=2.0
ProjBBRadius=0.5

[Bot Profile]
Name=target
CharacterProfile=target_char
DodgeProfileNames=Mimic
AimingProfileNames=Default

[Dodge Profile]
Name=Mimic
MinLRTimeChange=0.2
MaxLRTimeChange=0.5
JumpFrequency=0.5

[Map Data]
reflex map version 8
global
\tentity
\t\ttype WorldSpawn
\tentity
\t\ttype PlayerSpawn
\t\tVector3 position 0.000000 0.000000 -960.000000
\t\tBool8 teamB 0
""" + "".join(
    f"\tentity\n\t\ttype PlayerSpawn\n"
    f"\t\tVector3 position {x}.000000 {y}.000000 960.000000\n"
    f"\t\tVector3 angles 180.000000 0.000000 0.000000\n"
    f"\t\tBool8 teamA 0\n"
    for x in (-800, -400, 0, 400, 800) for y in (200, 600, 1000)
)

PLAYER_XYZ = (0.0, 0.0, -960.0)
WALL_XY = {(float(x), float(y)) for x in (-800, -400, 0, 400, 800)
           for y in (200, 600, 1000)}


@pytest.fixture
def mini_sce(tmp_path: Path) -> Path:
    p = tmp_path / "mini test.sce"
    p.write_text(MINI_SCE, encoding="utf-8")
    return p


def _wall_pts(pts: list[SpawnPoint]) -> list[SpawnPoint]:
    return [p for p in pts if (p.x, p.y, p.z) != PLAYER_XYZ]


def test_sce_header_and_sections(mini_sce: Path):
    sce = SceFile.read(mini_sce)
    assert sce.get_header("Name") == "mini test"
    assert sce.get_in_section("Character Profile", "target_char", "MainBBRadius") == "0.5"
    sce.set_in_section("Character Profile", "target_char", "MainBBRadius", 0.75)
    assert sce.get_in_section("Character Profile", "target_char", "MainBBRadius") == "0.75"
    # Player profile untouched
    assert sce.get_in_section("Character Profile", "Player", "MainBBRadius") == "1.0"


def test_sce_spawn_parse(mini_sce: Path):
    sce = SceFile.read(mini_sce)
    pts = sce.spawn_points()
    assert len(pts) == 16
    wall = _wall_pts(pts)
    assert {p.y for p in wall} == {200.0, 600.0, 1000.0}
    assert {p.z for p in wall} == {960.0}
    assert sum(1 for p in pts if (p.x, p.y, p.z) == PLAYER_XYZ) == 1


def test_sce_spawn_blocks_exclude_trailing_blank_lines(mini_sce: Path):
    # The file ends with a newline; that blank line belongs to the file,
    # not to the last PlayerSpawn block.
    sce = SceFile.read(mini_sce)
    last = sce.spawn_points()[-1]
    assert all(ln.strip() for ln in last.lines)
    assert sce.lines[-1] == ""  # trailing newline still in the file itself


def test_sce_roundtrip_identity(mini_sce: Path, tmp_path: Path):
    sce = SceFile.read(mini_sce)
    out = tmp_path / "copy.sce"
    sce.write(out)
    assert out.read_bytes() == mini_sce.read_bytes()


def test_sce_bom_roundtrip(tmp_path: Path):
    # Workshop scenarios often carry a UTF-8 BOM; parsing must see through it
    # and writing must re-emit it byte-identically.
    src = tmp_path / "bom.sce"
    src.write_text(MINI_SCE, encoding="utf-8-sig")
    sce = SceFile.read(src)
    assert sce.get_header("Name") == "mini test"  # BOM does not hide line 0
    out = tmp_path / "bom_copy.sce"
    sce.write(out)
    assert out.read_bytes() == src.read_bytes()


def test_sce_crlf_roundtrip(tmp_path: Path):
    # Every game-written .sce is CRLF; an untouched read->write must not
    # rewrite the newline convention.
    src = tmp_path / "crlf.sce"
    src.write_bytes(MINI_SCE.replace("\n", "\r\n").encode("utf-8"))
    sce = SceFile.read(src)
    assert sce.get_header("Name") == "mini test"
    assert len(sce.spawn_points()) == 16
    out = tmp_path / "crlf_copy.sce"
    sce.write(out)
    assert out.read_bytes() == src.read_bytes()


def test_target_profiles_follow_bot_chain(mini_sce: Path):
    # AddedBots "target.bot" -> [Bot Profile] Name=target ->
    # CharacterProfile=target_char -> [Character Profile] Name=target_char
    bots, chars = _target_profiles(SceFile.read(mini_sce))
    assert bots == ["target"]
    assert chars == ["target_char"]


def test_target_profiles_fallback_without_chain(tmp_path: Path):
    # Files whose Bot Profile has no CharacterProfile key (or no Bot Profile
    # at all) fall back to the bot name as the character name.
    text = (
        "Name=simple\nAddedBots=target.bot\n\n"
        "[Character Profile]\nName=target\nMainBBRadius=0.5\n\n"
        "[Bot Profile]\nName=target\nDodgeProfileNames=Mimic\n"
    )
    p = tmp_path / "simple.sce"
    p.write_text(text, encoding="utf-8")
    bots, chars = _target_profiles(SceFile.read(p))
    assert bots == ["target"]
    assert chars == ["target"]


def test_region_grid_wall_axes(mini_sce: Path):
    # Wall spans x (horizontal) and y (vertical): columns must bin x,
    # rows must bin y — up = higher row, right = higher col, matching
    # analysis/movement.py:region_deficits.
    wall = _wall_pts(SceFile.read(mini_sce).spawn_points())
    col, row = _region_grid(wall)
    assert col[0] == 0  # x
    assert row[0] == 1  # y
    top_mid = next(p for p in wall if (p.x, p.y) == (0.0, 1000.0))
    right_mid = next(p for p in wall if (p.x, p.y) == (800.0, 600.0))
    bottom_left = next(p for p in wall if (p.x, p.y) == (-800.0, 200.0))
    assert _region_of(top_mid, col, row, 3, 3) == "r2c1"
    assert _region_of(right_mid, col, row, 3, 3) == "r1c2"
    assert _region_of(bottom_left, col, row, 3, 3) == "r0c0"


def test_region_grid_matches_movement_deficit_keys(mini_sce: Path):
    # Cross-module contract: a flick aimed up must credit the same region
    # key that the generator assigns to the top of the wall.
    from kovadapt.analysis.movement import Flick, region_deficits

    def flick(angle: float) -> Flick:
        return Flick(t_click=0.0, t_onset=0.0, duration=0.1, amplitude=100.0,
                     angle=angle, peak_speed=1.0, time_to_peak=0.05,
                     overshoot=0.1, corrections=0)

    up, right = np.pi / 2, 0.0
    deficits = region_deficits([flick(up)] * 3 + [flick(right)] * 3, cols=3, rows=3)
    wall = _wall_pts(SceFile.read(mini_sce).spawn_points())
    col, row = _region_grid(wall)
    top_mid = next(p for p in wall if (p.x, p.y) == (0.0, 1000.0))
    right_mid = next(p for p in wall if (p.x, p.y) == (800.0, 600.0))
    assert set(deficits) == {
        _region_of(top_mid, col, row, 3, 3),     # up -> r2c1
        _region_of(right_mid, col, row, 3, 3),   # right -> r1c2
    }


def test_region_grid_flat_layout_falls_back_to_depth_rows():
    # Ground arena (x/z spread, no vertical spread): rows bin depth so the
    # grid stays two-dimensional.
    pts = [SpawnPoint(0, 0, float(x), 288.0, float(z), lines=[])
           for x in (-800, -400, 0, 400, 800) for z in (200, 600, 1000)]
    col, row = _region_grid(pts)
    assert col[0] == 0  # x
    assert row[0] == 2  # z (depth) fallback
    far_mid = next(p for p in pts if (p.x, p.z) == (0.0, 1000.0))
    assert _region_of(far_mid, col, row, 3, 3) == "r2c1"


def test_generate_variant(mini_sce: Path, tmp_path: Path):
    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(7))
    prof = PlayerProfile(scenario="mini test [Adaptive]")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(mini_sce, plan, s, tmp_path / "out.sce")

    sce = SceFile.read(out)
    assert sce.get_header("Name") == "mini test [Adaptive]"
    # target resized via the Bot Profile -> CharacterProfile chain
    r = float(sce.get_in_section("Character Profile", "target_char", "MainBBRadius"))
    assert abs(r - 0.5 * plan.target_scale) < 1e-6
    assert sce.get_in_section("Character Profile", "Player", "MainBBRadius") == "1.0"
    # dodge params patched (Bot Profile located by the *bot* name), and the
    # hold window moved relative to the author's own numbers
    base = SceFile.read(mini_sce)
    for key in ("MinLRTimeChange", "MaxLRTimeChange"):
        was = float(base.get_in_section("Dodge Profile", "Mimic", key))
        now = float(sce.get_in_section("Dodge Profile", "Mimic", key))
        assert now != was, f"{key} was not patched at all"
        assert abs(now - was) <= was * (s.dodge_relative_span + 0.05)
    # This fixture carries no JumpVelocity or Gravity at all, so the jump
    # channel is UNKNOWN rather than NO — an old file, not a claim of
    # stillness — and UNKNOWN attempts, matching what drivable_motion does
    # with the same absence. The value still moves relative to the author's.
    was = float(base.get_in_section("Dodge Profile", "Mimic", "JumpFrequency"))
    now = float(sce.get_in_section("Dodge Profile", "Mimic", "JumpFrequency"))
    assert abs(now - was) <= was * (s.dodge_relative_span + 0.05)
    # wall spawn count preserved (within rounding), all positions original
    pts = sce.spawn_points()
    wall = _wall_pts(pts)
    assert 12 <= len(wall) <= 18
    assert all((p.x, p.y) in WALL_XY and p.z == 960.0 for p in wall)


def test_generate_variant_player_spawn_passes_through(mini_sce: Path, tmp_path: Path):
    # The player-side spawn (teamB here) must never be duplicated into the
    # target pool: exactly one copy, byte-identical to the base block.
    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(11))
    prof = PlayerProfile(scenario="mini test [Adaptive]")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(mini_sce, plan, s, tmp_path / "out.sce")

    base_player = [p for p in SceFile.read(mini_sce).spawn_points()
                   if _spawn_team(p) == "teamB"]
    assert len(base_player) == 1
    gen_pts = SceFile.read(out).spawn_points()
    gen_player = [p for p in gen_pts if _spawn_team(p) == "teamB"]
    assert len(gen_player) == 1
    assert gen_player[0].lines == base_player[0].lines
    assert (gen_player[0].x, gen_player[0].y, gen_player[0].z) == PLAYER_XYZ
    # and no target spawn was relocated onto the player's position
    assert all((p.x, p.y, p.z) != PLAYER_XYZ for p in _wall_pts(gen_pts)
               if _spawn_team(p) == "teamA")


def test_generate_variant_map_data_stays_clean(mini_sce: Path, tmp_path: Path):
    # Resampling must not splice blank lines into [Map Data] or drop the
    # file's final newline.
    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(5))
    prof = PlayerProfile(scenario="mini test [Adaptive]")
    out = generate_adaptive_variant(mini_sce, engine.plan(prof, None), s,
                                    tmp_path / "out.sce")
    text = out.read_text(encoding="utf-8")
    assert text.endswith("\n") and not text.endswith("\n\n")
    map_data = text[text.index("[Map Data]"):]
    assert "\n\n" not in map_data


def test_generate_variant_preserves_bom_and_single_name(mini_sce: Path, tmp_path: Path):
    # A BOM'd base must yield a variant with the BOM intact and exactly one
    # Name= header (no duplicate inserted above a hidden original).
    src = tmp_path / "bom base.sce"
    src.write_text(MINI_SCE, encoding="utf-8-sig")
    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(2))
    prof = PlayerProfile(scenario="bom base [Adaptive]")
    out = generate_adaptive_variant(src, engine.plan(prof, None), s,
                                    tmp_path / "bom out.sce")
    raw = out.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    assert "\ufeff" not in text  # no stray BOM mid-file
    header = text[: text.index("[")]
    assert sum(1 for ln in header.split("\n") if ln.startswith("Name=")) == 1
    assert SceFile.read(out).get_header("Name") == "mini test [Adaptive]"


def test_generate_variant_preserves_crlf(mini_sce: Path, tmp_path: Path):
    # A CRLF base (the game-native convention) must yield a CRLF variant.
    src = tmp_path / "crlf base.sce"
    src.write_bytes(MINI_SCE.replace("\n", "\r\n").encode("utf-8"))
    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(4))
    prof = PlayerProfile(scenario="crlf base [Adaptive]")
    out = generate_adaptive_variant(src, engine.plan(prof, None), s,
                                    tmp_path / "crlf out.sce")
    raw = out.read_bytes()
    assert raw.count(b"\n") > 0
    assert raw.count(b"\r\n") == raw.count(b"\n")  # no bare LF anywhere


def test_focus_region_gets_more_density(mini_sce: Path, tmp_path: Path):
    # pinned at 3x3: the mini fixture has too few spawns to exercise the
    # (v0.4-default) 5x5 grid; the mechanism under test is density, not dims
    s = Settings(kovaaks_root=str(tmp_path), focus_weight=0.6,
                 region_cols=3, region_rows=3)
    engine = AdaptationEngine(s, rng=np.random.default_rng(3))
    prof = PlayerProfile(scenario="x")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(mini_sce, plan, s, tmp_path / "o.sce")
    # grid axes/extents come from the base wall cloud (what the generator used)
    base_wall = _wall_pts(SceFile.read(mini_sce).spawn_points())
    col, row = _region_grid(base_wall)
    pts = _wall_pts(SceFile.read(out).spawn_points())
    n_focus = sum(
        1 for p in pts
        if _region_of(p, col, row, s.region_cols, s.region_rows) == plan.focus_region
    )
    assert n_focus >= len(pts) * 0.4  # well above uniform 1/9


# ------------------------------------------------------------ real install
def test_real_sce_roundtrip(kovaaks_root: Path, tmp_path: Path):
    """Read->write every installed scenario byte-identically (BOM, CRLF,
    trailing newline and all)."""
    scen_dir = kovaaks_root / "Saved" / "SaveGames" / "Scenarios"
    files = sorted(scen_dir.glob("*.sce")) if scen_dir.is_dir() else []
    if not files:
        pytest.skip("no scenarios installed")
    out = tmp_path / "rt.sce"
    for src in files:
        sce = SceFile.read(src)
        sce.write(out)
        assert out.read_bytes() == src.read_bytes(), src.name
    wall = scen_dir / "1wall 6targets small.sce"
    if wall.is_file():
        assert len(SceFile.read(wall).spawn_points()) > 100


def test_real_generate_variant_1wall(kovaaks_root: Path, tmp_path: Path):
    """The real 1wall base: the single player spawn passes through verbatim,
    wall density is preserved, and every region can receive spawns."""
    src = kovaaks_root / "Saved" / "SaveGames" / "Scenarios" / "1wall 6targets small.sce"
    if not src.is_file():
        pytest.skip("scenario not installed")
    s = Settings(kovaaks_root=str(kovaaks_root))
    engine = AdaptationEngine(s, rng=np.random.default_rng(9))
    prof = PlayerProfile(scenario="1wall 6targets small [Adaptive]")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(src, plan, s, tmp_path / "gen.sce")

    base_pts = SceFile.read(src).spawn_points()
    base_player = [p for p in base_pts if _spawn_team(p) == "teamB"]
    assert len(base_player) == 1
    gen_pts = SceFile.read(out).spawn_points()
    gen_player = [p for p in gen_pts if _spawn_team(p) == "teamB"]
    assert len(gen_player) == 1
    assert gen_player[0].lines == base_player[0].lines
    n_targets = len(base_pts) - 1
    gen_targets = [p for p in gen_pts if _spawn_team(p) == "teamA"]
    assert abs(len(gen_targets) - n_targets) <= s.region_cols * s.region_rows
    # rows bin the wall's vertical axis: all 9 regions hold candidates
    col, row = _region_grid([p for p in base_pts if _spawn_team(p) == "teamA"])
    regions = {_region_of(p, col, row, 3, 3) for p in gen_targets}
    assert regions == {f"r{r}c{c}" for r in range(3) for c in range(3)}


def test_real_generate_variant_cata(kovaaks_root: Path, tmp_path: Path):
    """Cata links AddedBots -> Bot Profile -> CharacterProfile: size, speed
    and dodge edits must actually land."""
    src = kovaaks_root / "Saved" / "SaveGames" / "Scenarios" / "Cata IC Fast Strafes.sce"
    if not src.is_file():
        pytest.skip("scenario not installed")
    s = Settings(kovaaks_root=str(kovaaks_root))
    engine = AdaptationEngine(s, rng=np.random.default_rng(9))
    prof = PlayerProfile(scenario="Cata IC Fast Strafes [Adaptive]")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(src, plan, s, tmp_path / "gen.sce")

    base = SceFile.read(src)
    gen = SceFile.read(out)
    base_r = float(base.get_in_section("Character Profile", "Quaker", "MainBBRadius"))
    gen_r = float(gen.get_in_section("Character Profile", "Quaker", "MainBBRadius"))
    assert abs(gen_r - base_r * plan.target_scale) < 1e-6
    if plan.target_max_speed > 0:
        # Quaker has an AUTHORED MaxSpeed (1300): the variant must modulate
        # it, never replace it with the absolute static-wall ramp.
        base_speed = float(base.get_in_section("Character Profile", "Quaker", "MaxSpeed"))
        assert base_speed > 0
        assert float(gen.get_in_section("Character Profile", "Quaker", "MaxSpeed")) == \
            pytest.approx(round(base_speed * plan.target_speed_mult, 1))
    # Dodge values are RELATIVE to what the author wrote, not the plan's
    # absolute intent: within +-span of the authored number, so kovadapt
    # nudges this scenario rather than relocating it to a fixed difficulty.
    span = s.dodge_relative_span
    for key in ("MinLRTimeChange", "MaxLRTimeChange", "LeftStrafeTimeMult"):
        was = base.get_in_section("Dodge Profile", "Short Strafes", key)
        got = gen.get_in_section("Dodge Profile", "Short Strafes", key)
        if was is None:
            continue
        assert got is not None, key
        lo, hi = float(was) * (1 - span) * 0.95, float(was) * (1 + span) * 1.05
        assert lo <= float(got) <= hi, f"{key}: {was} -> {got} left the band"
    # and the hold window is still a window
    assert float(gen.get_in_section("Dodge Profile", "Short Strafes",
                                    "MinLRTimeChange")) <= \
        float(gen.get_in_section("Dodge Profile", "Short Strafes",
                                 "MaxLRTimeChange"))


def test_generate_variant_authored_speed_is_scaled_not_replaced(tmp_path: Path):
    """A bot with authored MaxSpeed > 0 keeps its speed class: the variant
    multiplies by plan.target_speed_mult (0.65-1.35) instead of writing the
    0-170 absolute ramp meant for static walls."""
    fast = MINI_SCE.replace("MaxSpeed=0.0", "MaxSpeed=1300.0")
    src = tmp_path / "fast.sce"
    src.write_text(fast)
    s = Settings(kovaaks_root=str(tmp_path))
    plan = AdaptationEngine(s, rng=np.random.default_rng(3)).plan(
        PlayerProfile(scenario="fast [Adaptive]"), None)
    out = generate_adaptive_variant(src, plan, s, tmp_path / "fast gen.sce")
    got = float(SceFile.read(out).get_in_section(
        "Character Profile", "target_char", "MaxSpeed"))
    if plan.target_max_speed > 0:
        assert got == pytest.approx(round(1300.0 * plan.target_speed_mult, 1))
        assert got > 170.0     # never collapsed to the absolute ramp's range
    else:
        assert got == pytest.approx(1300.0)   # untouched when speed edit is off


def test_a_static_wall_gets_no_speed_skew_or_jump_written_into_it(mini_sce, tmp_path):
    """Played in the real game: it did not move.

    The variant carried MaxSpeed 102.5, LeftStrafeTimeMult 1.591,
    RightStrafeTimeMult 0.650 and JumpFrequency 0.211 and the targets did not
    move, strafe or jump. A KovaaK's bot needs Acceleration above zero to
    reach any MaxSpeed at all, and a static wall authors it as 0 — so all
    three of those were numbers the file carried and the game ignored.

    Across all 33 local scenarios every one whose targets move carries
    Acceleration 450-20000 and every static one carries 0, with no exceptions.

    kovadapt does not ADD acceleration: a static click-timing wall is that by
    design, and making it move would change what the task trains rather than
    how hard it is. It writes nothing on those axes and says so on the page.
    """
    txt = mini_sce.read_text(encoding="utf-8")
    assert "MaxSpeed=0.0" in txt
    static = tmp_path / "static.sce"
    static.write_text(txt.replace("MaxSpeed=0.0", "MaxSpeed=0.0\nAcceleration=0.0"),
                      encoding="utf-8")

    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(7))
    prof = PlayerProfile(scenario="mini test [Adaptive]")
    prof.movement = 0.9                      # the planner really does want speed
    plan = engine.plan(prof, None)
    assert plan.target_max_speed > 0, "the plan must be asking for motion"

    base = SceFile.read(static)
    out = SceFile.read(generate_adaptive_variant(static, plan, s, tmp_path / "o.sce"))

    for char in ("target_char",):
        assert out.get_in_section("Character Profile", char, "MaxSpeed") == \
            base.get_in_section("Character Profile", char, "MaxSpeed")
    for key in ("JumpFrequency", "LeftStrafeTimeMult", "RightStrafeTimeMult"):
        assert out.get_in_section("Dodge Profile", "Mimic", key) == \
            base.get_in_section("Dodge Profile", "Mimic", key), key
    # ...and the generator reports it, the same contract as focus_applied
    assert plan.motion_applied is False

    # SIZE AND SPAWNS STILL ADAPT. Both were confirmed working in the same
    # in-game session, and the gate must not touch them.
    assert float(out.get_in_section("Character Profile", "target_char",
                                    "MainBBRadius")) != \
        float(base.get_in_section("Character Profile", "target_char", "MainBBRadius"))


def test_a_scenario_with_acceleration_still_gets_its_motion(mini_sce, tmp_path):
    """The gate reads the FILE's capability, so a base that can move keeps
    every change — and a base with no Acceleration line at all is not making
    a claim either way and keeps them too."""
    txt = mini_sce.read_text(encoding="utf-8")
    s = Settings(kovaaks_root=str(tmp_path))
    prof = PlayerProfile(scenario="mini test [Adaptive]")
    prof.movement = 0.9

    # BOTH keys. Acceleration alone is not motion: `1wall 2targets small -
    # valorant` authors Acceleration=16000 against MaxSpeed=0 and stands
    # still, and an earlier version of this test encoded that wrong rule.
    for label, body in (("self-propelled", txt.replace(
                            "MaxSpeed=0.0", "MaxSpeed=1300.0\nAcceleration=9000.0")),
                        ("no accel key", txt)):
        src = tmp_path / f"{label.replace(' ', '_')}.sce"
        src.write_text(body, encoding="utf-8")
        engine = AdaptationEngine(s, rng=np.random.default_rng(7))
        plan = engine.plan(prof, None)
        out = SceFile.read(generate_adaptive_variant(
            src, plan, s, tmp_path / f"{label.replace(' ', '_')}_out.sce"))
        got = float(out.get_in_section("Character Profile", "target_char", "MaxSpeed"))
        # Two speed paths, chosen by what the AUTHOR wrote. A base that
        # authors a speed is modulated around it; a base that authors none
        # takes the absolute ramp. Asserting the ramp for both was the bug in
        # the first version of this test.
        if label == "self-propelled":
            assert got == pytest.approx(1300.0 * plan.target_speed_mult), label
        else:
            assert got == pytest.approx(plan.target_max_speed), label
        assert plan.motion_applied is True, label


def test_a_rotation_entry_in_addedbots_resolves_to_its_bots(tmp_path):
    """AddedBots takes "X.bot" and also "X.rot". Only `.bot` was ever stripped,
    so a rotation entry matched no [Bot Profile], resolved to no character, and
    the generator wrote nothing but Name and Description — while that
    Description asserted a full plan.

    Measured on a copy of the real Reactive Flick.sce: 2 of 1493 lines changed,
    and the file still claimed `scale=1.18 movement=0.50 focus=r1c4 speed=86`.
    Four of the 33 base scenarios here were in that state. The rotation
    resolves entirely inside the .sce.
    """
    from kovadapt.scenario.generator import _expand_added_bots, _target_profiles

    body = """Name=rot test
AddedBots=pool.rot;solo.bot

[Bot Rotation Profile]
Name=pool
ProfileNames=A;B;A
ProfileWeights=1.0;1.0;1.0
Randomized=true

[Bot Profile]
Name=A
CharacterProfile=charA

[Bot Profile]
Name=B
CharacterProfile=charB

[Bot Profile]
Name=solo
CharacterProfile=charA

[Character Profile]
Name=charA
MaxSpeed=0.0

[Character Profile]
Name=charB
MaxSpeed=0.0
"""
    p = tmp_path / "rot.sce"
    p.write_text(body, encoding="utf-8")
    sce = SceFile.read(p)

    # the repeat is collapsed: every downstream use rewrites a SECTION, which
    # must not be applied twice to the same one
    assert _expand_added_bots(sce) == ["A", "B", "solo"]
    bots, chars = _target_profiles(sce)
    assert bots == ["A", "B", "solo"]
    assert chars == ["charA", "charB"]


def test_an_unresolvable_rotation_yields_nothing_rather_than_a_bad_name(tmp_path):
    """A .rot naming no rotation profile, or one listing bots that do not
    exist, must resolve to nothing — never to the literal "pool.rot" or
    "pool", which would then miss every section lookup silently."""
    from kovadapt.scenario.generator import _expand_added_bots, _target_profiles

    p = tmp_path / "bad.sce"
    p.write_text("""Name=bad
AddedBots=ghost.rot

[Bot Profile]
Name=real
CharacterProfile=charA

[Character Profile]
Name=charA
MaxSpeed=0.0
""", encoding="utf-8")
    sce = SceFile.read(p)
    assert _expand_added_bots(sce) == []
    assert _target_profiles(sce) == ([], [])

    # a rotation that exists but lists a bot with no character profile
    p2 = tmp_path / "partial.sce"
    p2.write_text("""Name=partial
AddedBots=pool.rot

[Bot Rotation Profile]
Name=pool
ProfileNames=A;missing

[Bot Profile]
Name=A
CharacterProfile=charA

[Bot Profile]
Name=missing
CharacterProfile=nope

[Character Profile]
Name=charA
MaxSpeed=0.0
""", encoding="utf-8")
    sce2 = SceFile.read(p2)
    assert _expand_added_bots(sce2) == ["A", "missing"]
    bots, chars = _target_profiles(sce2)
    assert chars == ["charA"], "a bot whose character does not exist was counted"


def test_target_motion_is_four_valued_because_two_is_not_enough(tmp_path):
    """A boolean motion test is wrong in BOTH directions on the real corpus.

    `1wall 2targets small - valorant` and `1wall 6targets small Horizontalish`
    author Acceleration=16000 against MaxSpeed=0 and stand still; Pressure
    Aiming's balloons author both as 0 and cross the room on a movement
    ability at MainVelocity 5000. Six of the eight ability-propelled units in
    the corpus would pass a bare `Acceleration > 0` check.

    The page has to say something TRUE about each, which is why kovadapt
    distinguishes "does it move" from "can kovadapt drive it".
    """
    from kovadapt.scenario.capability import (GRAVITY, IMPULSE, SELF, STATIC,
                                              UNKNOWN, can_express_motion,
                                              drivable_motion, target_motion)

    def sce(**over):
        keys = {"MaxSpeed": "0.0", "Acceleration": "0.0", "Gravity": "0.0",
                "AbilityProfileNames": ";;;"}
        keys.update(over)
        body = "Name=t\n\n[Character Profile]\nName=c\n" + "".join(
            f"{k}={v}\n" for k, v in keys.items() if v is not None)
        body += ("\n[Movement Ability Profile]\nName=Push\n"
                 "MainVelocity=5000.0\nUpVelocity=0.0\n"
                 "\n[Movement Ability Profile]\nName=Still\n"
                 "MainVelocity=0.0\nUpVelocity=0.0\n")
        p = tmp_path / "m.sce"
        p.write_text(body, encoding="utf-8")
        return SceFile.read(p)

    assert target_motion(sce(MaxSpeed="1300.0", Acceleration="9000.0"), "c") == SELF
    # each key alone is NOT motion
    assert target_motion(sce(Acceleration="16000.0"), "c") == STATIC
    assert target_motion(sce(MaxSpeed="1300.0"), "c") == STATIC
    assert target_motion(sce(AbilityProfileNames="Push.abilmov"), "c") == IMPULSE
    assert target_motion(sce(Gravity="1.875"), "c") == GRAVITY
    assert target_motion(sce(), "c") == STATIC
    # a missing key is not a claim of stillness
    assert target_motion(sce(Acceleration=None), "c") == UNKNOWN

    # an ability that is named but carries no velocity is not a movement source
    assert target_motion(sce(AbilityProfileNames="Still.abilmov"), "c") == STATIC
    # ...nor is a dangling reference to one that does not exist
    assert target_motion(sce(AbilityProfileNames="Ghost.abilmov"), "c") == STATIC

    # MOVES vs KOVADAPT CAN DRIVE IT are different questions
    assert can_express_motion({IMPULSE}) and not drivable_motion({IMPULSE})
    assert can_express_motion({GRAVITY}) and not drivable_motion({GRAVITY})
    assert can_express_motion({SELF}) and drivable_motion({SELF})
    assert not can_express_motion({STATIC}) and not drivable_motion({STATIC})
    # nothing resolved is never "no motion"
    assert not can_express_motion(set()) and not drivable_motion(set())
    assert can_express_motion({UNKNOWN}) and drivable_motion({UNKNOWN})
    assert not can_express_motion({UNKNOWN}, unknown_is_capable=False)


def _cap_sce(tmp_path, *, player=("0.0", "0.0"), target=("0.0", "0.0"),
             dodge=True, lr="true", fb="false", jump=("0.0", "0.0"),
             score=("100.0", "0.0"), invincible="false", spawns=0, extra=""):
    """A minimal .sce with every capability-bearing key under control."""
    body = f"""Name=cap
AddedBots=t.bot
PlayerProfile=Me
PlayerTeam=1
ScorePerKill={score[0]}
ScorePerDamage={score[1]}
InvincibleBots={invincible}

[Character Profile]
Name=Me
MaxSpeed={player[0]}
Acceleration={player[1]}

[Character Profile]
Name=tc
MaxSpeed={target[0]}
Acceleration={target[1]}
JumpVelocity={jump[0]}
Gravity={jump[1]}
MainBBRadius=50.0

[Bot Profile]
Name=t
CharacterProfile=tc
{'DodgeProfileNames=D' if dodge else 'DodgeProfileNames='}

[Dodge Profile]
Name=D
ToggleLeftRight={lr}
ToggleForwardBack={fb}
{extra}
[Map Data]
reflex map version 8
global
\tentity
\t\ttype PlayerSpawn
\t\tVector3 position 0.000000 0.000000 -960.000000
\t\tBool8 teamB 0
"""
    for i in range(spawns):
        body += (f"\tentity\n\t\ttype PlayerSpawn\n\t\tVector3 position "
                 f"{i * 30}.000000 {i * 20}.000000 960.000000\n\t\tBool8 teamA 0\n")
    p = tmp_path / f"cap{abs(hash((player, target, dodge, lr, spawns)))}.sce"
    p.write_text(body, encoding="utf-8")
    return SceFile.read(p)


def test_strafe_and_jump_are_separate_channels_from_motion(tmp_path):
    """`Revolving Tracking` is SELF-propelled (MaxSpeed 1024, Acceleration
    9000) and carries NO dodge channel — so motion does not imply a strafe
    timer to scale. And `JumpFrequency` is meaningless without something to
    jump with: kovadapt writes it today onto characters whose JumpVelocity is
    0, where no frequency produces a jump.
    """
    from kovadapt.scenario.capability import jump_channel, strafe_channel
    from kovadapt.scenario.generator import _target_profiles

    def tags(**kw):
        sce = _cap_sce(tmp_path, **kw)
        bots, chars = _target_profiles(sce)
        return strafe_channel(sce, bots, chars), jump_channel(sce, bots, chars)

    mover = {"target": ("1300.0", "9000.0")}
    (strafe, fb), jump = tags(**mover)
    assert strafe == "YES" and fb is False and jump == "NO"

    # self-propelled but no dodge profile at all
    (strafe, _), jump = tags(**mover, dodge=False)
    assert strafe == "NO" and jump == "NO"

    # dodge profile that toggles nothing is not a strafe channel
    (strafe, _), _ = tags(**mover, lr="false")
    assert strafe == "NO"
    (_, fb), _ = tags(**mover, lr="false", fb="true")
    assert fb is True

    # a static target has no strafe timer however many dodge blocks it has
    (strafe, _), _ = tags(target=("0.0", "0.0"))
    assert strafe == "NO"

    # jump needs BOTH a rise and something to fall back down
    _, jump = tags(**mover, jump=("800.0", "1.5"))
    assert jump == "YES"
    _, jump = tags(**mover, jump=("800.0", "0.0"))
    assert jump == "UNKNOWN", "rise with no fall is not settled by the file"
    _, jump = tags(**mover, jump=("0.0", "1.5"))
    assert jump == "NO"


def test_score_frame_and_the_player_frame_that_gates_measurement(tmp_path):
    """SCORE_FRAME gates the size controller's INPUT: `Run.accuracy` means
    "shots that connected" on a KILL scenario and "damage ticks that landed"
    on a DAMAGE one, against the same band.

    PLAYER_FRAME gates measurement itself. A strafing player must counter-move
    to hold even a stationary target, so overshoot becomes compensation error.
    Both keys are required for the same reason as target motion, and the
    counterexamples are real: Revolving Tracking gives the player MaxSpeed
    1024 with Acceleration 0, Narrow Strafe the reverse.
    """
    from kovadapt.scenario.capability import (player_frame, read_capability,
                                              score_frame)
    from kovadapt.scenario.generator import _target_profiles

    def frame(**kw):
        sce = _cap_sce(tmp_path, **kw)
        return score_frame(sce, _target_profiles(sce)[1])

    assert frame(score=("100.0", "0.0")) == ("KILL", False)
    assert frame(score=("0.0", "1.0")) == ("DAMAGE", False)
    assert frame(score=("10.0", "1.0")) == ("HYBRID", False)
    assert frame(score=("0.0", "1.0"), invincible="true") == ("DAMAGE", True)

    assert player_frame(_cap_sce(tmp_path, player=("0.0", "0.0"))) == "STATIC"
    assert player_frame(_cap_sce(tmp_path, player=("1300.0", "9000.0"))) == "MOBILE"
    # either key alone cannot produce motion, and must not read as MOBILE
    assert player_frame(_cap_sce(tmp_path, player=("1024.0", "0.0"))) == "PARTIAL"
    assert player_frame(_cap_sce(tmp_path, player=("0.0", "16000.0"))) == "PARTIAL"

    cap = read_capability(_cap_sce(tmp_path, player=("1300.0", "9000.0")))
    assert cap.player_frame == "MOBILE"
    assert cap.measurement_frame_is_static is False, \
        "flick microstructure was called trustworthy in a moving frame"
    assert read_capability(
        _cap_sce(tmp_path, player=("0.0", "16000.0"))).measurement_frame_is_static


def test_a_json_map_is_spawn_blind_not_spawn_free(tmp_path):
    """The reflex entity parser cannot see the JSON `[Map Data]` form at all,
    so zero spawns there is a failure to read rather than a fact about the
    layout. Ten files here are in that state; calling them "no spawn field"
    would assert something never looked at, and the page would then say spawn
    placement does not adapt when nobody checked."""
    from kovadapt.scenario.capability import read_capability, spawn_field

    enough = _cap_sce(tmp_path, spawns=30)
    assert spawn_field(enough, 5, 5) == ("YES", 30)
    # fewer target spawns than grid cells: resample_spawns refuses to reweight
    assert spawn_field(_cap_sce(tmp_path, spawns=9), 5, 5) == ("NO", 9)
    assert spawn_field(_cap_sce(tmp_path, spawns=9), 3, 3) == ("YES", 9)

    json_map = tmp_path / "json.sce"
    json_map.write_text(
        "\n".join(_cap_sce(tmp_path, spawns=30).lines).replace(
            "Name=cap", "Name=cap\nMapName=Sandbox.json", 1), encoding="utf-8")
    cap = read_capability(SceFile.read(json_map))
    assert cap.spawn_field == "BLIND"
    assert cap.spawn_field != "NO", "asserted an absence it never read"


def test_dodge_writes_stay_inside_the_authors_own_band(tmp_path):
    """Arjun's decision: kovadapt nudges a scenario rather than relocating it
    to an absolute difficulty point.

    The absolute form failed two ways that a relative one cannot. Its ceiling
    could sit BELOW an author's floor — JumpFrequency capped at 0.35 while 13
    of the 54 authored values in the corpus exceed it, so maximum difficulty
    made `1wall 6targets small` and `mccoyfrozentrack` (both 0.50) jump LESS
    than intended. And it wrote one value to every profile, erasing deliberate
    contrast like Surge Tags' 0.0 / 0.9 / 0.8 / 0.02 / 0.02.
    """
    from kovadapt.adapt.stochastic import relative_dodge_writes

    authored = {"JumpFrequency": 0.5, "MinLRTimeChange": 0.2,
                "MaxLRTimeChange": 0.5, "LeftStrafeTimeMult": 1.0}
    rng = np.random.default_rng(4)

    # maximum intensity makes it HARDER than the author wrote, never easier
    hard = relative_dodge_writes(authored, 1.0, rng=np.random.default_rng(4))
    assert hard["JumpFrequency"] > 0.5, "max difficulty jumped less than authored"
    assert hard["MinLRTimeChange"] < 0.2, "max difficulty held longer than authored"

    # minimum makes it easier, and the neutral point returns the author's own
    easy = relative_dodge_writes(authored, 0.0, rng=np.random.default_rng(4))
    assert easy["JumpFrequency"] < 0.5 and easy["MinLRTimeChange"] > 0.2
    mid = relative_dodge_writes(authored, 0.5, rng=np.random.default_rng(4))
    assert mid["JumpFrequency"] == pytest.approx(0.5, rel=0.06)

    # NO INVERSION. One common factor per profile, so a profile the author
    # wrote higher can never come out lower. Order can TIE at the top — 0.8
    # and 0.9 both reach the probability ceiling of 1.0 — and a tie is a loss
    # of resolution, not an inversion. Asserting an exact permutation would be
    # asserting that the clamp does not exist.
    surge = [0.0, 0.9, 0.8, 0.02, 0.02]
    out = [relative_dodge_writes({"JumpFrequency": v}, 1.0,
                                 rng=np.random.default_rng(9))["JumpFrequency"]
           for v in surge]
    for i, a in enumerate(surge):
        for j, b in enumerate(surge):
            if a < b:
                assert out[i] <= out[j], (
                    f"authored {a} < {b} but emitted {out[i]} > {out[j]}")

    # a frequency used as a probability never exceeds 1
    assert relative_dodge_writes({"JumpFrequency": 0.95}, 1.0,
                                 rng=rng)["JumpFrequency"] <= 1.0
    # a key the author never wrote is never invented: set_in_section only
    # rewrites an existing line, so an invented value goes nowhere
    assert "MinFBTimeChange" not in relative_dodge_writes(authored, 1.0, rng=rng)

    # the hold window stays a window, however the jitter falls
    for seed in range(300):
        r = relative_dodge_writes({"MinLRTimeChange": 0.30, "MaxLRTimeChange": 0.31},
                                  0.5, rng=np.random.default_rng(seed))
        assert r["MinLRTimeChange"] <= r["MaxLRTimeChange"], seed


def test_a_jump_frequency_is_not_written_where_nothing_can_jump(tmp_path):
    """A frequency is inert without something to jump WITH. kovadapt wrote one
    onto characters whose JumpVelocity is 0, where no value produces a jump —
    a number in the file that the game reads and cannot act on.

    NO suppresses; UNKNOWN does not. A file with no JumpVelocity line has not
    said its targets are grounded, and suppressing there would contradict what
    `drivable_motion` does with the same absence.
    """
    base = _cap_sce(tmp_path, target=("1300.0", "9000.0"),
                    jump=("0.0", "0.0"), extra="JumpFrequency=0.5\n")
    p = tmp_path / "nojump.sce"
    p.write_text("\n".join(base.lines), encoding="utf-8")

    s = Settings(kovaaks_root=str(tmp_path))
    prof = PlayerProfile(scenario="cap [Adaptive]")
    prof.movement = 0.9
    plan = AdaptationEngine(s, rng=np.random.default_rng(5)).plan(prof, None)
    out = SceFile.read(generate_adaptive_variant(p, plan, s, tmp_path / "o.sce"))

    assert out.get_in_section("Dodge Profile", "D", "JumpFrequency") == "0.5", \
        "wrote a jump frequency onto a target that cannot jump"
    # ...while the strafe keys, which this target CAN act on, did move
    assert out.get_in_section("Dodge Profile", "D", "ToggleLeftRight") == "true"

    # and with a real jump channel it is written
    jumper = _cap_sce(tmp_path, target=("1300.0", "9000.0"),
                      jump=("800.0", "1.5"), extra="JumpFrequency=0.5\n")
    p2 = tmp_path / "jump.sce"
    p2.write_text("\n".join(jumper.lines), encoding="utf-8")
    out2 = SceFile.read(generate_adaptive_variant(p2, plan, s, tmp_path / "o2.sce"))
    assert out2.get_in_section("Dodge Profile", "D", "JumpFrequency") != "0.5"


def test_strafe_timings_are_not_written_where_nothing_strafes(tmp_path):
    """`MinLRTimeChange` and the strafe multipliers scale a left/right timer.
    A dodge profile with `ToggleLeftRight=false` has no such timer, so those
    keys are inert there however the target moves — `Revolving Tracking` is
    self-propelled and carries exactly that shape."""
    base = _cap_sce(tmp_path, target=("1300.0", "9000.0"), lr="false",
                    extra="MinLRTimeChange=0.2\nMaxLRTimeChange=0.5\n"
                          "LeftStrafeTimeMult=1.0\nRightStrafeTimeMult=1.0\n"
                          "JumpFrequency=0.5\n", jump=("800.0", "1.5"))
    p = tmp_path / "nostrafe.sce"
    p.write_text("\n".join(base.lines), encoding="utf-8")

    s = Settings(kovaaks_root=str(tmp_path))
    prof = PlayerProfile(scenario="cap [Adaptive]")
    prof.movement = 0.9
    plan = AdaptationEngine(s, rng=np.random.default_rng(6)).plan(prof, None)
    out = SceFile.read(generate_adaptive_variant(p, plan, s, tmp_path / "o.sce"))

    for key, was in (("MinLRTimeChange", "0.2"), ("MaxLRTimeChange", "0.5"),
                     ("LeftStrafeTimeMult", "1.0"), ("RightStrafeTimeMult", "1.0")):
        assert out.get_in_section("Dodge Profile", "D", key) == was, \
            f"{key} was written onto a profile that does not strafe"
    # the jump channel is real here and still moves
    assert out.get_in_section("Dodge Profile", "D", "JumpFrequency") != "0.5"
