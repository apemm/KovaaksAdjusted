from pathlib import Path

import numpy as np
import pytest

from kovadapt.adapt.engine import AdaptationEngine
from kovadapt.config import Settings
from kovadapt.profile.player import PlayerProfile
from kovadapt.scenario.sce import SceFile
from kovadapt.scenario.generator import generate_adaptive_variant

MINI_SCE = """Name=mini test
AddedBots=target.bot;target.bot
Timelimit=60.0

[Character Profile]
Name=Player
MaxHealth=100.0
MainBBRadius=1.0

[Character Profile]
Name=target
MaxHealth=1.0
MaxSpeed=0.0
MainBBRadius=0.5
MainBBHeight=2.0
ProjBBRadius=0.5

[Bot Profile]
Name=target
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
""" + "".join(
    f"\tentity\n\t\ttype PlayerSpawn\n\t\tVector3 position {x}.000000 288.000000 {z}.000000\n"
    f"\t\tVector3 angles 180.000000 0.000000 0.000000\n"
    for x in (-800, -400, 0, 400, 800) for z in (200, 600, 1000)
)


@pytest.fixture
def mini_sce(tmp_path: Path) -> Path:
    p = tmp_path / "mini test.sce"
    p.write_text(MINI_SCE, encoding="utf-8")
    return p


def test_sce_header_and_sections(mini_sce: Path):
    sce = SceFile.read(mini_sce)
    assert sce.get_header("Name") == "mini test"
    assert sce.get_in_section("Character Profile", "target", "MainBBRadius") == "0.5"
    sce.set_in_section("Character Profile", "target", "MainBBRadius", 0.75)
    assert sce.get_in_section("Character Profile", "target", "MainBBRadius") == "0.75"
    # Player profile untouched
    assert sce.get_in_section("Character Profile", "Player", "MainBBRadius") == "1.0"


def test_sce_spawn_parse(mini_sce: Path):
    sce = SceFile.read(mini_sce)
    pts = sce.spawn_points()
    assert len(pts) == 15
    assert {p.z for p in pts} == {200.0, 600.0, 1000.0}


def test_sce_roundtrip_identity(mini_sce: Path, tmp_path: Path):
    sce = SceFile.read(mini_sce)
    out = tmp_path / "copy.sce"
    sce.write(out)
    assert out.read_text(encoding="utf-8") == mini_sce.read_text(encoding="utf-8")


def test_generate_variant(mini_sce: Path, tmp_path: Path):
    s = Settings(kovaaks_root=str(tmp_path))
    engine = AdaptationEngine(s, rng=np.random.default_rng(7))
    prof = PlayerProfile(scenario="mini test [Adaptive]")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(mini_sce, plan, s, tmp_path / "out.sce")

    sce = SceFile.read(out)
    assert sce.get_header("Name") == "mini test [Adaptive]"
    # target resized, player untouched
    r = float(sce.get_in_section("Character Profile", "target", "MainBBRadius"))
    assert abs(r - 0.5 * plan.target_scale) < 1e-6
    assert sce.get_in_section("Character Profile", "Player", "MainBBRadius") == "1.0"
    # dodge params patched
    got = float(sce.get_in_section("Dodge Profile", "Mimic", "JumpFrequency"))
    assert abs(got - plan.dodge_params["JumpFrequency"]) < 1e-6
    # spawn count preserved (within rounding), all positions from original set
    pts = sce.spawn_points()
    assert 12 <= len(pts) <= 18
    orig_xz = {(x, z) for x in (-800.0, -400.0, 0.0, 400.0, 800.0)
               for z in (200.0, 600.0, 1000.0)}
    assert all((p.x, p.z) in orig_xz for p in pts)


def test_focus_region_gets_more_density(mini_sce: Path, tmp_path: Path):
    s = Settings(kovaaks_root=str(tmp_path), focus_weight=0.6)
    engine = AdaptationEngine(s, rng=np.random.default_rng(3))
    prof = PlayerProfile(scenario="x")
    plan = engine.plan(prof, None)
    out = generate_adaptive_variant(mini_sce, plan, s, tmp_path / "o.sce")
    pts = SceFile.read(out).spawn_points()
    xs = sorted({p.x for p in pts} | {-800.0, 800.0})
    # count spawns in focus cell vs uniform expectation
    from kovadapt.scenario.generator import _region_of
    x_ext, z_ext = (-800.0, 800.0), (200.0, 1000.0)
    n_focus = sum(
        1 for p in pts
        if _region_of(p, x_ext, z_ext, s.region_cols, s.region_rows) == plan.focus_region
    )
    assert n_focus >= len(pts) * 0.4  # well above uniform 1/9


def test_real_sce_roundtrip(kovaaks_root: Path, tmp_path: Path):
    """Read->write a real scenario byte-identically (modulo trailing newline)."""
    src = kovaaks_root / "Saved" / "SaveGames" / "Scenarios" / "1wall 6targets small.sce"
    if not src.is_file():
        pytest.skip("scenario not installed")
    sce = SceFile.read(src)
    out = tmp_path / "rt.sce"
    sce.write(out)
    assert out.read_text(encoding="utf-8") == src.read_text(encoding="utf-8", errors="replace")
    assert len(sce.spawn_points()) > 100
