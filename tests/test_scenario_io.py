"""Durability of the .sce write path + what spawn resampling can't do.

Everything here works on files under tmp_path; no real install, no ~/.kovadapt.
"""

from pathlib import Path

import pytest

from kovadapt.scenario.generator import (
    _region_grid,
    _region_of,
    _spawn_team,
    resample_spawns,
)
from kovadapt.scenario.sce import SceFile

COLS = ROWS = 5
ALL_KEYS = [f"r{r}c{c}" for r in range(ROWS) for c in range(COLS)]
HOLE = "r4c0"  # top-left cell, emptied by the holed_wall fixture

_HEAD = """Name=io probe
AddedBots=target.bot
PlayerTeam=1

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


def _wall(coords: list[tuple[float, float]]) -> str:
    """A 1wall-shaped .sce: player spawn on teamB, targets on teamA."""
    return _HEAD + "".join(
        f"\tentity\n\t\ttype PlayerSpawn\n"
        f"\t\tVector3 position {x:.6f} {y:.6f} 960.000000\n"
        f"\t\tBool8 teamA 0\n"
        for x, y in coords
    )


def _targets(sce: SceFile):
    return [p for p in sce.spawn_points() if _spawn_team(p) == "teamA"]


def _regions(pts, col, row) -> set[str]:
    return {_region_of(p, col, row, COLS, ROWS) for p in pts}


@pytest.fixture
def full_wall(tmp_path: Path) -> Path:
    p = tmp_path / "full wall.sce"
    p.write_text(_wall([(float(x), float(y))
                        for x in range(0, 501, 50) for y in range(0, 501, 50)]),
                 encoding="utf-8")
    return p


@pytest.fixture
def holed_wall(tmp_path: Path, full_wall: Path) -> Path:
    """Same wall with every spawn in HOLE removed — a layout the bandit can
    still pick as focus but the generator can never spawn into."""
    pts = _targets(SceFile.read(full_wall))
    col, row = _region_grid(pts)
    keep = [(p.x, p.y) for p in pts
            if _region_of(p, col, row, COLS, ROWS) != HOLE]
    p = tmp_path / "holed wall.sce"
    p.write_text(_wall(keep), encoding="utf-8")
    return p


# --------------------------------------------------------------- atomic write
def test_write_leaves_no_debris(full_wall: Path, tmp_path: Path):
    out = tmp_path / "out" / "x.sce"
    out.parent.mkdir()
    SceFile.read(full_wall).write(out)
    assert {p.name for p in out.parent.iterdir()} == {"x.sce"}
    assert out.read_bytes() == full_wall.read_bytes()


def test_write_interrupted_leaves_the_previous_variant_intact(
    full_wall: Path, holed_wall: Path, tmp_path: Path, monkeypatch
):
    # The variant lives in the game's Scenarios folder; a torn write there is
    # unrecoverable (watch() only bootstraps a *missing* file). Simulate dying
    # at the last step: everything before the rename must be invisible.
    dest = tmp_path / "Adaptive.sce"
    previous = holed_wall.read_bytes()
    dest.write_bytes(previous)

    def boom(self, target):
        raise OSError("simulated crash before the rename")

    monkeypatch.setattr(Path, "replace", boom)
    with pytest.raises(OSError):
        SceFile.read(full_wall).write(dest)

    assert dest.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_write_roundtrip_stays_byte_identical(tmp_path: Path):
    # The atomic path must not touch the BOM/CRLF round-trip contract.
    body = _wall([(0.0, 0.0), (100.0, 100.0)])
    for name, raw in (
        ("plain.sce", body.encode("utf-8")),
        ("bom.sce", body.encode("utf-8-sig")),
        ("crlf.sce", body.replace("\n", "\r\n").encode("utf-8")),
    ):
        src = tmp_path / name
        src.write_bytes(raw)
        out = tmp_path / f"copy {name}"
        SceFile.read(src).write(out)
        assert out.read_bytes() == raw, name


# ------------------------------------------------------- unactionable focus
def test_resample_reports_the_regions_it_filled(holed_wall: Path):
    sce = SceFile.read(holed_wall)
    before = _targets(sce)
    col, row = _region_grid(before)
    populated = _regions(before, col, row)
    assert HOLE not in populated and len(populated) == COLS * ROWS - 1

    # Plan asks for half the mass on the empty cell (what the bandit does
    # whenever it focuses a region the layout has no spawn for).
    weights = {k: 0.5 / (len(ALL_KEYS) - 1) for k in ALL_KEYS}
    weights[HOLE] = 0.5

    used = resample_spawns(sce, weights, COLS, ROWS, seed=1)
    assert used == populated                      # focus was not honoured...
    assert HOLE not in used                       # ...and the caller can tell
    assert HOLE not in _regions(_targets(sce), col, row)


def test_resample_reports_nothing_when_it_cannot_reweight(tmp_path: Path):
    # Fewer targets than grid cells: the layout is left exactly as authored,
    # so no region received planned mass at all.
    src = tmp_path / "sparse.sce"
    src.write_text(_wall([(float(x), 0.0) for x in range(0, 500, 50)]),
                   encoding="utf-8")
    sce = SceFile.read(src)
    before = [p.lines for p in sce.spawn_points()]
    assert resample_spawns(sce, {k: 1.0 / len(ALL_KEYS) for k in ALL_KEYS},
                           COLS, ROWS, seed=1) == set()
    assert [p.lines for p in sce.spawn_points()] == before
