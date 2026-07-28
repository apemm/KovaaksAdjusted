"""Apply an AdaptationPlan to a base .sce, producing the adaptive variant.

Edits performed:
  - Name                       -> "<Base> [Adaptive]"
  - target [Character Profile] -> MainBBRadius/Height, ProjBB*, MaxSpeed scaled
  - target [Dodge Profile]s    -> OU-sampled timing params (micro-movement)
  - [Map Data] PlayerSpawns    -> resampled by region weights (weak regions
                                  get proportionally more spawn density)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..adapt.engine import AdaptationPlan
from ..config import ADAPTIVE_SUFFIX, Settings
from .sce import SceFile, SpawnPoint

_SIZE_KEYS = ("MainBBRadius", "MainBBHeight", "MainBBHeadRadius",
              "ProjBBRadius", "ProjBBHeight", "ProjBBHeadRadius")

_TEAM_FLAG_RE = re.compile(r"^\s*Bool8 (team[AB]) ")

# One grid axis: (index into (x, y, z), lo extent, hi extent).
Axis = tuple[int, float, float]


def _player_team_flag(sce: SceFile) -> str | None:
    """Reflex spawn flag ("teamA"/"teamB") marking the *player's* side.

    KovaaK's stores team indices in the PlayerTeam/BotTeams headers; in the
    embedded reflex map, even indices spawn on teamA and odd on teamB
    (verified across stock + workshop scenarios: 1wall has PlayerTeam=1 and
    a single teamB block where the player stands; Cata has PlayerTeam=2 and
    a single teamA block)."""
    raw = sce.get_header("PlayerTeam")
    if raw is None:
        return None
    try:
        return "teamB" if int(raw.strip()) % 2 else "teamA"
    except ValueError:
        return None


def _spawn_team(p: SpawnPoint) -> str | None:
    for ln in p.lines:
        m = _TEAM_FLAG_RE.match(ln)
        if m:
            return m.group(1)
    return None


def _region_grid(pts: list[SpawnPoint]) -> tuple[Axis, Axis]:
    """(column axis, row axis) of the region grid over the target spawn cloud.

    Reflex map coordinates are y-up: wall scenarios spread targets across a
    world-horizontal axis (x or z) and vertical y, with the remaining axis
    the (constant) wall depth. Columns bin the dominant horizontal axis
    left->right, rows bin y bottom->top — matching
    analysis/movement.py:region_deficits, where a rightward flick credits a
    higher col and an upward flick a higher row. Flat ground layouts with no
    vertical spread fall back to binning depth as rows (farther targets sit
    higher on screen) so the grid stays two-dimensional."""
    coords = np.array([(p.x, p.y, p.z) for p in pts], dtype=float)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    spread = hi - lo
    col_ax = 0 if spread[0] >= spread[2] else 2
    row_ax = 1
    if spread[1] <= max(1e-9, 0.05 * spread[col_ax]):
        row_ax = 2 if col_ax == 0 else 0
    return ((col_ax, float(lo[col_ax]), float(hi[col_ax])),
            (row_ax, float(lo[row_ax]), float(hi[row_ax])))


def _region_of(p: SpawnPoint, col: Axis, row: Axis, cols: int, rows: int) -> str:
    """Grid cell key (r{row}c{col}) of a spawn point."""
    v = (p.x, p.y, p.z)
    c = min(int((v[col[0]] - col[1]) / max(col[2] - col[1], 1e-9) * cols), cols - 1)
    r = min(int((v[row[0]] - row[1]) / max(row[2] - row[1], 1e-9) * rows), rows - 1)
    return f"r{r}c{c}"


def _scale_line(line: str, factor: float) -> str:
    key, _, val = line.partition("=")
    try:
        return f"{key}={round(float(val) * factor, 6)}"
    except ValueError:
        return line


def resample_spawns(sce: SceFile, weights: dict[str, float],
                    cols: int, rows: int, seed: int) -> None:
    """Rebuild the target PlayerSpawn list: same total count, region densities
    proportional to `weights`. Sampling with replacement within regions keeps
    every emitted spawn a physically valid original location. Player-side
    spawns (the side the PlayerTeam header maps to) pass through verbatim —
    they never enter the resample pool or the region extents."""
    pts = sce.spawn_points()
    player_flag = _player_team_flag(sce)
    if player_flag is not None:
        player_pts = [p for p in pts if _spawn_team(p) == player_flag]
        targets = [p for p in pts if _spawn_team(p) != player_flag]
    else:
        player_pts, targets = [], pts
    if len(targets) < cols * rows:
        return  # too few spawns to meaningfully reweight
    col_ax, row_ax = _region_grid(targets)

    by_region: dict[str, list[SpawnPoint]] = {}
    for p in targets:
        by_region.setdefault(_region_of(p, col_ax, row_ax, cols, rows), []).append(p)

    rng = np.random.default_rng(seed)
    total = len(targets)
    # Only regions that actually contain candidate spawns can receive mass.
    valid = {k: w for k, w in weights.items() if k in by_region}
    if not valid:
        return
    wsum = sum(valid.values())
    alloc = {k: max(1, round(w / wsum * total)) for k, w in valid.items()}

    new_blocks: list[list[str]] = []
    for key, count in alloc.items():
        pool = by_region[key]
        idx = rng.integers(0, len(pool), size=count)
        new_blocks.extend(pool[i].lines for i in idx)
    rng.shuffle(new_blocks)
    sce.replace_spawn_points([p.lines for p in player_pts] + new_blocks)


def _target_profiles(sce: SceFile) -> tuple[list[str], list[str]]:
    """(bot names, character profile names) for the AddedBots targets.

    Real scenarios link AddedBots "X.bot" -> [Bot Profile] Name=X ->
    CharacterProfile=<char> -> [Character Profile] Name=<char>. Files whose
    Bot Profile lacks the CharacterProfile key (older/simple scenarios) fall
    back to using the bot name as the character name directly."""
    bots = sce.get_header("AddedBots") or ""
    bot_names = sorted({re.sub(r"\.bot$", "", b) for b in bots.split(";") if b})
    chars: set[str] = set()
    for bot in bot_names:
        char = sce.get_in_section("Bot Profile", bot, "CharacterProfile")
        char = char.strip() if char else bot
        if sce.find_section("Character Profile", char):
            chars.add(char)
    return bot_names, sorted(chars)


def generate_adaptive_variant(
    base_sce: Path | str,
    plan: AdaptationPlan,
    settings: Settings,
    out_path: Path | str | None = None,
) -> Path:
    base_sce = Path(base_sce)
    sce = SceFile.read(base_sce)

    base_name = sce.get_header("Name") or base_sce.stem
    if not base_name.endswith(ADAPTIVE_SUFFIX):
        sce.set_header("Name", base_name + ADAPTIVE_SUFFIX)
    sce.set_header(
        "Description",
        f"kovadapt auto-generated | {plan.describe()} | base: {base_name}",
    )

    bot_names, char_names = _target_profiles(sce)

    # --- target size + speed ---------------------------------------------
    for char in char_names:
        span = sce.find_section("Character Profile", char)
        assert span is not None
        for i in range(span[0], span[1]):
            key = sce.lines[i].partition("=")[0]
            if key in _SIZE_KEYS:
                sce.lines[i] = _scale_line(sce.lines[i], plan.target_scale)
        if plan.target_max_speed > 0:
            try:
                base_speed = float(
                    sce.get_in_section("Character Profile", char, "MaxSpeed") or 0.0)
            except ValueError:
                base_speed = 0.0
            if base_speed > 0:
                # Authored speed (strafe/tracking bots): modulate around the
                # author's value. Writing the absolute static-wall ramp here
                # would slow a 1300-speed bot to a crawl.
                sce.set_in_section("Character Profile", char, "MaxSpeed",
                                   round(base_speed * plan.target_speed_mult, 1))
            else:
                sce.set_in_section("Character Profile", char, "MaxSpeed",
                                   plan.target_max_speed)

    # --- micro-movement: patch every dodge profile the target bots use ----
    # ([Bot Profile] sections are keyed by the *bot* name, not the character.)
    dodge_names: set[str] = set()
    for span in [sce.find_section("Bot Profile", n) for n in bot_names]:
        if span is None:
            continue
        for ln in sce.lines[span[0]: span[1]]:
            if ln.startswith("DodgeProfileNames="):
                dodge_names.update(d for d in ln.partition("=")[2].split(";") if d)
    for dodge in sorted(dodge_names):
        for key, val in plan.dodge_params.items():
            sce.set_in_section("Dodge Profile", dodge, key, val)

    # --- spawn region reweighting ------------------------------------------
    resample_spawns(sce, plan.spawn_weights, settings.region_cols,
                    settings.region_rows, plan.seed)

    if out_path is None:
        out_path = base_sce.parent / f"{base_name}{ADAPTIVE_SUFFIX}.sce"
    out_path = Path(out_path)
    sce.write(out_path)
    return out_path
