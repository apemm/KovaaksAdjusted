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
from ..adapt.stochastic import relative_dodge_writes
from .capability import (drivable_motion, jump_channel,
                         scenario_motion, strafe_channel)
from .sce import SceFile, SpawnPoint

_SIZE_KEYS = ("MainBBRadius", "MainBBHeight", "MainBBHeadRadius",
              "ProjBBRadius", "ProjBBHeight", "ProjBBHeadRadius")

_TEAM_FLAG_RE = re.compile(r"^\s*Bool8 (team[AB]) ")

#: Every [Dodge Profile] key kovadapt may touch. Read from the file first: a
#: key the author did not write is never invented, because `set_in_section`
#: only rewrites an existing line and a value that goes nowhere is worse than
#: no value at all.
_DODGE_KEYS = ("MinLRTimeChange", "MaxLRTimeChange", "MinFBTimeChange",
               "MaxFBTimeChange", "StrafeSwapMinPause", "StrafeSwapMaxPause",
               "JumpFrequency", "LeftStrafeTimeMult", "RightStrafeTimeMult")
#: The subset that only means something when a strafe timer exists.
_STRAFE_KEYS = ("MinLRTimeChange", "MaxLRTimeChange", "StrafeSwapMinPause",
                "StrafeSwapMaxPause", "LeftStrafeTimeMult",
                "RightStrafeTimeMult")

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
                    cols: int, rows: int, seed: int) -> set[str]:
    """Rebuild the target PlayerSpawn list: same total count, region densities
    proportional to `weights`. Sampling with replacement within regions keeps
    every emitted spawn a physically valid original location. Player-side
    spawns (the side the PlayerTeam header maps to) pass through verbatim —
    they never enter the resample pool or the region extents.

    Returns the regions that actually received spawns (empty when the layout
    was left untouched). A region holding no candidate spawn point cannot be
    emphasized — we never invent coordinates — so its weight is absorbed by
    the rest; when that region is the plan's focus, the emitted scenario has
    no focus at all. Callers that credit a bandit arm for the run this file
    produces need to know that, hence the return value."""
    pts = sce.spawn_points()
    player_flag = _player_team_flag(sce)
    if player_flag is not None:
        player_pts = [p for p in pts if _spawn_team(p) == player_flag]
        targets = [p for p in pts if _spawn_team(p) != player_flag]
    else:
        player_pts, targets = [], pts
    if len(targets) < cols * rows:
        return set()  # too few spawns to meaningfully reweight
    col_ax, row_ax = _region_grid(targets)

    by_region: dict[str, list[SpawnPoint]] = {}
    for p in targets:
        by_region.setdefault(_region_of(p, col_ax, row_ax, cols, rows), []).append(p)

    rng = np.random.default_rng(seed)
    total = len(targets)
    # Only regions that actually contain candidate spawns can receive mass.
    valid = {k: w for k, w in weights.items() if k in by_region}
    if not valid:
        return set()
    wsum = sum(valid.values())
    alloc = {k: max(1, round(w / wsum * total)) for k, w in valid.items()}

    new_blocks: list[list[str]] = []
    for key, count in alloc.items():
        pool = by_region[key]
        idx = rng.integers(0, len(pool), size=count)
        new_blocks.extend(pool[i].lines for i in idx)
    rng.shuffle(new_blocks)
    sce.replace_spawn_points([p.lines for p in player_pts] + new_blocks)
    return set(alloc)


def _expand_added_bots(sce: SceFile) -> list[str]:
    """Every [Bot Profile] name the scenario adds, with rotations expanded.

    AddedBots entries come in two flavours:

      "X.bot"  -> [Bot Profile] Name=X
      "X.rot"  -> [Bot Rotation Profile] Name=X -> ProfileNames=A;B;C
                  -> [Bot Profile] Name=A, =B, =C

    Only `.bot` was ever stripped, so a `.rot` entry matched no Bot Profile,
    resolved to no character, and the generator wrote NOTHING but the Name and
    Description — while that Description asserted a full plan. Measured on a
    copy of Reactive Flick.sce: 2 of 1493 lines changed, and the file still
    claimed `scale=1.18 movement=0.50 focus=r1c4 speed=86`.

    4 of the 33 base scenarios here are in that state: Fisher Simulator,
    KovaaKs Sandbox Intro Scenario, Reactive Flick, lgc3 Reborn Varied Easy
    Meso. The rotation resolves entirely inside the .sce — no second file
    format, no external lookup.

    ProfileNames may repeat (KovaaKs Sandbox lists `target` three times to
    weight it); the return is deduplicated because every downstream use is a
    section rewrite, which is idempotent per section and must not be applied
    N times to the same one.
    """
    raw = sce.get_header("AddedBots") or ""
    names: set[str] = set()
    for entry in (e.strip() for e in raw.split(";")):
        if not entry:
            continue
        if entry.lower().endswith(".rot"):
            rot = entry[: -len(".rot")]
            listed = sce.get_in_section("Bot Rotation Profile", rot, "ProfileNames")
            for bot in (b.strip() for b in (listed or "").split(";")):
                if bot:
                    names.add(bot)
            continue
        names.add(re.sub(r"\.bot$", "", entry))
    return sorted(names)


def _target_profiles(sce: SceFile) -> tuple[list[str], list[str]]:
    """(bot names, character profile names) for the AddedBots targets.

    Real scenarios link AddedBots "X.bot" -> [Bot Profile] Name=X ->
    CharacterProfile=<char> -> [Character Profile] Name=<char>, and "X.rot"
    through a [Bot Rotation Profile] first (see `_expand_added_bots`). Files
    whose Bot Profile lacks the CharacterProfile key (older/simple scenarios)
    fall back to using the bot name as the character name directly."""
    bot_names = _expand_added_bots(sce)
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
    # CAN KOVAAK'S DRIVE THESE BOTS? Not "do they move" — the narrower
    # question of whether a MaxSpeed and a dodge timer are what moves them.
    # An Acceleration test alone was wrong in BOTH directions on the real
    # corpus: `1wall 2targets small - valorant` authors Acceleration=16000
    # against MaxSpeed=0 and stands still, while Pressure Aiming's balloons
    # author both as 0 and cross the room on a movement ability. See
    # scenario/capability.py.
    kinds = scenario_motion(sce, char_names)
    can_move = drivable_motion(kinds)

    for char in char_names:
        span = sce.find_section("Character Profile", char)
        assert span is not None
        for i in range(span[0], span[1]):
            key = sce.lines[i].partition("=")[0]
            if key in _SIZE_KEYS:
                sce.lines[i] = _scale_line(sce.lines[i], plan.target_scale)
        if can_move and plan.target_max_speed > 0:
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
    # Dodge params modify HOW something moves, so on a scenario that cannot
    # move they are numbers written into a file that no longer describes
    # anything. Left unwritten rather than written-and-ignored.
    #
    # RELATIVE TO WHAT THE AUTHOR WROTE, per profile. The absolute form had a
    # ceiling that could sit below an author's floor — JumpFrequency capped at
    # 0.35 while 13 of 54 authored values in the corpus exceed it, so maximum
    # difficulty made those scenarios jump LESS than intended — and it wrote
    # one value to every profile, erasing deliberate contrast like Surge Tags'
    # 0.0/0.9/0.8/0.02/0.02. Scaling each profile by a common factor keeps the
    # rank order the author chose.
    strafe, _fb = strafe_channel(sce, bot_names, char_names)
    jump = jump_channel(sce, bot_names, char_names)
    if can_move:
        for dodge in sorted(dodge_names):
            authored: dict[str, float] = {}
            for key in _DODGE_KEYS:
                raw = sce.get_in_section("Dodge Profile", dodge, key)
                if raw is None:
                    continue          # never invent a key the author omitted
                try:
                    authored[key] = float(raw)
                except ValueError:
                    continue
            writes = relative_dodge_writes(
                authored, plan.movement, plan.dodge_bias,
                span=settings.dodge_relative_span,
                rng=np.random.default_rng(plan.seed or None))
            for key, val in writes.items():
                # A jump frequency is inert without something to jump WITH:
                # kovadapt wrote one onto characters with JumpVelocity=0,
                # where no frequency produces a jump.
                # Suppress on a definite NO only. UNKNOWN means the keys that
                # decide are missing — an older file, not a claim — and it has
                # to answer the same way `drivable_motion` does, or two gates
                # reach opposite conclusions about one scenario.
                if key == "JumpFrequency" and jump == "NO":
                    continue
                if key in _STRAFE_KEYS and strafe == "NO":
                    continue
                sce.set_in_section("Dodge Profile", dodge, key, val)

    # --- spawn region reweighting ------------------------------------------
    # The focus region only exists in the emitted scenario if the layout
    # actually has spawn points there — we resample originals and never invent
    # coordinates. `focus_applied` carries that answer out to the caller,
    # because crediting the bandit arm for a focus the .sce could not express
    # attributes an effect that never happened.
    used = resample_spawns(sce, plan.spawn_weights, settings.region_cols,
                           settings.region_rows, plan.seed)
    # Membership is the whole test, including when `used` is empty: an empty
    # set means the layout was left alone entirely (fewer target spawns than
    # grid cells, or no weighted region holding a candidate), and an untouched
    # layout expresses no focus at all. Reading empty as "applied" would have
    # credited the arm on exactly the scenarios where the region bandit has no
    # causal channel.
    plan.focus_applied = plan.focus_region in used
    # Same contract as focus_applied: the GENERATOR is the only thing that has
    # read the file, so it is the only thing that can say whether the plan's
    # motion terms survived contact with it.
    plan.motion_applied = can_move

    if out_path is None:
        out_path = base_sce.parent / f"{base_name}{ADAPTIVE_SUFFIX}.sce"
    out_path = Path(out_path)
    sce.write(out_path)
    return out_path
