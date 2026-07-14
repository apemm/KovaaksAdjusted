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


def _region_of(p: SpawnPoint, xs: np.ndarray, zs: np.ndarray,
               cols: int, rows: int) -> str:
    """Grid cell of a spawn point. Columns bin X (left->right), rows bin Z
    (bottom->top), using min/max extents of the player-facing spawn cloud."""
    c = min(int((p.x - xs[0]) / max(xs[1] - xs[0], 1e-9) * cols), cols - 1)
    r = min(int((p.z - zs[0]) / max(zs[1] - zs[0], 1e-9) * rows), rows - 1)
    return f"r{r}c{c}"


def _scale_line(line: str, factor: float) -> str:
    key, _, val = line.partition("=")
    try:
        return f"{key}={round(float(val) * factor, 6)}"
    except ValueError:
        return line


def resample_spawns(sce: SceFile, weights: dict[str, float],
                    cols: int, rows: int, seed: int) -> None:
    """Rebuild the PlayerSpawn list: same total count, region densities
    proportional to `weights`. Sampling with replacement within regions keeps
    every emitted spawn a physically valid original location."""
    pts = sce.spawn_points()
    if len(pts) < cols * rows:
        return  # too few spawns to meaningfully reweight
    xs = np.array([p.x for p in pts])
    zs = np.array([p.z for p in pts])
    x_ext = (xs.min(), xs.max())
    z_ext = (zs.min(), zs.max())

    by_region: dict[str, list[SpawnPoint]] = {}
    for p in pts:
        by_region.setdefault(_region_of(p, x_ext, z_ext, cols, rows), []).append(p)

    rng = np.random.default_rng(seed)
    total = len(pts)
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
    sce.replace_spawn_points(new_blocks)


def _target_character_names(sce: SceFile) -> list[str]:
    """Character profiles referenced by AddedBots (the actual targets)."""
    bots = sce.get_header("AddedBots") or ""
    names = {re.sub(r"\.bot$", "", b) for b in bots.split(";") if b}
    return [n for n in names if sce.find_section("Character Profile", n)]


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

    # --- target size + speed ---------------------------------------------
    for char in _target_character_names(sce):
        span = sce.find_section("Character Profile", char)
        assert span is not None
        for i in range(span[0], span[1]):
            key = sce.lines[i].partition("=")[0]
            if key in _SIZE_KEYS:
                sce.lines[i] = _scale_line(sce.lines[i], plan.target_scale)
        if plan.target_max_speed > 0:
            sce.set_in_section("Character Profile", char, "MaxSpeed", plan.target_max_speed)

    # --- micro-movement: patch every dodge profile the target bots use ----
    bot_names = _target_character_names(sce)
    dodge_names: set[str] = set()
    for span in [sce.find_section("Bot Profile", n) for n in bot_names]:
        if span is None:
            continue
        for ln in sce.lines[span[0]: span[1]]:
            if ln.startswith("DodgeProfileNames="):
                dodge_names.update(d for d in ln.partition("=")[2].split(";") if d)
    for dodge in dodge_names:
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
