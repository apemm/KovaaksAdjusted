"""Per-task adaptation ledger: what kovadapt has actually changed about ONE
scenario, and the runs that made it decide to.

Three questions, in reading order, and none of them answered without its
evidence:

    WHAT MOVED     the five criteria the engine actually turns — target size,
                   target speed, spawn mass on the focus region, dodge-
                   direction skew, movement/pace — each as BASELINE -> NOW on
                   the knob's own real bounds, read from the archetype's
                   EFFECTIVE settings (`Settings.for_archetype`) because that
                   is the only view the engine itself ever uses. Every rail
                   contains the values it plots and every baseline is either a
                   fresh profile's own reachable default or a number read out
                   of the base `.sce`; a criterion the files cannot support
                   goes unmeasured and prints a dash. NOW is measured back OUT
                   OF THE WRITTEN VARIANT for the two criteria a `.sce` can be
                   read for — target speed and spawn mass — because the emitted
                   plan is what the game loads and `plan(fatigue=...)` eases the
                   emitted values while persisted state stays un-eased. The
                   model's number and the file's are two different claims, this
                   page makes the file's, and it says which one it is.
    WHY            per criterion, the runs behind it: `profile.history` for
                   the controller inputs, `RegionPosterior.n` per arm, and the
                   per-run `region_deficits`/`bias` in the RunReport JSONs
                   under `<profile_dir>/reports/`. A criterion with no
                   evidence says so in those words and is drawn unlit — a
                   cold-start default IS a real value in the emitted file, but
                   it is not a learned change, and this page must never let
                   the two read alike.
    IN THE FILE    the actual numbers, base `.sce` against
                   `<name> [Adaptive].sce`. The generator always applies the
                   plan to the BASE file and never to the previous variant, so
                   reading the two side by side is a true before/after rather
                   than a reconstruction: authored MaxSpeed against written
                   MaxSpeed, the dodge timings, and the target spawn count per
                   region — binned with the generator's OWN grid functions,
                   because the r{row}c{col} keys have to stay byte-identical
                   or the comparison is fiction.

Nothing here mutates anything. `plan()` moves persisted profile state, so it
is never called; the region bandit is only ever asked for weights, and only
on a deep copy, because `profile.region()` creates arms as a side effect.

House style: data and structure are character art (QPainter glyph grids,
theme colours read at paint time), controls are real Qt widgets. Motion comes
from gui/motion.py alone — one glyph-rate clock for a staggered reveal, no
ambient loop, and the clock never runs while the page is hidden.
"""

from __future__ import annotations

import copy
import json
import math
import re
import time
from dataclasses import dataclass, field, replace
from html import escape as _html_escape
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFontMetricsF, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from ..adapt.bandit import ThompsonRegionBandit
from ..adapt.stochastic import movement_speed, speed_multiplier
from ..analysis.insights import _region_words
from ..analysis.report import input_degraded
from ..config import ADAPTIVE_SUFFIX, Settings
from ..profile.player import PlayerProfile, _slug
# The generator's own grid + target resolution. Importing these rather than
# re-deriving them is the whole point: region keys must be byte-identical
# across bandit.py, generator.py and analysis/movement.py, and a fourth
# private copy of the binning would make this page's before/after fiction the
# day any of them changed.
from ..scenario.generator import (
    _player_team_flag,
    _region_grid,
    _region_of,
    _spawn_team,
    _target_profiles,
)
from ..scenario.sce import SceFile
from . import motion, theme
from .onboarding import HintBar

# Glyph ramps, same convention as gui/viz.py: dense at the anchor, light at
# the tip, so a run of characters reads as a magnitude rather than a bar.
#
# The run ramp's DENSEST glyph must never be the marker glyph. It was "@" —
# the same character the caption defines as "where the model stands now" — so
# a long run rendered "|@@@@#####*****+++++=====-----:::::@" and four of the
# five @s in that row were ramp rather than value. Three of the five rows were
# unreadable for exactly that reason.
_RUN_RAMP = "#*+=-:."
_DENSITY_RAMP = " .:-=+*#%@"

# The rail's three marks, named here because the caption promises exactly
# these and nothing else. A row whose value sits ON its baseline used to draw
# the marker alone (the painter tested `col == marker` before `col == anchor`),
# so a cold profile showed a lone "@" and no "|" at all.
_ANCHOR_GLYPH = "|"         # the baseline the move is measured from
_MARKER_GLYPH = "@"         # where the model stands now
_ON_BASELINE_GLYPH = "0"    # both in one cell: this criterion has not moved

# Rail geometry. Below MIN_RAIL_CELLS the rail cannot tell a move from a
# no-move — at 8 cells a real -20% size move rendered "@|" and a -6% speed
# move rendered as a lone "@" — so it is dropped entirely and the row falls
# back to its numbers, which stay true at any width.
MIN_RAIL_CELLS = 24
MAX_RAIL_CELLS = 55
# A move `Knob.moved` calls real must RENDER as a move. MOVE_EPS_FRAC is 1% of
# a knob's range, which is a fifth of a cell even at full width, so anchor and
# marker are held at least this far apart whenever the knob moved. The rail is
# a magnitude run, not a measuring scale, and the exact numbers sit beside it —
# the caption says both.
MIN_RUN_CELLS = 2

# A move smaller than this fraction of the knob's own range is not a move.
# Expressed against the range rather than absolutely because the five knobs
# live on wildly different scales (0.65-1.35 next to 0-170).
MOVE_EPS_FRAC = 0.01

# Directional-bias evidence gate, mirroring watcher.py's own: a run only
# reaches observe_bias with >= 8 flicks AND >= 3 per side, because
# analysis.directional_bias returns a flat 0.0 below that and a 0.0 that means
# "not measured" must never be counted as an observation.
_BIAS_MIN_FLICKS = 8
_BIAS_MIN_PER_SIDE = 3

# Newest-first cap on report JSONs read for evidence. Reports accumulate one
# per run forever; the counts below are evidence, not analysis, and a training
# week is well inside this.
_MAX_REPORTS = 240

# A spawn count that moved by one point is allocation rounding, not emphasis:
# resample_spawns allocates `max(1, round(weight * total))` per region.
_SPAWN_NOISE = 1

# Density anchor for the spawn map: three times the uniform share fills the
# ramp. An ABSOLUTE anchor, not min-max — min-max normalisation is what once
# turned 25 regions of noise into a screaming "weakest zone" in viz.py.
_SPAWN_FULL_MULT = 3.0


# --------------------------------------------------------------- data model
@dataclass(frozen=True)
class Knob:
    """One criterion the engine moves: BASELINE -> NOW, plus its evidence.

    `evidence` is a required constructor field and validated non-empty, the
    same structural trick gui/dashboard.py:Hero uses for its because-clause.
    A knob that cannot say what moved it cannot be built at all.

    `measured` is NOT "did it move" — it is "does any run back this value".
    False means the number is a cold-start default: real in the emitted .sce,
    but never learned, and the ladder draws it unlit so the two can never read
    as the same thing.

    THE RAIL MUST CONTAIN THE VALUES IT PLOTS, and that is enforced in the
    constructor rather than trusted: a tracking profile was built with
    lo=min_movement (0.35) and baseline=the dataclass default (0.15), and
    because the painter clamps a fraction into [0, 1] the row printed 0.15 in
    the baseline column while painting "|" on the 0.35 tick. Silently
    disagreeing with itself is the one thing this page may not do.
    """

    key: str
    name: str
    lo: float
    hi: float
    baseline: float
    now: float
    fmt: str = "{:.2f}"
    evidence: str = ""
    delta_text: str = ""
    unit: str = ""
    tip: str = ""
    measured: bool = True
    note: str = ""
    # What the delta column says INSTEAD of a delta when nothing backs the
    # value. "no evidence" is the honest default, but a bandit exploration
    # draw and a cold-start default are different absences and reading them
    # as one would flatten the distinction this page exists to make.
    flag: str = "no evidence"
    # False = there is no number to plot at all (the base file cannot say which
    # speed path applies, or the scenario mixes both). The row then prints a
    # dash and its flag instead of a value on a rail, because putting the
    # rail's own floor there would read as a measurement.
    rail: bool = True
    # True = the value is in the MODEL but no [Adaptive] file carries it yet.
    # A planned change and a written one must never read alike.
    pending: bool = False

    def __post_init__(self) -> None:
        if not self.evidence:
            raise ValueError(f"knob {self.key!r} has no evidence — "
                             "every criterion must name the runs behind it")
        if self.hi < self.lo:
            raise ValueError(f"knob {self.key!r} has an inverted rail "
                             f"({self.lo} .. {self.hi})")
        tol = max(abs(self.hi - self.lo), 1.0) * 1e-9
        for label, value in (("baseline", self.baseline), ("now", self.now)):
            if not (self.lo - tol <= value <= self.hi + tol):
                raise ValueError(
                    f"knob {self.key!r} plots {label}={value!r} outside its own "
                    f"rail [{self.lo}, {self.hi}] — the rail would paint the "
                    "clamped position next to the unclamped number")

    def text(self, value: float) -> str:
        return self.fmt.format(value)

    @property
    def eps(self) -> float:
        return max(abs(self.hi - self.lo) * MOVE_EPS_FRAC, 1e-9)

    @property
    def moved(self) -> bool:
        return abs(self.now - self.baseline) > self.eps

    @property
    def at_bound(self) -> bool:
        return (abs(self.now - self.lo) <= self.eps
                or abs(self.now - self.hi) <= self.eps)


@dataclass(frozen=True)
class ProfileEntry:
    """One trainable task that has a profile on disk."""

    base: str        # scenario name with any [Adaptive] suffix stripped
    stored: str      # the profile's own `scenario` field, verbatim
    runs: int
    last: str        # ISO ts of the last run ("" = never)


@dataclass(frozen=True)
class ReportEvidence:
    """Aggregated per-run telemetry evidence from the RunReport JSONs.

    Reports for ONE task land in two slug directories — runs of the base
    scenario and runs of the [Adaptive] variant both feed the same profile —
    so both are scanned and counted.
    """

    files: int = 0
    dirs: tuple[str, ...] = ()
    region_n: dict[str, int] = field(default_factory=dict)
    region_mean: dict[str, float] = field(default_factory=dict)
    bias_runs: int = 0
    bias_mean: float = 0.0
    degraded: int = 0
    latest: str = ""


@dataclass(frozen=True)
class SpawnMap:
    """Target spawn counts per region, base vs variant, on ONE grid."""

    cols: int
    rows: int
    base: dict[str, int] = field(default_factory=dict)
    adaptive: dict[str, int] = field(default_factory=dict)
    planned: dict[str, float] = field(default_factory=dict)
    focus: str | None = None
    reason: str = ""

    @property
    def total_base(self) -> int:
        return sum(self.base.values())

    @property
    def total_adaptive(self) -> int:
        return sum(self.adaptive.values())

    @property
    def cells(self) -> int:
        return max(self.cols * self.rows, 1)

    @property
    def uniform(self) -> float:
        return 1.0 / self.cells

    @property
    def untouched(self) -> bool:
        """The generator provably left this layout alone (`reason` is set only
        when resample_spawns bails out), so NOTHING here is an adaptation.

        The grid has to agree with the spawn knob about that. It did not: five
        target spawns against a 5x5 grid gave each occupied cell five times an
        even share, the absolute anchor saturated at three times, and the panel
        painted solid "@" blocks — maximum emphasis — one line above the knob
        saying no spawn focus can be applied to this scenario at all.
        """
        return bool(self.reason)

    def share(self, key: str) -> float:
        """Share of target spawns in `key` as the variant actually stands —
        measured from the file when there is one, else the planned weight,
        else the base layout's own share."""
        if self.adaptive:
            return self.adaptive.get(key, 0) / max(self.total_adaptive, 1)
        if self.planned:
            return float(self.planned.get(key, 0.0))
        return self.base.get(key, 0) / max(self.total_base, 1)

    def base_share(self, key: str) -> float:
        return self.base.get(key, 0) / max(self.total_base, 1)


@dataclass(frozen=True)
class LedgerRow:
    label: str
    base: str
    adaptive: str
    delta: str = ""
    tone: str = "fg"     # fg | accent | warn


@dataclass(frozen=True)
class SceFacts:
    """Everything read out of the two .sce files, or why it could not be."""

    base_path: Path | None = None
    variant_path: Path | None = None
    have_base: bool = False
    have_variant: bool = False
    # The variant EXISTS on disk, whether or not it could be parsed and whether
    # or not the base could. `have_variant` is the stronger claim — parsed, and
    # comparable against a base — and only THIS field may back a sentence about
    # what is or is not on disk. read_sce_facts returns early when the base is
    # missing without ever looking at the variant, so the page said "no
    # [Adaptive] .sce on disk carries this value", five times over, about a file
    # sitting right next to the one it could not find.
    variant_on_disk: bool = False
    error: str = ""
    chars: tuple[str, ...] = ()
    # Target characters whose base MaxSpeed could actually be READ. A character
    # with no MaxSpeed line at all is not a 0 — set_in_section only rewrites a
    # key that already exists, so the generator writes no speed for it — and
    # recording it as 0.0 would put it on the static-wall ramp in this page's
    # arithmetic while the file never changes.
    authored: dict[str, float] = field(default_factory=dict)
    no_speed_key: tuple[str, ...] = ()
    # The MaxSpeed those same characters carry in the VARIANT. The ladder plots
    # its speed reading from these rather than from the model, because the
    # emitted plan is what the game loads: `plan(fatigue=...)` eases the emitted
    # values while the persisted profile stays un-eased by contract, so a
    # variant written while tired legitimately holds a different number.
    written_speeds: dict[str, float] = field(default_factory=dict)
    rows: tuple[LedgerRow, ...] = ()
    extra_sections: int = 0
    spawns: SpawnMap | None = None
    description: str = ""
    written: str = ""

    @property
    def speed_paths(self) -> dict[str, str]:
        """Which speed path applies PER TARGET CHARACTER.

        generate_adaptive_variant decides this per character, inside the loop
        over char_names: base MaxSpeed > 0 is modulated by target_speed_mult,
        base MaxSpeed == 0 gets the absolute target_max_speed ramp. Any summary
        of this that is not per character is a summary of something else.
        """
        return {c: ("multiplier" if v > 0 else "ramp")
                for c, v in self.authored.items()}

    @property
    def speed_path(self) -> str:
        """"multiplier" (every target authors a speed of its own), "ramp"
        (every target is a base-MaxSpeed-0 static wall), "mixed" (both, in one
        scenario), or "unknown".

        The first two are mutually exclusive by contract and confusing them
        destroys a scenario: writing the absolute 0-170 static-wall ramp onto a
        1300-speed strafe bot collapses its difficulty. This returned
        "multiplier" as soon as ANY character authored a speed, which is how a
        mixed scenario got a page-wide claim that the ramp is "never written
        here" printed directly above a ledger row showing the ramp written.
        """
        if not self.have_base or not self.chars:
            return "unknown"
        if self.no_speed_key or len(self.authored) != len(self.chars):
            return "unknown"
        paths = set(self.speed_paths.values())
        return paths.pop() if len(paths) == 1 else "mixed"


# --------------------------------------------------------------- disk layer
def scan_profiles(profile_dir: Path | str) -> list[ProfileEntry]:
    """Every task with a profile on disk, most recently played first.

    Read straight out of each JSON's own `scenario` field rather than by
    un-slugging the filename: the slug replaces runs of characters and is
    lossy, so "1w6ts [Adaptive]" and "1w6ts_Adaptive_" cannot be mapped back.
    This also means the picker works with no KovaaK's install present.
    """
    root = Path(profile_dir) / "profiles"
    if not root.is_dir():
        return []
    found: dict[str, ProfileEntry] = {}
    for path in sorted(root.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue                      # a corrupt profile hides itself, not the page
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("scenario") or "").strip()
        if not name:
            continue
        base = name[:-len(ADAPTIVE_SUFFIX)] if name.endswith(ADAPTIVE_SUFFIX) else name
        entry = ProfileEntry(base=base, stored=name,
                             runs=int(raw.get("run_count") or 0),
                             last=str(raw.get("last_run_ts") or ""))
        # Both "<base>" and "<base> [Adaptive]" can exist on disk for one task
        # (profiles are keyed on the suffixed name, but older files and manual
        # runs are not). They are the same task: keep the one with the runs.
        prev = found.get(base)
        if prev is None or entry.runs > prev.runs:
            found[base] = entry
    out = sorted(found.values(), key=lambda e: e.base.lower())
    out.sort(key=lambda e: e.last, reverse=True)      # stable: ties stay by name
    return out


def read_report_evidence(profile_dir: Path | str, base: str) -> ReportEvidence:
    """Per-run telemetry evidence for one task from its RunReport JSONs."""
    root = Path(profile_dir) / "reports"
    names = [base, base + ADAPTIVE_SUFFIX]
    paths: list[Path] = []
    dirs: list[str] = []
    for name in names:
        d = root / _slug(name)
        if not d.is_dir():
            continue
        files = sorted(d.glob("*.json"))
        if files:
            dirs.append(d.name)
        paths.extend(files)
    if not paths:
        return ReportEvidence()
    paths.sort(key=lambda p: p.name)
    paths = paths[-_MAX_REPORTS:]

    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    bias_vals: list[float] = []
    degraded = 0
    read = 0
    for path in paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        read += 1
        for key, z in (raw.get("region_deficits") or {}).items():
            try:
                val = float(z)
            except (TypeError, ValueError):
                continue
            totals[key] = totals.get(key, 0.0) + val
            counts[key] = counts.get(key, 0) + 1
        bias = raw.get("bias") or {}
        left = int((bias.get("left") or {}).get("n", 0) or 0)
        right = int((bias.get("right") or {}).get("n", 0) or 0)
        if (int(raw.get("n_flicks") or 0) >= _BIAS_MIN_FLICKS
                and left >= _BIAS_MIN_PER_SIDE and right >= _BIAS_MIN_PER_SIDE):
            bias_vals.append(float(bias.get("bias_score") or 0.0))
            # input_degraded reads an ATTRIBUTE, so a dict will silently pass;
            # the shim keeps the one shared definition (analysis/report.py) as
            # the only gate this page ever applies.
            if input_degraded(SimpleNamespace(input_health=raw.get("input_health"))):
                degraded += 1
    return ReportEvidence(
        files=read,
        dirs=tuple(dirs),
        region_n=counts,
        region_mean={k: totals[k] / counts[k] for k in counts},
        bias_runs=len(bias_vals),
        bias_mean=(sum(bias_vals) / len(bias_vals)) if bias_vals else 0.0,
        degraded=degraded,
        latest=paths[-1].stem if paths else "",
    )


def _esc(text: str) -> str:
    """HTML-escape a value bound for a RichText label.

    quote=False deliberately: none of this lands inside an attribute, and the
    default turns "the game's Scenarios folder" into "game&#x27;s" in the one
    sentence a reader most needs. Scenario names and character names do contain
    "&", so nothing interpolated into these labels may go through unescaped.
    """
    return _html_escape(text, quote=False)


def _num(text: str | None) -> float | None:
    try:
        return float((text or "").strip())
    except (TypeError, ValueError):
        return None


def _fmt_num(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e6:
        return f"{int(round(value))}"
    return f"{value:.4g}"


def _delta_text(base: float | None, now: float | None) -> tuple[str, str]:
    """(text, tone) for a base -> variant pair of raw .sce numbers."""
    if base is None or now is None:
        return "", "fg"
    if abs(now - base) < 1e-9:
        return "unchanged", "fg"
    if abs(base) > 1e-9:
        return f"x{now / base:.3g}", "accent"
    return f"{now - base:+.4g}", "accent"


def _spawn_counts(sce: SceFile, col_ax, row_ax, cols: int, rows: int) -> dict[str, int]:
    """Target PlayerSpawns per region, player-side spawns excluded exactly as
    resample_spawns excludes them."""
    flag = _player_team_flag(sce)
    out: dict[str, int] = {}
    for point in sce.spawn_points():
        if flag is not None and _spawn_team(point) == flag:
            continue
        key = _region_of(point, col_ax, row_ax, cols, rows)
        out[key] = out.get(key, 0) + 1
    return out


def _target_spawns(sce: SceFile) -> list:
    flag = _player_team_flag(sce)
    pts = sce.spawn_points()
    if flag is None:
        return pts
    return [p for p in pts if _spawn_team(p) != flag]


# The keys the generator actually writes, in the order it writes them.
_CHAR_KEYS = ("MainBBRadius", "MainBBHeight", "MaxSpeed")
_DODGE_KEYS = ("MinLRTimeChange", "MaxLRTimeChange", "LeftStrafeTimeMult",
               "RightStrafeTimeMult", "JumpFrequency")
_MAX_CHARS = 2          # sections shown in the ledger before it says "+N more"
_MAX_DODGES = 2


def _dodge_names(sce: SceFile, bots: list[str]) -> list[str]:
    """Dodge profiles the target bots use — generator.py's own lookup path
    ([Bot Profile] sections are keyed by BOT name, not character)."""
    names: set[str] = set()
    for bot in bots:
        span = sce.find_section("Bot Profile", bot)
        if span is None:
            continue
        for line in sce.lines[span[0]:span[1]]:
            if line.startswith("DodgeProfileNames="):
                names.update(d for d in line.partition("=")[2].split(";") if d)
    return sorted(names)


def read_sce_facts(settings: Settings, base: str,
                   focus: str | None = None) -> SceFacts:
    """Read the base .sce and its [Adaptive] variant side by side.

    A true before/after, not a reconstruction: the generator always applies
    the plan to the BASE file (multipliers are absolute and edits never
    compound), so the pair on disk IS the change. A missing variant is a
    normal state — never generated yet — and reported as such.
    """
    base_path = settings.scenarios_dir / f"{base}.sce"
    var_path = settings.scenarios_dir / f"{base}{ADAPTIVE_SUFFIX}.sce"
    on_disk = var_path.is_file()
    if not base_path.is_file():
        # Short, because this string lands in four panels; the full path goes
        # in the provenance label's tooltip. `variant_on_disk` is still recorded:
        # without it the page claimed the variant had never been written, having
        # never looked.
        return SceFacts(base_path=base_path, variant_path=var_path,
                        variant_on_disk=on_disk,
                        error=f"{base}.sce is not in the game's Scenarios folder")
    try:
        src = SceFile.read(base_path)
    except (OSError, ValueError) as exc:
        return SceFacts(base_path=base_path, variant_path=var_path,
                        variant_on_disk=on_disk,
                        error=f"could not read {base}.sce: {exc.__class__.__name__}")
    var = None
    if on_disk:
        try:
            var = SceFile.read(var_path)
        except (OSError, ValueError):
            # A variant that will not open is NOT the base's failure. One try
            # block around both reads reported "could not read <base>.sce" for a
            # file that had already parsed fine, and threw away every fact it
            # gave up. `variant_on_disk` then carries the real state: on disk,
            # unreadable — which is neither "written" nor "never written".
            var = None
        else:
            # SceFile is a deliberately TOLERANT verbatim line editor: it does
            # not raise on arbitrary bytes, it just yields a file with no
            # headers and no spawns. So "read() returned" is not evidence of a
            # usable variant — a corrupt file would otherwise be reported as a
            # written one and every criterion would read its values out of
            # nothing. The generator always writes Name=, so its absence is
            # the cheapest honest test that this is one of our variants.
            if var.get_header("Name") is None:
                var = None

    bots, chars = _target_profiles(src)
    # A MISSING MaxSpeed line and an authored MaxSpeed=0 are different facts:
    # the generator's set_in_section only rewrites a key that is already there,
    # so a character with no MaxSpeed line never receives a speed at all. Both
    # collapsed to 0.0 here, which put such a scenario on the static-wall ramp
    # in this page's arithmetic while its file never moved.
    authored: dict[str, float] = {}
    no_key: list[str] = []
    written_speeds: dict[str, float] = {}
    for char in chars:
        raw = src.get_in_section("Character Profile", char, "MaxSpeed")
        if raw is None:
            no_key.append(char)
            continue
        authored[char] = _num(raw) or 0.0
        # Only for characters the BASE authors a readable speed for: without the
        # author's own number there is nothing to read the written one against.
        wrote = _num(var.get_in_section("Character Profile", char, "MaxSpeed")) \
            if var else None
        if wrote is not None:
            written_speeds[char] = wrote

    rows: list[LedgerRow] = []
    for char in chars[:_MAX_CHARS]:
        for key in _CHAR_KEYS:
            b = _num(src.get_in_section("Character Profile", char, key))
            if b is None:
                continue
            a = _num(var.get_in_section("Character Profile", char, key)) if var else None
            delta, tone = _delta_text(b, a)
            rows.append(LedgerRow(f"{char} - {key}", _fmt_num(b), _fmt_num(a),
                                  delta, tone))
    dodges = _dodge_names(src, bots)
    for dodge in dodges[:_MAX_DODGES]:
        for key in _DODGE_KEYS:
            b = _num(src.get_in_section("Dodge Profile", dodge, key))
            if b is None:
                continue
            a = _num(var.get_in_section("Dodge Profile", dodge, key)) if var else None
            delta, tone = _delta_text(b, a)
            rows.append(LedgerRow(f"{dodge} - {key}", _fmt_num(b), _fmt_num(a),
                                  delta, tone))
    extra = max(len(chars) - _MAX_CHARS, 0) + max(len(dodges) - _MAX_DODGES, 0)

    spawns = _read_spawn_map(src, var, settings, focus)
    written = ""
    if var_path.is_file():
        try:
            written = time.strftime("%Y-%m-%d %H:%M",
                                    time.localtime(var_path.stat().st_mtime))
        except OSError:
            written = ""
    return SceFacts(
        base_path=base_path, variant_path=var_path, have_base=True,
        have_variant=var is not None, variant_on_disk=on_disk,
        chars=tuple(chars), authored=authored,
        no_speed_key=tuple(no_key), written_speeds=written_speeds,
        rows=tuple(rows), extra_sections=extra,
        spawns=spawns,
        description=(var.get_header("Description") or "") if var else "",
        written=written,
    )


def _read_spawn_map(src: SceFile, var: SceFile | None, settings: Settings,
                    focus: str | None) -> SpawnMap:
    """Bin BOTH files' target spawns on the BASE file's grid.

    One grid for both, deliberately: _region_grid derives its extents from the
    point cloud it is handed, and the variant's cloud is a resample of the
    base's, so letting each file pick its own extents would compare two
    different griddings and invent a difference that is not there.
    """
    cols, rows = settings.region_cols, settings.region_rows
    targets = _target_spawns(src)
    if not targets:
        return SpawnMap(cols=cols, rows=rows, focus=focus,
                        reason="this layout has no target PlayerSpawn entities")
    col_ax, row_ax = _region_grid(targets)
    base_counts = _spawn_counts(src, col_ax, row_ax, cols, rows)
    reason = ""
    if len(targets) < cols * rows:
        # resample_spawns bails out entirely below one candidate per cell.
        reason = (f"{len(targets)} target spawns for a {cols}x{rows} grid — the "
                  f"generator leaves the layout untouched below {cols * rows}, "
                  "so no spawn focus can be applied to this scenario")
    adaptive = ({} if var is None or reason
                else _spawn_counts(var, col_ax, row_ax, cols, rows))
    return SpawnMap(cols=cols, rows=rows, base=base_counts, adaptive=adaptive,
                    focus=focus, reason=reason)


def planned_weights(profile: PlayerProfile, settings: Settings) -> dict[str, float]:
    """The spawn weights the engine would emit for the CURRENT focus.

    On a deep copy: `profile.region()` creates missing arms as a side effect,
    and this page must not touch model state. choose_focus() is never called —
    it is a random draw, and the focus that matters is the one already
    persisted in `last_focus`.
    """
    if not profile.last_focus:
        return {}
    s = settings.for_archetype(profile.archetype)
    clone = copy.deepcopy(profile)
    bandit = ThompsonRegionBandit(clone, s.region_cols, s.region_rows,
                                  prior_var=s.bandit_prior_var,
                                  obs_noise=s.bandit_obs_noise)
    return bandit.spawn_weights(profile.last_focus, s.focus_weight)


# ------------------------------------------------------------ knob builders
def _fresh_defaults() -> PlayerProfile:
    """A never-observed profile — the honest source of every BASELINE below.
    Hardcoding 1.0 / 0.15 / 0.0 here would drift from the model in silence."""
    return PlayerProfile(scenario="")


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


def _reachable(fresh_value: float, lo: float, hi: float) -> tuple[float, str]:
    """(baseline, note) for a fresh-profile default against the ARCHETYPE's own
    clamps.

    `PlayerProfile`'s field defaults are archetype-blind, and the engine
    squashes into [lo, hi] on the very first plan — under the tracking override
    min_movement is 0.35, so a fresh tracking profile can never emit the 0.15
    dataclass default and printing it as the baseline claimed a starting point
    the controller cannot occupy.
    """
    value = _clamp(fresh_value, lo, hi)
    if abs(value - fresh_value) < 1e-9:
        return value, ""
    return value, (f"a fresh profile's {fresh_value:.2f} default is outside this "
                   f"archetype's [{lo:.2f}, {hi:.2f}] clamps, so the first value "
                   f"the engine can emit — and the baseline plotted here — is "
                   f"{value:.2f}")


def _widen(lo: float, hi: float, *values: float) -> tuple[float, float, str]:
    """A rail that CONTAINS every value it will plot.

    Only a persisted value from a differently-clamped past (a profile whose
    archetype overrides have since changed) can fall outside, and that is worth
    saying out loud rather than painting on the nearest tick.
    """
    out_lo, out_hi = min([lo, *values]), max([hi, *values])
    if abs(out_lo - lo) < 1e-9 and abs(out_hi - hi) < 1e-9:
        return lo, hi, ""
    return out_lo, out_hi, (
        f"the persisted value sits outside the controller's current "
        f"[{lo:.2f}, {hi:.2f}] clamps, so the rail is widened to hold it — the "
        "next plan will clip it back")


def _ratio_delta(baseline: float, now: float) -> str:
    if abs(now - baseline) < 1e-9:
        return ""
    if abs(baseline) < 1e-9:
        return f"{now:+.2f}"
    return f"{(now / baseline - 1.0) * 100.0:+.0f}%"


def _played_range(profile: PlayerProfile, key: str) -> str:
    vals = [float(h[key]) for h in profile.history if key in h]
    if not vals:
        return ""
    if len(vals) == 1:
        return f"{vals[0]:.2f}"
    return f"{vals[0]:.2f} to {vals[-1]:.2f}"


def _size_knob(profile: PlayerProfile, s: Settings, fresh: PlayerProfile) -> Knob:
    low, high = s.target_accuracy_low, s.target_accuracy_high
    accs = [float(h["accuracy"]) for h in profile.history if "accuracy" in h]
    above = sum(1 for a in accs if a > high)
    below = sum(1 for a in accs if a < low)
    inside = len(accs) - above - below
    # The Fitts sub-controller lives in the deadband's `elif`: it only ever
    # runs on a run whose accuracy was INSIDE the band. Appending its clause on
    # the observation count alone described a mechanism that had never fired —
    # this very evidence line read "0 sat inside it" and then claimed the extra
    # shrink step in the same sentence.
    stalled = (s.fitts_control_gain > 0 and profile.fitts_obs >= 5
               and profile.slow_fitts_ms > 0
               and profile.ewma_fitts_ms >= profile.slow_fitts_ms)
    if profile.run_count == 0:
        evidence = ("no completed runs — the deadband size controller has never "
                    "fired for this task, so targets are still exactly the size "
                    "the scenario's author gave them")
        measured = False
    else:
        evidence = (
            f"because {above + below} of {len(accs)} recorded runs sat outside the "
            f"{low:.0%}-{high:.0%} band ({above} above, {below} below) and the "
            f"deadband acted only on those; {inside} sat inside it and held size "
            "still")
        if stalled:
            reading = (f"ms-per-bit has also stalled ({profile.ewma_fitts_ms:.0f} "
                       f"fast vs {profile.slow_fitts_ms:.0f} slow across "
                       f"{profile.fitts_obs} telemetry runs)")
            evidence += (
                f"; {reading}, which added one extra shrink step on each of those "
                f"{inside} in-band runs" if inside else
                f"; {reading}, but the sub-controller only acts on a run INSIDE "
                "the band and none of your recorded runs was, so it has never "
                "fired here")
        played = _played_range(profile, "target_scale")
        if played:
            evidence += f"; the model stood at {played} across those runs"
        measured = True
    note = ""
    if abs(profile.target_scale - s.min_target_scale) < 1e-6:
        note = (f"at the {s.min_target_scale:.2f} floor — the controller cannot "
                "shrink targets any further")
    elif abs(profile.target_scale - s.max_target_scale) < 1e-6:
        note = (f"at the {s.max_target_scale:.2f} ceiling — the controller cannot "
                "grow targets any further")
    baseline, clamp_note = _reachable(fresh.target_scale, s.min_target_scale,
                                      s.max_target_scale)
    lo, hi, wide_note = _widen(s.min_target_scale, s.max_target_scale,
                               baseline, profile.target_scale)
    note = "; ".join(n for n in (note, clamp_note, wide_note) if n)
    return Knob(
        key="target_scale", name="target size", lo=lo, hi=hi, baseline=baseline,
        now=profile.target_scale, fmt="{:.2f}", unit="x",
        delta_text=_ratio_delta(baseline, profile.target_scale),
        evidence=evidence, measured=measured, note=note, flag="no runs yet",
        tip=("Deadband controller on the archetype's accuracy band: outside it, "
             "size moves multiplicatively against the EXCESS beyond the nearest "
             "edge only (falling below the floor grows targets 1.4x harder than "
             "sitting above it shrinks them). The emitted .sce multiplies this "
             f"by (1 + {s.size_speed_coupling:.2f} x movement) and clips to "
             f"[{s.min_target_scale:.2f}, {s.max_target_scale:.2f}], so the "
             "file's number can legitimately differ from the model's — the file "
             "ledger below carries what was actually written."))


class _WrittenSpeed(NamedTuple):
    """What the [Adaptive] file on disk actually carries for target speed."""

    value: float | None    # on the knob's own rail (a multiplier, or u/s)
    detail: str = ""       # the per-character read behind it
    why: str = ""          # why the file cannot answer, when value is None


# The per-character speed reads in one variant agree to within the file's own
# rounding (the generator writes round(base * mult, 1) from ONE multiplier), so
# a spread wider than these is not rounding — it is two different decisions, and
# collapsing them onto one rail would be the mixed-path lie in a new place.
_SPEED_AGREE_MULT = 0.02       # on the 0.65-1.35 multiplier rail
_SPEED_AGREE_RAMP = 0.2        # on the absolute 0-170 u/s rail


def _written_speed(facts: SceFacts) -> _WrittenSpeed:
    """The speed the VARIANT carries, on the same scale as the knob's rail.

    The ladder plots this in preference to what the model implies, for the same
    reason the spawn knob measures its share from the file: the emitted plan is
    what the game loads, and `plan(fatigue=...)` eases the EMITTED values while
    the persisted profile stays un-eased by contract. On an eased variant the
    model implied x1.17 while the file carried x1.00 — and the ladder printed
    the former one panel above a ledger row showing the latter.

    Only ever called with a parsed variant; returns value=None with a reason
    whenever the two files cannot answer between them.
    """
    path = facts.speed_path
    if path not in ("multiplier", "ramp"):
        return _WrittenSpeed(None, why="which speed path applies cannot be read")
    pairs = [(c, facts.authored[c], facts.written_speeds[c])
             for c in sorted(facts.authored) if c in facts.written_speeds]
    if not pairs:
        return _WrittenSpeed(None, why="the variant carries no MaxSpeed line to read")
    # _fmt_num, not a fresh format string: this text sits one panel above the
    # ledger rows for the very same keys, and the two must read identically.
    if path == "ramp":
        detail = ", ".join(f"{c} 0 -> {_fmt_num(w)}" for c, _, w in pairs)
        vals = [w for _, _, w in pairs]
        if max(vals) - min(vals) > _SPEED_AGREE_RAMP:
            return _WrittenSpeed(None, why=(
                f"the variant writes a different speed to each static wall "
                f"({detail}), which is not one reading"))
        return _WrittenSpeed(vals[0], detail)
    ratios = [(c, a, w, w / a) for c, a, w in pairs if a > 0]
    if not ratios:
        return _WrittenSpeed(None, why="no authored speed to measure against")
    detail = ", ".join(f"{c} {_fmt_num(a)} -> {_fmt_num(w)} (x{r:.3g})"
                       for c, a, w, r in ratios)
    if max(r for *_, r in ratios) - min(r for *_, r in ratios) > _SPEED_AGREE_MULT:
        return _WrittenSpeed(None, why=(
            f"the variant does not carry one multiplier for every target "
            f"({detail}), which is not one reading"))
    # The largest authored speed carries the least of the file's rounding to one
    # decimal, so it is the truest read of the multiplier that was written.
    return _WrittenSpeed(max(ratios, key=lambda t: t[1])[3], detail)


def _speed_divergence(profile: PlayerProfile, facts: SceFacts,
                      model_text: str) -> str:
    """Why the speed in the FILE differs from what the model implies.

    Read off the variant's own plan record, the same discipline size_check
    applies: `plan(fatigue=...)` eases only the emitted values, so a variant
    written while tired legitimately holds a lower speed than the persisted
    (un-eased) model does. Without a record the difference is unattributable and
    this says so rather than guessing.
    """
    record = _plan_record(facts.description)
    lead = f"; the model's own state implies {model_text} instead"
    if record is None:
        return (f"{lead}, and the variant's Description carries no kovadapt plan "
                "record, so why the two differ cannot be read off the file")
    eased = _clamp(record.get("eased", 0.0), 0.0, 1.0)
    if eased > 0:
        return (f"{lead} — the variant recorded eased={eased:.2f} for itself, and "
                "plan(fatigue=...) eases only the EMITTED values while the "
                "persisted profile stays un-eased on purpose, so the next session "
                "resumes true difficulty")
    return (f"{lead}, so the variant on disk was written from an earlier model "
            f"state (its own record reads movement={record.get('movement', 0.0):.2f} "
            f"against the model's {profile.movement:.2f}) — playing a run, or "
            "regenerating the variant, brings them back into step")


def _speed_knob(profile: PlayerProfile, s: Settings, facts: SceFacts) -> Knob:
    """Which speed path applies is a property of the BASE FILE, per character,
    never of the model.

    Every number here is read out of the .sce files. The baseline is the base
    file's own authored speed (the generator always applies the plan to the base
    and never to the previous variant, so the base IS the baseline) and NOW is
    the number the [Adaptive] file carries whenever one has been written —
    because that is what the game loads. Reading NOW off the model instead let
    the row print "1.00 -> 1.05x +5%" one panel above a ledger row saying
    "MaxSpeed 1300 -> 1197 x0.921": both computed from the same variant, and
    both wrong about the other. The gap is real and lawful — `plan(fatigue=...)`
    eases the emitted plan while persisted state stays un-eased — so the row
    names which number it is and where the difference came from.

    Where the files cannot say, this criterion goes unmeasured and says why. It
    used to hard-code the baseline (1.0 for the multiplier path, 0.0 for the
    ramp) and to resolve the path with `any(authored > 0)`, which on a scenario
    holding both a strafe bot and a static wall printed "never writes the
    absolute 0-170 ramp here" directly above a ledger row showing that exact
    ramp written.
    """
    path = facts.speed_path
    authored = ", ".join(f"{c} authors MaxSpeed={v:.0f}"
                         for c, v in sorted(facts.authored.items()))
    cold = profile.run_count == 0
    if path in ("multiplier", "ramp"):
        ramp = path == "ramp"
        model = (movement_speed(profile.movement) if ramp
                 else speed_multiplier(profile.movement))
        # "{:.4g}" on the ramp is _fmt_num's own rendering for anything on a
        # 0-170 rail written to one decimal, so the NOW column and the ledger
        # row for the same MaxSpeed print the same characters — at "{:.0f}" the
        # ladder said 61 u/s over a ledger row reading 61.4.
        fmt, unit = ("{:.4g}", "u/s") if ramp else ("{:.2f}", "x")
        # In prose the ramp reads "108 u/s" and the multiplier "1.03x"; the
        # ladder's NOW column concatenates the same unit without the space.
        say = (lambda v: f"{fmt.format(v)} {unit}") if ramp else \
              (lambda v: f"{fmt.format(v)}{unit}")
        lo, hi = (0.0, 170.0) if ramp else (0.65, 1.35)
        model_text = say(model)
        if ramp:
            # 1.00x on the other rail is not a chosen number either: it is the
            # author's own MaxSpeed expressed against itself.
            evidence = (f"because every target character's MaxSpeed line in the "
                        f"base file reads 0 ({authored} — a static wall), movement "
                        f"intensity is the only thing that can make them move at "
                        f"all, so the absolute 0-170 ramp is the path that applies "
                        f"and the author's own 0 is the baseline. ")
        else:
            evidence = (f"because the base file gives its targets a speed of their "
                        f"own ({authored}), kovadapt modulates AROUND the author's "
                        f"value (0.65-1.35x of it) and never writes the absolute "
                        f"0-170 static-wall ramp here — that ramp on a fast strafe "
                        f"bot would collapse the scenario. 1.00x below IS that "
                        f"authored value, read out of the base .sce. ")
        wrote = _written_speed(facts) if facts.have_variant else _WrittenSpeed(None)
        note = ""
        if wrote.value is None:
            now = model
            evidence += (f"Movement stands at {profile.movement:.2f}, which maps "
                         f"to {model_text}")
            if wrote.why:
                note = (f"{wrote.why}, so this row plots what the model implies "
                        "rather than what the variant carries")
        else:
            now = wrote.value
            evidence += (f"{say(now)} is read out of the [Adaptive] file on disk "
                         f"({wrote.detail}) — the number the game actually loads, "
                         f"which is why this row plots it")
            eps = max((hi - lo) * MOVE_EPS_FRAC, 1e-9)
            if abs(now - model) > eps:
                evidence += _speed_divergence(profile, facts, model_text)
        if cold:
            note = "; ".join(x for x in (
                f"cold-start default (movement {profile.movement:.2f}), not a "
                "learned value — the OU walk takes its first step on your first "
                "run" + (", but the file already carries this number"
                         if facts.have_variant else ""), note) if x)
        # A hand-edited or foreign variant can carry a speed the controller
        # itself could never emit. The rail has to hold what it plots.
        if not lo <= now <= hi:
            lo, hi = min(lo, now), max(hi, now)
            note = "; ".join(x for x in (note, (
                f"the variant carries {say(now)}, outside the controller's own "
                "range, so the rail is widened to hold it — that number was not "
                "written by these clamps")) if x)
        return Knob(
            key="target_speed",
            name="target speed (absolute ramp)" if ramp
            else "target speed (x authored)",
            lo=lo, hi=hi, baseline=0.0 if ramp else 1.0, now=now, fmt=fmt,
            unit=unit,
            delta_text=((f"{now:+.4g}" if now else "") if ramp
                        else _ratio_delta(1.0, now)),
            evidence=evidence, measured=not cold, note=note, flag="cold start",
            tip=("adapt/stochastic.py:movement_speed. This path exists only for "
                 "base-speed-0 characters; anything with an authored speed is "
                 "modulated instead. NOW is read out of the written variant "
                 "whenever there is one." if ramp else
                 "adapt/stochastic.py:speed_multiplier maps movement intensity "
                 "[0,1] onto 0.65-1.35; the generator writes base_speed x that, "
                 "per target character. NOW is that ratio measured back out of "
                 "the written variant whenever there is one, so it agrees with "
                 "the file ledger below even when the plan was eased."))
    if path == "mixed":
        per = "; ".join(
            f"{c} authors MaxSpeed={facts.authored[c]:.0f}, so it is "
            + ("modulated 0.65-1.35x around that" if facts.authored[c] > 0
               else "written the absolute 0-170 ramp instead")
            for c in sorted(facts.authored))
        return Knob(key="target_speed", name="target speed", lo=0.0, hi=1.0,
                    baseline=0.0, now=0.0, fmt="{:.2f}", measured=False,
                    rail=False, flag="mixed paths",
                    evidence=("this scenario mixes BOTH speed paths, so target "
                              f"speed is not one number: {per}. The generator "
                              "decides per character, and collapsing the two into "
                              "a single reading is exactly how the static-wall "
                              "ramp ends up described as applying to a fast strafe "
                              "bot — the file ledger below reports each character "
                              "on its own row"),
                    note=("reported per character rather than as one criterion, "
                          "because no single rail can hold an authored-speed "
                          "multiplier and an absolute ramp at once"))
    if not facts.have_base:
        why = facts.error or "the base .sce could not be read here"
    elif facts.no_speed_key:
        why = (f"{', '.join(facts.no_speed_key)} has no MaxSpeed line in the base "
               ".sce at all, and the generator only ever rewrites a key that is "
               "already there — so no speed is written for it, and reading a "
               "missing line as 0 would put this scenario on the static-wall ramp "
               "here while its file never moved")
    else:
        why = "the base .sce names no target character to read a speed from"
    return Knob(key="target_speed", name="target speed", lo=0.0, hi=1.0,
                baseline=0.0, now=0.0, fmt="{:.2f}", measured=False, rail=False,
                flag="unreadable",
                evidence=("cannot say which speed path applies: " + why
                          + ". The two paths are mutually exclusive — an authored "
                          "speed is modulated 0.65-1.35x, and only a base-speed-0 "
                          "wall gets the absolute 0-170 ramp — so naming one "
                          "would be a claim the data cannot support"))


def _spawn_knob(profile: PlayerProfile, s: Settings, facts: SceFacts,
                ev: ReportEvidence) -> Knob:
    """Spawn mass on the bandit's focus region.

    `now` is MEASURED FROM THE VARIANT FILE whenever there is one — the plan's
    focus_weight is only what was asked for, and a region holding no candidate
    spawn point cannot be emphasised at all (resampling reuses original
    coordinates and never invents them), so the file is the only honest source.
    """
    cells = max(s.region_cols * s.region_rows, 1)
    uniform = 1.0 / cells
    focus = profile.last_focus
    spawns = facts.spawns
    mapped = sum(1 for p in profile.regions.values() if p.n >= PlayerProfile.REGION_OBS)
    # The rail runs from 0, not from the even share: a focus region the base
    # layout holds no spawn point in really does carry 0% of the fire, and with
    # lo=uniform the row printed 0% in its NOW column while painting the marker
    # on the 4% tick.
    common = dict(key="spawn_focus", name="spawn mass on focus region",
                  lo=0.0, hi=1.0, baseline=uniform, fmt="{:.0%}",
                  tip=("Thompson sampling over the r{row}c{col} wall grid: one "
                       "draw from every cell's Gaussian belief, the worst draw "
                       f"takes the focus and {s.focus_weight:.0%} of the spawn "
                       "mass, with the remainder softmaxed over the other cells "
                       "by their posterior means. The variant's spawn list is a "
                       "resample of the base's own coordinates — a region with "
                       "no candidate point there cannot be emphasised, and "
                       "AdaptationPlan.focus_applied records that."))
    if not focus:
        return Knob(**common, now=uniform, measured=False,
                    evidence=("no focus region has been chosen yet — the bandit "
                              "picks one when the first plan is written, so spawns "
                              f"are the author's layout and every one of the {cells} "
                              f"cells carries its share of it"))
    where = _region_words(focus, s)
    arm = profile.regions.get(focus)
    # An arm with zero observations was picked by a draw from its UNTOUCHED
    # prior. The spawn shift is real and verifiable, but calling it evidenced
    # would dress up the bandit's exploration as a measured weakness.
    has_obs = arm is not None and arm.n > 0
    explore_note = (
        "" if has_obs else
        f"exploration, not a finding: {focus} has no observations behind it, so "
        "the shift you can see in the file is the bandit deliberately spending "
        "a run on an unmapped region — roughly a fifth of focus picks are")
    arm_text = (f"arm {focus} ({where}) carries n={arm.n} observations, posterior "
                f"mean {arm.mean:+.2f} (mean > 0 = weaker there)" if has_obs else
                f"arm {focus} ({where}) carries no observations at all and its "
                "Thompson draw came from the untouched prior")
    arm_text += f"; {mapped} of {cells} arms have the {PlayerProfile.REGION_OBS}+ " \
                "observations the model calls mapped"
    rn = ev.region_n.get(focus, 0)
    if rn:
        arm_text += (f"; {rn} run reports carry a telemetry deficit for it, "
                     f"mean z {ev.region_mean.get(focus, 0.0):+.2f}")

    if spawns is None or not spawns.base:
        return Knob(**common, now=uniform, measured=has_obs,
                    evidence=arm_text,
                    note=("the base .sce could not be read here, so the share "
                          "actually written into the variant is unknown"))
    if spawns.reason:
        return Knob(**common, now=spawns.base_share(focus), measured=False,
                    evidence=arm_text, note=spawns.reason)
    if spawns.base.get(focus, 0) == 0:
        return Knob(**common, now=spawns.base_share(focus), measured=False,
                    evidence=arm_text,
                    note=(f"the base layout has no spawn point in {focus}, so this "
                          "focus cannot be expressed at all — the generator never "
                          "invents coordinates, its weight is absorbed by the "
                          "other regions, and the emitted scenario carries no "
                          "focus (focus_applied=False)"))
    if spawns.adaptive:
        now = spawns.share(focus)
        evidence = (f"because {arm_text}. Verified in the file: {focus} holds "
                    f"{spawns.adaptive.get(focus, 0)} of {spawns.total_adaptive} "
                    f"target spawns in the variant against "
                    f"{spawns.base.get(focus, 0)} of {spawns.total_base} in the "
                    f"base — {spawns.base_share(focus):.0%} to {now:.0%}, where "
                    f"an even layout would be {uniform:.0%}")
        return Knob(**common, now=now,
                    delta_text=f"{(now - uniform) * 100:+.0f}pts",
                    measured=has_obs, evidence=evidence, note=explore_note,
                    flag="exploration")
    # No variant on disk: this is the share the NEXT generation would ask for.
    # The planned-not-written clause is no longer written here — build_knobs
    # applies it to every criterion, because it used to appear on this knob
    # alone while the other four read as though the game had already seen them.
    return Knob(**common, now=float(s.focus_weight), measured=has_obs,
                delta_text=f"{(s.focus_weight - uniform) * 100:+.0f}pts",
                evidence=f"because {arm_text}",
                # The flag only surfaces when nothing backs the value, so it has
                # to name the ACTUAL absence rather than the last one written.
                flag=("exploration" if not has_obs else "no evidence"),
                note=explore_note)


def _dodge_knob(profile: PlayerProfile, s: Settings, ev: ReportEvidence) -> Knob:
    """Strafe-timing skew toward the weak flick side (+ = LEFT is weaker)."""
    gated = abs(profile.ewma_bias) > 0.05
    now = 0.0
    if s.dodge_bias_enabled and gated:
        now = max(min(s.dodge_bias_gain * profile.ewma_bias, 1.0), -1.0)
    side = "left" if profile.ewma_bias > 0 else "right"
    measured = profile.bias_obs > 0 or abs(profile.ewma_bias) > 0.0
    if not measured:
        evidence = ("no run has produced a usable directional-bias measurement "
                    f"yet — that needs {_BIAS_MIN_FLICKS}+ flicks in a run with "
                    f"{_BIAS_MIN_PER_SIDE}+ per side, so strafing stays exactly "
                    "as symmetric as the author wrote it")
    else:
        evidence = (f"because {profile.bias_obs} runs produced a usable bias "
                    f"measurement and the EWMA sits at {profile.ewma_bias:+.2f}, "
                    f"i.e. your {side} flicks measurably cost more, so targets "
                    f"strafe longer toward the {side}")
        if ev.bias_runs:
            evidence += (f"; {ev.bias_runs} run reports clear the "
                         f"{_BIAS_MIN_FLICKS}-flick / {_BIAS_MIN_PER_SIDE}-per-side "
                         f"gate, mean score {ev.bias_mean:+.2f}")
    note = ""
    if not s.dodge_bias_enabled:
        note = ("dodge_bias_enabled is off in settings, so nothing is written "
                "however strong the measurement gets")
    elif measured and not gated:
        note = (f"under the engine's 0.05 gate ({abs(profile.ewma_bias):.3f}), so "
                "no skew is written yet — a weak measurement must not become a "
                "prescription")
    elif ev.degraded:
        note = (f"{ev.degraded} of the {ev.bias_runs} contributing runs had input "
                "timing too noisy to read flick microstructure from "
                "(analysis/report.py:input_degraded); the engine folded them in "
                "anyway, so treat the EWMA as that much softer")
    return Knob(key="dodge_bias", name="dodge direction skew", lo=-1.0, hi=1.0,
                baseline=_fresh_defaults().ewma_bias, now=now, fmt="{:+.2f}",
                # The value is already in the NOW column; the delta column
                # earns its space by naming the direction instead.
                delta_text=("" if abs(now) < 1e-9 else
                            f"toward {'left' if now > 0 else 'right'}"),
                evidence=evidence, measured=measured and gated, note=note,
                flag=("gated off" if measured else "no evidence"),
                tip=("bias_score > 0 means the LEFT side is weaker "
                     "(analysis/movement.py convention). The engine writes "
                     f"clip({s.dodge_bias_gain:.2f} x EWMA, -1, 1) into the dodge "
                     "profiles as reciprocal Left/RightStrafeTimeMult, so total "
                     "strafe time stays about constant while the weak side sees "
                     "more traffic."))


def _movement_knob(profile: PlayerProfile, s: Settings,
                   fresh: PlayerProfile) -> Knob:
    kpss = [float(h.get("kps", 0.0)) for h in profile.history[-10:]]
    half = len(kpss) // 2
    a0 = sum(kpss[:half]) / max(half, 1)
    a1 = sum(kpss[half:]) / max(len(kpss) - half, 1)
    flat = a0 > 0 and abs(a1 / a0 - 1.0) < 0.03
    last = profile.history[-1] if profile.history else {}
    last_kps = float(last.get("kps", 0.0) or 0.0)
    last_acc = float(last.get("accuracy", 0.0) or 0.0)
    in_band = s.target_accuracy_low <= last_acc <= s.target_accuracy_high
    plateau = (s.pace_progression_gain > 0 and profile.run_count >= 10
               and flat and in_band)
    # A fresh profile's 0.15 is the dataclass default, which the engine squashes
    # into the ARCHETYPE's clamps on the very first plan — under the tracking
    # override that floor is 0.35, so 0.15 is a starting point the controller
    # can never occupy. Plotted unclamped it printed 0.15 in the baseline column
    # while the painter put "|" on the 0.35 tick.
    baseline, clamp_note = _reachable(fresh.movement, s.min_movement,
                                      s.max_movement)
    if profile.run_count == 0:
        evidence = (f"no runs — movement is the profile's cold-start default "
                    f"({baseline:.2f}) and the OU walk has never stepped")
        measured = False
    else:
        evidence = (f"because the Ornstein-Uhlenbeck state stands at "
                    f"{profile.ou_state:+.2f} after {profile.run_count} steps "
                    f"(one per run), squashed into "
                    f"[{s.min_movement:.2f}, {s.max_movement:.2f}]")
        if profile.ewma_kps > 0 and last_kps > 0 and profile.run_count >= 3:
            rel = last_kps / profile.ewma_kps - 1.0
            evidence += (f"; your last run ran {abs(rel) * 100:.0f}% "
                         f"{'hotter' if rel > 0 else 'cooler'} than your own "
                         f"{profile.ewma_kps:.2f} kills/s EWMA, which pushes the "
                         f"walk {'up' if rel > 0 else 'down'}")
        played = _played_range(profile, "movement")
        if played:
            evidence += f"; the model stood at {played} across those runs"
        measured = True
    note = ""
    if plateau:
        note = (f"pace-plateau push active: kills/s flat within 3% across the last "
                f"{len(kpss)} runs ({a0:.2f} then {a1:.2f}) with accuracy in band, "
                "so movement gets a bounded upward nudge — growth shows up as "
                "speed, not accuracy")
    lo, hi, wide_note = _widen(s.min_movement, s.max_movement, baseline,
                               profile.movement)
    note = "; ".join(n for n in (note, clamp_note, wide_note) if n)
    moved = profile.movement - baseline
    return Knob(key="movement", name="movement / pace", lo=lo, hi=hi,
                baseline=baseline, now=profile.movement,
                fmt="{:.2f}",
                # An intensity on [0, 1], so the honest delta is absolute: a
                # ratio against the 0.15 cold start turns +0.29 into "+194%",
                # which reads as a far bigger claim than the number supports.
                delta_text=("" if abs(moved) < 1e-9 else f"{moved:+.2f}"),
                evidence=evidence, measured=measured, note=note,
                tip=("A mean-reverting random walk stepped once per run: drift is "
                     "smooth (no jarring jump between variants) but unpredictable "
                     "(no rhythm to memorise). Pace couples in twice — running "
                     "hotter than your own norm raises it immediately, and a flat "
                     "pace with accuracy parked in band adds the progression "
                     "push. Movement also sets target speed and, through "
                     f"size_speed_coupling={s.size_speed_coupling:.2f}, a size "
                     "floor."))


# One clause, applied to every criterion when nothing has been written yet. It
# lived on the spawn knob alone, so a task with no [Adaptive] file on disk had
# four criteria reading as changes the game had already seen while every VARIANT
# cell in the ledger below read "—".
# Short on purpose: it repeats five times, and the headline, the ladder's own
# PLANNED column header and the provenance line below carry the long form.
_PENDING_NOTE = "not written yet — no [Adaptive] .sce on disk carries this value"


def _pending_note(facts: SceFacts) -> str:
    """Why the ladder is showing model values rather than written ones.

    THREE different absences, and they must not read alike. Only one of them is
    "nothing has been written": read_sce_facts gives up on a missing base .sce
    before it ever looks at the variant, so a task whose base file had been
    renamed printed "no [Adaptive] .sce on disk carries this value" — five times,
    plus the same claim in the headline — about a file sitting right beside the
    one it could not find. `variant_on_disk` is the only field that may back a
    sentence about what is on disk.
    """
    if not facts.have_base:
        return ("unverified — the base .sce is missing, so what any [Adaptive] "
                "file carries cannot be read"
                if not facts.variant_on_disk else
                "unverified — an [Adaptive] .sce IS on disk, but without its base "
                "file there is no before/after to read it against")
    if facts.variant_on_disk:
        return ("the [Adaptive] .sce on disk could not be read, so what it "
                "carries is unknown")
    return _PENDING_NOTE


def build_knobs(profile: PlayerProfile, settings: Settings, facts: SceFacts,
                ev: ReportEvidence) -> list[Knob]:
    """The five criteria the engine actually moves, in the order it moves them.

    Bounds and bands come from `Settings.for_archetype(profile.archetype)`
    because that is the only view engine code is allowed to read (the contract:
    tunables through `_effective()`, never `self.s`). Showing the raw defaults
    would print the clicking band over a tracking scenario.
    """
    s = settings.for_archetype(profile.archetype)
    fresh = _fresh_defaults()
    knobs = [
        _size_knob(profile, s, fresh),
        _speed_knob(profile, s, facts),
        _spawn_knob(profile, s, facts, ev),
        _dodge_knob(profile, s, ev),
        _movement_knob(profile, s, fresh),
    ]
    if not facts.have_variant:
        # The clause lands on every criterion whose number would otherwise read
        # as a change the game has already seen. An UNMEASURED knob already
        # prints its own absence in the delta column ("no runs yet", "cold
        # start", "exploration") and its note already says why nothing landed,
        # so repeating this there would only bury that reason.
        clause = _pending_note(facts)
        knobs = [replace(k, pending=True,
                         note=". ".join(x for x in (clause, k.note) if x))
                 if k.measured else replace(k, pending=True)
                 for k in knobs]
    return knobs


def _headline_unwritten(facts: SceFacts | None) -> str:
    """The headline's own version of _pending_note — the same three absences, and
    the same rule: only `variant_on_disk` may back a claim about the disk."""
    if facts is None or facts.have_variant:
        return ""
    if not facts.have_base:
        return (", and none of it can be checked against the game's files: the "
                "base .sce is missing from the Scenarios folder"
                + (", and the [Adaptive] file on disk has nothing to be compared "
                   "against" if facts.variant_on_disk else ""))
    if facts.variant_on_disk:
        return (", but the [Adaptive] file on disk could not be read, so none of "
                "it is verified against the game's own copy")
    return ", but no [Adaptive] file has been written yet, so none of it has " \
           "reached the game"


def takeaway(knobs: list[Knob], profile: PlayerProfile | None,
             facts: SceFacts | None = None) -> str:
    """The page's headline: what the data SHOWS, never more than that."""
    if profile is None or not knobs:
        return "pick a scenario to see what kovadapt has changed about it"
    # A move in the model and a move the game has seen are different claims:
    # with no variant on disk this said "have moved" over a ledger whose every
    # VARIANT cell was a dash.
    written = facts is None or facts.have_variant
    unwritten = _headline_unwritten(facts)
    if profile.run_count == 0:
        return (f"{profile.scenario} has a profile but no completed runs — nothing "
                "here is learned yet, and every value below is a cold-start default"
                + ("" if written else unwritten))
    moved = [k for k in knobs if k.measured and k.moved]
    n = profile.run_count
    if not moved:
        held = next((k for k in knobs if k.measured and not k.moved), None)
        why = f" — {held.name} has held: {held.evidence}" if held else ""
        return (f"after {n} runs nothing has moved from baseline on evidence"
                f"{'' if written else unwritten}{why}")
    biggest = max(moved, key=lambda k: abs(k.now - k.baseline) / max(k.hi - k.lo, 1e-9))
    detail = (f"largest: {biggest.name}, {biggest.text(biggest.baseline)} to "
              f"{biggest.text(biggest.now)}")
    if written:
        return (f"{len(moved)} of {len(knobs)} criteria have moved on evidence after "
                f"{n} runs — {detail}")
    return (f"{len(moved)} of {len(knobs)} criteria have moved in the model after "
            f"{n} runs{unwritten} — {detail}")


# ----------------------------------------------------------------- painters
def _title_band(p: QPainter, pal, title: str, width: float) -> float:
    """Dim uppercase mono header line; returns the content top y."""
    if not title:
        return 8.0
    font = theme.mono(12)
    p.setFont(font)
    p.setPen(QColor(pal.fg_dim))
    p.drawText(QRectF(10, 4, width - 20, 18), Qt.AlignLeft | Qt.AlignVCenter,
               title.upper())
    return 26.0


def _empty_band(p: QPainter, pal, rect: QRectF, text: str) -> None:
    p.setFont(theme.mono(14))
    col = QColor(pal.fg_dim)
    col.setAlphaF(0.75)
    p.setPen(col)
    p.drawText(rect, Qt.AlignCenter, f"- {text} -")


class _Art(QWidget):
    """Base for the three character-art panels.

    Colours are read from theme.current() inside paintEvent, so restyle() is
    nothing but update() and no palette is ever cached. The staggered reveal
    is driven from outside: gui/motion.py owns the timing and ChangesView owns
    the single clock, so these widgets hold no timer of their own — one clock
    for the page means one propagating event rather than three.
    """

    DONE = 1e9

    # motion.STAGGER_PER_CELL is calibrated for a FINE glyph grid (~80 cells
    # across). A 5x5 zone map spans five units, so at 11 ms/unit its whole
    # propagation resolves inside one glyph frame and reads as a pop rather
    # than an event; `per_unit` is the knob motion.stagger exposes for exactly
    # this, so coarse grids scale the unit instead of inventing a duration.
    PER_UNIT = motion.STAGGER_PER_CELL

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self._reveal = self.DONE
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _delay(self, distance: float, settings: Settings | None = None) -> int:
        """Reveal delay for something `distance` units from the origin."""
        return motion.stagger(settings if settings is not None else self.s,
                              distance, per_unit=self.PER_UNIT)

    def set_reveal(self, elapsed_ms: float) -> None:
        self._reveal = float(elapsed_ms)
        self.update()

    def reveal_done(self) -> None:
        self.set_reveal(self.DONE)

    def reveal_total(self, settings: Settings) -> int:
        """Longest per-cell delay in this panel's reveal (0 = nothing to run)."""
        return 0

    def _lit(self, delay: float) -> bool:
        """Cells pop rather than fade: the output is quantised to a character
        ramp, so a sub-glyph alpha tween costs a full repaint for no visible
        change. The propagation IS the animation, at motion.GLYPH_HZ."""
        return self._reveal >= delay

    def restyle(self, *_pal) -> None:
        self.update()


class _Cols(NamedTuple):
    """KnobLadder column geometry. `cells == 0` means the rail does not fit."""

    x_read: float
    base_w: float
    now_w: float
    delta_w: float
    x_rail: float
    cw: float
    x_end: float
    cells: int


class KnobLadder(_Art):
    """BASELINE -> NOW for every criterion, on the criterion's own bounds.

    Each row is a rail spanning the controller's real clamp range, with `|` at
    the baseline, `@` at the current value, `0` where the two coincide, and a
    glyph run between them that fades away from the anchor (gui/viz.py's
    convention: dense at the anchor, light at the tip). A knob with no evidence
    behind it draws that run dim instead of in the accent, because a cold-start
    default is a real number in the file but is not a change the model earned.

    Two rules keep the rail from contradicting the numbers printed beside it:

    * below MIN_RAIL_CELLS the rail is DROPPED, not squeezed. At 8 cells a real
      -20% size move rendered "@|" and a -6% speed move rendered as a lone "@" —
      the row said "no move" while its own delta column said otherwise.
    * a move the knob calls real is held at least MIN_RUN_CELLS apart, because
      1% of a range (MOVE_EPS_FRAC) is a fifth of a cell even at full width.

    The reveal propagates outward from EACH row's own baseline anchor, so the
    five baselines ignite together and the moves grow out of them.
    """

    HEAD_H = 24.0
    ROW_H = 26.0

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(settings, parent)
        self._knobs: list[Knob] = []
        self.setFixedHeight(int(self.HEAD_H + self.ROW_H + 10))

    def set_knobs(self, knobs: list[Knob]) -> None:
        self._knobs = list(knobs)
        rows = max(len(self._knobs), 1)
        self.setFixedHeight(int(self.HEAD_H + rows * self.ROW_H + 10))
        self.update()

    # ------------------------------------------------------------------
    def _layout(self) -> _Cols:
        """Column geometry, with `cells == 0` when the rail cannot be honest.

        One line per criterion, and the numbers sit next to the NAME rather
        than at the far edge of a 1400px column — the first draft put the
        reading a screen-width away from the label it belonged to, and the two
        could not be connected by eye.
        """
        knobs = self._knobs
        fm_name = QFontMetricsF(theme.mono(12))
        fm_num = QFontMetricsF(theme.mono(14))
        name_w = max([fm_name.horizontalAdvance(k.name.upper()) for k in knobs]
                     + [120.0])
        base_w = max([fm_num.horizontalAdvance(self._read_of(k)[0]) for k in knobs]
                     + [40.0])
        now_w = max([fm_num.horizontalAdvance(self._read_of(k)[1])
                     for k in knobs] + [46.0])
        delta_w = max([fm_num.horizontalAdvance(self._delta_of(k)) for k in knobs]
                      + [fm_name.horizontalAdvance("PLANNED") + 2.0])
        gutter = max([fm_name.horizontalAdvance(k.text(k.lo)) for k in knobs]
                     + [fm_name.horizontalAdvance(k.text(k.hi)) for k in knobs]
                     + [28.0])
        arrow_w = fm_num.horizontalAdvance("->")
        x_read = 14.0 + name_w + 18.0
        read_w = base_w + 6.0 + arrow_w + 6.0 + now_w + 12.0 + delta_w
        cw = max(fm_num.horizontalAdvance("@"), 6.0)
        x_rail = x_read + read_w + 18.0 + gutter + 8.0
        # Capped, not full-bleed: over a 1400px column the rail became ~900px
        # of dots with the actual move a short run inside it, so the noise
        # dominated the signal. 55 cells is plenty of positional resolution.
        room = float(self.width()) - 14.0 - gutter - 8.0 - x_rail
        cells = int(min(room, cw * MAX_RAIL_CELLS) // cw)
        if cells < MIN_RAIL_CELLS:
            # Too cramped to tell a move from a no-move. Drop the rail rather
            # than draw one that contradicts the delta printed beside it.
            return _Cols(x_read, base_w, now_w, delta_w, x_rail, cw, x_rail, 0)
        return _Cols(x_read, base_w, now_w, delta_w, x_rail, cw,
                     x_rail + cells * cw, cells)

    @staticmethod
    def _delta_of(knob: Knob) -> str:
        return knob.delta_text if knob.measured else knob.flag

    def delta_header(self) -> str:
        """The delta column names itself PLANNED while nothing has been written,
        so the deltas under it cannot read as changes the game has seen."""
        return "PLANNED" if any(k.pending for k in self._knobs) else "CHANGE"

    @staticmethod
    def _read_of(knob: Knob) -> tuple[str, str]:
        """(baseline, now) as printed. A knob with no plottable value prints
        dashes: putting its rail's own floor in the column would read as a
        measurement of something the file cannot say."""
        if not knob.rail:
            return "—", "—"
        return knob.text(knob.baseline), knob.text(knob.now) + knob.unit

    @staticmethod
    def _col(knob: Knob, value: float, cells: int) -> int:
        span = knob.hi - knob.lo
        if span <= 0 or cells <= 1:
            return 0
        frac = (value - knob.lo) / span
        return int(round(min(max(frac, 0.0), 1.0) * (cells - 1)))

    def _rail_cols(self, knob: Knob, cells: int) -> tuple[int, int]:
        """(anchor, marker) cells, with a real move guaranteed to RENDER as one.

        MOVE_EPS_FRAC calls a 1%-of-range change a move, and 1% of a 24-55 cell
        rail is a fifth of a cell, so a real move could land in the anchor's own
        cell and draw the no-move glyph. The pair is pushed apart in the move's
        own direction (both shifted when that would run off the end), which
        trades exact position for a true reading — the rail is a magnitude run,
        the numbers beside it are the measurement, and the caption says so.
        """
        anchor = self._col(knob, knob.baseline, cells)
        marker = self._col(knob, knob.now, cells)
        if (not knob.moved or cells < MIN_RUN_CELLS * 2 + 1
                or abs(marker - anchor) >= MIN_RUN_CELLS):
            return anchor, marker
        step = MIN_RUN_CELLS if knob.now > knob.baseline else -MIN_RUN_CELLS
        marker = anchor + step
        if marker < 0:
            return -step, 0
        if marker > cells - 1:
            return cells - 1 - step, cells - 1
        return anchor, marker

    @staticmethod
    def _glyph(col: int, anchor: int, marker: int, span: int) -> str:
        """The character ONE rail cell carries.

        One definition, shared by the painter and `row_glyphs`, so what a test
        reads is literally what is drawn. The coincident case is tested FIRST:
        with the marker test ahead of the anchor test, an unmoved criterion drew
        a lone "@" and no baseline tick at all.
        """
        if col == marker and col == anchor:
            return _ON_BASELINE_GLYPH
        if col == marker:
            return _MARKER_GLYPH
        if col == anchor:
            return _ANCHOR_GLYPH
        if min(anchor, marker) < col < max(anchor, marker):
            frac = abs(col - anchor) / span
            return _RUN_RAMP[int(frac * (len(_RUN_RAMP) - 1))]
        return "."

    def row_glyphs(self, knob: Knob, cells: int | None = None) -> str:
        """One knob's whole rail as characters ("" when no rail is drawn)."""
        cells = self._layout().cells if cells is None else cells
        if cells <= 0 or not knob.rail:
            return ""
        anchor, marker = self._rail_cols(knob, cells)
        span = max(abs(marker - anchor), 1)
        return "".join(self._glyph(c, anchor, marker, span)
                       for c in range(cells))

    def _origin_row(self) -> float:
        return (len(self._knobs) - 1) / 2.0

    def _label_delay(self, row: int) -> int:
        """Labels land one FAST step behind their row's rail, so the numbers
        arrive as the move resolves rather than before it."""
        return (motion.stagger(self.s, abs(row - self._origin_row()))
                + motion.ms(self.s, motion.FAST))

    def reveal_total(self, settings: Settings) -> int:
        if not self._knobs or not motion.animates(settings):
            return 0
        cells = self._layout().cells
        mid = self._origin_row()
        worst = 0
        for row, knob in enumerate(self._knobs):
            worst = max(worst, self._label_delay(row))
            if cells <= 0 or not knob.rail:
                continue
            anchor = self._rail_cols(knob, cells)[0]
            for col in (0, cells - 1):
                worst = max(worst, self._delay(
                    motion.grid_distance(row, col, (mid, anchor)), settings))
        return worst + motion.GLYPH_MS

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        width = float(self.width())
        if not self._knobs:
            _empty_band(p, pal, QRectF(0, 8, width, self.height() - 8),
                        "no scenario selected")
            return

        geo = self._layout()
        x_read, base_w, now_w = geo.x_read, geo.base_w, geo.now_w
        x_rail, cw, x_end, cells = geo.x_rail, geo.cw, geo.x_end, geo.cells
        has_rail = cells > 0
        mid = self._origin_row()
        name_f = theme.mono(12)
        num_f = theme.mono(14)
        fm_num = QFontMetricsF(num_f)
        arrow_w = fm_num.horizontalAdvance("->")
        line_h = max(fm_num.height(), 16.0)
        x_arrow = x_read + base_w + 6.0
        x_now = x_arrow + arrow_w + 6.0
        x_delta = x_now + now_w + 12.0
        # Where the lo..hi range prints when there is no rail to tick.
        x_range = x_delta + geo.delta_w + 12.0
        range_w = max(width - x_range - 10.0, 40.0)

        # Column headers: the panel sits inside a group box that already names
        # it, so the band earns its space by labelling the columns instead of
        # repeating the title.
        p.setFont(name_f)
        p.setPen(QColor(pal.fg_dim))
        p.drawText(QRectF(14, 4, x_read - 20, line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, "CRITERION")
        p.drawText(QRectF(x_read, 4, base_w + arrow_w + now_w + 12, line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, "BASELINE -> NOW")
        # The delta column names itself, and names itself PLANNED while nothing
        # has been written: every delta below it is then a model value the game
        # has never seen.
        p.drawText(QRectF(x_delta, 4, geo.delta_w, line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, self.delta_header())
        if has_rail:
            p.drawText(QRectF(x_rail, 4, x_end - x_rail, line_h),
                       Qt.AlignLeft | Qt.AlignVCenter, "CONTROLLER RANGE")
        else:
            p.drawText(QRectF(x_range, 4, range_w, line_h),
                       Qt.AlignLeft | Qt.AlignVCenter, "RANGE")

        for row, knob in enumerate(self._knobs):
            y = self.HEAD_H + row * self.ROW_H
            drawn = has_rail and knob.rail
            anchor, marker = (self._rail_cols(knob, cells) if drawn else (0, 0))
            run_col = QColor(pal.accent if knob.measured else pal.fg_dim)
            mark_col = QColor(pal.warn if knob.at_bound and knob.measured
                              else run_col)
            base_txt, now_txt = self._read_of(knob)

            if self._lit(self._label_delay(row)):
                p.setFont(name_f)
                p.setPen(QColor(pal.fg))
                p.drawText(QRectF(14, y, x_read - 20, line_h),
                           Qt.AlignLeft | Qt.AlignVCenter, knob.name.upper())
                p.setFont(num_f)
                p.setPen(QColor(pal.fg_dim))
                p.drawText(QRectF(x_read, y, base_w, line_h),
                           Qt.AlignRight | Qt.AlignVCenter, base_txt)
                p.setPen(QColor(pal.border))
                p.drawText(QRectF(x_arrow, y, arrow_w + 2, line_h),
                           Qt.AlignLeft | Qt.AlignVCenter, "->")
                p.setPen(QColor(pal.fg))
                p.drawText(QRectF(x_now, y, now_w, line_h),
                           Qt.AlignLeft | Qt.AlignVCenter, now_txt)
                p.setPen(QColor(mark_col if knob.measured else pal.fg_dim))
                p.drawText(QRectF(x_delta, y, geo.delta_w, line_h),
                           Qt.AlignLeft | Qt.AlignVCenter, self._delta_of(knob))
                p.setFont(name_f)
                p.setPen(QColor(pal.fg_dim))
                if not knob.rail:
                    pass                    # no range to print for an unreadable knob
                elif has_rail:
                    p.drawText(QRectF(x_rail - 60.0, y, 52.0, line_h),
                               Qt.AlignRight | Qt.AlignVCenter, knob.text(knob.lo))
                    p.drawText(QRectF(x_end + 8, y, width - x_end - 10, line_h),
                               Qt.AlignLeft | Qt.AlignVCenter, knob.text(knob.hi))
                else:
                    p.drawText(QRectF(x_range, y, range_w, line_h),
                               Qt.AlignLeft | Qt.AlignVCenter,
                               f"{knob.text(knob.lo)} - {knob.text(knob.hi)}")

            if not drawn:
                continue
            # The rail is mostly unfilled dots, and one drawText per cell put
            # ~275 calls a frame on a 240 Hz panel's budget. The lit set is
            # contiguous (stagger is monotonic in distance from the anchor), so
            # consecutive dots batch into one string in the same mono cells.
            p.setFont(num_f)
            dot_col = QColor(pal.border)
            span = max(abs(marker - anchor), 1)
            dots: list[int] = []
            for col in range(cells + 1):
                lit = (col < cells
                       and self._lit(self._delay(
                           motion.grid_distance(row, col, (mid, anchor)))))
                ch = self._glyph(col, anchor, marker, span) if lit else ""
                if ch == ".":
                    dots.append(col)
                    continue
                if dots:
                    p.setPen(dot_col)
                    p.drawText(QRectF(x_rail + dots[0] * cw, y,
                                      cw * (len(dots) + 2), line_h),
                               Qt.AlignLeft | Qt.AlignVCenter, "." * len(dots))
                    dots = []
                if not lit:
                    continue
                if ch == _MARKER_GLYPH:
                    col_q = QColor(mark_col)
                elif ch in (_ANCHOR_GLYPH, _ON_BASELINE_GLYPH):
                    col_q = QColor(pal.fg)
                else:
                    frac = abs(col - anchor) / span
                    col_q = QColor(run_col)
                    col_q.setAlphaF(1.0 - 0.4 * frac)
                p.setPen(col_q)
                p.drawText(QRectF(x_rail + col * cw, y, cw * 2, line_h),
                           Qt.AlignLeft | Qt.AlignVCenter, ch)


class SpawnGrid(_Art):
    """Target spawn density per wall region, base against variant.

    Density is anchored ABSOLUTELY (three times an even share fills the ramp),
    never min-max normalised, so a busy layout is not stretched until one cell
    screams. A region the base layout has no spawn point in is drawn as a
    hairline outline and nothing else — that region cannot be emphasised at
    all, and painting it as "zero fire" would read as a decision rather than an
    impossibility.

    When the generator provably left the layout ALONE (`SpawnMap.untouched`),
    the ramp and the focus ring are dropped entirely and the cells carry their
    counts only. An absolute anchor is not enough on its own there: five target
    spawns across a 5x5 grid gave each occupied cell five times an even share,
    saturated the anchor, and painted maximum emphasis for a layout that no
    plan can touch — while the spawn knob one line below correctly said so.

    The reveal ignites at the focus cell and propagates outward.
    """

    TITLE_H = 26.0
    CELL_H = 62.0
    CELL_ASPECT = 1.7      # a wall is wider than tall, but not 5x
    PER_UNIT = 70          # five zone-units across -> ~350 ms of propagation

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(settings, parent)
        self._map: SpawnMap | None = None
        self.setMouseTracking(True)
        self.setFixedHeight(int(self.TITLE_H + 5 * self.CELL_H + 12))

    def set_map(self, spawn_map: SpawnMap | None) -> None:
        self._map = spawn_map
        if spawn_map is None or not spawn_map.base:
            # Nothing to lattice: an empty five-row grid left ~350px of void
            # with one line of explanation floating in it.
            self.setFixedHeight(int(self.TITLE_H + 44))
        else:
            self.setFixedHeight(
                int(self.TITLE_H + max(spawn_map.rows, 1) * self.CELL_H + 12))
        self.update()

    # ------------------------------------------------------------------
    def _geom(self):
        """(x0, y0, cell_w, cell_h, gap, rows, cols); screen row 0 is the TOP
        row, i.e. data row rows-1 (aim convention, row 0 = bottom)."""
        sm = self._map
        if sm is None or not sm.base:
            return None
        gap = 4.0
        y0 = self.TITLE_H
        ch = (self.height() - y0 - 10.0 - gap * (sm.rows - 1)) / max(sm.rows, 1)
        avail = self.width() - 24.0 - gap * (sm.cols - 1)
        # A wall map has to look like a wall: given the whole 1400px column the
        # cells came out 270px wide and 50 tall, which read as a table of
        # dotted bars rather than a grid over the wall. Cap the aspect and
        # centre what is left.
        cw = min(avail / max(sm.cols, 1), ch * self.CELL_ASPECT)
        if cw <= 6 or ch <= 6:
            return None
        used = cw * sm.cols + gap * (sm.cols - 1)
        x0 = max((self.width() - used) / 2.0, 12.0)
        return x0, y0, cw, ch, gap, sm.rows, sm.cols

    def title_text(self) -> str:
        """The panel's own header line — it has to name what it is showing."""
        sm = self._map
        if sm is not None and sm.untouched:
            return "spawn layout per wall region - the author's own, left untouched"
        return "spawn density per wall region - base vs variant"

    def cell_density(self, key: str) -> float:
        """Ramp fill for one cell, on [0, 1].

        ZERO for every cell of a layout the generator provably left alone: there
        the absolute anchor is not enough on its own, because five spawns across
        a 5x5 grid put five times an even share in each occupied cell and painted
        maximum emphasis on a layout no plan can touch.
        """
        sm = self._map
        if sm is None or sm.untouched or sm.base.get(key, 0) == 0:
            return 0.0
        full = max(sm.uniform * _SPAWN_FULL_MULT, 1e-9)
        return min(max(sm.share(key) / full, 0.0), 1.0)

    def shows_focus_ring(self, key: str) -> bool:
        """The accent ring is the loudest mark on the panel, so it may only
        appear where the generator could actually apply the focus."""
        sm = self._map
        return (sm is not None and key == sm.focus and not sm.untouched
                and sm.base.get(key, 0) > 0)

    def _focus_cell(self) -> tuple[int, int]:
        sm = self._map
        if sm is None:
            return (0, 0)
        if sm.focus:
            try:
                r, c = sm.focus[1:].split("c")
                return int(r), int(c)
            except ValueError:
                pass
        return ((sm.rows - 1) // 2, (sm.cols - 1) // 2)

    def reveal_total(self, settings: Settings) -> int:
        sm = self._map
        if sm is None or not sm.base or not motion.animates(settings):
            return 0
        origin = self._focus_cell()
        worst = max(self._delay(motion.grid_distance(r, c, origin), settings)
                    for r in range(sm.rows) for c in range(sm.cols))
        return worst + motion.GLYPH_MS

    def zone_info(self, x: float, y: float) -> str | None:
        geom = self._geom()
        sm = self._map
        if geom is None or sm is None:
            return None
        x0, y0, cw, ch, gap, rows, cols = geom
        col = int((x - x0) // (cw + gap))
        disp = int((y - y0) // (ch + gap))
        if not (0 <= col < cols and 0 <= disp < rows):
            return None
        if (x - x0) - col * (cw + gap) > cw or (y - y0) - disp * (ch + gap) > ch:
            return None                                    # in the gutter
        row = rows - 1 - disp
        key = f"r{row}c{col}"
        base = sm.base.get(key, 0)
        if base == 0:
            return f"{key} - no spawn point here in the base layout"
        out = f"{key} - base {base} spawns ({sm.base_share(key):.1%})"
        if sm.adaptive:
            out += (f" -> variant {sm.adaptive.get(key, 0)} "
                    f"({sm.share(key):.1%})")
        elif sm.planned:
            out += f" -> planned {sm.planned.get(key, 0.0):.1%}"
        if key == sm.focus:
            out += ("  [focus - NOT applied, the generator left this layout alone]"
                    if sm.untouched else "  [focus]")
        return out

    def mouseMoveEvent(self, event) -> None:
        info = self.zone_info(event.position().x(), event.position().y())
        self.setToolTip(info or "")
        if info:
            QToolTip.showText(event.globalPosition().toPoint(), info, self)
        super().mouseMoveEvent(event)

    # ------------------------------------------------------------------
    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        width, height = float(self.width()), float(self.height())
        sm = self._map
        untouched = sm is not None and sm.untouched
        top = _title_band(p, pal, self.title_text(), width)
        geom = self._geom()
        if geom is None:
            reason = (sm.reason if sm is not None and sm.reason
                      else "no spawn data for this scenario")
            _empty_band(p, pal, QRectF(0, top, width, height - top), reason)
            return
        x0, y0, cw, ch, gap, rows, cols = geom
        origin = self._focus_cell()

        glyph_f = theme.mono(12)
        count_f = theme.mono(12)
        fm = QFontMetricsF(glyph_f)
        gw = max(fm.horizontalAdvance("@"), 4.0)
        gh = max(fm.height() * 0.92, 6.0)

        for disp in range(rows):
            row = rows - 1 - disp
            for col in range(cols):
                key = f"r{row}c{col}"
                if not self._lit(self._delay(
                        motion.grid_distance(row, col, origin))):
                    continue
                cx = x0 + col * (cw + gap)
                cy = y0 + disp * (ch + gap)
                rect = QRectF(cx, cy, cw, ch)
                base = sm.base.get(key, 0)
                if base == 0:
                    # Cannot be emphasised: the generator reuses original
                    # coordinates and never invents them.
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QColor(pal.border))
                    p.drawRoundedRect(rect, 4, 4)
                    continue
                # An untouched layout has no emphasis to draw, and cell_density
                # returns 0 for every cell of one. The counts below still carry
                # the layout; nothing pretends to be a decision.
                density = self.cell_density(key)
                delta = sm.adaptive.get(key, base) - base if sm.adaptive else 0
                if delta > _SPAWN_NOISE:
                    colour = QColor(pal.accent)
                elif delta < -_SPAWN_NOISE:
                    colour = QColor(pal.fg_dim)
                else:
                    colour = QColor(pal.fg)

                back = QColor(colour)
                back.setAlphaF(0.12 if untouched else 0.06 + 0.20 * density)
                p.setPen(Qt.NoPen)
                p.setBrush(back)
                p.drawRoundedRect(rect, 4, 4)
                if untouched:
                    # A hairline so an occupied cell still reads as occupied
                    # without the ramp doing the talking.
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QColor(pal.border))
                    p.drawRoundedRect(rect, 4, 4)

                p.setFont(glyph_f)
                gcols = max(int(cw // gw), 1)
                grows = max(int((ch - 14.0) // gh), 1)
                mx = cx + (cw - gcols * gw) / 2
                my = cy + 3.0
                ch_glyph = _DENSITY_RAMP[int(round(density * (len(_DENSITY_RAMP) - 1)))]
                if ch_glyph != " ":
                    # One call per glyph ROW, not per glyph: a monospace run
                    # lands on exactly the same cells and 25 zones x ~42 glyphs
                    # was 1000+ drawText calls a frame. The jitter moves to the
                    # row's alpha — and it stays on ALPHA, never on the value,
                    # or a zone's density would stop matching its own number.
                    line = ch_glyph * gcols
                    for gr in range(grows):
                        jit = (_cell_noise(disp * 31 + gr, col * 17) - 0.5) * 0.26
                        cq = QColor(colour)
                        cq.setAlphaF(min(max(0.5 + 0.45 * density + jit, 0.0), 1.0))
                        p.setPen(cq)
                        p.drawText(QRectF(mx, my + gr * gh,
                                          gw * (gcols + 1), gh * 1.4),
                                   Qt.AlignLeft | Qt.AlignVCenter, line)

                if cw > 46 and ch > 30:
                    p.setFont(count_f)
                    p.setPen(QColor(pal.fg if key == sm.focus and not untouched
                                    else pal.fg_dim))
                    label = (f"{base}>{sm.adaptive.get(key, 0)}" if sm.adaptive
                             else f"{base}")
                    p.drawText(QRectF(cx + 3, cy + ch - 15, cw - 6, 14),
                               Qt.AlignHCenter | Qt.AlignVCenter, label)
                # No focus ring on an untouched layout: the generator applied no
                # focus there at all (resample_spawns returned an empty set), so
                # an accent ring around the bandit's pick would be the loudest
                # claim on the panel and a false one.
                if self.shows_focus_ring(key):
                    pen = QColor(pal.accent)
                    p.setBrush(Qt.NoBrush)
                    p.setPen(pen)
                    p.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4)


def _cell_noise(a: int, b: int) -> float:
    """Stable per-cell jitter in [0, 1) — the sin hash gui/viz.py uses, so the
    two grids share one texture family."""
    return (math.sin(b * 12.9898 + a * 78.233) * 43758.5453) % 1.0


class FileLedger(_Art):
    """The real numbers, base .sce -> [Adaptive].sce, one key per row.

    Not a reconstruction: the generator always edits the BASE file, so these
    two values are literally the before and the after. Rows reveal as a wave
    over the ledger's own grid of values (row, field).
    """

    TITLE_H = 26.0
    ROW_H = 20.0
    FIELDS = 4
    PER_UNIT = 35          # ~9 units of diagonal -> ~320 ms of propagation

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(settings, parent)
        self._rows: tuple[LedgerRow, ...] = ()
        self._reason = ""
        self.setFixedHeight(int(self.TITLE_H + 3 * self.ROW_H + 12))

    def set_rows(self, rows, reason: str = "") -> None:
        self._rows = tuple(rows)
        self._reason = reason
        if not self._rows:
            self.setFixedHeight(int(self.TITLE_H + 30))
        else:
            self.setFixedHeight(int(self.TITLE_H + len(self._rows) * self.ROW_H + 14))
        self.update()

    def reveal_total(self, settings: Settings) -> int:
        if not self._rows or not motion.animates(settings):
            return 0
        worst = max(self._delay(motion.grid_distance(i, k * 2, (0, 0)), settings)
                    for i in range(len(self._rows)) for k in range(self.FIELDS))
        return worst + motion.GLYPH_MS

    def paintEvent(self, event) -> None:
        pal = theme.current()
        p = QPainter(self)
        p.setRenderHint(QPainter.TextAntialiasing)
        width, height = float(self.width()), float(self.height())
        top = self.TITLE_H          # the header band is drawn per column below
        if not self._rows:
            _empty_band(p, pal, QRectF(0, top, width, height - top),
                        self._reason or "no scenario file to compare")
            return

        key_f = theme.mono(12)
        num_f = theme.mono(14)
        fm_key = QFontMetricsF(key_f)
        fm_num = QFontMetricsF(num_f)
        key_w = min(max([fm_key.horizontalAdvance(r.label) for r in self._rows]),
                    width * 0.45)
        # The column HEADER is part of the column: sized on the values alone,
        # "VARIANT" clipped to "'ARIANT" whenever every variant cell was a dash.
        num_w = max([fm_num.horizontalAdvance(r.base) for r in self._rows]
                    + [fm_num.horizontalAdvance(r.adaptive) for r in self._rows]
                    + [fm_key.horizontalAdvance("VARIANT") + 2.0])
        line_h = max(fm_num.height(), 16.0)
        x_key = 14.0
        x_base = x_key + key_w + 14.0
        x_arrow = x_base + num_w + 8.0
        x_adap = x_arrow + fm_num.horizontalAdvance("->") + 8.0
        x_delta = x_adap + num_w + 14.0
        tones = {"accent": pal.accent, "warn": pal.warn}

        # Column headers over their OWN columns: one centred caption above four
        # measured columns read as a misprint.
        p.setFont(key_f)
        p.setPen(QColor(pal.fg_dim))
        p.drawText(QRectF(x_key, 4, key_w, line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, "KEY")
        p.drawText(QRectF(x_base, 4, num_w, line_h),
                   Qt.AlignRight | Qt.AlignVCenter, "BASE")
        p.drawText(QRectF(x_adap, 4, num_w, line_h),
                   Qt.AlignRight | Qt.AlignVCenter, "VARIANT")
        p.drawText(QRectF(x_delta, 4, max(width - x_delta - 10.0, 40.0), line_h),
                   Qt.AlignLeft | Qt.AlignVCenter, "CHANGE")

        for i, row in enumerate(self._rows):
            y = top + i * self.ROW_H
            fields = (
                (x_key, key_w, row.label, QColor(pal.fg_dim), key_f,
                 Qt.AlignLeft),
                (x_base, num_w, row.base, QColor(pal.fg_dim), num_f,
                 Qt.AlignRight),
                (x_adap, num_w, row.adaptive, QColor(pal.fg), num_f,
                 Qt.AlignRight),
                (x_delta, max(width - x_delta - 10.0, 40.0), row.delta,
                 QColor(tones.get(row.tone, pal.fg_dim)), key_f, Qt.AlignLeft),
            )
            for k, (x, w, text, colour, font, align) in enumerate(fields):
                delay = self._delay(motion.grid_distance(i, k * 2, (0, 0)))
                if not text or not self._lit(delay):
                    continue
                p.setFont(font)
                p.setPen(colour)
                p.drawText(QRectF(x, y, w, line_h), align | Qt.AlignVCenter, text)
            arrow_delay = self._delay(motion.grid_distance(i, 2, (0, 0)))
            if self._lit(arrow_delay):
                p.setFont(num_f)
                p.setPen(QColor(pal.border))
                p.drawText(QRectF(x_arrow, y, 24.0, line_h),
                           Qt.AlignLeft | Qt.AlignVCenter, "->")


# --------------------------------------------------------------------- page
_LADDER_CAPTION = (
    "The rail spans each controller's real clamp range, and always contains the "
    "values it plots. <b>|</b> is the baseline the move is measured from — a "
    "fresh profile's own default, clamped to this archetype's limits where they "
    "are tighter, and for target speed the scenario author's own number read "
    "out of the base <code>.sce</code>. <b>@</b> is where things stand now: for "
    "target speed and spawn mass that is <i>measured back out of the written "
    "<code>[Adaptive] .sce</code></i> whenever one exists, because the emitted "
    "plan is what the game loads and an eased plan writes numbers the un-eased "
    "model does not hold; for the other criteria it is the model's own state. "
    "With no variant on disk every row is the model and the column above says "
    "PLANNED. <b>0</b> means baseline and now coincide and the criterion has "
    "not moved, and the run between them is the move. A move too small for the "
    "rail's resolution "
    "still shows one glyph rather than none, so read the numbers for the "
    "magnitude. A dim run means the value is a cold-start default with no runs "
    "behind it — real in the file, but not learned. Amber marks a value sitting "
    "on its clamp, where the controller has no room left. A dash means the "
    "files cannot support a reading at all.")
_GRID_CAPTION = (
    "One glyph block per wall region, denser where more target spawns land "
    "(three times an even share fills the ramp — an absolute anchor, never "
    "min-max). Counts read base&gt;variant. Accent = the variant put more fire "
    "there than the base; outlined cells hold no spawn point at all in the base "
    "layout, so they can never be emphasised. Where the generator provably "
    "leaves a layout alone — fewer target spawns than grid cells — the ramp and "
    "the focus ring are dropped and the cells carry their counts only, because "
    "there is no emphasis to draw. Row 0 is the bottom of the wall. Hover a "
    "cell.")
_LEDGER_CAPTION = (
    "Read straight out of the two files. The generator always applies the plan "
    "to the BASE .sce and never to the previous variant, so these are literally "
    "the before and the after — not a reconstruction.")


def _clear_layout(layout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


class ChangesView(QWidget):
    """What kovadapt has done to one scenario, and why.

    Constructible as `ChangesView(settings)` with nothing on disk; `refresh()`
    rescans the profile store and `show_scenario(name)` drives it from outside
    (the trailing [Adaptive] suffix is stripped, the same compounding guard the
    CLI applies). The shell mounts it; this file never wires itself in.
    """

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        self.setObjectName("tabPage")           # transparent, per theme.py
        self._entries: list[ProfileEntry] = []
        self._base = ""
        self._profile: PlayerProfile | None = None
        self._facts = SceFacts()
        self._ev = ReportEvidence()
        self._knobs: list[Knob] = []
        self._loading = False

        self.picker = QComboBox()
        self.picker.setMinimumWidth(320)
        self.picker.setToolTip("Scenarios kovadapt has a player model for")
        self.picker.currentIndexChanged.connect(self._picked)
        self.reload_btn = QPushButton("Refresh")
        self.reload_btn.setToolTip("Rescan profiles, run reports and scenario files")
        self.reload_btn.clicked.connect(self.refresh)

        row = QHBoxLayout()
        row.setSpacing(10)
        label = QLabel("Task")
        label.setProperty("dim", True)
        row.addWidget(label)
        row.addWidget(self.picker)
        row.addWidget(self.reload_btn)
        row.addStretch(1)

        self.headline = QLabel("")
        self.headline.setProperty("headline", True)
        self.headline.setWordWrap(True)
        self.subhead = QLabel("")
        self.subhead.setProperty("dim", True)
        self.subhead.setWordWrap(True)

        self.ladder = KnobLadder(settings)
        self.grid = SpawnGrid(settings)
        self.ledger = FileLedger(settings)
        self._art: tuple[_Art, ...] = (self.ladder, self.grid, self.ledger)

        self.ev_box = QVBoxLayout()
        self.ev_box.setSpacing(6)
        self.ev_box.setContentsMargins(0, 4, 0, 0)

        moved_box = QGroupBox("WHAT MOVED, AND ON WHAT EVIDENCE")
        moved_lay = QVBoxLayout(moved_box)
        moved_lay.setSpacing(8)
        moved_lay.addWidget(self.ladder)
        moved_lay.addWidget(_caption(_LADDER_CAPTION))
        moved_lay.addLayout(self.ev_box)

        spawn_box = QGroupBox("WHERE THE SPAWNS WENT")
        spawn_lay = QVBoxLayout(spawn_box)
        spawn_lay.setSpacing(8)
        spawn_lay.addWidget(self.grid)
        self.spawn_note = QLabel("")
        self.spawn_note.setWordWrap(True)
        spawn_lay.addWidget(self.spawn_note)
        spawn_lay.addWidget(_caption(_GRID_CAPTION))

        file_box = QGroupBox("IN THE SCENARIO FILE")
        file_lay = QVBoxLayout(file_box)
        file_lay.setSpacing(8)
        file_lay.addWidget(self.ledger)
        self.provenance = QLabel("")
        self.provenance.setWordWrap(True)
        file_lay.addWidget(self.provenance)
        file_lay.addWidget(_caption(_LEDGER_CAPTION))

        lay = QVBoxLayout(self)
        lay.setSpacing(12)
        lay.addWidget(HintBar(settings, (
            "Every number on this page carries the runs behind it. A criterion "
            "with no evidence yet says so instead of implying a change — and "
            "the file section is read out of the two <code>.sce</code> files on "
            "disk, so it is a measured before/after rather than a guess.")))
        lay.addLayout(row)
        lay.addWidget(self.headline)
        lay.addWidget(self.subhead)
        lay.addWidget(moved_box)
        lay.addWidget(spawn_box)
        lay.addWidget(file_box)
        # The slack gets an explicit home: handed to a panel instead, a fixed
        # art widget stretches and a caption fills a section.
        lay.addStretch(1)

        # One glyph-rate clock for the whole page: three panels revealing on
        # three timers would be three events, and motion.py's rule is that a
        # character-quantised reveal runs at GLYPH_HZ, not the display rate.
        self._clock = QTimer(self)
        self._clock.setInterval(motion.GLYPH_MS)
        self._clock.timeout.connect(self._tick)
        self._t0 = 0.0
        self._total = 0
        self._pending = False

        self.refresh()

    # ------------------------------------------------------------------ API
    @property
    def scenario(self) -> str:
        """The base scenario currently shown ("" when there is none)."""
        return self._base

    def refresh(self) -> None:
        """Rescan the profile store and re-read everything for the selection."""
        keep = self._base or self.picker.currentData() or ""
        self._entries = scan_profiles(self.s.profile_path)
        self._loading = True
        self.picker.clear()
        if self._entries:
            for entry in self._entries:
                runs = f"{entry.runs} run{'s' if entry.runs != 1 else ''}"
                self.picker.addItem(f"{entry.base}  ·  {runs}", entry.base)
            self.picker.setEnabled(True)
        else:
            self.picker.addItem("no player models yet", "")
            self.picker.setEnabled(False)
        self._loading = False
        target = keep if any(e.base == keep for e in self._entries) else (
            self._entries[0].base if self._entries else "")
        if target:
            self.show_scenario(target)
        else:
            self._show_empty()

    def show_scenario(self, name: str) -> None:
        """Render the ledger for `name`. A trailing [Adaptive] is stripped, so
        either spelling of the same task lands on the same page."""
        base = name[:-len(ADAPTIVE_SUFFIX)] if name.endswith(ADAPTIVE_SUFFIX) else name
        base = base.strip()
        if not base:
            self._show_empty()
            return
        index = self.picker.findData(base)
        if index >= 0 and self.picker.currentIndex() != index:
            self._loading = True
            self.picker.setCurrentIndex(index)
            self._loading = False
        self._base = base
        self._load(base)

    def restyle(self, *_pal) -> None:
        for art in self._art:
            art.restyle()
        self._render_text()

    # -------------------------------------------------------------- internals
    def _picked(self, _index: int) -> None:
        if self._loading:
            return
        data = self.picker.currentData()
        if data:
            self.show_scenario(str(data))

    def _show_empty(self) -> None:
        self._base = ""
        self._profile = None
        self._facts = SceFacts()
        self._ev = ReportEvidence()
        self._knobs = []
        self.ladder.set_knobs([])
        self.grid.set_map(None)
        self.ledger.set_rows((), "no scenario selected")
        self._render_text()

    def _load(self, base: str) -> None:
        entry = next((e for e in self._entries if e.base == base), None)
        # Profiles are keyed on base + ADAPTIVE_SUFFIX, but an older file can
        # sit under the bare name; load the exact name whose file was found.
        stored = entry.stored if entry is not None else base + ADAPTIVE_SUFFIX
        profile = PlayerProfile.load(stored, self.s.profile_path)
        ev = read_report_evidence(self.s.profile_path, base)
        facts = read_sce_facts(self.s, base, profile.last_focus)

        spawns = facts.spawns
        if spawns is None:
            spawns = SpawnMap(cols=self.s.region_cols, rows=self.s.region_rows,
                              focus=profile.last_focus,
                              reason=facts.error or "no scenario file to read")
        if not spawns.adaptive and not spawns.reason:
            # No variant on disk: show what the next generation would ask for,
            # labelled as planned rather than written.
            spawns = replace(spawns, planned=planned_weights(profile, self.s))
        facts = replace(facts, spawns=spawns)

        self._profile = profile
        self._facts = facts
        self._ev = ev
        self._knobs = build_knobs(profile, self.s, facts, ev)

        self.ladder.set_knobs(self._knobs)
        self.grid.set_map(spawns)
        # Short panel reason; the detail belongs to the provenance line below
        # it, which has room for a sentence and a tooltip.
        self.ledger.set_rows(facts.rows,
                             "no base scenario file to read" if not facts.have_base
                             else "no [Adaptive] variant on disk yet"
                             if not facts.have_variant
                             else "no editable target sections in this scenario")
        self._render_text()
        self._begin_reveal()

    # ---------------------------------------------------------------- text
    def _render_text(self) -> None:
        """Rebuild every text surface from the cached model.

        Called on load AND on restyle: the evidence lines colour their own
        severity, so they are re-rendered rather than holding a palette.
        """
        pal = theme.current()
        profile, facts = self._profile, self._facts
        self.headline.setText(takeaway(self._knobs, profile, facts))
        self.subhead.setText(self._subhead())
        _clear_layout(self.ev_box)
        for knob in self._knobs:
            self.ev_box.addWidget(_evidence_label(knob, pal))
        spawn_text = self._spawn_text(pal)
        self.spawn_note.setText(spawn_text)
        self.spawn_note.setVisible(bool(spawn_text))
        self.provenance.setText(self._provenance_text(pal))

    def _subhead(self) -> str:
        profile, facts = self._profile, self._facts
        if profile is None:
            return ("No scenario has a player model yet. kovadapt writes one on "
                    "your first completed run, and there is nothing to show "
                    "before then.")
        cells = max(self.s.region_cols * self.s.region_rows, 1)
        ready = profile.readiness(cells)
        arch = profile.archetype or "not stamped yet"
        bits = [f"archetype {arch}", f"{profile.run_count} runs",
                f"calibration {ready['score']:.0%} ({ready['stage']})"]
        if profile.last_run_ts:
            bits.append("last run " + profile.last_run_ts.replace("T", " ")[:16])
        if self._ev.files:
            bits.append(f"{self._ev.files} run reports read from "
                        + " + ".join(self._ev.dirs))
        else:
            bits.append("no run reports on disk — telemetry evidence unavailable")
        out = " · ".join(bits)
        if not profile.archetype:
            out += ("  —  the archetype is stamped on the first run, so the bands "
                    "and clamps below are the clicking defaults")
        if not facts.have_base and facts.error:
            out += f"  —  {facts.error}"
        return out

    def _spawn_text(self, pal) -> str:
        spawns = self._facts.spawns
        if spawns is None or not spawns.base:
            # The panel's own empty band already carries the reason; repeating
            # it verbatim one line lower read as a stutter.
            return ""
        parts = [f"{spawns.total_base} target spawns in the base layout across "
                 f"{len(spawns.base)} of {spawns.cells} regions"]
        if spawns.adaptive:
            parts.append(f"{spawns.total_adaptive} in the variant")
        if spawns.focus:
            where = _region_words(spawns.focus, self.s)
            # A bandit pick the generator could not act on is not a focus the
            # scenario has; say so on the same line rather than one line later.
            parts.append(f"focus {spawns.focus} ({where})"
                         + (" — not applied" if spawns.untouched else ""))
        text = " · ".join(parts)
        if spawns.reason:
            text += (f"<br><span style='color:{pal.warn}'>{spawns.reason}</span>")
        elif not spawns.adaptive:
            # Why there is nothing measured to show, from the same three states
            # the ladder's own clause uses: an unreadable [Adaptive] file is not
            # an absent one, and this line claimed it was.
            why = ("the [Adaptive] file on disk could not be read"
                   if self._facts.variant_on_disk else
                   "no variant has been written yet")
            text += (f"<br><span style='color:{pal.fg_dim}'>densities are the "
                     f"weights the next generation would ask for — {why}</span>")
        return text

    def _provenance_text(self, pal) -> str:
        facts = self._facts
        name = _esc(facts.variant_path.name if facts.variant_path else "")
        if not facts.have_base:
            self.provenance.setToolTip(str(facts.base_path or ""))
            # An [Adaptive] file that IS on disk gets said out loud: the page used
            # to report only the missing base and then claim, five knobs and one
            # headline over, that nothing had ever been written.
            return (f"<span style='color:{pal.fg_dim}'>"
                    f"{_esc(facts.error) or 'no scenario file found'} — kovadapt "
                    "can still show the model, but there is no file to compare "
                    "against"
                    + (f"; <b>{name}</b> is on disk and cannot be read without its "
                       "base." if facts.variant_on_disk else ".")
                    + "</span>")
        # The verbatim header lives in the tooltip, not on the page: every number
        # printed on the page has to be one that applied here, and the raw string
        # carries a static-wall ramp figure even when the multiplier path ran.
        tip = str(facts.variant_path or "")
        if facts.description:
            tip += "\n\nDescription header, verbatim:\n" + facts.description
        self.provenance.setToolTip(tip)
        if not facts.have_variant:
            if facts.variant_on_disk:
                return (f"<span style='color:{pal.warn}'><b>{name}</b> is on disk "
                        "but could not be read, so the column above has nothing to "
                        "compare against and nothing here is verified against the "
                        "game's own copy.</span>")
            return (f"<span style='color:{pal.fg_dim}'>no <b>{name}</b> "
                    "on disk — kovadapt has not written anything for this "
                    "scenario yet, so the column above has nothing to compare "
                    "against.</span>")
        out = (f"<span style='color:{pal.fg_dim}'>variant written "
               f"<b>{_esc(facts.written) or 'unknown'}</b>")
        summary, unapplied = _plan_summary(facts)
        if summary:
            out += f" — {summary}"
        out += "</span>"
        if unapplied:
            out += f"<br><span style='color:{pal.fg_dim}'>{unapplied}</span>"
        check = size_check(self._profile, self.s, facts) if self._profile else ""
        if check:
            stale = "does not match" in check
            out += (f"<br><span style='color:{pal.warn if stale else pal.fg_dim}'>"
                    f"{check}</span>")
        if facts.extra_sections:
            out += (f"<br><span style='color:{pal.fg_dim}'>"
                    f"+{facts.extra_sections} more target/dodge sections are "
                    "edited the same way and not listed here</span>")
        return out

    # -------------------------------------------------------------- motion
    def _begin_reveal(self) -> None:
        total = max((art.reveal_total(self.s) for art in self._art), default=0)
        if total <= 0 or not motion.animates(self.s):
            # Motion off: jump to the end state. Never run a zero-length
            # animation, and never leave a panel half-drawn.
            self._pending = False
            self._clock.stop()
            for art in self._art:
                art.reveal_done()
            return
        self._total = total
        self._pending = True
        self._kick()

    def _kick(self) -> None:
        """Start the reveal, but only while visible — a timer must never run
        behind a hidden page."""
        if not self._pending or not self.isVisible():
            return
        self._pending = False
        # Recomputed here, not just in _begin_reveal: the first load runs from
        # __init__, before the widget has its real width, and the rail's cell
        # count (hence its longest delay) depends on that width.
        self._total = max((art.reveal_total(self.s) for art in self._art), default=0)
        self._t0 = time.monotonic()
        for art in self._art:
            art.set_reveal(0.0)
        self._clock.start()

    def _tick(self) -> None:
        elapsed = (time.monotonic() - self._t0) * 1000.0
        for art in self._art:
            art.set_reveal(elapsed)
        if elapsed >= self._total:
            self._clock.stop()
            # Force the end state rather than trusting the estimate: a cell
            # whose delay outran _total would otherwise stay dark forever.
            for art in self._art:
                art.reveal_done()

    def showEvent(self, event) -> None:
        self._kick()            # a reveal asked for while hidden runs now
        super().showEvent(event)

    def hideEvent(self, event) -> None:
        if self._clock.isActive():
            self._clock.stop()
            self._pending = True    # replay it when the page comes back
        super().hideEvent(event)


# The fields describe() records, in its own order, and what each one did to the
# file. `speed` is deliberately NOT here: it is the absolute static-wall ramp,
# it lands only on base-MaxSpeed-0 characters, and which characters those are is
# a property of the base .sce — so it is rendered separately and gated on the
# path the FILES say applied.
_PLAN_LABELS = (
    ("scale", "target size x{}"),
    ("movement", "movement {} as emitted"),
    ("focus", "focus region requested {}"),
    ("dodge", "dodge skew {}"),
    ("eased", "eased for fatigue {}"),
)


def _plan_fields(description: str) -> dict[str, str]:
    """The `key=value` tokens of a variant's own plan record.

    Only the MIDDLE segment of the header is scanned: the trailing `base: <name>`
    carries a scenario name, and a name holding an `=` would otherwise become a
    plan field that never existed.
    """
    if "kovadapt auto-generated" not in description:
        return {}
    parts = description.split("|")
    return dict(re.findall(r"(\w+)=([^\s|]+)",
                           parts[1] if len(parts) > 1 else description))


def _plan_summary(facts: SceFacts) -> tuple[str, str]:
    """(what the plan record says applied here, what it says did NOT).

    Echoing the Description verbatim was cheaper and wrong. describe() always
    writes `speed=<absolute 0-170 ramp>`, and on a scenario whose targets author
    a MaxSpeed of their own that ramp is never written at all — the generator
    takes the multiplier path, per character. Quoting a stored string is not
    evidence that every number in it landed, and the absolute ramp and the
    authored multiplier are exactly the two things this page may not blur:
    writing the ramp onto a 1300-speed strafe bot collapses the scenario.
    """
    if not facts.description:
        return "it carries no Description header, so it records no plan", ""
    fields = _plan_fields(facts.description)
    if not fields:
        return ("its Description header is not a kovadapt plan record, so nothing "
                "in it can be read as a record of what was written"), ""
    applied = [text.format(_esc(fields[key])) for key, text in _PLAN_LABELS
               if key in fields]
    ramp, tail = fields.get("speed"), ""
    if ramp is not None:
        ramp = _esc(ramp)
        path = facts.speed_path
        if path == "ramp":
            applied.append(f"absolute MaxSpeed ramp {ramp} u/s — every target "
                           "here authors 0, so that is the path that applied")
        elif path == "mixed":
            walls = ", ".join(c for c, v in sorted(facts.authored.items())
                              if v == 0)
            applied.append(f"absolute MaxSpeed ramp {ramp} u/s, written only to "
                           f"the base-speed-0 targets ({_esc(walls)}) — the "
                           "rest were modulated around their own authored speed")
        elif path == "multiplier":
            tail = (f"its record also carries speed={ramp}, the absolute 0-170 "
                    "static-wall ramp. That number was NOT written here: every "
                    "target authors a MaxSpeed of its own, so the generator took "
                    "the multiplier path per character — the MaxSpeed rows above "
                    "are what landed.")
        else:
            tail = (f"its record also carries speed={ramp}, the absolute "
                    "static-wall ramp, but the base .sce cannot say which speed "
                    "path applied here, so whether that number was written is "
                    "unknown.")
    if not applied:
        return "its plan record carries no readable fields", tail
    return ("its own plan record, showing only what applied to this scenario: "
            + ", ".join(applied)), tail


def _plan_record(description: str) -> dict[str, float] | None:
    """The plan a variant records about ITSELF, out of its Description header.

    generate_adaptive_variant writes `AdaptationPlan.describe()` in there, which
    makes the file the only place the FATIGUE EASING of that plan survives:
    `plan(fatigue=...)` eases the emitted values while the persisted profile
    stays un-eased by contract, and the tracker itself is session-scoped and
    never written to disk. Without this header, an eased variant and a stale one
    are indistinguishable — and size_check called every eased variant stale.

    Returns None when the header is not a kovadapt plan record, because then the
    easing is genuinely unknowable and no verdict may be given.
    """
    if "kovadapt auto-generated" not in description:
        return None
    out: dict[str, float] = {}
    for token in ("scale", "movement", "eased"):
        m = re.search(rf"\b{token}=(-?\d+(?:\.\d+)?)", description)
        if m:
            out[token] = float(m.group(1))
    return out if "scale" in out else None


def size_check(profile: PlayerProfile, settings: Settings,
               facts: SceFacts) -> str:
    """Reconcile the size the FILE carries with the size the MODEL holds.

    They are legitimately different numbers: the emitted plan multiplies the
    model's scale by (1 + size_speed_coupling x movement), then by the fatigue
    easing, clipping at each step — so the ladder can read 0.86 while the .sce
    moved by x0.99. A reader will spot that and be right to ask, so the
    arithmetic is shown — and when the two do NOT reconcile, that is said
    instead, because a stale variant is exactly the thing this page exists to
    catch.

    Ignoring the easing made this accuse CURRENT variants of being stale: an
    eased plan writes bigger targets on purpose, so a run finished while tired
    produced an amber "written from an earlier model state" verdict directly
    below the variant's own Description line reading `eased=0.80`.
    """
    if not facts.have_variant or not facts.rows or profile.run_count == 0:
        return ""
    row = next((r for r in facts.rows if r.label.endswith("MainBBRadius")), None)
    base = _num(row.base) if row else None
    now = _num(row.adaptive) if row else None
    if not base or now is None:
        return ""
    s = settings.for_archetype(profile.archetype)
    written = now / base
    coupled = _clamp(profile.target_scale * (1.0 + s.size_speed_coupling
                                             * profile.movement),
                     s.min_target_scale, s.max_target_scale)
    arithmetic = (f"the model's {profile.target_scale:.2f} times the "
                  f"(1 + {s.size_speed_coupling:.2f} x {profile.movement:.2f}) "
                  f"movement coupling = x{coupled:.3f}")
    record = _plan_record(facts.description)
    if record is None:
        # No plan record, so whether that variant was eased is unknowable. State
        # the arithmetic, give no verdict: a staleness claim would be a guess.
        return (f"the size rows moved by x{written:.3f}, against {arithmetic} from "
                "the model. The variant's Description carries no kovadapt plan "
                "record, so there is no way to tell an eased plan from a stale "
                "one and this page will not call it either way.")
    fatigue = _clamp(record.get("eased", 0.0), 0.0, 1.0)
    expected = _clamp(coupled * (1.0 + 0.20 * fatigue),
                      s.min_target_scale, s.max_target_scale)
    if fatigue > 0:
        arithmetic += (f", eased by the fatigue the plan recorded for itself "
                       f"(eased={fatigue:.2f} -> x{expected:.3f}); the profile "
                       "keeps the un-eased state on purpose, so the next session "
                       "resumes true difficulty")
    if abs(written - expected) <= max(0.01 * expected, 0.005):
        return (f"the size rows moved by x{written:.3f}, which is {arithmetic} — "
                "the file and the model reconcile.")
    return (f"the size rows moved by x{written:.3f} but {arithmetic}: the variant "
            "on disk was written from an earlier model state and does not match "
            "the numbers above. Playing a run, or regenerating the variant, "
            "brings them back into step.")


def _no_stretch(label: QLabel) -> None:
    """Stop a label absorbing a layout's spare vertical space WITHOUT losing
    its height-for-width.

    Building a fresh QSizePolicy(Preferred, Maximum) resets the
    height-for-width flag QLabel sets when word wrap is on, so the layout sized
    every wrapped evidence line to one line and they drew on top of each other.
    Mutating the existing policy keeps the flag.
    """
    policy = label.sizePolicy()
    policy.setVerticalPolicy(QSizePolicy.Maximum)
    label.setSizePolicy(policy)


def _caption(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setTextFormat(Qt.RichText)
    lab.setWordWrap(True)
    lab.setProperty("dim", True)
    _no_stretch(lab)
    return lab


def _evidence_label(knob: Knob, pal) -> QLabel:
    """One criterion's because-clause, with its derivation in the tooltip."""
    dot = pal.accent if (knob.measured and knob.moved) else pal.fg_dim
    text = (f"<span style='color:{dot}'>&#9679;</span> "
            f"<b>{knob.name}</b> — {knob.evidence}")
    if knob.note:
        text += f"<br><span style='color:{pal.warn}'>&#8627; {knob.note}</span>"
    lab = QLabel(text)
    lab.setTextFormat(Qt.RichText)
    lab.setWordWrap(True)
    lab.setToolTip(knob.tip or knob.evidence)
    _no_stretch(lab)
    return lab
