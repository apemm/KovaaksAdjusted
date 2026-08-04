"""Per-task sensitivity reasoning: the case both ways, never a directive.

Given the player's mouse DPI and in-game sensitivity (``Settings.mouse_dpi``
/ ``Settings.game_sens``), compute cm/360 and assemble a BOTH-SIDED case:
arguments for LOWER sensitivity (more cm/360) and for HIGHER sensitivity
(less cm/360), each grounded in the player's own numbers from the RunReport
and cited back to analysis/kb.py. Sens-stability doctrine is genuinely
contested (kb GAPS: Voltaic endorses changes, Aimer7 forbids score-inflating
changes yet prescribes temporary ones, pro practice spans both poles) and
per-signal sens attribution is medium-confidence at best — so this module
NEVER emits a directive. It states evidence for both directions, or nothing:
when no side has substantive evidence, ``sens_case`` returns ``None``.

Discipline (mirrors analysis/insights.py):
- every argument line carries its live numbers and a ``kb:`` citation tag;
- numeric cutoffs with no primary source are labeled editorial;
- bad input health suppresses the flick-microstructure arguments.

Pure leaf over (RunReport, PlayerProfile, Settings[, SkillTrends]): no I/O,
no randomness — the same inputs always yield the same case.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..profile.player import PlayerProfile
from . import kb
from .report import RunReport, input_degraded
from .skill import SkillTrends

# KovaaK's (Quake/Source lineage) yaw: degrees turned per mouse count at
# in-game sensitivity 1.0. cm/360 = 2.54 * 360 / (dpi * sens * YAW).
# Defined in `movement` (the leaf) and re-exported here, where it is named:
# two copies of 0.022 is two places for the flick floor and cm/360 to drift
# apart, and they are the same physical fact.
from .movement import YAW_DEG_PER_COUNT  # noqa: E402

# Aimer7's community cm/360 ranges by play style (kb p-sensitivity-doctrine:
# tracking 20-25, click-timing 30+, versatile 21-27). Keyed by kovadapt
# archetype; switching maps to "versatile" because Aimer7 classes target
# switching as the hybrid style. These are HIS ranges, cited as such — not
# rules (Voltaic's broader 20-50 "acceptable" band overlaps all of them).
_STYLE_RANGES: dict[str, tuple[str, float, float | None]] = {
    "clicking": ("click-timing", 30.0, None),
    "tracking": ("tracking", 20.0, 25.0),
    "switching": ("versatile", 21.0, 27.0),
}

# Editorial calibration (kb GAPS: no primary source defines numeric telemetry
# cutoffs). Overshoot/corrections/flick-count/input-health values mirror
# analysis/insights.py so both modules read the same evidence the same way.
_MIN_FLICKS = 8
_OVERSHOOT_HIGH = 0.30
_CORRECTIONS_CHAIN = 2.0
_OVERSHOOT_LOW = 0.15            # "overshoot low" branch of dx-undershoot-slow
_FLICK_SLOW_MS = 220.0           # mean flick duration that reads as cautious/slow
_FITTS_SLOW_MS_PER_BIT = 150.0   # ms/bit that reads as a positive Fitts residual
_TRAVEL_CM_PER_KILL = 12.0       # hand travel per kill that reads as excursion/clutch risk


def cm_per_360_at(dpi: float, deg_per_count: float) -> float:
    """Centimetres of mousepad travel for one full 360-degree turn, given the
    angle a count is actually worth.

    This is the primitive, because `deg_per_count` is the quantity that
    survives a change of game: sens alone is meaningless without the scale it
    is written in. Raises ValueError on non-positive inputs."""
    if dpi <= 0 or deg_per_count <= 0:
        raise ValueError(
            f"dpi and deg/count must be positive (got {dpi}, {deg_per_count})")
    return 2.54 * 360.0 / (dpi * deg_per_count)


def cm_per_360(dpi: float, sens: float) -> float:
    """cm/360 for a sens assumed to be in KovaaK's-NATIVE units.

    Kept for callers that have only the settings pair, and weaker for exactly
    that reason: on this machine the configured 800 dpi at sens 1.0 lands 1.8%
    from the truth (1600 dpi, Valorant 0.16) purely because two wrong inputs
    cancel in this one formula. Prefer `cm_per_360_at` with a run-derived
    angle."""
    if sens <= 0:
        raise ValueError(f"dpi and sens must be positive (got {dpi}, {sens})")
    return cm_per_360_at(dpi, sens * YAW_DEG_PER_COUNT)


@dataclass(frozen=True)
class SensCase:
    """Both-sided sensitivity evidence. Never a directive: consumers must
    render BOTH sides (or neither) — a side with no evidence is rendered as
    "no evidence points that way", never dropped to leave one imperative."""

    cm360: float
    style_range: tuple | None       # (style_label, lo_cm, hi_cm | None) — Aimer7's ranges
    for_lower: list[str]            # evidence FOR lower sens (more cm/360)
    for_higher: list[str]           # evidence FOR higher sens (less cm/360)
    neutral: str                    # side-free framing (always present)
    sources: tuple[str, ...] = ()   # merged kb citations for every line above


def _range_str(lo: float, hi: float | None) -> str:
    return f"{lo:.0f}+ cm/360" if hi is None else f"{lo:.0f}-{hi:.0f} cm/360"


def _dedup(xs: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(xs))


def sens_case(
    rep: RunReport,
    profile: PlayerProfile,
    settings: Settings,
    trends: SkillTrends | None = None,
) -> SensCase | None:
    """Build the both-sided case, or ``None`` when sensitivity is not
    configured (dpi/sens <= 0) or no side has substantive evidence."""
    # THE RUN FIRST. `rep` carries the dpi and the angle-per-count KovaaK's
    # itself recorded for this run; the settings pair is a fallback that
    # assumes the configured sens is in KovaaK's-native units. Both wrong here
    # (800/1.0 against a real 1600/Valorant-0.16) and cm/360 still came out
    # 1.8% right, because the errors cancel in this formula and nowhere else.
    dpi = float(getattr(rep, "mouse_dpi", 0.0) or 0.0) or float(settings.mouse_dpi)
    per_count = float(getattr(rep, "deg_per_count", 0.0) or 0.0)
    if per_count <= 0:
        per_count = float(settings.game_sens) * YAW_DEG_PER_COUNT
    if dpi <= 0 or per_count <= 0:
        return None
    cm360 = cm_per_360_at(dpi, per_count)
    sens = per_count / YAW_DEG_PER_COUNT      # KovaaK's-native equivalent
    mm_per_deg = 10.0 * cm360 / 360.0    # hand travel cost of one degree of aim

    style = _STYLE_RANGES.get(profile.archetype or "")
    eff = settings.for_archetype(profile.archetype)

    # Shared gate (report.input_degraded) — this file used to re-derive it
    # from its own copy of the thresholds, one call away from insights.py's.
    has_flicks = rep.n_flicks >= _MIN_FLICKS and not input_degraded(rep)

    lower: list[str] = []
    higher: list[str] = []
    src: list[str] = list(kb.principle("p-sensitivity-doctrine")["sources"])

    # ---- FOR LOWER (more cm/360) ------------------------------------------
    # Sustained overshoot with correction chains: the CD-gain mechanism.
    if (has_flicks and rep.overshoot_rate > _OVERSHOOT_HIGH
            and rep.mean_corrections >= _CORRECTIONS_CHAIN):
        n_over = round(rep.overshoot_rate * rep.n_flicks)
        lower.append(
            f"{rep.overshoot_rate:.0%} of {rep.n_flicks} flicks overshot (~{n_over} flicks, each "
            "past the target by >10% of its own amplitude — kovadapt's segmentation cutoff) with "
            f"{rep.mean_corrections:.1f} corrective submovements per flick: a hypermetric "
            "ballistic phase plus repair chain, not a swipe. Higher control-display gain "
            "increases exactly this overshooting, worst on small/far targets, and at your "
            f"{cm360:.1f} cm/360 every degree overshot costs {mm_per_deg:.1f} mm of hand travel "
            "to undo (kb: dx-overshoot-control — Casiez & Vogel 2008; sens attribution medium)."
        )
        src += list(kb.diagnostic("dx-overshoot-control")["sources"])

    # Overshoot rising across sessions: the persists-across-sessions branch.
    if trends is not None:
        over_t = trends.overall.get("overshoot_rate")
        if over_t is not None and over_t.classification == "declining":
            lower.append(
                f"Overshoot rose {abs(over_t.rel_change):.0%} across your last {over_t.n} saved "
                f"runs (Theil-Sen {over_t.slope:+.4f}/run) — the persists-across-sessions "
                "pattern under which the sourced playbook considers a modest sens reduction "
                "worth weighing (kb: dx-overshoot-control — CD-gain research + community rule, "
                "sens attribution medium)."
            )
            src += list(kb.diagnostic("dx-overshoot-control")["sources"])

    # ---- FOR HIGHER (less cm/360) -----------------------------------------
    # Accurate but slow at amplitude: hypometric creep / positive Fitts residual.
    slow_flick = rep.mean_flick_ms >= _FLICK_SLOW_MS
    slow_fitts = rep.fitts_slope_ms >= _FITTS_SLOW_MS_PER_BIT
    if (has_flicks and (slow_flick or slow_fitts)
            and rep.overshoot_rate <= _OVERSHOOT_LOW
            and rep.accuracy >= eff.target_accuracy_low):
        bits = []
        if slow_flick:
            bits.append(f"mean flick {rep.mean_flick_ms:.0f} ms")
        if slow_fitts:
            bits.append(f"Fitts slope {rep.fitts_slope_ms:.0f} ms per bit of distance")
        higher.append(
            f"{' and '.join(bits)} with only {rep.overshoot_rate:.0%} overshoot at "
            f"{rep.accuracy:.0%} accuracy — accurate but slow, the hypometric-creep pattern "
            f"(positive Fitts residual; cutoffs {_FLICK_SLOW_MS:.0f} ms and "
            f"{_FITTS_SLOW_MS_PER_BIT:.0f} ms/bit are editorial). Very low control-display gain "
            "costs time through clutching and limb-speed limits, and at your "
            f"{cm360:.1f} cm/360 a 90-degree flick demands {cm360 / 4.0:.1f} cm of hand travel "
            "(kb: dx-undershoot-slow — Casiez & Vogel 2008 clutching cost)."
        )
        src += list(kb.diagnostic("dx-undershoot-slow")["sources"])

    # Large hand excursions per kill: clutching risk at low gain. Gross count
    # sums, not flick microstructure, so not gated on input health.
    travel_cm = rep.total_travel_counts / dpi * 2.54
    per_kill = travel_cm / rep.kills if rep.kills > 0 else 0.0
    if per_kill >= _TRAVEL_CM_PER_KILL:
        higher.append(
            f"Your hand travelled {travel_cm:.0f} cm over {rep.kills} kills — {per_kill:.1f} cm "
            f"per kill ({rep.total_travel_counts:.0f} counts / {dpi:.0f} dpi x 2.54; the "
            f"{_TRAVEL_CM_PER_KILL:.0f} cm cutoff is editorial). Excursions this large invite "
            "clutching, and clutching is the measured time cost of very low CD gain "
            "(kb: dx-undershoot-slow / p-sensitivity-doctrine — Casiez & Vogel 2008)."
        )
        src += list(kb.diagnostic("dx-undershoot-slow")["sources"])

    # ---- Style range (Aimer7's, cited as his) -----------------------------
    if style is not None:
        label, lo_cm, hi_cm = style
        rng = _range_str(lo_cm, hi_cm)
        if cm360 < lo_cm:
            lower.append(
                f"Your {cm360:.1f} cm/360 is {lo_cm - cm360:.1f} cm faster than Aimer7's "
                f"{label} range ({rng}) — his community ranges, not a law, and Voltaic calls "
                "20-50 acceptable (kb: p-sensitivity-doctrine — Aimer7 sec 3)."
            )
        elif hi_cm is not None and cm360 > hi_cm:
            higher.append(
                f"Your {cm360:.1f} cm/360 is {cm360 - hi_cm:.1f} cm slower than the top of "
                f"Aimer7's {label} range ({rng}) — his community ranges, not a law, and Voltaic "
                "calls 20-50 acceptable (kb: p-sensitivity-doctrine — Aimer7 sec 3)."
            )

    if not lower and not higher:
        return None

    if style is not None:
        label, lo_cm, hi_cm = style
        if cm360 < lo_cm:
            pos = "below (faster than)"
        elif hi_cm is not None and cm360 > hi_cm:
            pos = "above (slower than)"
        else:
            pos = "inside"
        range_note = (
            f" Aimer7's {label} range is {_range_str(lo_cm, hi_cm)}; yours sits {pos} it — his "
            "ranges, cited as such, not a rule."
        )
    else:
        range_note = ""
    neutral = (
        f"At {dpi:g} dpi and {per_count:.4f} degrees of turn per mouse count "
        f"(KovaaK's-native sens {sens:.3g}) you "
        f"turn 360 degrees in {cm360:.1f} cm of mousepad.{range_note} Performance vs sensitivity "
        "is U-shaped with a broad usable middle, and whether to change sens at all is genuinely "
        "contested (s1mple held one sens for years; TenZ tweaks constantly) — so the evidence is "
        "argued both ways and no direction is recommended (kb: p-sensitivity-doctrine + GAPS)."
    )

    return SensCase(cm360=cm360, style_range=style, for_lower=lower,
                    for_higher=higher, neutral=neutral, sources=_dedup(src))


#: Degrees turned per mouse count at in-game sensitivity 1.0, per sens SCALE.
#: KovaaK's lets you enter sensitivity in another game's units and records
#: which in every stats file ("Sens Scale:"), so the number a player types
#: means nothing without it — Valorant 0.16 and KovaaK's 0.16 differ by 3.2x.
#:
#: This app had no concept of scale at all: it applied the Source-lineage
#: 0.022 unconditionally against a `game_sens` that defaults to 1.0 and that
#: nothing ever asks for. All 398 real stats files here read "Valorant" at
#: 0.16, so every angular claim was computed 1.96x out.
#:
#: Keys are lowercased KovaaK's scale strings. Only scales whose yaw is
#: established go in — an unknown scale returns 0.0 from `deg_per_count` and
#: the caller says so rather than guessing.
YAW_BY_SCALE: dict[str, float] = {
    "kovaak's": YAW_DEG_PER_COUNT,
    "kovaaks": YAW_DEG_PER_COUNT,
    "quake": YAW_DEG_PER_COUNT,          # the lineage 0.022 comes from
    "source": YAW_DEG_PER_COUNT,
    "cs:go": YAW_DEG_PER_COUNT,
    "csgo": YAW_DEG_PER_COUNT,
    "cs2": YAW_DEG_PER_COUNT,
    "apex legends": YAW_DEG_PER_COUNT,
    "apex": YAW_DEG_PER_COUNT,
    "valorant": 0.07,
}


def deg_per_count(run=None, settings=None) -> tuple[float, str]:
    """(degrees of view turn per mouse count, where the number came from).

    THE RUN WINS. KovaaK's writes "Sens Scale:", "Horiz Sens:" and "DPI:" into
    every stats file, so each run carries the sensitivity it was actually
    played at — authoritative, per run, and already in `Run.summary`. A
    settings field is a thing a user might have set once and never revisited;
    on this machine it never got set at all, while the sensitivity behind the
    398 recorded runs moved across five values.

    Sources, strongest first: "run" (the game's own record), "settings" (the
    configured value, assumed to be KovaaK's-native scale), and "" with 0.0
    when neither can be resolved — an unknown SCALE included, because a
    sensitivity number without its scale is not a quantity.
    """
    if run is not None:
        summary = getattr(run, "summary", None) or {}
        scale = str(summary.get("Sens Scale:", "") or "").strip().lower()
        try:
            sens = float(summary.get("Horiz Sens:", "") or 0.0)
        except (TypeError, ValueError):
            sens = 0.0
        if sens > 0 and scale in YAW_BY_SCALE:
            return YAW_BY_SCALE[scale] * sens, "run"
        if sens > 0 and scale:
            return 0.0, ""      # a scale we cannot convert is not a guess
    if settings is not None:
        sens = float(getattr(settings, "game_sens", 0) or 0)
        if sens > 0:
            return YAW_DEG_PER_COUNT * sens, "settings"
    return 0.0, ""


def min_flick_counts_for(run=None, settings=None) -> float:
    """The flick floor in COUNTS for the run that produced this telemetry.

    Falls back to the sens-1.0 reference when the scale cannot be resolved:
    the reference is a worse answer than a converted one and a much better
    answer than no floor at all, and `flick_floor_deg` on the report records
    which it was.
    """
    from .movement import MIN_FLICK_COUNTS, MIN_FLICK_DEG

    per_count, _src = deg_per_count(run, settings)
    return MIN_FLICK_DEG / per_count if per_count > 0 else MIN_FLICK_COUNTS


def min_flick_counts(settings) -> float:
    """The flick-amplitude floor in mouse counts for THIS player's mouse.

    Settings-only, and therefore the weaker path — prefer
    `min_flick_counts_for(run, settings)`, which reads the sensitivity the run
    was actually played at out of the game's own record. This one assumes the
    configured sens is in KovaaK's-native units, which is exactly the
    assumption that made the v0.5.2 floor 1.96x wrong on this machine.

    The floor means an ANGLE — below about a degree the overshoot ratio stops
    measuring aim and starts measuring segmentation error — and the angle a
    count is worth is `YAW_DEG_PER_COUNT * sens`.

    NOT DPI. A count is a count: DPI decides how many counts a centimetre of
    desk produces, not how far the view turns for one of them. So the floor
    moves with in-game sensitivity alone, and a 1600 DPI player at sens 1.0
    gets exactly the same count floor as an 800 DPI player at sens 1.0 — they
    just reach it with less hand movement.

    `movement.MIN_FLICK_COUNTS` is the sens-1.0 reference, returned unchanged
    when sens is unconfigured: without it there is nothing to convert with,
    and the reference is a better answer than no floor at all.
    """
    from .movement import MIN_FLICK_COUNTS, MIN_FLICK_DEG

    sens = float(getattr(settings, "game_sens", 0) or 0)
    per_count = YAW_DEG_PER_COUNT * sens
    return MIN_FLICK_DEG / per_count if per_count > 0 else MIN_FLICK_COUNTS
