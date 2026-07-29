"""The local coach: deterministic insights from a run + profile, grounded in
the cited knowledge base (analysis/kb.py).

Every Insight carries its full chain of custody — the measured signal and
threshold that fired (`reasoning`, with the live numbers), the sourced
interpretation and prescription (verbatim KB text), the KB entry id, its
confidence, and its citations. Nothing is forced on the player: output is
evidence plus suggestion, thresholds that are kovadapt's own calibration are
labeled as such, and when input health is bad the flick-microstructure
diagnoses are suppressed outright rather than reported on noisy data.

Pure function of (RunReport, PlayerProfile, Settings): no file or network
I/O, no randomness — the same run always yields the same insights.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Settings
from ..profile.player import PlayerProfile
from . import kb
from .report import RunReport
from .sens import SensCase, sens_case
from .skill import SkillTrends

# kovadapt's own calibration cutoffs (labeled "editorial" in reasoning text
# wherever the KB provides no primary-source number).
_OVERSHOOT_HIGH = 0.30
_CORRECTIONS_CHAIN = 2.0     # mean corrections/flick that reads as a repair chain
_CORRECTIONS_CLEAN = 1.0
_BIAS_SUSTAINED = 0.15
_REGION_DEFICIT = 0.15
_JITTER_BAD_MS = 2.0
_POLLING_LOW_HZ = 490.0      # below any competitive polling class
_MIN_FLICKS = 8              # too few flicks = no microstructure claims
_CROSS_SESSION_MIN_RUNS = 15  # runs before a cross-session progress claim


@dataclass(frozen=True)
class Insight:
    id: str                  # KB entry id (diagnostic or principle) this is grounded in
    kind: str                # "diagnosis" | "positive" | "health" | "progress" | "info"
    severity: str            # "info" | "attention" | "warning"
    title: str
    body: str                # sourced interpretation (KB text)
    reasoning: str           # live numbers + the condition that fired
    prescription: str        # sourced suggestion (KB text) — never a command
    confidence: str
    sources: tuple[str, ...]


def _from_kb(did: str, kind: str, severity: str, title: str, reasoning: str) -> Insight:
    d = kb.diagnostic(did)
    return Insight(
        id=did, kind=kind, severity=severity, title=title,
        body=d["interpretation"], reasoning=reasoning,
        prescription=d["prescription"], confidence=d["confidence"],
        sources=tuple(d["sources"]),
    )


def _recent(profile: PlayerProfile, key: str, n: int) -> list[float]:
    return [float(h.get(key, 0.0)) for h in profile.history[-n:]]


def generate_insights(
    rep: RunReport, profile: PlayerProfile, settings: Settings,
    *, trends: SkillTrends | None = None,
) -> list[Insight]:
    """All insights for one run, most actionable first.

    ``trends`` (optional) is a cross-session ``SkillTrends`` fit over the
    saved report history (analysis/skill.py); when provided it can add
    cross-session cards on top of the within-session rules."""
    out: list[Insight] = []
    eff = settings.for_archetype(profile.archetype)
    arche = profile.archetype or "clicking"
    has_flicks = rep.n_flicks >= _MIN_FLICKS

    # ---- input health gates everything microstructural -------------------
    jitter = float(rep.input_health.get("jitter_ms", 0.0) or 0.0)
    polling = float(rep.input_health.get("polling_hz_est", 0.0) or 0.0)
    input_bad = jitter > _JITTER_BAD_MS or (0.0 < polling < _POLLING_LOW_HZ)
    if input_bad:
        out.append(_from_kb(
            "dx-input-health", "health", "warning", "Input health is degrading the data",
            f"Timing jitter {jitter:.1f} ms"
            + (f", estimated polling {polling:.0f} Hz" if polling else "")
            + f" (kovadapt's cutoffs — jitter > {_JITTER_BAD_MS:.0f} ms or polling "
              "below the competitive class — are editorial calibration; no primary "
              "source defines numeric limits). Flick-microstructure diagnoses for "
              "this run are suppressed rather than reported on noisy data."))

    # ---- accuracy vs the archetype band ----------------------------------
    lo, hi = eff.target_accuracy_low, eff.target_accuracy_high
    band_note = (
        "the 85–95% band is primary-sourced for clicking; kovadapt's "
        f"{arche} band is an extrapolation of the same control law"
        if arche != "clicking" else "primary-sourced band"
    )
    accs = _recent(profile, "accuracy", 3)
    if len(accs) >= 3 and all(a > hi for a in accs):
        out.append(_from_kb(
            "dx-acc-above-band", "diagnosis", "attention", "Accuracy parked above the band",
            f"Last {len(accs)} runs all above the {hi:.0%} ceiling for the "
            f"{arche} band ({band_note}); this run {rep.accuracy:.0%}."))
    elif len(accs) >= 3 and all(a < lo for a in accs):
        out.append(_from_kb(
            "dx-acc-below-band", "diagnosis", "attention", "Accuracy below the band floor",
            f"Last {len(accs)} runs all under the {lo:.0%} floor for the "
            f"{arche} band ({band_note}); this run {rep.accuracy:.0%}."))

    # ---- overshoot: control failure vs deliberate speed ------------------
    if has_flicks and not input_bad:
        in_band = lo <= rep.accuracy <= hi
        if rep.overshoot_rate > _OVERSHOOT_HIGH and rep.mean_corrections >= _CORRECTIONS_CHAIN:
            out.append(_from_kb(
                "dx-overshoot-control", "diagnosis", "attention",
                "Overshooting, then repairing",
                f"{rep.overshoot_rate:.0%} of {rep.n_flicks} flicks overshot AND "
                f"{rep.mean_corrections:.1f} corrective submovements per flick "
                f"(cutoffs {_OVERSHOOT_HIGH:.0%} / {_CORRECTIONS_CHAIN:.0f} are "
                "editorial calibration of the sourced pattern)."))
        elif (rep.overshoot_rate > _OVERSHOOT_HIGH
              and rep.mean_corrections <= _CORRECTIONS_CLEAN and in_band):
            out.append(_from_kb(
                "dx-overshoot-strategic", "positive", "info",
                "Overshoot without repair — likely a speed strategy",
                f"{rep.overshoot_rate:.0%} flicks overshot but only "
                f"{rep.mean_corrections:.1f} corrections per flick with accuracy "
                f"{rep.accuracy:.0%} inside the band. (kovadapt cannot yet see "
                "shot timing along the flick, so the swipe pattern is inferred "
                "from the correction profile — a known approximation.)"))

        # ---- archetype-specific correction profiles ----------------------
        if arche == "tracking" and rep.mean_corrections > _CORRECTIONS_CHAIN:
            out.append(_from_kb(
                "dx-tracking-jitter", "diagnosis", "attention", "Tracking is jittery",
                f"{rep.mean_corrections:.1f} corrective submovements per flick in a "
                f"tracking scenario (editorial cutoff {_CORRECTIONS_CHAIN:.0f})."))
        if arche == "switching" and rep.mean_corrections > _CORRECTIONS_CLEAN:
            out.append(_from_kb(
                "dx-switch-corrections", "diagnosis", "attention",
                "Switches need a correction tax",
                f"{rep.mean_corrections:.1f} corrections per acquisition vs the "
                "sourced ideal of landing the first flick clean (≈0)."))

    # ---- directional bias -------------------------------------------------
    if abs(profile.ewma_bias) > _BIAS_SUSTAINED and profile.run_count >= 5:
        weak = "left" if profile.ewma_bias > 0 else "right"
        out.append(_from_kb(
            "dx-bias", "diagnosis", "attention", f"Your {weak} side is weaker",
            f"Directional bias EWMA {profile.ewma_bias:+.2f} over {profile.run_count} "
            f"runs (threshold {_BIAS_SUSTAINED} is editorial; the drill is sourced, "
            "prevalence claims are anecdotal). kovadapt is already skewing strafes "
            f"toward your {weak} side."))

    # ---- region deficits --------------------------------------------------
    worst_key, worst_mean = "", 0.0
    for key, post in profile.regions.items():
        if post.n >= 2 and post.mean > max(worst_mean, _REGION_DEFICIT):
            worst_key, worst_mean = key, post.mean
    if worst_key:
        out.append(_from_kb(
            "dx-region-deficit", "diagnosis", "attention",
            f"Weakest wall region: {_region_words(worst_key, settings)}",
            f"Region {worst_key} posterior deficit {worst_mean:+.2f} "
            f"(n={profile.regions[worst_key].n}); spawns are already being "
            "resampled toward it."))

    # ---- fatigue ----------------------------------------------------------
    level = (rep.fatigue or {}).get("level", "")
    if level and level != "fresh":
        out.append(_from_kb(
            "dx-fatigue", "health", "warning", f"Session fatigue: {level}",
            f"Theil-Sen trend over {rep.fatigue.get('runs', 0)} runs shows overshoot "
            "and flick duration worsening together (the composite detector is "
            "kovadapt's own construct; the stop-when-tired doctrine is sourced)."))

    # ---- progress framing: speed is the growth axis -----------------------
    scores = _recent(profile, "score", 10)
    kpss = _recent(profile, "kps", 10)
    if len(scores) >= 10:
        half = len(scores) // 2
        score_flat = abs(_mean(scores[half:]) - _mean(scores[:half])) \
            <= 0.05 * max(_mean(scores[:half]), 1e-9)
        kps_up = _mean(kpss[half:]) > 1.05 * max(_mean(kpss[:half]), 1e-9)
        if score_flat and kps_up:
            out.append(_from_kb(
                "dx-fitts-progress", "progress", "info",
                "Score plateau, but you are getting faster",
                f"Mean score flat across your last {len(scores)} runs "
                f"({_mean(scores[:half]):.0f} → {_mean(scores[half:]):.0f}) while "
                f"pace rose {_mean(kpss[:half]):.2f} → {_mean(kpss[half:]):.2f} "
                "kills/s. Speed is the axis long-term improvement shows up on."))

    # ---- cross-session trends (saved report history) ----------------------
    if trends is not None:
        out.extend(_cross_session_insights(trends))

    # ---- sensitivity: the case both ways (never a directive) --------------
    if settings.mouse_dpi > 0 and settings.game_sens > 0:
        case = sens_case(rep, profile, settings, trends=trends)
        if case is not None and (case.for_lower or case.for_higher):
            out.append(_sens_insight(case, settings))

    return out


def _sens_insight(case: SensCase, settings: Settings) -> Insight:
    """Composite card over p-sensitivity-doctrine plus the diagnostics that
    contributed lines (merged sources come with the SensCase). Both sides
    are ALWAYS rendered — an evidence-free side is stated as such, never
    dropped — so the card can never read as a one-way directive."""
    lower = " ".join(case.for_lower) or (
        "nothing in this run's evidence argues that way.")
    higher = " ".join(case.for_higher) or (
        "nothing in this run's evidence argues that way.")
    body = (
        f"The case for LOWER sensitivity (more cm/360): {lower} "
        f"The case for HIGHER sensitivity (less cm/360): {higher} "
        f"{case.neutral}"
    )
    reasoning = (
        f"cm/360 = 2.54 x 360 / (dpi x sens x 0.022) = 2.54 x 360 / "
        f"({settings.mouse_dpi:g} x {settings.game_sens:g} x 0.022) = {case.cm360:.2f} cm. "
        f"{len(case.for_lower)} evidence line(s) argue for lower and {len(case.for_higher)} "
        "for higher, each carrying its own live numbers and kb citation; every numeric cutoff "
        "without a primary source is labeled editorial (kb GAPS)."
    )
    prescription = (
        "No direction is recommended — the evidence argues both ways and sens-stability "
        "doctrine is contested: Voltaic endorses sens changes for practice and says not to "
        "obsess over it; Aimer7 forbids changing settings to inflate a score yet prescribes "
        "temporary sens changes in his own protocols; pro practice spans both poles (s1mple "
        "vs TenZ). If you do trial a change, change one variable and judge it by averages "
        "over runs, not single scores."
    )
    return Insight(
        id="p-sensitivity-doctrine", kind="info", severity="info",
        title="Your sensitivity: the case both ways",
        body=body, reasoning=reasoning, prescription=prescription,
        confidence=("high (cm/360 math, Aimer7/Voltaic ranges); medium (per-signal sens "
                    "attribution); contested (whether to change at all)"),
        sources=case.sources,
    )


def _cross_session_insights(trends: SkillTrends) -> list[Insight]:
    """Cards that need the whole saved-report history, not one run."""
    out: list[Insight] = []
    o = trends.overall
    fitts_t, kps_t = o.get("fitts_slope_ms"), o.get("kps")
    score_t, over_t = o.get("score"), o.get("overshoot_rate")

    # Speed improving across >= _CROSS_SESSION_MIN_RUNS runs under a flat
    # score trend: the cross-session form of dx-fitts-progress.
    driver = None
    for cand in (fitts_t, kps_t):
        if (cand is not None and cand.classification == "improving"
                and cand.n >= _CROSS_SESSION_MIN_RUNS):
            driver = cand
            break
    if driver is not None and score_t is not None and score_t.classification == "flat":
        if driver.metric == "fitts_slope_ms":
            what = (f"flick time per bit of distance fell {abs(driver.rel_change):.0%} "
                    f"(Theil-Sen {driver.slope:+.2f} ms/bit per run) over "
                    f"{driver.n} runs")
        else:
            what = (f"kill pace rose {abs(driver.rel_change):.0%} "
                    f"(Theil-Sen {driver.slope:+.3f} kills/s per run) over "
                    f"{driver.n} runs")
        out.append(_from_kb(
            "dx-fitts-progress", "progress", "info",
            "Cross-session: improving under a flat scoreboard",
            f"Across your saved reports, {what} while the score trend stayed "
            f"flat ({score_t.rel_change:+.0%} over {score_t.n} runs). Robust "
            "trends over run history, not PBs."))

    # Overshoot rising across sessions (not just within one run/session).
    if over_t is not None and over_t.classification == "declining":
        out.append(_from_kb(
            "dx-overshoot-control", "diagnosis", "attention",
            "Overshoot is rising across sessions",
            f"Overshoot rate rose {abs(over_t.rel_change):.0%} across "
            f"{over_t.n} saved runs (Theil-Sen {over_t.slope:+.4f}/run) — a "
            "sustained cross-session pattern, not one bad day; the "
            "persists-across-sessions branch of the sourced prescription "
            "applies."))
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _region_words(key: str, settings: Settings) -> str:
    """r{row}c{col} -> plain words ('upper left', 'center', ...)."""
    try:
        r, c = key[1:].split("c")
        r, c = int(r), int(c)
    except ValueError:
        return key
    rows, cols = settings.region_rows, settings.region_cols
    vert = "upper" if r >= rows - max(rows // 3, 1) else \
        ("lower" if r < max(rows // 3, 1) else "middle")
    horiz = "left" if c < max(cols // 3, 1) else \
        ("right" if c >= cols - max(cols // 3, 1) else "center")
    return "center" if (vert, horiz) == ("middle", "center") else f"{vert} {horiz}"
