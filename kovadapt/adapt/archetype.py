"""Scenario archetype detection: clicking vs tracking vs target-switching.

Different archetypes need different adaptation dynamics (see
config.default_archetype_overrides). Detection is name-keyword first
(scenario authors are consistent about naming), stats-heuristic second,
"clicking" as the safe default. The result is cached on the PlayerProfile
so one detection sticks for the scenario's lifetime.
"""

from __future__ import annotations

from ..stats.models import Run

# Lowercase substrings of scenario names, checked in order. Switching is
# checked before tracking because hybrid names ("tracking switch") are
# switching-style tasks.
_NAME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("switching", ("switch", " ts ", "psalmts", "voltaic ts", "pokeball",
                   "bounce", "multi", "reflex flick")),
    ("tracking", ("track", "smoothbot", "controlsphere", "air ", " air",
                  "thin gauntlet", "ground plaza", "fuglaa", "orbit",
                  "close strafes", "follow")),
)

# Stats heuristic: tracking weapons register a hit/miss per damage tick, so
# shots-per-kill runs an order of magnitude above clicking scenarios. There is
# no comparably reliable stats signature for switching (clicking runs also
# show low TTK at high pace), so switching is keyword-only.
_TRACKING_SHOTS_PER_KILL = 20.0

# Pure-tracking scenarios use INVINCIBLE targets: nothing ever dies, so the
# CSV reports Kills: 0 and shots-per-kill is undefined. Guarding the ratio on
# kill_count > 0 therefore skipped the heuristic for exactly the scenarios it
# exists to catch, and they fell through to "clicking" — where their
# structurally lower per-tick accuracy reads as permanently below the clicking
# band and grows targets every run. Sustained hits with zero kills is instead
# the STRONGEST tracking signal available: a clicking hit kills, so a clicking
# run cannot land hundreds of hits and kill nothing.
# Calibrated on 392 real stats files: of 162 zero-kill runs, the lowest hit
# count was 182 and none had zero hits — so this threshold has a wide margin
# on both sides, and a genuine no-shot (AFK) run still scores 0 and is left
# to the caller's minimum-evidence gate.
_TRACKING_MIN_HITS = 30


#: Evidence behind an archetype, weakest first. A stamp may be replaced by
#: one of a STRONGER kind and never by a weaker or equal one, which is what
#: keeps a correction to exactly one step and stops it flapping run to run.
EVIDENCE = ("default", "name", "stats")


def detect_archetype(scenario_name: str, run: Run | None = None) -> str:
    """Best-effort archetype for a scenario. Never raises; defaults to
    "clicking" when the evidence is thin."""
    return classify_archetype(scenario_name, run)[0]


def classify_archetype(scenario_name: str,
                       run: Run | None = None) -> tuple[str, str]:
    """(archetype, which EVIDENCE produced it).

    The source is the load-bearing half. `detect_archetype` alone cannot tell
    "clicking because the stats say so" from "clicking because we had nothing
    to go on", and every caller latched the answer behind `if not
    profile.archetype`. Two of the three call sites — the browser's Start
    adapting and `kovadapt generate` — run BEFORE any run exists, so they
    stamp a name-only guess permanently and the stats heuristic never gets a
    turn.

    Measured on the real 95-scenario library here: 31 scenarios come out
    `clicking` from the name and `tracking` from their own stats — Whisphere,
    WhisphereRawControl, waldoTS, cloverRawControl, the Polarized Hell set.
    Every one of them would have been scored against an accuracy band its
    invincible targets can never reach, forever, from one pre-run click.
    """
    name = f" {scenario_name.lower()} "
    for arch, keywords in _NAME_KEYWORDS:
        if any(k in name for k in keywords):
            return arch, "name"
    if run is not None:
        if run.kill_count > 0:
            shots = run.hit_count + run.miss_count
            if shots / run.kill_count >= _TRACKING_SHOTS_PER_KILL:
                return "tracking", "stats"
        elif run.hit_count >= _TRACKING_MIN_HITS:
            return "tracking", "stats"   # invincible targets: ticks, no deaths
        # A run that does NOT trip the heuristic is real evidence of clicking,
        # not an absence of evidence — that distinction is the whole point.
        return "clicking", "stats"
    return "clicking", "default"


def stamp_archetype(profile, scenario_name: str, run: Run | None = None):
    """Record the archetype on `profile` when this evidence beats what is
    already there. Returns (old, new) if it changed, else None.

    ONE rule, four call sites. `if not profile.archetype` was repeated at each
    of them, which made any first stamp permanent — and two of the four run
    before the scenario has ever been played, so the permanent stamp was a
    name-only guess. 31 of the 95 scenarios in the real library here take
    `clicking` from their name and `tracking` from their own stats.

    A profile written before `archetype_source` existed has "", which sorts
    below every real level, so its next run re-evaluates it. That is
    deliberate: those are exactly the profiles that may be carrying a guess.
    """
    arch, source = classify_archetype(scenario_name, run)
    have = profile.archetype_source or "default"
    if profile.archetype and EVIDENCE.index(source) <= EVIDENCE.index(have):
        return None
    was = profile.archetype
    profile.archetype, profile.archetype_source = arch, source
    return (was, arch) if was and was != arch else None
