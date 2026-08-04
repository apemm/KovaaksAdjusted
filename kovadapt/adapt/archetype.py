"""Scenario archetype detection: clicking vs tracking vs target-switching.

Different archetypes need different adaptation dynamics (see
config.default_archetype_overrides). Detection is name-keyword first
(scenario authors are consistent about naming), stats-heuristic second,
"clicking" as the safe default. The result is cached on the PlayerProfile
so one detection sticks for the scenario's lifetime.
"""

from __future__ import annotations

import re

from ..stats.models import Run

# Lowercase substrings of scenario names, checked in order. Switching is
# checked before tracking because hybrid names ("tracking switch") are
# switching-style tasks.
_NAME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("switching", ("switch", "psalmts", "pokeball",
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


#: The community writes target-switching as a TS token, and it is almost
#: always a SUFFIX — waldoTS, beanTS, FloatTS, devTS — or the Voltaic numeric
#: wall form, 1w2ts / 1w3ts. The keyword was `" ts "`, space-delimited on both
#: sides, which matched **none** of the 16 TS-named scenarios in the real
#: 95-scenario library here. Nine of them landed on `clicking` and were scored
#: against an 0.85-0.95 accuracy band instead of switching's 0.65-0.85, so the
#: size controller shrank targets every run chasing a number a switching task
#: does not produce. `" ts "` and `"voltaic ts"` are gone from the keyword list
#: below: `_TS_WORD` subsumes both. `"psalmts"` stays — it is an all-lowercase
#: spelling no pattern here can see.
#:
#: CASE IS THE SIGNAL, and lowercasing the name before matching is exactly what
#: threw it away. `waldoTS` carries a capital TS; `targets` does not. A naive
#: case-insensitive word-bounded `ts` matches "6targets" and "4 Targets" —
#: which would have relabelled `1wall 6targets small [Adaptive]`, the main
#: adaptive scenario on this machine, as target-switching. So the suffix
#: pattern is case-SENSITIVE and requires a lowercase letter or digit in front,
#: which "TARGETS" cannot satisfy either.
_TS_SUFFIX = re.compile(r"[a-z0-9]TS\b")            # waldoTS, beanTS, devTS
_TS_NUMERIC = re.compile(r"\b\d+w\d+ts\b", re.I)    # 1w2ts, 1w3ts
_TS_WORD = re.compile(r"\bts\b", re.I)              # "psalm TS", "voltaic ts"

#: How far a run's shots-per-kill must sit from `_TRACKING_SHOTS_PER_KILL`
#: before it is allowed to overturn a name-derived stamp. A name keyword is a
#: claim about authoring convention and is usually right, so a marginal run
#: must not flip it — but `Controlsphere Click Easy` takes "tracking" from the
#: `controlsphere` keyword while its own stats read **1.8** shots per kill
#: against a threshold of 20, and no reading of 1.8 is tracking.
#:
#: Measured across the 55 real scenarios that record kills: 39 sit below 10,
#: 6 sit above 40, and every one of the 10 in between is a TS or Switch
#: scenario — a genuine hybrid where the name is the better authority. So a
#: factor-of-two band fires on the one real error and on none of the hybrids.
_DECISIVE_FACTOR = 2.0


def _is_switching_name(scenario_name: str) -> bool:
    return bool(_TS_SUFFIX.search(scenario_name)
                or _TS_NUMERIC.search(scenario_name)
                or _TS_WORD.search(scenario_name))


def _stats_archetype(run: Run) -> tuple[str, bool]:
    """(archetype, is the evidence decisive enough to overturn a name?).

    Note this can only ever answer tracking or clicking. Switching has no
    reliable stats signature — a clicking run and a switching run look alike
    — so this function's disagreement with a `switching` stamp is not
    evidence against it, and the caller must not treat it as such.
    """
    if run.kill_count > 0:
        ratio = (run.hit_count + run.miss_count) / run.kill_count
        if ratio >= _TRACKING_SHOTS_PER_KILL:
            return "tracking", ratio >= _TRACKING_SHOTS_PER_KILL * _DECISIVE_FACTOR
        return "clicking", ratio <= _TRACKING_SHOTS_PER_KILL / _DECISIVE_FACTOR
    # Invincible targets: sustained hits and nothing ever dies. Structural,
    # not marginal — a clicking hit kills, so this cannot be a clicking run.
    if run.hit_count >= _TRACKING_MIN_HITS:
        return "tracking", True
    return "clicking", False


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
    padded = f" {scenario_name.lower()} "
    named = ""
    if _is_switching_name(scenario_name):
        named = "switching"
    else:
        for arch, keywords in _NAME_KEYWORDS:
            if any(k in padded for k in keywords):
                named = arch
                break

    # SWITCHING IS TERMINAL. `_stats_archetype` can only ever answer tracking
    # or clicking, so when it disagrees with a switching stamp that is not
    # evidence against it — it is the heuristic reporting from outside its own
    # range. Eight scenarios here prove the point: domiSwitch, voxTargetSwitch
    # and tamTargetSwitch read 30-216 shots per kill, because you TRACK a
    # target and then switch. The name is the better authority and there is no
    # stats signature that could ever say otherwise.
    if named == "switching":
        return "switching", "name"

    if run is None:
        return (named, "name") if named else ("clicking", "default")

    arch, decisive = _stats_archetype(run)
    # A run that does NOT trip the tracking heuristic is real evidence of
    # clicking, not an absence of evidence — that distinction is the whole
    # point, and it is why an unnamed scenario takes the stats answer outright.
    if not named:
        return arch, "stats"
    # With a name keyword present, the stats have to be DECISIVE to overturn
    # it: a keyword is a claim about authoring convention and is usually
    # right. `Controlsphere Click Easy` is why the door is not simply shut —
    # it takes "tracking" from the `controlsphere` keyword while its own stats
    # read 1.8 shots per kill against a threshold of 20.
    if decisive and arch != named:
        return arch, "stats"
    return named, "name"


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
