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


def detect_archetype(scenario_name: str, run: Run | None = None) -> str:
    """Best-effort archetype for a scenario. Never raises; defaults to
    "clicking" when the evidence is thin."""
    name = f" {scenario_name.lower()} "
    for arch, keywords in _NAME_KEYWORDS:
        if any(k in name for k in keywords):
            return arch
    if run is not None:
        if run.kill_count > 0:
            shots = run.hit_count + run.miss_count
            if shots / run.kill_count >= _TRACKING_SHOTS_PER_KILL:
                return "tracking"
        elif run.hit_count >= _TRACKING_MIN_HITS:
            return "tracking"          # invincible targets: ticks, no deaths
    return "clicking"
