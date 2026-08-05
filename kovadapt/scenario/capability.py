"""What a scenario is structurally CAPABLE of, read from its .sce.

An adaptation parameter only means something if the scenario has the structure
it acts on. kovadapt learned this the hard way: it wrote MaxSpeed into static
walls for months, and the only reason anyone found out was playing one and
watching the targets not move.

This module answers capability questions and nothing else. It does not decide
what to write, and it never mutates. The v0.6 taxonomy grows here; today it
covers target motion, which is the axis that has already drawn blood.

THREE-VALUED, ALWAYS. Absent is not zero and neither is unresolved:

    "the file says the targets are static"        -> a fact
    "the file has no Acceleration line"           -> an older file, not a claim
    "no target character resolved at all"         -> our reader failed

Collapsing those is how a reader defect gets mistaken for a scenario property.
"""

from __future__ import annotations

from .sce import SceFile

#: A target that moves under its own locomotion: it has a speed to reach AND
#: the acceleration to reach it. BOTH are required, and requiring only one is
#: wrong in both directions on the real corpus — `1wall 2targets small -
#: valorant` and `1wall 6targets small Horizontalish` author
#: `Acceleration=16000` against `MaxSpeed=0` and do not move, while the
#: Pressure Aiming balloons author both as 0 and cross the room.
SELF = "SELF"
#: Driven by a [Movement Ability Profile] — velocity comes from the ability's
#: MainVelocity/UpVelocity, not from MaxSpeed. Pressure Aiming's balloons
#: (MainVelocity 5000 approaching, -7500 departing), Skeet Tracking, Reactive
#: Flick. Six of the eight ability-propelled units here would pass a bare
#: `Acceleration > 0` check, which is why that check is not the test.
IMPULSE = "IMPULSE"
#: Falls. MaxSpeed 0, no movement ability, gravity on — Piano Tiles' author
#: named the character "falling bot without movement".
GRAVITY = "GRAVITY"
#: Authored to hold still. A click-timing wall is this by design.
STATIC = "STATIC"
#: The keys that decide are missing, or nothing resolved. Never a synonym for
#: STATIC — see the module docstring.
UNKNOWN = "UNKNOWN"

MOVING = (SELF, IMPULSE, GRAVITY)

_MOVEMENT_ABILITY_SUFFIX = ".abilmov"


def _num(sce: SceFile, char: str, key: str) -> float | None:
    raw = sce.get_in_section("Character Profile", char, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def target_motion(sce: SceFile, char: str) -> str:
    """How one target [Character Profile] moves, if at all."""
    speed = _num(sce, char, "MaxSpeed")
    accel = _num(sce, char, "Acceleration")
    if speed is None or accel is None:
        return UNKNOWN
    if speed > 0 and accel > 0:
        return SELF
    abilities = sce.get_in_section("Character Profile", char,
                                   "AbilityProfileNames") or ""
    for name in (a.strip() for a in abilities.split(";")):
        if not name.lower().endswith(_MOVEMENT_ABILITY_SUFFIX):
            continue
        # The ability must exist AND carry a velocity. A named-but-absent
        # profile is a dangling reference, and one with no velocity is not a
        # movement source however it is spelled.
        stem = name[: -len(_MOVEMENT_ABILITY_SUFFIX)]
        for key in ("MainVelocity", "UpVelocity"):
            raw = sce.get_in_section("Movement Ability Profile", stem, key)
            try:
                if raw is not None and abs(float(raw)) > 0:
                    return IMPULSE
            except (TypeError, ValueError):
                continue
    if speed == 0 and (_num(sce, char, "Gravity") or 0) > 0:
        return GRAVITY
    return STATIC


def scenario_motion(sce: SceFile, chars: list[str] | tuple[str, ...]) -> set[str]:
    """The set of motion kinds present across a scenario's target characters.

    A set rather than one verdict because scenarios genuinely mix: Frog
    Simulator and KovaaKs Sandbox Intro each carry SELF and STATIC targets
    together, and an adaptation that moves the movers while leaving the props
    alone is a different thing from one that treats the file as uniform.

    Empty means nothing resolved — the caller must treat that as UNKNOWN, not
    as "no motion".
    """
    return {target_motion(sce, c) for c in chars}


def can_express_motion(kinds: set[str], *, unknown_is_capable: bool = True) -> bool:
    """Whether motion adaptation can act on a scenario with these kinds.

    `unknown_is_capable` preserves today's behaviour: a file missing the keys
    is attempted rather than suppressed, which is right for an older scenario
    and wrong for a reader failure. Measured exposure is currently zero —
    Acceleration is present in every [Character Profile] across the corpus —
    but key drift between game versions is real here, so this stays a named
    parameter rather than an assumption, pending a decision on it.

    Only SELF is actually drivable by kovadapt's speed and dodge writes:
    IMPULSE velocity lives in the ability profile and GRAVITY is a constant,
    neither of which the current write path touches. They are reported as
    moving because they ARE moving — a page that says "these targets cannot
    move" about Pressure Aiming's balloons is simply wrong — but they are not
    yet adaptable, and `drivable_motion` is the narrower question.
    """
    if not kinds:
        return False
    if UNKNOWN in kinds and unknown_is_capable:
        return True
    return any(k in MOVING for k in kinds)


def drivable_motion(kinds: set[str], *, unknown_is_capable: bool = True) -> bool:
    """Whether kovadapt's MaxSpeed and dodge writes can actually drive it.

    Narrower than `can_express_motion` on purpose: writing MaxSpeed into an
    IMPULSE target changes a number the ability path never reads.
    """
    if not kinds:
        return False
    if UNKNOWN in kinds and unknown_is_capable:
        return True
    return SELF in kinds
