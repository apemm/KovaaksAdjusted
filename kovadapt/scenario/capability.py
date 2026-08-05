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

from dataclasses import dataclass, field

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


# ---------------------------------------------------------------- the channels
#: A dodge profile can steer left/right, forward/back, both, or neither. Only a
#: SELF-propelled target has a strafe TIMER for those multipliers to scale --
#: `Revolving Tracking` is SELF (MaxSpeed 1024, Acceleration 9000) and carries
#: no dodge channel at all, which is why this is its own tag and not a
#: restatement of T1.
YES, NO = "YES", "NO"

#: `[Map Data]` in the newer JSON form, which the reflex entity parser cannot
#: see. Not the same as having no spawns: eight files here are in this state,
#: and calling them "no spawn field" would assert something never read.
BLIND = "BLIND"

KILL, DAMAGE, HYBRID = "KILL", "DAMAGE", "HYBRID"
MOBILE, PARTIAL = "MOBILE", "PARTIAL"


@dataclass(frozen=True)
class Capability:
    """What a scenario can express, and what can honestly be measured on it.

    Every field is three-valued or a set. Nothing here is a boolean, because
    the two defects this module exists to prevent were both a boolean standing
    in for "we did not look".
    """

    motion: dict[str, str] = field(default_factory=dict)   # char -> T1
    strafe: str = UNKNOWN                                   # T2
    strafe_fb: bool = False                                 # sub-flag
    jump: str = UNKNOWN                                     # T3
    spawn_field: str = UNKNOWN                              # T4
    n_target_spawns: int = 0
    score_frame: str = UNKNOWN                              # T5
    invincible: bool = False
    player_frame: str = UNKNOWN                             # T6

    @property
    def motion_kinds(self) -> set[str]:
        return set(self.motion.values())

    @property
    def drives_motion(self) -> bool:
        """Can kovadapt's MaxSpeed and dodge writes actually move these?"""
        return drivable_motion(self.motion_kinds)

    @property
    def moves_somehow(self) -> bool:
        return can_express_motion(self.motion_kinds)

    @property
    def measurement_frame_is_static(self) -> bool:
        """Whether flick microstructure means what it claims to mean.

        `segment_flicks` integrates mouse displacement into a view-space path
        and assumes the view moves only when the mouse does. A strafing player
        must counter-move to hold even a stationary target, so overshoot
        becomes compensation error and directional bias is confounded by which
        way they were strafing. Hit/miss survives; the microstructure does not.

        PARTIAL counts as static: those files give the player one of the two
        keys and no way to reach a speed, so no counter-motion occurs.
        """
        return self.player_frame in (STATIC, PARTIAL, UNKNOWN)


def _dodge_profiles(sce: SceFile, bots: list[str] | tuple[str, ...]) -> list[str]:
    """Dodge profile names reachable from these bots, honouring NoDodging."""
    names: list[str] = []
    for bot in bots:
        if (sce.get_in_section("Bot Profile", bot, "NoDodging") or "").strip().lower() == "true":
            continue
        listed = sce.get_in_section("Bot Profile", bot, "DodgeProfileNames") or ""
        for d in (x.strip() for x in listed.split(";")):
            if d and sce.find_section("Dodge Profile", d) and d not in names:
                names.append(d)
    return names


def _toggle(sce: SceFile, dodge: str, key: str) -> bool:
    return (sce.get_in_section("Dodge Profile", dodge, key) or "").strip().lower() == "true"


def strafe_channel(sce: SceFile, bots, chars) -> tuple[str, bool]:
    """(T2, forward-back sub-flag).

    UNKNOWN when the targets are IMPULSE-driven and still carry a dodge
    profile: the block may be steering the impulse rather than scaling a
    strafe timer, and nothing in the corpus separates the two hypotheses.
    """
    kinds = scenario_motion(sce, chars)
    dodges = _dodge_profiles(sce, bots)
    lr = any(_toggle(sce, d, "ToggleLeftRight") for d in dodges)
    fb = any(_toggle(sce, d, "ToggleForwardBack") for d in dodges)
    if not dodges:
        return NO, False
    if SELF in kinds:
        return (YES if lr else NO), fb
    if IMPULSE in kinds or UNKNOWN in kinds:
        # IMPULSE: the dodge block may be steering the impulse rather than
        # scaling a strafe timer, and nothing in the corpus separates those.
        # UNKNOWN: the motion keys are missing, so this is an old file, not a
        # claim of stillness — and it must answer the same way `drivable_motion`
        # does or the two gates disagree about the same scenario.
        return UNKNOWN, fb
    return NO, fb


def jump_channel(sce: SceFile, bots, chars) -> str:
    """T3. `JumpFrequency` is meaningless without something to jump WITH.

    Measured on the emitted files: kovadapt writes a jump frequency onto
    characters with `JumpVelocity=0`, where no frequency produces a jump.
    UNKNOWN is rise-with-no-fall (JumpVelocity > 0, Gravity == 0), which the
    files alone do not settle.
    """
    if not _dodge_profiles(sce, bots):
        return NO
    seen = set()
    for c in chars:
        jv, g = _num(sce, c, "JumpVelocity"), _num(sce, c, "Gravity")
        if jv is None or g is None:
            seen.add(UNKNOWN)
        elif jv > 0 and g > 0:
            seen.add(YES)
        elif jv > 0:
            seen.add(UNKNOWN)
        else:
            seen.add(NO)
    if YES in seen:
        return YES
    return UNKNOWN if UNKNOWN in seen else NO


def score_frame(sce: SceFile, chars) -> tuple[str, bool]:
    """(T5, invincible). Gates the size controller's INPUT, not a write.

    `Run.accuracy` means "shots that connected" on a KILL scenario and "damage
    ticks that landed" on a DAMAGE one, and those are not the same quantity
    against the same band. Invincible targets never die, so kills are
    structurally zero rather than bad.
    """
    def head(k):
        try:
            return float(sce.get_header(k) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    per_kill, per_damage = head("ScorePerKill"), head("ScorePerDamage")
    invincible = (sce.get_header("InvincibleBots") or "").strip().lower() == "true" or any(
        (sce.get_in_section("Character Profile", c, "InvincibleBots") or "").strip().lower() == "true"
        for c in chars)
    if per_kill > 0 and per_damage > 0:
        return HYBRID, invincible
    if per_damage > 0:
        return DAMAGE, invincible
    if per_kill > 0:
        return KILL, invincible
    return UNKNOWN, invincible


def player_frame(sce: SceFile) -> str:
    """T6. Whether the PLAYER moves — the axis that gates MEASUREMENT.

    Both keys again, and for the same reason as T1: the counterexamples are
    real. `Revolving Tracking` gives the player MaxSpeed 1024 with
    Acceleration 0; `Narrow Strafe` gives Acceleration 16000 with MaxSpeed 0.
    Neither can actually move, and either key alone would call one of them
    mobile.

    This is a statement about the FILE'S PERMISSION, never an observation that
    the player moved. `Distance Traveled` takes two distinct values across all
    401 stats files here and cannot be distinguished from a dead field, so
    there is no way to know whether they used it.
    """
    name = (sce.get_header("PlayerProfile") or "Player").strip()
    if not sce.find_section("Character Profile", name):
        return UNKNOWN
    speed, accel = _num(sce, name, "MaxSpeed"), _num(sce, name, "Acceleration")
    if speed is None or accel is None:
        return UNKNOWN
    if speed > 0 and accel > 0:
        return MOBILE
    if speed > 0 or accel > 0:
        return PARTIAL
    return STATIC


def spawn_field(sce: SceFile, cols: int, rows: int) -> tuple[str, int]:
    """(T4, target spawn count).

    BLIND when `[Map Data]` is the newer JSON form: `SceFile.spawn_points`
    reads reflex entities and cannot see it at all, so zero spawns there is a
    failure to read rather than a fact about the layout. Eight files here are
    in that state, and reporting them as "no spawn field" would assert
    something never looked at.

    NO when there are genuinely fewer target spawns than grid cells —
    `resample_spawns` refuses to reweight below that, because a region holding
    no candidate point cannot be emphasised and we never invent coordinates.
    """
    from .generator import _player_team_flag, _spawn_team

    if (sce.get_header("MapName") or "").strip().lower().endswith(".json"):
        return BLIND, 0
    pts = sce.spawn_points()
    flag = _player_team_flag(sce)
    targets = [p for p in pts if _spawn_team(p) != flag] if flag else pts
    n = len(targets)
    if not pts:
        # A reflex map with no entities at all is unreadable, not empty.
        return BLIND, 0
    return (YES if n >= max(cols * rows, 1) else NO), n


def read_capability(sce: SceFile, cols: int = 5, rows: int = 5) -> Capability:
    """Every tag for one scenario, read from the file and nothing else.

    Deliberately takes the parsed `SceFile` rather than a path: the caller has
    already read it for other reasons, and re-reading is how two views of one
    file drift apart.
    """
    from .generator import _target_profiles

    bots, chars = _target_profiles(sce)
    motion = {c: target_motion(sce, c) for c in chars}
    strafe, fb = strafe_channel(sce, bots, chars)
    frame, invincible = score_frame(sce, chars)
    field_kind, n_spawns = spawn_field(sce, cols, rows)
    return Capability(
        motion=motion,
        strafe=strafe,
        strafe_fb=fb,
        jump=jump_channel(sce, bots, chars),
        spawn_field=field_kind,
        n_target_spawns=n_spawns,
        score_frame=frame,
        invincible=invincible,
        player_frame=player_frame(sce),
    )
