"""Which adaptation channels a scenario actually offers.

kovadapt has five ways to make a task harder. A given scenario offers some
subset of them, decided by its capability tags — and until now the engine
computed all five regardless, the generator dropped the ones that could not
land, and the page explained away the difference afterwards.

This module is the parameter-by-tag matrix as code, and it is the seam the
per-tag strategies compose through: a scenario's tag set selects the channels,
and difficulty is asked of the channels that exist rather than of all five.

It decides AVAILABILITY only. How hard to push a channel stays with the
controller that owns it — this module never returns a value, only whether a
value would mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..scenario.capability import (BLIND, Capability, GRAVITY, IMPULSE, NO,
                                   SELF, UNKNOWN)

#: The five things kovadapt can change about a scenario.
SIZE = "size"
SPEED = "speed"
STRAFE = "strafe"
JUMP = "jump"
SPAWN = "spawn"

ALL = (SIZE, SPEED, STRAFE, JUMP, SPAWN)


@dataclass(frozen=True)
class Channel:
    """One adaptation channel, and whether this scenario offers it.

    `reason` is written for the player, not the log: it is what the Changes
    page says when a criterion cannot move, so it names the property of the
    file rather than the name of a flag.
    """

    key: str
    available: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.available


def _size(cap: Capability) -> Channel:
    """Universal. Every target has a hit volume, on every shape of scenario.

    Deliberately not gated on whether the box is VISIBLE. Nine target
    characters in the corpus carry `MainBBHide=true`, so the hitbox really
    does change and the player cannot see that it did — but the change is
    real, and refusing to make it would be worse than making it quietly.
    """
    return Channel(SIZE, True)


def _speed(cap: Capability) -> Channel:
    """Needs a target kovadapt can actually drive.

    Not "does it move" — six scenarios move on a movement ability or under
    gravity, and writing MaxSpeed at them changes a number the ability path
    never reads.
    """
    kinds = cap.motion_kinds
    if not kinds:
        return Channel(SPEED, False, "no target character could be resolved")
    if SELF in kinds or UNKNOWN in kinds:
        return Channel(SPEED, True)
    if IMPULSE in kinds:
        return Channel(SPEED, False,
                       "these targets move on a movement ability, which is not "
                       "a speed kovadapt can set")
    if GRAVITY in kinds:
        return Channel(SPEED, False,
                       "these targets fall rather than travel, and gravity is "
                       "not a speed kovadapt can set")
    return Channel(SPEED, False, "these targets are authored to hold still")


def _strafe(cap: Capability) -> Channel:
    if cap.strafe == NO:
        return Channel(STRAFE, False,
                       "nothing here strafes: there is no left/right dodge "
                       "timer for a skew to scale")
    return Channel(STRAFE, True)


def _jump(cap: Capability) -> Channel:
    if cap.jump == NO:
        return Channel(JUMP, False,
                       "these targets cannot jump, so no frequency produces one")
    return Channel(JUMP, True)


def _spawn(cap: Capability) -> Channel:
    if cap.spawn_field == BLIND:
        return Channel(SPAWN, False,
                       "this scenario's layout is in a format kovadapt cannot "
                       "read, so nothing is claimed about its spawns")
    if cap.spawn_field == NO:
        return Channel(SPAWN, False,
                       f"only {cap.n_target_spawns} target spawn points, fewer "
                       "than the grid has cells — a region with no candidate "
                       "point cannot be emphasised and coordinates are never "
                       "invented")
    return Channel(SPAWN, True)


_RULES = {SIZE: _size, SPEED: _speed, STRAFE: _strafe, JUMP: _jump,
          SPAWN: _spawn}


def channels_for(cap: Capability) -> dict[str, Channel]:
    """Every channel, available or not, with a reason when not.

    Returns all five rather than only the usable ones on purpose. A caller
    that iterates the available set silently loses the ability to say why the
    others are missing, and saying why is most of this project's value.
    """
    return {key: rule(cap) for key, rule in _RULES.items()}


def available(cap: Capability) -> set[str]:
    return {k for k, ch in channels_for(cap).items() if ch.available}


def measurement_mask(cap: Capability) -> dict[str, bool]:
    """Which ANALYSES this scenario's shape lets us trust.

    Tags gate what can be measured as well as what can be written, and this
    is the half that is easy to forget: nothing here changes a file, so a
    wrong answer shows up as a confident claim rather than a broken scenario.

    `flick_microstructure` covers overshoot, corrections and directional bias.
    `segment_flicks` integrates mouse displacement into a view-space path and
    assumes the view moves only when the mouse does — a strafing player must
    counter-move to hold even a stationary target, so overshoot becomes
    compensation error and bias is confounded by strafe direction. 16 of the
    49 scenarios here are in that state.

    `hit_rate` survives everything: whether a shot connected does not depend
    on the reference frame it was taken in.
    """
    static_frame = cap.measurement_frame_is_static
    return {
        "hit_rate": True,
        "flick_microstructure": static_frame,
        "directional_bias": static_frame,
        "region_deficits": static_frame,
    }
