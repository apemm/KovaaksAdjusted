"""One motion system: durations, easing, stagger, and the intensity dial.

Every animated surface in the app reads its timing from here instead of
inventing its own, so "full show" is a coherent thing rather than a pile of
independent timers. `Settings.motion` selects an intensity — full / reduced /
off — and everything scales off that single value:

    full      everything, including ambient idle loops (the shipped default)
    reduced   meaningful reveals only; no ambient, no idle, no parallax drift
    off       instant; nothing animates at all

Two clocks, because they have different jobs:

* GLYPH_HZ (15) drives anything whose output is quantized to a character
  ramp. Between-frames there produce no visible change while costing a full
  repaint, and above ~13-15 Hz a character grid starts to read as flicker
  rather than motion because adjacent cells cross ramp thresholds at
  different moments. This is the correct rate, not a compromise.
* Transform/colour motion (alpha, position) can run at the display's own
  cadence; it is continuous, so more frames genuinely look smoother.

The duration ladder is deliberately short. Anything a user sees a hundred
times a day earns less time, so foreground moves get exactly one sub-300 ms
step and ambient motion never resolves faster than ~800 ms.

Pure timing helpers over `Settings` — no widgets, no Qt event loop — so the
rules are testable without a display.
"""

from __future__ import annotations

import math

# --- intensity -------------------------------------------------------------
FULL, REDUCED, OFF = "full", "reduced", "off"
_LEVELS = {FULL: 1.0, REDUCED: 0.6, OFF: 0.0}


def level(settings) -> str:
    """The configured intensity, defaulting to FULL for an unknown value.

    Read at USE time, never cached: the setting is user-editable while the app
    runs, exactly like the palette.
    """
    value = str(getattr(settings, "motion", FULL) or FULL).lower()
    return value if value in _LEVELS else FULL


def scale(settings) -> float:
    """Duration multiplier: 1.0 full, 0.6 reduced, 0.0 off."""
    return _LEVELS[level(settings)]


def animates(settings) -> bool:
    """False when motion is off — callers should jump straight to the end
    state rather than run a zero-length animation."""
    return scale(settings) > 0.0


def ambient(settings) -> bool:
    """Whether AMBIENT motion runs: idle loops, parallax drift, the backdrop
    eye's own life. These are the frames a user sees on every single visit,
    so `reduced` drops them entirely while keeping purposeful reveals."""
    return level(settings) == FULL


# --- durations (ms at full intensity) --------------------------------------
# A short ladder beats per-widget guesses: pick the rung, not a number.
INSTANT = 70        # state flip that still wants to be seen (toggle, hover)
FAST = 110          # small local change
BASE = 150          # the default for a foreground move
SLOW = 240          # something entering or leaving
SLOWER = 400        # a whole panel changing
CEREMONY = 700      # deliberate, once-per-event (a run landing, a PB)

# Ambient motion never resolves faster than this, or it reads as twitch.
AMBIENT_MIN = 800

GLYPH_HZ = 15               # character/structure animation
GLYPH_MS = int(1000 / GLYPH_HZ)


def ms(settings, base: int) -> int:
    """A duration in ms at the configured intensity (0 when motion is off)."""
    return int(round(base * scale(settings)))


# --- stagger ---------------------------------------------------------------
# Reveals stagger by DISTANCE from an origin, not by list index: a report's
# charts ignite outward from the weakest zone, bars run from the zero anchor.
# Index order makes a grid look like it is being typed; distance makes it look
# like one event propagating.
STAGGER_PER_CELL = 11       # ms of delay per unit of distance
STAGGER_CAP = 420           # total spread ceiling, so nothing waits too long


def stagger(settings, distance: float, *, per_unit: int = STAGGER_PER_CELL,
            cap: int = STAGGER_CAP) -> int:
    """Delay in ms for something `distance` units from the reveal's origin."""
    if not animates(settings):
        return 0
    return int(min(distance * per_unit, cap) * scale(settings))


def grid_distance(row: int, col: int, origin: tuple[int, int]) -> float:
    """Euclidean distance in cells from a reveal origin — the ordering term
    for a staggered grid reveal."""
    return math.hypot(row - origin[0], col - origin[1])


# --- easing ----------------------------------------------------------------
def ease_out(t: float) -> float:
    """Decelerating: the curve for anything ARRIVING. Cheap cubic rather than
    a QEasingCurve so this module stays importable without a widget stack."""
    t = min(max(t, 0.0), 1.0)
    return 1.0 - (1.0 - t) ** 3


def ease_in(t: float) -> float:
    """Accelerating: the curve for anything LEAVING. Exits also run shorter
    than entrances — a thing on its way out should not hold attention."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t


def ease_in_out(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 3.0 * t * t - 2.0 * t * t * t


def exit_ms(settings, base: int = BASE) -> int:
    """Exits are ~20% faster than the matching entrance."""
    return int(round(ms(settings, base) * 0.8))
