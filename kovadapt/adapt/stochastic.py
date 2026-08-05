"""Stochastic processes for anti-autopilot micro-movement.

An Ornstein-Uhlenbeck process drives movement intensity *across runs* so
difficulty drifts smoothly (no jarring jumps) but unpredictably (no muscle-
memory lock-in). Within a run, sampled dodge-profile parameters inject
per-target timing jitter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrnsteinUhlenbeck:
    """dX = theta * (mu - X) dt + sigma dW, exact discretization.

    Exact transition (no Euler bias):
        X_{t+dt} = mu + (X_t - mu) e^{-theta dt} + sigma * sqrt((1 - e^{-2 theta dt}) / (2 theta)) * N(0,1)
    """

    theta: float = 0.35
    mu: float = 0.0
    sigma: float = 0.25

    def step(self, x: float, dt: float = 1.0, rng: np.random.Generator | None = None) -> float:
        rng = rng or np.random.default_rng()
        e = np.exp(-self.theta * dt)
        std = self.sigma * np.sqrt((1.0 - e * e) / (2.0 * self.theta))
        return float(self.mu + (x - self.mu) * e + std * rng.standard_normal())

    def path(self, x0: float, n: int, dt: float = 1.0, seed: int | None = None) -> np.ndarray:
        """Vectorized sample path of length n+1 (includes x0)."""
        rng = np.random.default_rng(seed)
        e = np.exp(-self.theta * dt)
        std = self.sigma * np.sqrt((1.0 - e * e) / (2.0 * self.theta))
        out = np.empty(n + 1)
        out[0] = x0
        noise = rng.standard_normal(n) * std
        for i in range(n):
            out[i + 1] = self.mu + (out[i] - self.mu) * e + noise[i]
        return out

    def stationary_std(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.theta)


def squash(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Map an unbounded OU state into [lo, hi] via a logistic squash."""
    return lo + (hi - lo) / (1.0 + np.exp(-2.0 * x))


def sample_dodge_params(
    movement: float,
    rng: np.random.Generator | None = None,
    direction_bias: float = 0.0,
) -> dict[str, float]:
    """Map movement intensity [0,1] -> KovaaK's [Dodge Profile] parameters.

    movement 0   => effectively static targets
    movement 1   => fast, twitchy strafes with short direction hold times
    Randomized within bounds each generation so consecutive runs never share
    exact timings (anti-autopilot).

    direction_bias in [-1, 1] skews strafe hold times toward one side so
    targets travel longer in the direction that forces the player's weak-side
    flicks: positive = more leftward strafing (train a weak left), negative =
    more rightward. The multipliers are reciprocal so total strafe time stays
    roughly constant.
    """
    m = float(np.clip(movement, 0.0, 1.0))
    b = float(np.clip(direction_bias, -1.0, 1.0))
    rng = rng or np.random.default_rng()
    j = lambda v, pct=0.15: v * (1.0 + rng.uniform(-pct, pct))  # noqa: E731

    # Hold times shrink as intensity rises: 1.2s (calm) -> 0.15s (twitchy).
    min_hold = j(np.interp(m, [0, 1], [1.20, 0.15]))
    max_hold = min_hold + j(np.interp(m, [0, 1], [0.90, 0.20]))
    # Up to 1.8x longer strafes toward the weak side at |bias| = 1.
    left_mult = (1.0 + 0.8 * b) if b >= 0 else 1.0 / (1.0 - 0.8 * b)
    return {
        "MinLRTimeChange": round(min_hold, 3),
        "MaxLRTimeChange": round(max_hold, 3),
        "MinFBTimeChange": round(min_hold, 3),
        "MaxFBTimeChange": round(max_hold, 3),
        "LeftStrafeTimeMult": round(float(np.clip(j(left_mult, 0.10), 0.5, 2.0)), 3),
        "RightStrafeTimeMult": round(float(np.clip(j(1.0 / left_mult, 0.10), 0.5, 2.0)), 3),
        "StrafeSwapMinPause": round(j(np.interp(m, [0, 1], [0.10, 0.0])), 3),
        "StrafeSwapMaxPause": round(j(np.interp(m, [0, 1], [0.25, 0.05])), 3),
        "JumpFrequency": round(np.interp(m, [0, 1], [0.0, 0.35]), 3),
    }


def movement_speed(movement: float, base_speed: float = 170.0) -> float:
    """Absolute MaxSpeed ramp for characters whose base speed is 0 (static
    walls): movement intensity is the only thing that makes them move."""
    return round(float(np.interp(np.clip(movement, 0, 1), [0, 1], [0.0, base_speed])), 1)


def speed_multiplier(movement: float) -> float:
    """Scale factor for characters with an AUTHORED MaxSpeed (> 0): movement
    modulates around the scenario author's speed instead of replacing it —
    a 1300-speed strafe bot must never be overwritten with the 0-170
    static-wall ramp. 0 eases to 65% of authored, 1 pushes to 135%."""
    return round(float(np.interp(np.clip(movement, 0, 1), [0, 1], [0.65, 1.35])), 3)


#: Which way "harder" runs for each dodge key. Hold times and swap pauses get
#: SHORTER as intensity rises; jump frequency gets higher. Without this a
#: single relative formula would make half the parameters easier at maximum.
_DODGE_DIRECTION = {
    "MinLRTimeChange": -1,
    "MaxLRTimeChange": -1,
    "MinFBTimeChange": -1,
    "MaxFBTimeChange": -1,
    "StrafeSwapMinPause": -1,
    "StrafeSwapMaxPause": -1,
    "JumpFrequency": +1,
}

#: The strafe skew is a ratio, not an intensity: it multiplies whatever the
#: author wrote rather than moving it along a difficulty axis.
_DODGE_RATIO_KEYS = ("LeftStrafeTimeMult", "RightStrafeTimeMult")


def relative_dodge_writes(
    authored: dict[str, float],
    movement: float,
    direction_bias: float = 0.0,
    span: float = 0.35,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Dodge values expressed as a band around what the AUTHOR wrote.

    The absolute form had two failures that a relative one cannot have.

    ITS CEILING COULD SIT BELOW THE AUTHOR'S FLOOR. `JumpFrequency` maxed at
    0.35 while 13 of the 54 authored values in the real corpus are above it —
    `1wall 6targets small` and `mccoyfrozentrack` both author 0.50. Turning
    the dial to maximum made those scenarios jump LESS than their author
    intended, which is not a difficulty control, it is a different scenario.

    IT FLATTENED DELIBERATE CONTRAST. `Surge Tags` authors JumpFrequency
    0.0 / 0.9 / 0.8 / 0.02 / 0.02 across five profiles to make them behave
    differently; one absolute value written to all five erases the design.
    Scaling by a common factor preserves rank order by construction.

    `movement` 0.5 is the neutral point and writes the author's own value
    back. Below it the scenario is easier than authored, above it harder,
    bounded by `span` either way — so kovadapt nudges a scenario rather than
    relocating it to an absolute difficulty point.

    Keys the author did not write are not invented: `set_in_section` only
    rewrites an existing key, so returning one would be a value that silently
    goes nowhere.
    """
    m = float(np.clip(movement, 0.0, 1.0))
    span = float(np.clip(span, 0.0, 1.0))
    rng = rng or np.random.default_rng()
    out: dict[str, float] = {}

    for key, direction in _DODGE_DIRECTION.items():
        base = authored.get(key)
        if base is None:
            continue
        # 0.5 -> 1.0 (unchanged); the sign flips for keys where harder is less
        factor = 1.0 + direction * span * (2.0 * m - 1.0)
        jitter = 1.0 + rng.uniform(-0.05, 0.05)   # anti-autopilot, not a knob
        value = max(base * factor * jitter, 0.0)
        if key == "JumpFrequency":
            # A frequency used as a per-decision probability cannot exceed 1.
            # This can TIE two profiles the author separated — 0.8 and 0.9
            # both reach the ceiling at full intensity — which is a loss of
            # resolution at the top, not an inversion. Nothing here can avoid
            # it without seeing the whole profile set at once.
            # The highest any author here writes is 0.9, so the band can carry
            # a value past the top without a clamp; what KovaaK's does with
            # 1.19 is unknown and not worth finding out in a player's file.
            value = min(value, 1.0)
        out[key] = round(value, 4)

    # Min and Max are jittered independently, so they can cross — a hold
    # window whose floor sits above its ceiling is not a difficulty setting,
    # it is a malformed one. Measured at 894 crossings in 4000 draws on an
    # authored pair only 0.01s apart. Ordering is restored rather than
    # re-drawn, so the band still spans what it says it spans.
    for lo_key, hi_key in (("MinLRTimeChange", "MaxLRTimeChange"),
                           ("MinFBTimeChange", "MaxFBTimeChange"),
                           ("StrafeSwapMinPause", "StrafeSwapMaxPause")):
        lo, hi = out.get(lo_key), out.get(hi_key)
        if lo is not None and hi is not None and lo > hi:
            out[lo_key], out[hi_key] = hi, lo

    b = float(np.clip(direction_bias, -1.0, 1.0))
    left = (1.0 + 0.8 * b) if b >= 0 else 1.0 / (1.0 - 0.8 * b)
    for key, mult in zip(_DODGE_RATIO_KEYS, (left, 1.0 / left)):
        base = authored.get(key)
        if base is None:
            continue
        out[key] = round(float(np.clip(base * mult, 0.0, 10.0)), 4)
    return out
