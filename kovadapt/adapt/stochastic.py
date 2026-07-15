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
    """Character MaxSpeed for the target bots as a function of intensity."""
    return round(float(np.interp(np.clip(movement, 0, 1), [0, 1], [0.0, base_speed])), 1)
