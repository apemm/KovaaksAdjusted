"""Adaptation engine: run + profile -> next scenario parameters.

Three coupled controllers:
  1. Size controller  — multiplicative log-scale update keeping hit rate in the
     configured sweet spot (default 60-80%): flow-state difficulty.
  2. Region bandit    — Thompson sampling concentrates spawns on weak regions.
  3. Movement (OU)    — Ornstein-Uhlenbeck drift + speed-coupled sizing: when
     movement intensity rises, targets grow to keep the task fair; when the
     player's pace (kills/s) rises, movement rises to break autopilot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from ..profile.player import PlayerProfile
from ..stats.models import Run
from .bandit import ThompsonRegionBandit
from .stochastic import OrnsteinUhlenbeck, sample_dodge_params, movement_speed, squash


@dataclass
class AdaptationPlan:
    """Everything the scenario generator needs to write the next variant."""

    scenario: str
    target_scale: float
    movement: float
    focus_region: str
    spawn_weights: dict[str, float]
    dodge_params: dict[str, float] = field(default_factory=dict)
    target_max_speed: float = 0.0
    seed: int = 0
    dodge_bias: float = 0.0    # strafe skew toward the weak side (+ = left)
    fatigue: float = 0.0       # easing applied to this plan (0 = none)

    def describe(self) -> str:
        out = (
            f"scale={self.target_scale:.2f} movement={self.movement:.2f} "
            f"focus={self.focus_region} speed={self.target_max_speed:.0f}"
        )
        if abs(self.dodge_bias) > 0.01:
            out += f" dodge={'L' if self.dodge_bias > 0 else 'R'}{abs(self.dodge_bias):.2f}"
        if self.fatigue > 0.01:
            out += f" eased={self.fatigue:.2f}"
        return out


class AdaptationEngine:
    def __init__(self, settings: Settings, rng: np.random.Generator | None = None) -> None:
        self.s = settings
        self.rng = rng or np.random.default_rng()
        self.ou = OrnsteinUhlenbeck(theta=settings.ou_theta, sigma=settings.ou_sigma)

    def _effective(self, profile: PlayerProfile) -> Settings:
        """Settings with the profile's archetype overrides applied."""
        return self.s.for_archetype(profile.archetype)

    # ------------------------------------------------------------ observe
    def observe(
        self,
        profile: PlayerProfile,
        run: Run,
        region_deficits: dict[str, float] | None = None,
        bias_score: float | None = None,
    ) -> None:
        """Fold a finished run into the profile (must be called before plan).

        When flick telemetry produced per-region deficits, those feed the
        bandit directly (observed rewards for every region flicked toward);
        run-level focus attribution is the fallback without telemetry.
        bias_score (analysis.directional_bias convention: + = left weaker)
        drives the trace-informed dodge direction.
        """
        s = self._effective(profile)
        if s.bandit_posterior_decay > 0:
            profile.decay_regions(s.bandit_posterior_decay, s.bandit_prior_var)
        if region_deficits:
            profile.credit_observed_regions(
                region_deficits, s.telemetry_blend,
                prior_var=s.bandit_prior_var, obs_noise=s.bandit_obs_noise,
            )
        else:
            profile.credit_focus_region(         # bandit reward first (needs old EWMA)
                run, prior_var=s.bandit_prior_var, obs_noise=s.bandit_obs_noise,
            )
        profile.observe_run(run, s.ewma_half_life)
        if bias_score is not None:
            profile.observe_bias(bias_score, s.ewma_half_life)

    # --------------------------------------------------------------- plan
    def plan(
        self,
        profile: PlayerProfile,
        last_run: Run | None = None,
        fatigue: float = 0.0,
    ) -> AdaptationPlan:
        """fatigue in [0, 1] eases the EMITTED plan only (bigger targets,
        calmer movement); the persisted profile state stays un-eased so
        recovery next session resumes from the true difficulty."""
        s = self._effective(profile)
        fatigue = float(np.clip(fatigue, 0.0, 1.0))

        # -- 1. size controller (multiplicative, log-space) ----------------
        scale = profile.target_scale
        if last_run is not None and (last_run.hit_count + last_run.miss_count) >= s.min_shots_for_size:
            acc = last_run.accuracy
            mid = 0.5 * (s.target_accuracy_low + s.target_accuracy_high)
            band = 0.5 * (s.target_accuracy_high - s.target_accuracy_low)
            err = acc - mid
            if abs(err) > band:  # outside sweet spot: act on the excess only
                excess = err - math.copysign(band, err)
                # accuracy too high -> excess > 0 -> shrink targets, and vice versa
                scale *= math.exp(-s.size_learning_rate * excess)
        scale = float(np.clip(scale, s.min_target_scale, s.max_target_scale))

        # -- 2. movement: OU drift + pace coupling --------------------------
        pace_push = 0.0
        if profile.run_count >= 3 and profile.ewma_kps > 0 and last_run is not None:
            # Player running hotter than their norm => likely autopiloting.
            rel = last_run.kills_per_second() / profile.ewma_kps - 1.0
            pace_push = float(np.clip(rel, -0.5, 0.5))
        profile.ou_state = self.ou.step(
            profile.ou_state + s.pace_coupling_gain * pace_push, rng=self.rng
        )
        movement = squash(profile.ou_state, s.min_movement, s.max_movement)
        profile.movement = movement

        # -- 3. speed-coupled sizing: faster targets get a size floor -------
        # Keeps effective difficulty smooth: size *= (1 + coupling * movement).
        coupling = s.size_speed_coupling
        scale = float(np.clip(scale * (1.0 + coupling * movement),
                              s.min_target_scale, s.max_target_scale))
        profile.target_scale = scale / (1.0 + coupling * movement)  # persist base scale

        # -- 4. region bandit ------------------------------------------------
        bandit = ThompsonRegionBandit(
            profile, s.region_cols, s.region_rows, rng=self.rng,
            prior_var=s.bandit_prior_var, obs_noise=s.bandit_obs_noise,
        )
        focus = bandit.choose_focus()
        weights = bandit.spawn_weights(focus, s.focus_weight)
        profile.last_focus = focus

        # -- 5. trace-informed dodge direction -------------------------------
        # Strafe longer toward the weak flick side (bias sign: + = left weak).
        dodge_bias = 0.0
        if s.dodge_bias_enabled and abs(profile.ewma_bias) > 0.05:
            dodge_bias = float(np.clip(s.dodge_bias_gain * profile.ewma_bias, -1.0, 1.0))

        # -- 6. fatigue easing (emitted values only) --------------------------
        out_movement = movement * (1.0 - 0.4 * fatigue)
        out_scale = float(np.clip(scale * (1.0 + 0.20 * fatigue),
                                  s.min_target_scale, s.max_target_scale))

        seed = int(self.rng.integers(0, 2**31 - 1))
        return AdaptationPlan(
            scenario=profile.scenario,
            target_scale=out_scale,
            movement=out_movement,
            focus_region=focus,
            spawn_weights=weights,
            dodge_params=sample_dodge_params(out_movement, rng=self.rng,
                                             direction_bias=dodge_bias),
            target_max_speed=movement_speed(out_movement),
            seed=seed,
            dodge_bias=dodge_bias,
            fatigue=fatigue,
        )
