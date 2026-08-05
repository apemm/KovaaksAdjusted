"""Adaptation engine: run + profile -> next scenario parameters.

Coupled controllers:
  1. Size controller  — multiplicative log-scale update keeping hit rate in the
     configured sweet spot (default 85-95%, the primary-sourced clicking band —
     analysis/kb.py: p-speed-accuracy-governor): flow-state difficulty.
     A Fitts throughput sub-controller adds an extra shrink step when accuracy
     is comfortable in-band but ms-per-bit has stalled (challenge point).
  2. Region bandit    — Thompson sampling concentrates spawns on weak regions.
  3. Movement (OU)    — Ornstein-Uhlenbeck drift + speed-coupled sizing: when
     movement intensity rises, targets grow to keep the task fair; when the
     player's pace (kills/s) rises, movement rises to break autopilot — and
     when pace plateaus with accuracy parked in-band, a bounded progression
     push raises the movement target (speed is the growth axis).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..config import Settings
from ..profile.player import PlayerProfile
from ..stats.models import Run
from .bandit import ThompsonRegionBandit
from .stochastic import (
    OrnsteinUhlenbeck,
    movement_speed,
    sample_dodge_params,
    speed_multiplier,
    squash,
)

# How far the size controller may move target scale in ONE run. See the long
# note at the call site: without a cap a single unrepresentative run (wrong
# sensitivity, wrong scenario, spraying to end a run) traverses the entire
# 0.4-2.5 range in one step, and the best possible correction afterwards is
# 3.5% per run. The band is deliberately asymmetric in the same direction as
# the gains — growing targets after a bad run is the accuracy-first call, so
# it gets more room than shrinking them after a good one.
_MIN_SIZE_STEP, _MAX_SIZE_STEP = 0.80, 1.35


@dataclass
class AdaptationPlan:
    """Everything the scenario generator needs to write the next variant."""

    scenario: str
    target_scale: float
    movement: float
    focus_region: str
    spawn_weights: dict[str, float]
    dodge_params: dict[str, float] = field(default_factory=dict)
    target_max_speed: float = 0.0   # absolute ramp, used when base MaxSpeed is 0
    target_speed_mult: float = 1.0  # multiplier on an authored base MaxSpeed > 0
    seed: int = 0
    dodge_bias: float = 0.0    # strafe skew toward the weak side (+ = left)
    fatigue: float = 0.0       # easing applied to this plan (0 = none)
    # Written by the GENERATOR, not the planner: whether the emitted .sce could
    # actually express focus_region. A region holding no candidate spawn point
    # cannot be emphasized — resampling reuses original coordinates and never
    # invents them — so its weight is absorbed by the others and the scenario
    # carries no focus at all. The watcher reads this to avoid crediting a
    # bandit arm for an emphasis the player never actually saw. Defaults True
    # so a plan that never reaches the generator behaves exactly as before.
    focus_applied: bool = True
    # Also written by the GENERATOR: whether the target bots can MOVE at all.
    # A KovaaK's bot needs Acceleration > 0 to ever reach its MaxSpeed, and a
    # static-wall scenario is authored `MaxSpeed=0, Acceleration=0`. Setting
    # MaxSpeed alone on one of those changes nothing the game can act on — and
    # because nothing moves, the Left/RightStrafeTimeMult skew and the jump
    # frequency are inert with it.
    #
    # Verified in the running game rather than inferred: a variant written with
    # MaxSpeed 102.5, strafe skew 1.591/0.650 and JumpFrequency 0.211 was
    # played, and the targets did not move, strafe or jump. Across all 33 local
    # scenarios every one whose targets move carries Acceleration 450-20000 and
    # every static one carries 0, with no exceptions.
    #
    # kovadapt does NOT add acceleration to a scenario that has none: a static
    # click-timing wall is that by design, and making it move would change what
    # the task trains rather than how hard it is. Motion adaptation is reserved
    # for scenarios that already move.
    motion_applied: bool = True
    #: Which adaptation channels this scenario actually offered, by
    #: `adapt/channels.py`. Recorded on the plan rather than recomputed by
    #: each consumer, for the same reason `focus_applied` is: the answer
    #: depends on the FILE, and a second derivation is a second thing to
    #: drift. Empty means the planner was never given a capability — the
    #: pre-v0.6 path, where every channel was computed and the generator
    #: dropped what could not land.
    #:
    #: It also goes into the shadow log with the rest of the plan, which an
    #: off-policy learner needs: an action that was never available is not
    #: an action the policy declined to take.
    channels: dict[str, bool] = field(default_factory=dict)

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
        fitts_slope_ms: float | None = None,
    ) -> None:
        """Fold a finished run into the profile (must be called before plan).

        When flick telemetry produced per-region deficits, those feed the
        bandit directly (observed rewards for every region flicked toward);
        run-level focus attribution is the fallback without telemetry.
        bias_score (analysis.directional_bias convention: + = left weaker)
        drives the trace-informed dodge direction.

        A run in which nothing was fired carries no evidence and is dropped
        whole. Run.accuracy is 0.0 for zero shots (a guard, not a
        measurement), so folding one in would drag the accuracy EWMA toward
        zero AND — because credit_focus_region scores `ewma_accuracy -
        run.accuracy` — book the player's entire baseline as a deficit
        against whichever region the bandit last focused, inventing a strong
        weakness there. One alt-tab during a timed scenario was enough. The
        size controller already gates on min_shots_for_size; this is the
        same guard for the EWMAs and the bandit.
        """
        if run.hit_count + run.miss_count <= 0:
            return
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
        if fitts_slope_ms is not None and fitts_slope_ms > 0:
            profile.observe_fitts(fitts_slope_ms, s.ewma_half_life)

    # --------------------------------------------------------------- plan
    def plan(
        self,
        profile: PlayerProfile,
        last_run: Run | None = None,
        fatigue: float = 0.0,
        capability=None,
    ) -> AdaptationPlan:
        """fatigue in [0, 1] eases the EMITTED plan only (bigger targets,
        calmer movement); the persisted profile state stays un-eased so
        recovery next session resumes from the true difficulty.

        `capability` is the scenario's `scenario.capability.Capability`
        when the caller has read the file. Given one, the plan records
        which adaptation channels the scenario actually offers — a static
        wall has two of the five, and a plan that does not say so leaves
        every consumer to work it out again. Omitted, the plan is computed
        exactly as before and `channels` stays empty rather than claiming
        all five are available."""
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
                # accuracy too high -> excess > 0 -> shrink targets, and vice versa.
                # Deliberately accuracy-biased (Arjun's design call: accuracy
                # builds better habits for real gameplay): falling BELOW the
                # band grows targets 1.4x harder than sitting above it shrinks
                # them, so the model recovers accuracy before it chases speed.
                gain = s.size_learning_rate * (1.4 if excess < 0 else 0.8)
                # ONE RUN MAY NOT CROSS THE WHOLE RANGE. Measured: a 15-shot
                # run at 6.7% (wrong sensitivity after a config change, wrong
                # scenario loaded, spraying to end a run) gives excess -0.783
                # and exp(0.987) = 2.68x in a SINGLE step, which clips
                # straight to max_target_scale. The error term is structurally
                # asymmetric — accuracy is bounded in [0,1] with the band at
                # 0.85-0.95, so excess below can reach -0.85 while above it
                # caps at +0.05, a 17x asymmetry the 1.4/0.8 gain split
                # compounds rather than compensates. The best possible
                # down-step is 3.5%/run, so the excursion costs ~25-30 runs —
                # a whole session of training against targets that are
                # trivially easy, which is the opposite of the product.
                #
                # It does self-heal (a player facing 2.5x targets really does
                # hit ~100%, which drives the controller back down), so this
                # bounds the step rather than rejecting the run: a genuinely
                # struggling player still reaches the ceiling, over several
                # runs, and the deadband keeps intermediate values to work
                # from. observe()'s zero-shot guard already handles the case
                # where there is no evidence at all; this handles the case
                # where the evidence is real but wildly unrepresentative.
                step = math.exp(-gain * excess)
                scale *= min(max(step, _MIN_SIZE_STEP), _MAX_SIZE_STEP)
            elif (s.fitts_control_gain > 0 and profile.fitts_obs >= 5
                  and profile.ewma_fitts_ms >= profile.slow_fitts_ms):
                # Fitts throughput sub-controller: comfortable in the band but
                # ms-per-bit has stalled -> one extra gentle shrink step per
                # run toward the challenge point (analysis/kb.py:
                # p-challenge-point, p-fitts-throughput). ~1.75%/run at the
                # default gain; the deadband above reverses it if accuracy
                # falls out of the band.
                scale *= math.exp(-s.fitts_control_gain * 0.05)
        scale = float(np.clip(scale, s.min_target_scale, s.max_target_scale))

        # -- 2. movement: OU drift + pace coupling --------------------------
        pace_push = 0.0
        if profile.run_count >= 3 and profile.ewma_kps > 0 and last_run is not None:
            # Player running hotter than their norm => likely autopiloting.
            rel = last_run.kills_per_second() / profile.ewma_kps - 1.0
            pace_push = float(np.clip(rel, -0.5, 0.5))
        if (s.pace_progression_gain > 0 and last_run is not None
                and profile.run_count >= 10
                and s.target_accuracy_low <= last_run.accuracy <= s.target_accuracy_high):
            # Pace-plateau progression: accuracy parked in-band with kills/s
            # flat across recent history -> bounded upward push through the
            # existing OU/pace pathway (analysis/kb.py: p-speed-is-growth-axis).
            kpss = [float(h.get("kps", 0.0)) for h in profile.history[-10:]]
            half = len(kpss) // 2
            a0 = sum(kpss[:half]) / max(half, 1)
            a1 = sum(kpss[half:]) / max(len(kpss) - half, 1)
            if a0 > 0 and abs(a1 / a0 - 1.0) < 0.03:
                pace_push += 0.5 * s.pace_progression_gain
        # Built per-plan from the *effective* settings so archetype overrides
        # of ou_theta/ou_sigma apply (contract: tunables via _effective()).
        ou = OrnsteinUhlenbeck(theta=s.ou_theta, sigma=s.ou_sigma)
        profile.ou_state = ou.step(
            profile.ou_state + s.pace_coupling_gain * pace_push, rng=self.rng
        )
        movement = squash(profile.ou_state, s.min_movement, s.max_movement)
        profile.movement = movement

        # -- 3. speed-coupled sizing: faster targets get a size floor -------
        # Keeps effective difficulty smooth: size *= (1 + coupling * movement).
        coupling = s.size_speed_coupling
        # Persist the UN-coupled base, then couple + clip for the emitted plan.
        # Dividing the clipped value back out (the old order) only round-trips
        # while nothing clips: at the ceiling it wrote back
        # max_target_scale / (1 + coupling * movement) — strictly less than the
        # base that reached the ceiling — so touching the cap permanently gave
        # away earned scale, shrinking targets on a player the size controller
        # had been growing them for.
        profile.target_scale = float(
            np.clip(scale, s.min_target_scale, s.max_target_scale))
        scale = float(np.clip(profile.target_scale * (1.0 + coupling * movement),
                              s.min_target_scale, s.max_target_scale))

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
        # Availability, not magnitude: the controllers above have already
        # decided how hard to push. This records which of those pushes the
        # scenario can receive at all.
        from .channels import channels_for
        chans = ({k: ch.available for k, ch in channels_for(capability).items()}
                 if capability is not None else {})

        return AdaptationPlan(
            channels=chans,
            scenario=profile.scenario,
            target_scale=out_scale,
            movement=out_movement,
            focus_region=focus,
            spawn_weights=weights,
            dodge_params=sample_dodge_params(out_movement, rng=self.rng,
                                             direction_bias=dodge_bias),
            target_max_speed=movement_speed(out_movement),
            target_speed_mult=speed_multiplier(out_movement),
            seed=seed,
            dodge_bias=dodge_bias,
            fatigue=fatigue,
        )


def settle_focus(profile: PlayerProfile, plan: AdaptationPlan) -> bool:
    """Drop a focus region the emitted `.sce` could not actually express.

    Call this after `generate_adaptive_variant()` and BEFORE `profile.save()`,
    at every site that pairs a `plan()` with a generate. Returns True when the
    focus was cleared, so callers can say so.

    `plan()` stamps `profile.last_focus` unconditionally, but the generator
    only learns whether the focus survived once it has tried to resample
    spawns — `resample_spawns()` returns nothing when the layout has fewer
    target spawns than `region_cols * region_rows` (25 by default), or when
    the focus region holds no candidate. On the next run `credit_focus_region`
    books that run's accuracy deficit against an emphasis the player was never
    shown: evidence for or against a change that was silently dropped. It is
    not display-only — `bandit.spawn_weights` softmaxes the arm posteriors, so
    contaminated arms steer the spawn mass written into future variants.

    This lived inline in `SessionWatcher.process_run` and NOWHERE ELSE, while
    four other sites (`SessionWatcher.bootstrap`, `cli.cmd_generate`,
    `ScenarioBrowser._generate`, and the dashboard's Start via bootstrap) did
    the same plan/generate/save without it. Measured against a real install,
    19 of 31 `.sce` files carry fewer than 25 target spawns and 8 have none at
    all, so on those layouts no focus is ever actionable.
    """
    if plan.focus_applied:
        return False
    profile.last_focus = None
    return True
