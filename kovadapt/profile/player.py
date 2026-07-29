"""Persistent per-scenario player model.

Tracks exponentially-weighted performance statistics, region-bandit posteriors,
and the adaptation state (current target scale, OU movement state, run count).
Stored as JSON under ~/.kovadapt/profiles/<scenario>.json.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..stats.models import Run


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


@dataclass
class RegionPosterior:
    """Gaussian posterior over player *skill deficit* in one spawn region.

    mean > 0  => player underperforms when fire is concentrated here.
    Conjugate normal update with fixed observation noise.
    """

    mean: float = 0.0
    var: float = 1.0
    n: int = 0
    obs_noise: float = 0.25

    def update(self, deficit: float) -> None:
        precision = 1.0 / self.var + 1.0 / self.obs_noise
        self.mean = (self.mean / self.var + deficit / self.obs_noise) / precision
        self.var = 1.0 / precision
        self.n += 1


@dataclass
class PlayerProfile:
    scenario: str
    ewma_accuracy: float = 0.0
    ewma_ttk: float = 0.0
    ewma_kps: float = 0.0            # kills per second (pace)
    ewma_score: float = 0.0
    run_count: int = 0
    target_scale: float = 1.0        # current size multiplier applied to scenario
    movement: float = 0.15           # current micro-movement intensity [0, 1]
    ou_state: float = 0.0            # OU process state driving movement drift
    regions: dict[str, RegionPosterior] = field(default_factory=dict)
    last_focus: str | None = None    # region arm used for the run in flight
    last_run_ts: str = ""
    ewma_bias: float = 0.0           # directional flick bias (+ = left weaker)
    archetype: str = ""              # clicking | tracking | switching ("" = unknown)
    # Fitts throughput trend (ms of flick time per bit of difficulty; lower =
    # better). Two EWMAs at different half-lives form the minimal persisted
    # trend signal: fast at/above slow means throughput has stopped improving.
    # All default so pre-v0.4 profile JSON keeps loading.
    ewma_fitts_ms: float = 0.0       # fast EWMA of per-run Fitts slope (0 = no evidence)
    slow_fitts_ms: float = 0.0       # 4x-half-life reference EWMA (trend baseline)
    fitts_obs: int = 0               # telemetry runs folded into the Fitts EWMAs
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ ewma
    def _alpha(self, half_life: float) -> float:
        return 1.0 - math.exp(-math.log(2.0) / max(half_life, 1e-9))

    def observe_run(self, run: Run, half_life: float = 5.0) -> None:
        """Fold a completed run into the EWMA statistics."""
        a = self._alpha(half_life)
        acc, ttk, kps, score = run.accuracy, run.avg_ttk, run.kills_per_second(), run.score
        if self.run_count == 0:
            self.ewma_accuracy, self.ewma_ttk, self.ewma_kps, self.ewma_score = acc, ttk, kps, score
        else:
            self.ewma_accuracy += a * (acc - self.ewma_accuracy)
            self.ewma_ttk += a * (ttk - self.ewma_ttk)
            self.ewma_kps += a * (kps - self.ewma_kps)
            self.ewma_score += a * (score - self.ewma_score)
        self.run_count += 1
        self.last_run_ts = run.started.isoformat()
        self.history.append(
            {
                "ts": self.last_run_ts,
                "accuracy": round(acc, 4),
                "avg_ttk": round(ttk, 4),
                "kps": round(kps, 4),
                "score": round(score, 2),
                "target_scale": round(self.target_scale, 4),
                "movement": round(self.movement, 4),
                "focus": self.last_focus,
            }
        )
        self.history = self.history[-500:]

    def observe_bias(self, bias_score: float, half_life: float = 5.0) -> None:
        """Fold one run's directional bias score into its EWMA (same
        convention as analysis.directional_bias: positive = left weaker)."""
        if self.run_count <= 1:
            self.ewma_bias = float(bias_score)
        else:
            self.ewma_bias += self._alpha(half_life) * (float(bias_score) - self.ewma_bias)

    def observe_fitts(self, fitts_slope_ms: float, half_life: float = 5.0) -> None:
        """Fold one run's Fitts slope (analysis/report.py: ms of flick time
        per bit of distance) into the throughput trend. The slow EWMA runs at
        4x the half-life; the fast EWMA falling below it is genuine motor
        improvement, sitting at/above it is a throughput stall (the signal
        the engine's Fitts controller acts on)."""
        f = float(fitts_slope_ms)
        if f <= 0:
            return
        if self.fitts_obs == 0:
            self.ewma_fitts_ms = f
            self.slow_fitts_ms = f
        else:
            self.ewma_fitts_ms += self._alpha(half_life) * (f - self.ewma_fitts_ms)
            self.slow_fitts_ms += self._alpha(4.0 * half_life) * (f - self.slow_fitts_ms)
        self.fitts_obs += 1

    # --------------------------------------------------------------- regions
    def region(
        self,
        key: str,
        prior_var: float | None = None,
        obs_noise: float | None = None,
    ) -> RegionPosterior:
        """Posterior for one region arm; prior_var/obs_noise apply only when
        the arm is created (existing arms keep their persisted parameters)."""
        if key not in self.regions:
            self.regions[key] = RegionPosterior(
                var=prior_var if prior_var is not None else 1.0,
                obs_noise=obs_noise if obs_noise is not None else 0.25,
            )
        return self.regions[key]

    def decay_regions(self, decay: float, prior_var: float = 1.0) -> None:
        """Forget region evidence: shrink means toward 0 and relax variances
        toward the prior, so old weaknesses re-open for exploration."""
        d = max(0.0, min(1.0, float(decay)))
        for post in self.regions.values():
            post.mean *= 1.0 - d
            post.var += d * (prior_var - post.var)

    def credit_focus_region(
        self,
        run: Run,
        prior_var: float | None = None,
        obs_noise: float | None = None,
    ) -> None:
        """Bandit reward: how much worse was this run vs. the player's baseline,
        attributed to the region that fire was concentrated on."""
        if self.last_focus is None or self.run_count == 0:
            return
        baseline = self.ewma_accuracy if self.ewma_accuracy > 0 else run.accuracy
        deficit = baseline - run.accuracy  # positive => struggled in focus region
        self.region(self.last_focus, prior_var, obs_noise).update(deficit)

    def credit_observed_regions(
        self,
        deficits: dict[str, float],
        weight: float = 1.0,
        prior_var: float | None = None,
        obs_noise: float | None = None,
    ) -> None:
        """Direct per-region rewards from flick telemetry (z-scored deficits).

        Far higher signal than run-level attribution: every flick's direction
        maps to the wall region it targeted, so all regions touched in a run
        get an observation, scaled by `weight` into the posterior's units
        (~accuracy-deficit scale, so z-scores are damped to obs_noise * weight).
        """
        damp = obs_noise if obs_noise is not None else 0.25
        for key, z in deficits.items():
            self.region(key, prior_var, obs_noise).update(damp * weight * float(z))

    # ------------------------------------------------------------- readiness
    BASELINE_RUNS = 24      # runs before the EWMAs are considered settled
    REGION_OBS = 3          # observations per arm before it counts as mapped
    BIAS_RUNS = 8           # runs of smoothing before bias evidence is trusted

    def readiness(self, region_count: int = 9) -> dict:
        """How calibrated the adaptive model is, 0..1 + what's still needed.

        Adaptation runs from run 1, but its decisions sharpen as evidence
        accumulates, and the ceiling is deliberately high — "dialed in"
        means roughly a training week, not a warm-up. Three components,
        weighted by how much each controller depends on them:

          baseline  runs folded into the EWMAs vs BASELINE_RUNS (size/pace
                    controllers trust their baseline from here on)
          regions   region arms with >= REGION_OBS observations (bandit
                    exploitation beats exploration once arms carry evidence)
          bias      |EWMA| distinguishable after BIAS_RUNS of smoothing
                    (dodge direction stays neutral until then — correct)
        """
        baseline = min(self.run_count / float(self.BASELINE_RUNS), 1.0)
        observed = sum(1 for p in self.regions.values() if p.n >= self.REGION_OBS)
        regions = min(observed / max(region_count, 1), 1.0)
        bias = 1.0 if (self.run_count >= self.BIAS_RUNS
                       and abs(self.ewma_bias) > 0.0) else \
            min(self.run_count / float(self.BIAS_RUNS), 1.0)
        score = 0.5 * baseline + 0.35 * regions + 0.15 * bias

        if score >= 0.999:
            stage = "dialed in"
        elif score >= 0.6:
            stage = "calibrating"
        elif score >= 0.25:
            stage = "learning"
        else:
            stage = "cold start"

        detail = [
            f"runs {min(self.run_count, self.BASELINE_RUNS)}/{self.BASELINE_RUNS}",
            f"regions {observed}/{region_count} mapped "
            f"({self.REGION_OBS}+ observations each)",
            ("bias evidence collected" if bias >= 1.0 else
             f"bias evidence {min(self.run_count, self.BIAS_RUNS)}/{self.BIAS_RUNS} runs"),
        ]
        missing: list[str] = []
        if baseline < 1.0:
            missing.append(f"{self.BASELINE_RUNS - self.run_count} more runs "
                           "for a settled baseline")
        if regions < 1.0:
            missing.append(f"{max(region_count - observed, 0)} wall regions "
                           "still need evidence")
        if not missing:
            msg = "dialed in — adaptation is running on settled evidence"
        else:
            msg = "calibrating: " + "; ".join(missing)
        return {"score": round(score, 3), "baseline": round(baseline, 3),
                "regions": round(regions, 3), "bias": round(bias, 3),
                "stage": stage, "detail": detail, "message": msg}

    # ----------------------------------------------------------- persistence
    @staticmethod
    def path_for(scenario: str, profile_dir: Path) -> Path:
        return profile_dir / "profiles" / f"{_slug(scenario)}.json"

    def save(self, profile_dir: Path) -> Path:
        p = self.path_for(self.scenario, profile_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v for k, v in self.__dict__.items() if k != "regions"}
        d["regions"] = {k: r.__dict__ for k, r in self.regions.items()}
        # Atomic replace: the GUI thread reloads this file on every report,
        # and an unclean shutdown mid-write must never brick the profile.
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(d, indent=2))
        tmp.replace(p)
        return p

    @classmethod
    def load(cls, scenario: str, profile_dir: Path) -> "PlayerProfile":
        p = cls.path_for(scenario, profile_dir)
        if not p.is_file():
            return cls(scenario=scenario)
        d = json.loads(p.read_text())
        regions = {k: RegionPosterior(**r) for k, r in d.pop("regions", {}).items()}
        prof = cls(**d)
        prof.regions = regions
        return prof
