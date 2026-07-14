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

    # --------------------------------------------------------------- regions
    def region(self, key: str) -> RegionPosterior:
        if key not in self.regions:
            self.regions[key] = RegionPosterior()
        return self.regions[key]

    def credit_focus_region(self, run: Run) -> None:
        """Bandit reward: how much worse was this run vs. the player's baseline,
        attributed to the region that fire was concentrated on."""
        if self.last_focus is None or self.run_count == 0:
            return
        baseline = self.ewma_accuracy if self.ewma_accuracy > 0 else run.accuracy
        deficit = baseline - run.accuracy  # positive => struggled in focus region
        self.region(self.last_focus).update(deficit)

    def credit_observed_regions(self, deficits: dict[str, float], weight: float = 1.0) -> None:
        """Direct per-region rewards from flick telemetry (z-scored deficits).

        Far higher signal than run-level attribution: every flick's direction
        maps to the wall region it targeted, so all regions touched in a run
        get an observation, scaled by `weight` into the posterior's units
        (~accuracy-deficit scale, so z-scores are damped to +-0.25 * weight).
        """
        for key, z in deficits.items():
            self.region(key).update(0.25 * weight * float(z))

    # ----------------------------------------------------------- persistence
    @staticmethod
    def path_for(scenario: str, profile_dir: Path) -> Path:
        return profile_dir / "profiles" / f"{_slug(scenario)}.json"

    def save(self, profile_dir: Path) -> Path:
        p = self.path_for(self.scenario, profile_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v for k, v in self.__dict__.items() if k != "regions"}
        d["regions"] = {k: r.__dict__ for k, r in self.regions.items()}
        p.write_text(json.dumps(d, indent=2))
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
