"""Thompson-sampling bandit over spawn regions.

Arms = cells of a (cols x rows) grid over the wall plane. Each arm holds a
Gaussian posterior over the player's *skill deficit* there (see
profile.player.RegionPosterior). Sampling the posterior and picking the argmax
naturally balances exploring untested regions against exploiting known
weaknesses.
"""

from __future__ import annotations

import numpy as np

from ..profile.player import PlayerProfile


def region_keys(cols: int, rows: int) -> list[str]:
    return [f"r{r}c{c}" for r in range(rows) for c in range(cols)]


class ThompsonRegionBandit:
    def __init__(self, profile: PlayerProfile, cols: int, rows: int,
                 rng: np.random.Generator | None = None,
                 prior_var: float | None = None,
                 obs_noise: float | None = None) -> None:
        self.profile = profile
        self.keys = region_keys(cols, rows)
        self.rng = rng or np.random.default_rng()
        self.prior_var = prior_var    # applied only when an arm is first created
        self.obs_noise = obs_noise

    def _region(self, key: str):
        return self.profile.region(key, self.prior_var, self.obs_noise)

    def choose_focus(self) -> str:
        """Thompson sample: draw a deficit from each region's posterior, focus
        the region with the largest draw."""
        draws = []
        for k in self.keys:
            post = self._region(k)
            draws.append(self.rng.normal(post.mean, np.sqrt(post.var)))
        return self.keys[int(np.argmax(draws))]

    def spawn_weights(self, focus: str, focus_weight: float = 0.5) -> dict[str, float]:
        """Probability mass per region: `focus_weight` on the focus region,
        the remainder spread over the others proportional to posterior means
        (softmax), so secondary weak spots also get extra fire."""
        others = [k for k in self.keys if k != focus]
        if not others:
            return {focus: 1.0}
        means = np.array([self._region(k).mean for k in others])
        w = np.exp(means - means.max())
        w = w / w.sum() * (1.0 - focus_weight)
        out = {k: float(v) for k, v in zip(others, w)}
        out[focus] = float(focus_weight)
        return out
