"""Configuration and path discovery."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

_STEAM_CANDIDATES = (
    r"C:\Program Files (x86)\Steam\steamapps\common\FPSAimTrainer\FPSAimTrainer",
    r"C:\Program Files\Steam\steamapps\common\FPSAimTrainer\FPSAimTrainer",
    r"D:\SteamLibrary\steamapps\common\FPSAimTrainer\FPSAimTrainer",
)

ADAPTIVE_SUFFIX = " [Adaptive]"

# Scenario archetypes with distinct adaptation dynamics. "clicking" is the
# baseline; overrides below shift the controllers for the other two.
ARCHETYPES = ("clicking", "tracking", "switching")


def default_archetype_overrides() -> dict[str, dict[str, float]]:
    """Per-archetype Settings overrides (keys must be Settings field names).

    tracking:  accuracy is per-tick hit rate (runs much higher than clicking),
               so the sweet spot moves up; movement is the difficulty axis, so
               it never drops to zero and sizing reacts more gently.
    switching: flick volume is the point — slightly looser accuracy band and
               more spawn mass on the weak region.
    """
    return {
        "clicking": {},
        "tracking": {
            "target_accuracy_low": 0.70,
            "target_accuracy_high": 0.88,
            "size_learning_rate": 0.6,
            "min_movement": 0.35,
        },
        "switching": {
            "target_accuracy_low": 0.55,
            "target_accuracy_high": 0.75,
            "focus_weight": 0.6,
        },
    }


def find_kovaaks_root() -> Path | None:
    """Locate the FPSAimTrainer directory (env var KOVAAKS_ROOT wins)."""
    env = os.environ.get("KOVAAKS_ROOT")
    if env and Path(env).is_dir():
        return Path(env)
    for cand in _STEAM_CANDIDATES:
        p = Path(cand)
        if (p / "stats").is_dir():
            return p
    return None


@dataclass
class Settings:
    """All tunables for the adaptation loop."""

    kovaaks_root: str = ""
    # Difficulty controller: keep per-run hit rate inside this band ("training sweet spot").
    target_accuracy_low: float = 0.60
    target_accuracy_high: float = 0.80
    # Target size clamps (multipliers on the base scenario's bounding-box size).
    min_target_scale: float = 0.40
    max_target_scale: float = 2.50
    size_learning_rate: float = 0.9  # gain on log-scale size updates
    # Spawn region grid over the wall plane (columns x rows).
    region_cols: int = 3
    region_rows: int = 3
    focus_weight: float = 0.5  # probability mass concentrated on the bandit's focus region
    # Ornstein-Uhlenbeck micro-movement process.
    ou_theta: float = 0.35  # mean reversion rate (per run)
    ou_sigma: float = 0.25  # diffusion
    # Movement intensity clamps (0 = static targets, 1 = max configured jitter).
    min_movement: float = 0.0
    max_movement: float = 1.0
    # EWMA half-life (runs) for profile statistics.
    ewma_half_life: float = 5.0
    # --- Advanced engine internals (defaults preserve pre-v0.3 behavior) ---
    size_speed_coupling: float = 0.35    # size floor grows with target speed
    pace_coupling_gain: float = 0.4      # kills/s above norm -> OU push
    min_shots_for_size: int = 10         # size controller needs this many shots
    bandit_obs_noise: float = 0.25       # observation noise of region posteriors
    bandit_prior_var: float = 1.0        # prior variance of fresh region arms
    bandit_posterior_decay: float = 0.0  # per-run forgetting toward the prior
    # --- Trace-informed dodge direction ---
    dodge_bias_enabled: bool = True      # strafe longer toward the weak side
    dodge_bias_gain: float = 0.8         # scales EWMA bias into strafe asymmetry
    # --- Session fatigue detection ---
    fatigue_detection_enabled: bool = True
    fatigue_sensitivity: float = 1.0     # >1 = flags fatigue sooner
    fatigue_min_runs: int = 5            # runs before a trend is trusted
    fatigue_easing: bool = False         # ease difficulty when fatigued
    # --- Per-archetype adaptation ---
    archetype_enabled: bool = True
    archetype_overrides: dict = field(default_factory=default_archetype_overrides)
    # --- Telemetry & analysis (v0.2) ---
    telemetry_enabled: bool = True       # Raw Input mouse capture during watch
    telemetry_blend: float = 0.6         # weight of observed flick deficits vs run-level bandit credit
    # Rolling retention (minutes) of the live mouse recording; runs are sliced
    # out within seconds of ending, so only the recent window is ever needed.
    # Bounds a multi-hour watch session to ~230 MB at 8 kHz (vs ~460 MB/hour
    # unbounded). 0 = keep the whole session in memory.
    telemetry_retention_min: float = 30.0
    clips_enabled: bool = False          # dxcam ring-buffer clips of notable moments (needs [clips] extra)
    clip_fps: int = 30
    clip_buffer_seconds: float = 90.0
    clip_scale: float = 0.5              # downscale factor for buffered frames
    # --- App shell (v0.4+): theme, overlay, onboarding ---
    theme: str = "auto"                  # auto | dark | light ("auto" follows Windows)
    accent: str = "indigo"               # accent preset (gui/theme.py ACCENTS)
    overlay_opacity: float = 0.9         # in-game overlay window opacity (0.3-1.0)
    overlay_clickthrough: bool = True    # overlay ignores the mouse (position with Unlock)
    overlay_autoshow: bool = False       # pop the overlay whenever watching starts
    overlay_x: int = -1                  # -1 = default corner (top-right of primary screen)
    overlay_y: int = -1
    show_hints: bool = True              # contextual hint bars across the app
    onboarding_done: bool = False        # startup guide shown once until dismissed
    profile_dir: str = str(Path.home() / ".kovadapt")

    root: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.kovaaks_root:
            found = find_kovaaks_root()
            self.kovaaks_root = str(found) if found else ""
        self.root = Path(self.kovaaks_root) if self.kovaaks_root else Path(".")

    @property
    def stats_dir(self) -> Path:
        return self.root / "stats"

    @property
    def scenarios_dir(self) -> Path:
        return self.root / "Saved" / "SaveGames" / "Scenarios"

    @property
    def playlists_dir(self) -> Path:
        return self.root / "Saved" / "SaveGames" / "Playlists"

    @property
    def profile_path(self) -> Path:
        return Path(self.profile_dir)

    def for_archetype(self, archetype: str) -> "Settings":
        """Effective settings for a scenario archetype (self when no overrides)."""
        if not archetype or not self.archetype_enabled:
            return self
        ov = (self.archetype_overrides or {}).get(archetype) or {}
        known = {f for f in self.__dataclass_fields__ if f != "root"}
        ov = {k: v for k, v in ov.items() if k in known}
        return dataclasses.replace(self, **ov) if ov else self

    def save(self, path: Path | None = None) -> Path:
        # Default to the canonical bootstrap file load() reads. profile_dir is
        # itself a settings.json field, so saving next to a customized
        # profile_dir would write a file no startup ever loads again.
        path = path or Path.home() / ".kovadapt" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d.pop("root", None)
        path.write_text(json.dumps(d, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or Path.home() / ".kovadapt" / "settings.json"
        if path.is_file():
            d = json.loads(path.read_text())
            # Tolerate settings files written by other versions.
            known = {f for f in cls.__dataclass_fields__ if f != "root"}
            return cls(**{k: v for k, v in d.items() if k in known})
        return cls()
