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

    The clicking baseline band (0.85-0.95) is the primary-sourced doctrine
    (analysis/kb.py: p-speed-accuracy-governor). Per kb.py GAPS there is NO
    primary tracking or switching band, so both bands below are kovadapt
    extrapolations of the same control law:

    tracking:  accuracy is per-tick time-on-target, which runs lower than
               click accuracy (every off-target tick counts against you), so
               the band sits below the clicking doctrine at ~0.70-0.88;
               movement is the difficulty axis, so it never drops to zero
               and sizing reacts more gently.
    switching: flick volume is the point, so the band trades accuracy for
               attempts — ~0.65-0.85, looser than clicking but still tight
               enough that sprayed flicks pull difficulty back down — and
               more spawn mass lands on the weak region.
    """
    return {
        "clicking": {},
        "tracking": {  # kovadapt extrapolation (no primary tracking band)
            "target_accuracy_low": 0.70,
            "target_accuracy_high": 0.88,
            "size_learning_rate": 0.6,
            "min_movement": 0.35,
        },
        "switching": {  # kovadapt extrapolation (no primary switching band)
            "target_accuracy_low": 0.65,
            "target_accuracy_high": 0.85,
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
    # Defaults are the primary-sourced clicking doctrine — hold accuracy in an explicit
    # 85-95% band and train speed against it (analysis/kb.py: p-speed-accuracy-governor,
    # dx-acc-above-band). Tracking/switching bands are labeled extrapolations set in
    # default_archetype_overrides().
    target_accuracy_low: float = 0.85
    target_accuracy_high: float = 0.95
    # Target size clamps (multipliers on the base scenario's bounding-box size).
    min_target_scale: float = 0.40
    max_target_scale: float = 2.50
    size_learning_rate: float = 0.9  # gain on log-scale size updates
    # Spawn region grid over the wall plane (columns x rows). 5x5 pairs with the
    # amplitude-aware flick->region mapping (analysis/movement.py:region_deficits):
    # short flicks credit inner cells, long flicks the edges.
    region_cols: int = 5
    region_rows: int = 5
    focus_weight: float = 0.5  # probability mass concentrated on the bandit's focus region
    # Ornstein-Uhlenbeck micro-movement process.
    ou_theta: float = 0.35  # mean reversion rate (per run)
    ou_sigma: float = 0.25  # diffusion
    # Movement intensity clamps (0 = static targets, 1 = max configured jitter).
    min_movement: float = 0.0
    max_movement: float = 1.0
    # EWMA half-life (runs) for profile statistics.
    ewma_half_life: float = 5.0
    # --- Advanced engine internals ---
    size_speed_coupling: float = 0.35    # size floor grows with target speed
    pace_coupling_gain: float = 0.4      # kills/s above norm -> OU push
    min_shots_for_size: int = 10         # size controller needs this many shots
    bandit_obs_noise: float = 0.25       # observation noise of region posteriors
    bandit_prior_var: float = 1.0        # prior variance of fresh region arms
    # Per-run forgetting toward the prior. Default on since v0.4: weaknesses
    # re-open as you improve, so mapped arms must not stay pinned forever.
    bandit_posterior_decay: float = 0.03
    # --- Doctrine-aligned progression controllers (analysis/kb.py citations) ---
    # Fitts throughput controller: accuracy comfortable (inside the band) but the
    # ms-per-bit EWMA no longer improving -> shrink targets one extra step per run
    # to push difficulty back to the challenge point (p-challenge-point,
    # p-fitts-throughput). 0 disables.
    fitts_control_gain: float = 0.35
    # Pace plateau push: accuracy in band + flat kills/s across recent history ->
    # bounded upward push on the movement target through the existing OU/pace
    # pathway (p-speed-is-growth-axis). 0 disables.
    pace_progression_gain: float = 0.3
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
    # --- Sensitivity context: analysis input only ---
    # kovadapt never changes your sens; these feed the both-sided per-task
    # cm/360 reasoning in analysis/sens.py. cm/360 = 2.54*360/(dpi*sens*0.022)
    # — KovaaK's (Quake-lineage) yaw is 0.022 deg per mouse count at sens 1.0.
    # Defaults are the common 800 dpi / 1.0; set either to 0 to mark
    # sensitivity as not configured (disables the sensitivity card).
    mouse_dpi: float = 800.0
    game_sens: float = 1.0
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
    skip_splash: bool = False            # jump straight to the window (no LED opening)
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
