"""Configuration and path discovery."""

from __future__ import annotations

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
    # --- Telemetry & analysis (v0.2) ---
    telemetry_enabled: bool = True       # Raw Input mouse capture during watch
    telemetry_blend: float = 0.6         # weight of observed flick deficits vs run-level bandit credit
    clips_enabled: bool = False          # dxcam ring-buffer clips of notable moments (needs [clips] extra)
    clip_fps: int = 30
    clip_buffer_seconds: float = 90.0
    clip_scale: float = 0.5              # downscale factor for buffered frames
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
    def profile_path(self) -> Path:
        return Path(self.profile_dir)

    def save(self, path: Path | None = None) -> Path:
        path = path or self.profile_path / "settings.json"
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
