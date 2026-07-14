# kovadapt feature map

## Shipped

### v0.1 — adaptation core
- Stats CSV parser (per-kill events + summary) and per-scenario player profile (EWMA accuracy/TTK/pace/score).
- `.sce` surgical editor: byte-identical round-trip, target size scaling (`MainBBRadius/Height`), dodge-profile patching, spawn-point resampling by region weights.
- Thompson-sampling region bandit (Gaussian posteriors over a 3×3 wall grid).
- Size controller holding hit rate in the 60–80% sweet spot (multiplicative log-scale updates).
- Ornstein-Uhlenbeck micro-movement drift + per-run dodge jitter (anti-autopilot).
- Session watcher: stats folder polling → observe → plan → regenerate `[Adaptive]` variant. CLI: `scenarios`, `watch`, `generate`, `status`, `replay`.

### v0.2 — telemetry, analysis, GUI
- **Raw Input mouse telemetry** (`kovadapt/telemetry/`): message-only window on a background thread receives relative deltas exactly as the game does (immune to pointer ballistics/cursor clipping). Chunked preallocated buffers; `snapshot()` slices runs out of a live recording. Traces stored as compressed npz.
- **Flick analysis** (`kovadapt/analysis/`): click-anchored segmentation (onset at 8% of segment peak, lookback clamped at the previous click), overshoot along the flick axis, hysteresis-counted corrective submovements, directional bias score, per-region deficits (z-scored) that feed the bandit as *observed* rewards — replacing run-level attribution when telemetry exists.
- **Notable moments**: worst overshoots, hesitations (correction chains), Fitts-normalized slow flicks, plus one clean reference flick; serialized in per-run JSON reports.
- **Clip capture** (`kovadapt/capture/`, optional `[clips]` extra): dxcam ring buffer (~90s, downscaled) → mp4 segments around each notable moment.
- **GUI** (`kovadapt/gui/`, `[gui]` extra): dark PySide6 + pyqtgraph app — Dashboard (scenario picker, start/stop, live log, accuracy trend), Analysis (summary, direction-bias bars, aim-travel heatmap, notable-moment list with trajectory replays and clip playback side by side), Adaptability (full settings surface), Optimizer (below).
- **Optimizer tab (basics)**: detect `FPSAimTrainer`, set High priority, free CPU 0/1 (psutil) — the free replacement for the Process Lasso workflow — plus the researched manual checklist.

## Performance research notes (v0.2, feeds v0.3 auto-checkup)

- KovaaK's own FAQ recommends High process priority and keeping the game off CPU 0/1 because mouse input is processed on the first core (kovaaks.com FAQ).
- NVIDIA Reflex / Low Latency Mode is the largest input-latency win on supported GPUs.
- Exclusive fullscreen + disabling "fullscreen optimizations" on FPSAimTrainer.exe avoids DWM compositing latency (PCGamingWiki).
- Corrupt `GameUserSettings.ini` (`%localappdata%\FPSAimTrainer\Saved\Config\WindowsNoEditor\`) is a documented stutter cause; deleting it forces a clean rebuild.
- Ultimate Performance power plan prevents core parking/downclocking mid-run.
- HAGS helps on RTX 3000+/RX 6000+; hurts on older GPUs.
- Chromium-based apps (Discord, Spotify, browsers) degrade timer resolution and cause frame-time spikes; overlays worse.
- Stable FPS cap beats higher-but-unstable uncapped FPS; 1ms timer resolution matters on Windows 10 (Windows 11 handles it per-process).

## Roadmap

### v0.3 — optimizer window (full)
- Hardware detection (GPU/CPU/monitor Hz) → recommended in-game settings profile.
- One-click automatic checkup: power plan, HAGS state, fullscreen-optimization flags, config-file corruption scan, running Chromium apps, timer resolution.
- Watchdog mode: auto-apply priority/affinity whenever the game launches.
- Launch-options manager.

### v0.4+
- More adaptive task archetypes (tracking, target-switching; per-archetype knobs).
- Trace-informed dodge direction (dodge toward the player's weak flick direction).
- Session-level fatigue detection (flick quality decay over a session → suggest breaks).
- Packaged installer (PyInstaller) so no Python required.
