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

### v0.3 — optimizer window + deep adaptation (shipped)
- **Optimizer window** (`kovadapt/optimize/` + `gui/optimizer_window.py`): hardware detection (CPU/GPU/RAM/refresh/Windows build via registry, ctypes, one CIM query); checkup with per-item Fix buttons + "fix all safe" (power plan via powercfg, HAGS registry state, per-exe fullscreen-optimization flags, SPI mouse-acceleration probe, Chromium-app scan, GameUserSettings.ini corruption scan with backup-then-delete fix, live game priority/affinity); watchdog thread auto-tuning every game launch (SMT-aware: frees CPU 0+1 with hyperthreading, CPU 0 without) with optional HKCU Run startup entry; hardware-matched launch options (+ documented myths) and settings advice (Reflex by GPU gen, HAGS by architecture, honest frame-gen guidance for RT-core GPUs). CLI: `checkup`, `watchdog`.
- **Adaptability internals exposed** (Adaptability tab): EWMA half-life, size–speed coupling, pace gain, min-shots gate, bandit prior variance / observation noise / posterior decay, with tooltips and reset-to-defaults.
- **Trace-informed dodge direction**: directional-bias EWMA on the profile skews Left/RightStrafeTimeMult (reciprocal pair, clamped 0.5–2.0) so targets strafe longer toward the weak side.
- **Session fatigue detection** (`analysis/fatigue.py`): Theil-Sen trend over per-run overshoot + flick-duration composite; fresh/declining/fatigued with break suggestions; optional plan easing that never contaminates persisted difficulty.
- **Per-archetype adaptation** (`adapt/archetype.py`): clicking/tracking/switching detected by name keywords then a shots-per-kill heuristic; per-archetype Settings overrides editable in the GUI.
- **Comprehensive telemetry**: left-button releases recorded (click-hold times), input-health metrics per run (polling-rate estimate, timing jitter IQR, worst-gap p99) surfaced in run summaries — high jitter points at the optimizer checkup.
- **Calibration readiness**: profile-level 0–100% indicator (baseline runs, region coverage, bias evidence) on the Dashboard and in `status`.
- **Full-run replay**: animated crosshair playback with flick-quality overlays (green clean / red flawed / ✕ shots), scrubber, wall-clock-accurate speeds, point decimation to stay lightweight.
- **Packaging**: PyInstaller one-dir spec + build script (`packaging/`), `kovadapt.exe` runs GUI/CLI/watchdog from one binary.

### v0.4 — the hub update (shipped)
Stability and speed from the old roadmap, plus the app becoming the place you start KovaaK's from instead of a companion you alt-tab to.

- **Launch integration** (`kovadapt/launcher.py`): the Dashboard's **Play adaptive task** starts the watcher, writes an adaptive playlist, and jumps the game straight into the adaptive scenario via KovaaK's official deep links (`steam://run/824270//?action=jump-to-scenario&name=…&mode=challenge` — the exact template the 3.9.x binary emits; boots the game cold or retargets a running instance in place). Everything launches through Steam, which enforces ownership itself; the install check verifies the game folder and the owning library's appmanifest, tModLoader-style. Playlists are written in the game's own JSON schema (`Saved/SaveGames/Playlists`, key casing preserved, UTF-8 no BOM). CLI: `play`.
- **In-game overlay** (`gui/overlay.py`): frameless, always-on-top, translucent session card — last run vs baseline EWMA, session run count, current difficulty, fatigue, input health, accuracy sparkline. Click-through via Qt's WindowTransparentForInput (no hooks, no injection), unlock-to-drag with persisted position, opacity slider. Needs Borderless/Windowed.
- **Theme system** (`gui/theme.py`): dark and light palettes rendered through one QSS generator; "auto" follows the Windows scheme live (`QStyleHints.colorSchemeChanged`). Views restyle in place — plots re-penned, status dots re-colored — no restart.
- **Onboarding** (`gui/onboarding.py`): four-page startup guide on first run, dismissible TIP bars on every tab (one × tucks them all away), Help menu with the guide, a hints toggle, and an open-data-folder shortcut.
- **Test coverage for the previously untested surfaces**: watcher end-to-end against a fake stats tree (bootstrap, mirrored report paths, stop semantics, surviving corrupt runs), CLI output contracts, GUI offscreen smoke tests, launcher URL/playlist pinning. Suite: 71 → 136 tests, still <3 s.
- **Analysis performance**: `build_report` shares one resample grid (`ResampleCache`; the 250 Hz heatmap grid is derived bitwise-exactly from the 500 Hz binning) — ~16 % faster per-run analysis at 4 kHz polling, ~2× that at 8 kHz. Long-session memory ceiling: `telemetry_retention_min` rolls the live recording (default 30 min ≈ 230 MB at 8 kHz; 0 restores the old unbounded behavior).
- **KovaaK's boosts**: new checkup probes for Game Bar/GameDVR background capture (safe one-click fix), Windows Game Mode, live timer resolution (`NtQueryTimerResolution`), and the active plan's core-parking floor (`powercfg /qh` — hidden on Win11, and the default Balanced plan really does park half the cores). HAGS is now verified against the **live driver state** via `D3DKMTQueryAdapterInfo` (WDDM 2.7 caps), not just registry intent. The watchdog timestamps every applied tune, and the optimizer window turns per-run input-health jitter into before/after evidence — boosts as measurements, not folklore.
- **Watcher robustness**: a stats CSV vanishing mid-poll no longer kills the loop, and the write-settle wait respects stop requests.

## Roadmap

### Later
- Triage field reports from the public release (issue templates with `kovadapt status` + checkup output).
- Installer polish: signed builds, winget manifest, in-app update check.
- Richer ML: per-flick Fitts-law residual model for skill tracking over weeks; bandit over dodge parameters, not just regions.
- Fatigue-aware session planner (suggest scenario order from profile deficits).
- Rust extension for the Raw Input pump + flick segmentation, only if profiling ever shows pure Python as the bottleneck.
