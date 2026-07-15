# kovadapt — adaptive KovaaK's

Free, ML-driven adaptive training for [KovaaK's](https://store.steampowered.com/app/824270/KovaaKs/). kovadapt watches your runs and regenerates scenario variants between runs so the task itself trains your weaknesses:

- **Weak-side targeting** — real mouse telemetry (Raw Input) segments every flick, maps each one to the wall region it aimed at, and a Thompson-sampling bandit shifts spawns toward the regions where you overshoot, hesitate, or lag.
- **Adaptive target size** — a controller keeps your hit rate in the 60–80% training sweet spot, with a size floor coupled to target speed.
- **Anti-autopilot movement** — an Ornstein-Uhlenbeck process drifts micro-movement intensity unpredictably, and targets strafe longer toward your measured weak flick direction.
- **Per-archetype adaptation** — clicking, tracking, and target-switching scenarios get their own controller parameters (auto-detected).
- **Fatigue detection** — robust trend fitting over each session's flick quality; suggests breaks before you feel the decay, optionally easing difficulty.
- **Calibration indicator** — the dashboard shows how much baseline data the model still wants before its decisions are fully evidence-backed.
- **Post-run analysis & replay** — directional bias, aim-travel heatmap, notable moments (worst overshoots/hesitations, one clean reference flick), and an animated full-run replay with flick-quality overlays: green = clean, red = overshoot/correction, ✕ = shot.
- **Optimizer window** — the free Process Lasso replacement: hardware detection, a one-click system checkup with per-item fixes, a watchdog that applies High priority and frees the input-processing core on every game launch, and launch options + settings matched to your GPU/monitor.

KovaaK's has no modding API, so adaptation happens **between** runs: each finished run updates your profile and rewrites `<Scenario> [Adaptive].sce`. Load the adaptive variant in-game; every run reshapes the next.

## Install

```
pip install -e .[gui]          # core + desktop app
pip install -e .[gui,clips]    # + video clips of notable moments (dxcam)
```

Requires Python 3.10+ on Windows (analysis and tests run anywhere; capture is Windows-only). For a no-Python install, build the standalone exe: `powershell -File packaging/build.ps1` → `dist/kovadapt/kovadapt.exe`.

## Use

```
kovadapt gui                        # desktop app: dashboard, analysis, config, optimizer
kovadapt scenarios [filter]         # list installed scenarios
kovadapt watch "1wall 6targets small"   # headless adaptation loop
kovadapt status "1wall 6targets small"  # learned profile + region heatmap + calibration
kovadapt replay "1wall 6targets small"  # bootstrap profile from stats history
kovadapt checkup                    # print the system optimization checkup
kovadapt watchdog                   # headless auto-tune on every game launch
```

Start `kovadapt gui`, pick a scenario, hit **Start adapting**, and play `<Scenario> [Adaptive]` in KovaaK's. After each run the Analysis tab fills in with your movement report and replay.

Settings live at `~/.kovadapt/settings.json`; profiles, traces, reports, and clips under the same directory.

## Optimizer

The Optimizer tab opens a separate window that replaces the paid Process Lasso workflow for KovaaK's:

- **Checkup** scans power plan, HAGS, fullscreen optimizations, mouse acceleration, background Chromium apps, KovaaK's config-file health, and the game process — each finding has a plain-language explanation and its own Fix button (plus "Fix all safe items" for the per-user, reversible ones). Nothing is changed without a click.
- **Watchdog** re-applies High priority and frees the input core (CPU 0+1 with hyperthreading, CPU 0 without — Windows resets this every launch) automatically whenever the game starts, with an optional start-with-Windows entry (per-user registry Run key, no admin).
- **Advice** is matched to detected hardware: Reflex vs Ultra Low Latency by GPU generation, HAGS by architecture, fps caps from your monitor's refresh rate, and an honest note on why driver frame generation stays off for training.

Minimal invasiveness is a design rule: probes are read-only, automated fixes are per-user (HKCU) and reversible, invasive ones (power plan, config deletion) always require their own explicit click, and the only process kovadapt ever modifies is the game's.

## How it works

Each finished run's stats CSV is parsed and joined with the raw mouse trace for that run's time window. Flicks are segmented click-anchored (movement onset → click), and characterized by amplitude, direction, peak speed, overshoot along the flick axis, and corrective submovements. Per-region deficits (overshoot + corrections + slowness, z-scored) feed conjugate-normal posteriors; Thompson sampling picks the next focus region; spawn points are resampled toward it; target profiles are rescaled; dodge parameters are jittered by the OU state and skewed toward your weak side. Session fatigue is a Theil-Sen trend over per-run flick quality. See `FEATURES.md` for the full feature map and roadmap.

## Development

```
pip install -e .[dev]
pytest
```

Test suites run on any OS — Windows-only capture (Raw Input, dxcam) is import-guarded.
