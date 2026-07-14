# kovadapt — adaptive KovaaK's

Free, ML-driven adaptive training for [KovaaK's](https://store.steampowered.com/app/824270/KovaaKs/). kovadapt watches your runs and regenerates scenario variants between runs so the task itself trains your weaknesses:

- **Weak-side targeting** — real mouse telemetry (Raw Input) segments every flick, maps each one to the wall region it aimed at, and a Thompson-sampling bandit shifts spawns toward the regions where you overshoot, hesitate, or lag.
- **Adaptive target size** — a controller keeps your hit rate in the 60–80% training sweet spot, with a size floor coupled to target speed.
- **Anti-autopilot movement** — an Ornstein-Uhlenbeck process drifts micro-movement intensity unpredictably, so muscle memory alone never solves the task.
- **Post-run analysis** — directional bias, aim-travel heatmap, and clip-worthy notable moments (worst overshoots, hesitations, one clean reference flick) with trajectory replays and optional real video clips.
- **Optimizer** — the free basics of Process Lasso for KovaaK's: High priority + CPU 0/1 freed, plus a researched tuning checklist.

KovaaK's has no modding API, so adaptation happens **between** runs: each finished run updates your profile and rewrites `<Scenario> [Adaptive].sce`. Load the adaptive variant in-game; every run reshapes the next.

## Install

```
pip install -e .[gui]          # core + desktop app
pip install -e .[gui,clips]    # + video clips of notable moments (dxcam)
```

Requires Python 3.10+ on Windows (analysis and tests run anywhere; capture is Windows-only).

## Use

```
kovadapt gui                        # desktop app: dashboard, analysis, config, optimizer
kovadapt scenarios [filter]         # list installed scenarios
kovadapt watch "1wall 6targets small"   # headless adaptation loop
kovadapt status "1wall 6targets small"  # learned profile + region heatmap
kovadapt replay "1wall 6targets small"  # bootstrap profile from stats history
```

Start `kovadapt gui`, pick a scenario, hit **Start adapting**, and play `<Scenario> [Adaptive]` in KovaaK's. After each run the Analysis tab fills in with your movement report.

Settings live at `~/.kovadapt/settings.json`; profiles, traces, reports, and clips under the same directory.

## How it works

Each finished run's stats CSV is parsed and joined with the raw mouse trace for that run's time window. Flicks are segmented click-anchored (movement onset → click), and characterized by amplitude, direction, peak speed, overshoot along the flick axis, and corrective submovements. Per-region deficits (overshoot + corrections + slowness, z-scored) feed conjugate-normal posteriors; Thompson sampling picks the next focus region; spawn points are resampled toward it; target profiles are rescaled; dodge parameters are jittered by the OU state. See `FEATURES.md` for the full feature map and roadmap.

## Development

```
pip install -e .[dev]
pytest
```

Test suites run on any OS — Windows-only capture (Raw Input, dxcam) is import-guarded.
