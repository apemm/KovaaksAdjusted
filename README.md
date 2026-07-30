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
- **Play from the app** — one button starts the adaptation loop, queues an adaptive playlist, and jumps KovaaK's straight into the adaptive scenario through Steam's deep-link protocol (the game's own mechanism since 3.0.0). Ownership stays where it belongs: everything launches through Steam, which won't start a game your account doesn't own.
- **In-game overlay** — a toggleable, click-through, translucent card over the game: last run vs your baseline, session count, difficulty, fatigue, input health, accuracy sparkline. Drag it anywhere, tune its opacity; needs Borderless/Windowed mode.
- **Themes** — dark and light, or auto-synced to the Windows theme, switchable live.
- **Startup guide & hints** — a short first-run guide plus contextual TIP bars on every tab; one click tucks them all away, the Help menu brings them back.
- **A coach that cites its sources** — every run gets insight cards grounded in a research-distilled knowledge base (Fitts's law and throughput, the two-phase flick literature, Aimer7's guide, Voltaic doctrine): the exact numbers that triggered each insight, what they mean, what to try, and the citations behind it. Cross-session skill curves separate real progress from score plateaus. Nothing is forced, and nothing is claimed without evidence.
- **Scenario browser** — every installed scenario with its training state, searchable, one click to play or start adapting.

KovaaK's has no modding API, so adaptation happens **between** runs: each finished run updates your profile and rewrites `<Scenario> [Adaptive].sce`. Every run reshapes the next — and the app drops you straight into the current variant.

## Install

```
git clone https://github.com/apemm/KovaaksAdjusted.git
cd KovaaksAdjusted
pip install -e .[gui]          # core + desktop app
pip install -e .[gui,clips]    # + video clips of notable moments (dxcam)
```

Requires Python 3.10+ on Windows (analysis and tests run anywhere; capture is Windows-only). For a no-Python install, build the standalone exe: `powershell -File packaging/build.ps1` → `dist/kovadapt/kovadapt.exe`.

kovadapt looks for the game in the usual Steam locations (`C:\Program Files (x86)\Steam`, `C:\Program Files\Steam`, `D:\SteamLibrary`). If yours lives elsewhere, set the `KOVAAKS_ROOT` environment variable to the `FPSAimTrainer` folder, or put the path in the `kovaaks_root` field of `~/.kovadapt/settings.json` — a value there takes priority over the environment variable.

## Use

```
kovadapt gui                        # desktop app: dashboard, analysis, adaptability, optimizer
kovadapt scenarios [filter]         # list installed scenarios
kovadapt play "1wall 6targets small"    # jump the game into the adaptive variant
kovadapt watch "1wall 6targets small"   # headless adaptation loop
kovadapt status "1wall 6targets small"  # learned profile + region heatmap + calibration
kovadapt replay "1wall 6targets small"  # bootstrap profile from stats history
kovadapt checkup                    # print the system optimization checkup
kovadapt watchdog                   # headless auto-tune on every game launch
```

Start `kovadapt gui`, pick a scenario, hit **Play adaptive task** — the app starts watching, queues the adaptive playlist, and opens KovaaK's directly in `<Scenario> [Adaptive]` (Steam must be running). After each run the Analysis tab fills in with your movement report and replay, and the overlay (if toggled on) tracks the session over the game. **Start adapting** does the same without launching the game, for when it's already open on the base scenario.

Settings live at `~/.kovadapt/settings.json`; profiles, traces, reports, and clips under the same directory.

## Optimizer

The Optimizer tab opens a separate window that replaces the paid Process Lasso workflow for KovaaK's:

- **Checkup** scans power plan, HAGS, fullscreen optimizations, mouse acceleration, background Chromium apps, KovaaK's config-file health, and the game process — each finding has a plain-language explanation and its own Fix button (plus "Fix all safe items" for the per-user, reversible ones). Nothing is changed without a click.
- **Watchdog** re-applies High priority and frees the input core (CPU 0+1 with hyperthreading, CPU 0 without — Windows resets this every launch) automatically whenever the game starts, with an optional start-with-Windows entry (per-user registry Run key, no admin).
- **Advice** is matched to detected hardware: Reflex vs Ultra Low Latency by GPU generation, HAGS by architecture, fps caps from your monitor's refresh rate, and an honest note on why driver frame generation stays off for training.

Minimal invasiveness is a design rule: probes are read-only, automated fixes are per-user (HKCU) and reversible, invasive ones (power plan, config deletion) always require their own explicit click, and the only process kovadapt ever modifies is the game's.

## How it works

KovaaK's has no modding API, so kovadapt never touches a scenario you are playing. Instead, it treats every finished run as one observation in a longer experiment: when the game writes a run's stats CSV, kovadapt folds that run into a per-scenario model of your aim and rewrites `<Scenario> [Adaptive].sce` before the next load. The file the game reads is an ordinary scenario; the process that produced it is not.

The model draws on two kinds of measurement. The stats CSV records outcomes — what was hit and how long each kill took — but outcomes conflate distinct failures: a miss may indicate a badly aimed flick or a well-aimed flick that arrived too slowly. Hence the second source. A background thread registers for Raw Input and records relative mouse deltas exactly as the game receives them, immune to Windows pointer ballistics. When a run ends, its window of the recording is cut into flicks by working backward from each click to the onset of movement (the point where speed rises through 8% of the segment's peak, clamped at the previous click), and each flick is characterized by amplitude, direction, peak speed, overshoot along the flick axis, and the corrective submovements needed to settle. These features separate the miss that came from overshooting from the miss that came from hesitating, which is precisely the distinction accuracy alone cannot make.

Adaptation runs on three controllers. The wall is divided into a 5×5 grid, and each region carries a Gaussian posterior over how much weaker you are there than your own average; per-region deficits from telemetry — overshoot, corrections, and Fitts-normalized slowness, z-scored — update those posteriors after every run. Focus is chosen by Thompson sampling: draw one sample from each posterior and commit to the worst draw. Most draws exploit the model's current belief, but roughly a fifth land elsewhere, and this waste is deliberate: a weak side moves as you improve, so a purely greedy policy would keep drilling a weakness that no longer exists. Alongside the bandit, a size controller holds hit rate inside an 85–95% band — difficult enough to force adaptation, comfortable enough to permit it, and biased so that falling below the floor grows targets harder than sitting above it shrinks them — and an Ornstein-Uhlenbeck process drifts target micro-movement between runs so that no two variants reward the same memorized rhythm; strafe timing is skewed toward whichever direction your flicks measurably favor less.

Two constraints shape everything downstream. Edits always apply to the base scenario, never to the previous variant, so multipliers remain absolute and difficulty cannot compound silently across sessions. And the plan's seed is the only randomness that reaches the generator, so a given plan regenerates its `.sce` byte for byte — any variant you have ever played can be reproduced exactly.

Finally, a session-level Theil-Sen fit over per-run flick quality watches for fatigue. When the trend declines, kovadapt suggests a break and can ease the emitted plan — larger, calmer targets — without ever writing that easing into the profile, so the next session resumes at your true difficulty rather than the tired one. The intent throughout is the same as a good coach's: not to make the drill harder, but to keep it aimed at whatever you are currently worst at.

See `FEATURES.md` for the full feature map and roadmap.

## Development

```
pip install -e .[dev]
pytest
```

Test suites run on any OS — Windows-only capture (Raw Input, dxcam) is import-guarded.

## License

MIT — see `LICENSE`.
