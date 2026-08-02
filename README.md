# kovadapt — adaptive KovaaK's

Free, ML-driven adaptive training for [KovaaK's](https://store.steampowered.com/app/824270/KovaaKs/). KovaaK's has no modding API, so kovadapt adapts **between** runs: each finished run updates a per-scenario model of your aim and rewrites `<Scenario> [Adaptive].sce` before the next load, so the task itself shifts toward whatever you are currently worst at.

- **Weak-side targeting** — Raw Input telemetry segments every flick, maps it to the wall region it aimed at, and a Thompson-sampling bandit shifts spawns toward the 5×5 regions where you overshoot, hesitate, or lag.
- **Adaptive size, unpredictable movement** — a controller holds your hit rate inside an 85–95% band, biased so falling below the floor grows targets harder than sitting above it shrinks them; an Ornstein-Uhlenbeck process drifts micro-movement so no two variants reward the same memorized rhythm, and strafe timing skews toward whichever direction your flicks measurably favour less.
- **Per-archetype adaptation** — clicking, tracking and switching scenarios get their own controller parameters, auto-detected. Pure-tracking scenarios report `Kills: 0` because their targets are invincible, which is exactly the case the detector has to get right.
- **Fatigue detection** — a Theil-Sen fit over each session's flick quality suggests a break before you feel the decay, and can ease the emitted plan without writing that easing into your profile.
- **Post-run analysis & replay** — directional bias, aim-travel heatmap, notable moments, and an animated full-run replay with flick-quality overlays: green clean, red overshoot, ✕ shot.
- **What changed** — per task, what the model has actually done to that scenario and why, each row carrying the runs that moved it. Edits always apply to the base file, so the before-and-after is read out of two files rather than inferred: a measurement, not a reconstruction.
- **Findings withheld when the data cannot carry them** — flick microstructure is unreadable through noisy input timing, so when jitter or polling says the timing is bad, every overshoot and directional verdict on the page is withheld at once. One rule, applied everywhere, so no two panels can disagree about the same run.
- **A coach that cites its sources** — insight cards grounded in a research-distilled knowledge base (Fitts's law and throughput, the two-phase flick literature, Aimer7, Voltaic doctrine): the numbers that triggered each insight, what they mean, what to try, and the citations behind it. Nothing is forced, and nothing is claimed without evidence.
- **Optimizer window** — a free replacement for the Process Lasso workflow: hardware detection, a one-click checkup with per-item fixes, and a watchdog that re-applies High priority and frees the input-processing core on every game launch. Probes are read-only; nothing changes without a click.
- **The app itself** — one scrolling page of seven sections, an in-game click-through overlay, a scenario browser, four themes whose colours are derived from one set of parameters rather than hand-picked (contrast floors enforced by a test), and a motion dial. Character animation runs at 15 Hz on purpose: output quantized to a glyph ramp gains nothing above it.

Three numbers lead the home screen — readiness, form and load — and none can render as a bare score, because the clause naming the runs behind it is a required field rather than a convention.

## Install

```
git clone https://github.com/apemm/KovaaksAdjusted.git
cd KovaaksAdjusted
pip install -e .[gui]          # core + desktop app
pip install -e .[gui,clips]    # + video clips of notable moments (dxcam)
```

Python 3.10+ on Windows; analysis and the test suite run anywhere, capture is Windows-only. For a no-Python install, download the zip from [Releases](https://github.com/apemm/KovaaksAdjusted/releases), extract it, and run `kovadapt.exe`.

> **The Windows build is unsigned.** Smart App Control — on by default on a clean Windows 11 — will block it outright, and SmartScreen will warn on it otherwise. Installing from pip avoids this entirely; if you use the exe you will have to allow it explicitly. Code signing is on the list.

kovadapt looks for the game in the usual Steam locations. If yours lives elsewhere, set `KOVAAKS_ROOT` to the `FPSAimTrainer` folder, or put the path in `kovaaks_root` in `~/.kovadapt/settings.json` — a value there takes priority over the environment variable.

## Use

```
kovadapt gui                        # desktop app: one scrolling page, seven sections
kovadapt scenarios [filter]         # list installed scenarios
kovadapt play "1wall 6targets small"     # jump the game into the adaptive variant
kovadapt watch "1wall 6targets small"    # headless adaptation loop
kovadapt status "1wall 6targets small"   # learned profile + region heatmap + calibration
kovadapt replay "1wall 6targets small"   # bootstrap a profile from stats history
kovadapt generate "1wall 6targets small" # rewrite the variant without playing
kovadapt checkup                    # print the system optimization checkup
kovadapt watchdog                   # headless auto-tune on every game launch
kovadapt train                      # train the flick encoder (needs [ml])
```

Start `kovadapt gui`, pick a scenario, hit **Play adaptive task** — the app starts watching, writes a one-scenario playlist in the game's own format, and launches KovaaK's through Steam with the variant queued. A deep link cannot open a scenario generated on your own disk, tested in-game rather than assumed, so the playlist is the mechanism and the deep link stays best-effort; ownership stays where it belongs, since Steam won't start a game your account doesn't own. **Start adapting** does the same without launching, for when the game is already open.

Settings live at `~/.kovadapt/settings.json`; profiles, traces, reports and clips under the same directory.

## How it works

The stats CSV records outcomes, but outcomes conflate distinct failures: a miss may be a badly aimed flick or a well-aimed flick that arrived too slowly. So a background thread registers for Raw Input and records relative mouse deltas exactly as the game receives them, immune to Windows pointer ballistics. When a run ends, its window of the recording is cut into flicks by working backward from each click to the onset of movement, and each flick is characterized by amplitude, direction, peak speed, overshoot along the flick axis, and the corrective submovements needed to settle — which separates the miss that came from overshooting from the miss that came from hesitating, precisely the distinction accuracy alone cannot make.

Adaptation runs on three controllers. Each of the 5×5 wall regions carries a Gaussian posterior over how much weaker you are there than your own average, and focus is chosen by Thompson sampling: draw one sample per region and commit to the worst draw. Most draws exploit the model's current belief, but roughly a fifth land elsewhere, and that waste is deliberate — a weak side moves as you improve, so a purely greedy policy would keep drilling a weakness that no longer exists. Alongside it the size controller holds the accuracy band, and an Ornstein-Uhlenbeck process drifts movement. Two refinements ride inside those: a Fitts-normalized throughput term gives a little size back when movement time per unit of difficulty stops improving, and a pace-plateau term pushes movement when accuracy sits in the band while kills per second have gone flat. The band is a floor to build on, not a place to settle.

Two constraints shape everything downstream. Edits always apply to the base scenario, never to the previous variant, so multipliers stay absolute and difficulty cannot compound silently across sessions. And the plan's seed is the only randomness reaching the generator, so a given plan regenerates its `.sce` byte for byte — any variant you have ever played can be reproduced exactly.

See `FEATURES.md` for the full feature map and roadmap.

## Development

```
pip install -e .[dev]
pytest
```

Test suites run on any OS — Windows-only capture (Raw Input, dxcam) is import-guarded.

## License

MIT — see `LICENSE`.
