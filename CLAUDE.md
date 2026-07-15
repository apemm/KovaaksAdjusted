# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

kovadapt adapts KovaaK's aim-training scenarios **between** runs (the game has no modding API). It watches the game's stats folder; after each run it updates a per-scenario player model and rewrites `<Scenario> [Adaptive].sce` so the next load targets the player's weaknesses. Windows is the runtime target (Raw Input telemetry, dxcam clips, the game itself); analysis and the test suite run on any OS.

## Commands

```
pip install -e .[dev]                 # core (numpy only) + pytest + ruff
pip install -e .[gui]                 # + PySide6/pyqtgraph/psutil desktop app
pytest                                # full suite, cross-platform, <1s
pytest tests/test_adapt.py -k bandit  # single file / keyword
ruff check .                          # lint (line-length 100, target py310)
```

CLI (console script `kovadapt` or `python -m kovadapt`): `gui | scenarios [filter] | watch "<scenario>" | generate "<scenario>" | status "<scenario>" | replay "<scenario>" | checkup | watchdog`. Standalone exe: `powershell -File packaging/build.ps1` (PyInstaller one-dir; `packaging/entry.py` routes GUI/CLI/`--watchdog` through one binary).

- `tests/conftest.py` inserts the repo root into `sys.path`, so pytest works without installing the package.
- Integration tests `pytest.skip` unless a real KovaaK's install exists (`KOVAAKS_ROOT` env var, else the default Steam path) — a green run does not prove real-file parsing was exercised.

## Architecture

### The adaptation loop (`kovadapt/watcher.py`)

`SessionWatcher.watch()` polls `<kovaaks_root>/stats/` (1 s poll, 1.5 s write-settle so KovaaK's finishes writing) and for each new CSV of the base **or** adaptive scenario runs `process_run()`:

1. `stats/parser.py` parses the CSV into a `Run` (per-kill `KillEvent`s + summary dict).
2. `analysis/report.py:build_report()` joins the run with a slice of the live mouse recording (`run_time_window()` reconstructs the run's epoch window from CSV wall-clock kill timestamps), producing a `RunReport`: flick segmentation, directional bias, per-region deficits, notable moments, input-health metrics (`MouseTrace.input_health()`). The watcher's session-scoped `SessionFatigueTracker` (`analysis/fatigue.py`, Theil-Sen trend, reset every `watch()`) stamps `rep.fatigue`. Persisted as JSON; clips extracted from the dxcam ring buffer must happen immediately or the buffer rolls past.
3. `adapt/engine.py:AdaptationEngine.observe()` folds the run into the `PlayerProfile` — bandit credit **first** (reward is measured against the pre-run accuracy EWMA), then EWMAs, then the directional-bias EWMA. When telemetry region deficits exist they *replace* run-level focus attribution. First run also stamps `profile.archetype` (`adapt/archetype.py`: name keywords, then shots-per-kill heuristic, default "clicking") — every engine call resolves settings through `Settings.for_archetype()`.
4. `AdaptationEngine.plan()` runs the controllers — deadband size controller (hold hit rate in the archetype's band), Thompson-sampling bandit over the wall-region grid, Ornstein-Uhlenbeck movement drift with pace coupling, dodge-direction skew from `profile.ewma_bias` — and returns an `AdaptationPlan`. The optional `fatigue` argument eases only the **emitted** plan (bigger/calmer targets); persisted profile state stays un-eased.
5. `scenario/generator.py:generate_adaptive_variant()` applies the plan to the **base** `.sce` (never the previous variant — multipliers are absolute, edits never compound) and writes `<Base> [Adaptive].sce`.
6. The profile is saved. History predating `watch()` is ignored; the `replay` command is the only way to backfill it.

The GUI (`kovadapt gui`) wraps this loop in a single QThread (`gui/workers.py:WatcherWorker`); watcher callbacks fire on the watcher thread and are bridged into Qt as signals — never touch widgets from them. Stopping is cooperative (`request_stop()` flag, ~1 s latency).

### Packages

- `config.py` — `Settings` dataclass holding every tunable; KovaaK's path discovery (`KOVAAKS_ROOT` env var wins); `ADAPTIVE_SUFFIX`.
- `stats/` — CSV → `Run`/`KillEvent`. Positional kill-row columns and summary keys **with trailing colon** (`"Score:"`, `"Hit Count:"`) are the contract with KovaaK's 3.9.x output.
- `profile/` — `PlayerProfile`: EWMA stats, per-region Gaussian `RegionPosterior`s (mean > 0 = weaker there), adaptation state. All bandit state lives here — the bandit object itself is stateless and rebuilt each plan.
- `adapt/` — decision core (engine, bandit, OU process). Pure in-memory; zero file I/O.
- `scenario/` — `SceFile`, a verbatim line-based `.sce` editor with surgical accessors, + the variant generator.
- `telemetry/` — Raw Input recorder (ctypes message-only window on a daemon thread; packets arrive without focus via `RIDEV_INPUTSINK`) → `MouseTrace` (.npz). One continuous recording per watch session; runs are sliced out via `snapshot().window()`.
- `analysis/` — flick/bias/deficit/notable-moment computation and `RunReport`. Pure leaf over stats + telemetry; never imports adapt/scenario/gui.
- `capture/` — dxcam desktop ring buffer → mp4 clips (`[clips]` extra).
- `optimize/` — the free Process Lasso replacement: `hardware.py` (read-only probes: registry CPU name, ctypes RAM/refresh-rate, one PowerShell CIM call for GPU — wmic is gone on recent Win11), `checkup.py` (probe/fix pairs; `safe=True` marks per-user reversible fixes eligible for fix-all), `watchdog.py` (daemon-thread poller, tunes once per game PID; SMT-aware `cpus_to_free()` — [0,1] only when logical > physical cores; HKCU Run-key startup registration), `recommend.py` (pure advice rendering, gated on `HardwareInfo` properties). Every module imports on any OS; probes return unknowns off-Windows.
- `gui/` — PySide6 + pyqtgraph 4-tab app (Dashboard, Analysis, Adaptability, Optimizer). Thin presentation layer. The Optimizer tab is a launcher for `optimizer_window.OptimizerWindow` (separate top-level window; owns the `GameWatchdog`; closing hides it so the watchdog keeps running; checkup scans run on a QThread). `replay.TrajectoryReplay` draws flick overlays as exactly two NaN-separated curves and decimates above 50k points — keep it that way; per-flick plot items would tank redraws.

### On-disk state (`~/.kovadapt/`)

`settings.json` • `profiles/<slug>.json` • `traces/<slug>/<ts>.npz` • `reports/<slug>/<ts>.json` + sibling `clips/*.mp4`. Slug = scenario name with runs of chars outside `[A-Za-z0-9._-]` replaced by `_`; timestamps are ISO with `:` → `-`. `watcher._report_path` derives the reports path by path surgery on the traces path — the two trees mirror each other, and changing `TraceStore`'s directory depth silently breaks report/clip placement.

## Cross-module contracts (breaking these fails silently)

- **Region keys** `r{row}c{col}` (row-major, dims from `Settings.region_cols/rows`) must be byte-identical across `adapt/bandit.py:region_keys()`, `scenario/generator.py:_region_of()`, and `analysis/movement.py:region_deficits()`. A mismatch makes spawn resampling no-op and telemetry credit vanish — no errors raised.
- **`plan()` mutates the profile** (`ou_state`, `movement`, `target_scale`, `last_focus`). Callers must save the profile afterward or persisted state desyncs from the generated `.sce`; never call `plan()` speculatively. `plan(profile, None)` must stay valid (bootstrap path).
- **Ordering in `observe()`**: `credit_focus_region()` before `observe_run()` — folding the run into the EWMA first would contaminate the bandit's own baseline.
- **`ADAPTIVE_SUFFIX` is `" [Adaptive]"` — leading space included.** Profiles are keyed on base+suffix; both base and adaptive runs feed the same profile; the GUI picker hides adaptive variants but loads profiles by the suffixed name.
- **Byte-identical `.sce` round-trip** is test-pinned: writing an untouched `SceFile` must equal the input. `.sce` files contain *duplicate* section headers (`[Character Profile]` per bot) disambiguated by their `Name=` line — hence the (section, name, key) accessor API. Only bots listed in the `AddedBots` header get size/speed edits; spawn resampling only reuses existing coordinates, never invents them.
- **JSON dataclass round-trips** (`PlayerProfile`, `RunReport`) load via `cls(**dict)`: new fields need defaults; removing/renaming a field breaks every existing file under `~/.kovadapt`. There is no schema version key. Trace `.npz` files are versionless too — `MouseTrace.load` must keep tolerating missing keys (`clicks_up` is absent in pre-v0.3 traces).
- **Archetype overrides** (`Settings.archetype_overrides`) are plain dicts of Settings field names; `for_archetype()` silently drops unknown keys so stale settings.json files can't crash `dataclasses.replace`. Engine code must read tunables via `self._effective(profile)`, never `self.s` directly, or archetype behavior silently vanishes.
- **Fatigue never touches persisted state**: `plan(fatigue=...)` eases only the emitted `AdaptationPlan`; `profile.target_scale`/`movement` stay un-eased so the next session resumes true difficulty. The tracker itself is session-scoped — recreated in `watch()`, not persisted.
- **Y axis**: raw mouse `+y` is down. `analysis/` flips to aim-convention (+y up) internally, and `MouseTrace.path()` negates y for plotting — don't "correct" the flip in consumers. `bias_score > 0` means the **left** side is weaker.
- **Time base** is epoch `time.time()` everywhere (trace packets, click times, clip frames, run windows). Cross-correlation of traces/clips/runs works only because everything shares that clock. `MouseTrace.t` must stay monotonic.
- **Determinism**: `plan.seed` is the only randomness reaching the generator, so a given `AdaptationPlan` always regenerates the identical `.sce`.

## Optional dependencies & import guards

The core install needs only numpy; everything else is optional and **lazily imported** — keep it that way:

- `cli.cmd_gui` imports the Qt app inside the function and converts `ImportError` into an install hint; `gui/__init__.py` stays import-free so `import kovadapt` never pulls Qt.
- `telemetry/raw_input.py` defines all ctypes/Win32 code behind `RAW_INPUT_AVAILABLE` (`sys.platform == "win32"`); `capture/clips.py` sets `CLIPS_AVAILABLE` via try/except **`Exception`** (dxcam raises non-ImportError errors on unsupported platforms). Both modules must import cleanly on any OS — that is what lets analysis and tests run cross-platform. `watcher._start_capture` imports both lazily.
- In `raw_input.py`, the `WNDPROC` ctypes callback is kept alive only by the pump thread's stack frame — don't refactor it out of `_run()` or it gets garbage-collected and crashes. Keep the WndProc critical section tiny (runs per packet, up to ~8 kHz).

## Tests

The suite pins behavior, not just shapes: byte-identical `.sce` round-trip, the stats filename grammar (scenario names may contain `" - "`; the scenario regex group must stay greedy), the accuracy deadband (90 % → shrink, 40 % → grow, 66.7 % → no-op inside the band), and the +y-up flick-angle convention. `tests/test_telemetry.py:TraceBuilder` synthesizes ground-truth mouse traces — use it for new analysis tests. The Thompson bandit intentionally keeps ~15–25 % exploration; convergence tests assert a plurality (≥ 25 of last 50 picks), never a monopoly — don't tighten these assertions or "improve" exploitation. Untested: `watcher`, `cli`, `gui/*`, live Raw Input capture, clips.
