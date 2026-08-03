"""kovadapt command-line interface.

    kovadapt gui                     launch the desktop app
    kovadapt scenarios [filter]      list installed scenarios
    kovadapt play "<scenario>"       jump KovaaK's into the adaptive variant
    kovadapt watch "<scenario>"      run the adaptation loop for a scenario
    kovadapt generate "<scenario>"   one-shot: build/refresh the adaptive .sce
    kovadapt status "<scenario>"     show learned profile + region heatmap
    kovadapt replay "<scenario>"     rebuild profile from historical stats
    kovadapt watchdog                headless: auto-tune the game on every launch
    kovadapt checkup                 print the system optimization checkup
    kovadapt train                   train the neural flick model on recorded traces
"""

from __future__ import annotations

import argparse
import os
import sys

from .adapt.engine import AdaptationEngine, settle_focus
from .config import ADAPTIVE_SUFFIX, Settings
from .profile.player import PlayerProfile
from .scenario.generator import generate_adaptive_variant
from .stats.parser import iter_runs


def _settings() -> Settings:
    s = Settings.load()
    if not s.kovaaks_root:
        sys.exit("KovaaK's install not found. Set KOVAAKS_ROOT or edit ~/.kovadapt/settings.json")
    return s


def _base_scenario(name: str) -> str:
    """Strip trailing ADAPTIVE_SUFFIXes from a user-supplied scenario name.

    Adaptive edits are absolute against the base .sce and must never compound;
    passing "X [Adaptive]" to watch/generate would build
    "X [Adaptive] [Adaptive]" from the already-edited variant and fork a
    second profile. Repeated stripping also repairs doubled-suffix names left
    behind by earlier mistakes.
    """
    base = name
    while base.endswith(ADAPTIVE_SUFFIX):
        base = base[: -len(ADAPTIVE_SUFFIX)]
    if base != name:
        print(f'note: "{name}" is an adaptive variant — using base scenario "{base}"')
    return base


def cmd_scenarios(args) -> None:
    s = _settings()
    names = sorted(p.stem for p in s.scenarios_dir.glob("*.sce"))
    for n in names:
        if not args.filter or args.filter.lower() in n.lower():
            print(n)


def cmd_watch(args) -> None:
    from .watcher import SessionWatcher

    s = _settings()
    w = SessionWatcher(s, _base_scenario(args.scenario))
    if not w.base_sce_path().is_file():
        sys.exit(f"scenario file not found: {w.base_sce_path()}")
    try:
        w.watch()
    except KeyboardInterrupt:
        print("\nstopped")


def cmd_generate(args) -> None:
    from .adapt.archetype import detect_archetype

    s = _settings()
    scenario = _base_scenario(args.scenario)
    base_sce = s.scenarios_dir / f"{scenario}.sce"
    # cmd_watch has always guarded this; cmd_generate did not, so a typo'd
    # or missing scenario surfaced as an 11-frame FileNotFoundError traceback
    # out of SceFile.read.
    if not base_sce.is_file():
        sys.exit(f"scenario file not found: {base_sce}")
    adaptive = scenario + ADAPTIVE_SUFFIX
    profile = PlayerProfile.load(adaptive, s.profile_path)
    profile.scenario = adaptive
    if not profile.archetype:
        profile.archetype = detect_archetype(scenario)
    plan = AdaptationEngine(s).plan(profile, None)
    out = generate_adaptive_variant(
        s.scenarios_dir / f"{scenario}.sce", plan, s,
        s.scenarios_dir / f"{adaptive}.sce",
    )
    settle_focus(profile, plan)
    profile.save(s.profile_path)
    print(f"wrote {out}\nplan: {plan.describe()}")


def cmd_status(args) -> None:
    s = _settings()
    profile = PlayerProfile.load(_base_scenario(args.scenario) + ADAPTIVE_SUFFIX, s.profile_path)
    if profile.run_count == 0:
        print("no runs recorded yet")
        return
    print(f"scenario:      {profile.scenario}")
    print(f"runs:          {profile.run_count}")
    print(f"accuracy EWMA: {profile.ewma_accuracy:.1%}")
    print(f"avg TTK EWMA:  {profile.ewma_ttk:.3f}s")
    print(f"pace EWMA:     {profile.ewma_kps:.2f} kills/s")
    print(f"score EWMA:    {profile.ewma_score:.0f}")
    print(f"target scale:  {profile.target_scale:.2f}")
    print(f"movement:      {profile.movement:.2f}")
    if profile.archetype:
        print(f"archetype:     {profile.archetype}")
    ready = profile.readiness(s.region_cols * s.region_rows)
    print(f"calibration:   {ready['score']:.0%} — {ready['message']}")
    print("\nregion deficit heatmap (+ = weaker, more spawns there):")
    for r in range(s.region_rows - 1, -1, -1):  # top row printed first
        cells = []
        for c in range(s.region_cols):
            post = profile.regions.get(f"r{r}c{c}")
            # Gated on EVIDENCE, not on the arm object existing. plan()
            # materializes all region_cols*region_rows arms the first time it
            # runs, so after a single `kovadapt generate` every cell had an
            # object with n == 0 and the map printed "+0.00(0)" 25 times — a
            # measured-looking zero for regions nothing has ever observed.
            cells.append(f"{post.mean:+.2f}({post.n})"
                         if post is not None and post.n else "  --  ")
        print("   " + "  ".join(f"{x:>10}" for x in cells))


def cmd_replay(args) -> None:
    """Rebuild the profile from existing stats history (base + adaptive runs)."""
    s = _settings()
    scenario = _base_scenario(args.scenario)
    adaptive = scenario + ADAPTIVE_SUFFIX
    profile = PlayerProfile(scenario=adaptive)
    engine = AdaptationEngine(s)
    # EWMAs and bandit credit are order-sensitive: merge the base and adaptive
    # streams into true chronological order before folding them in.
    runs = [run for name in (scenario, adaptive)
            for run in iter_runs(s.stats_dir, scenario=name)]
    runs.sort(key=lambda r: r.started)
    if not runs:
        # Replay REBUILDS from scratch — `profile` above is deliberately a
        # fresh empty one — so saving it with nothing folded in overwrites
        # everything the scenario had learned. A misspelled name, or a stats
        # folder the user has cleared out, silently destroyed the profile and
        # exited 0 reporting "replayed 0 runs". Refusing to write is the only
        # safe answer: there is nothing here to rebuild FROM.
        existing = PlayerProfile.path_for(adaptive, s.profile_path)
        sys.exit(
            f"no stats found for {scenario!r} in {s.stats_dir}\n"
            + (f"{existing} left untouched — check the scenario name"
               if existing.is_file() else "nothing to replay"))
    for run in runs:
        engine.observe(profile, run)
    profile.save(s.profile_path)
    print(f"replayed {len(runs)} runs into {profile.scenario!r} "
          f"(accuracy EWMA {profile.ewma_accuracy:.1%})")


def cmd_play(args) -> None:
    """Queue the adaptive playlist and deep-link the game into the scenario."""
    from . import launcher

    s = _settings()
    scenario = _base_scenario(args.scenario)
    adaptive = scenario + ADAPTIVE_SUFFIX
    if (s.scenarios_dir / f"{adaptive}.sce").is_file():
        msg, ok = launcher.play_adaptive(s, scenario)
    else:
        print(f"no adaptive variant yet — launching the base scenario "
              f"(run `kovadapt watch \"{scenario}\"` to adapt)")
        msg, ok = launcher.launch_scenario(scenario)
    print(msg)
    if not ok:
        sys.exit(1)


# Lines kept in the watchdog's audit log. One tune per game launch, so this is
# months of history even for a heavy user.
_LOG_LINES = 2000


def cmd_watchdog(args) -> None:
    """Headless watchdog loop (used by the start-with-Windows entry).

    EVERYTHING IT DOES IS WRITTEN TO A LOG the user can read. When this is
    launched from the start-with-Windows entry it runs under pythonw with no
    console, so `print` goes nowhere: a background process that silently
    changes another process's priority and affinity, leaving no record it ever
    ran. That is the shape of software people are right to distrust, and the
    only thing separating this from it is that the record exists.

    Appended and trimmed from the front at _LOG_LINES, never deleted, so the
    history survives a restart.
    """
    import time
    from datetime import datetime

    from .optimize.watchdog import GameWatchdog

    s = _settings()
    log_path = s.profile_path / "watchdog.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_path = None

    def record(msg: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
        print(line)
        if log_path is None:
            return
        try:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            # Trimmed from the FRONT once it grows, never deleted: a log that
            # erases itself is no better than no log. One tune per game
            # launch, so _LOG_LINES is months of history.
            lines = log_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > _LOG_LINES:
                log_path.write_text(
                    "\n".join(lines[-(_LOG_LINES // 2):]) + "\n",
                    encoding="utf-8")
        except OSError:
            pass        # a watchdog that cannot log still has to keep working

    record(f"watchdog started (pid {os.getpid()}) — watching for the game; "
           f"it applies High priority and frees the input core, and nothing "
           f"else. Remove it under Optimizer > Start the watchdog with "
           f"Windows, or delete the 'kovadapt-watchdog' value under "
           f"HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run.")
    w = GameWatchdog(on_event=record)
    w.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        w.stop()
        record("watchdog stopped")


def cmd_checkup(args) -> None:
    from .optimize.checkup import SystemCheckup
    from .optimize.hardware import detect_hardware

    s = Settings.load()
    hw = detect_hardware()
    if hw.cpu_name or hw.gpu_name:
        print(f"{hw.cpu_name} | {hw.gpu_name} | {hw.ram_gb:.0f} GB | "
              f"{hw.monitor_hz} Hz | Windows {'11' if hw.is_windows_11 else '10'}")
    mark = {"ok": "+", "warn": "!", "bad": "X", "info": "i", "unknown": "?"}
    for r in SystemCheckup(s.kovaaks_root, hw).run_all():
        print(f"[{mark.get(r.status, '?')}] {r.title}")
        print(f"    {r.detail}")
        if r.can_fix:
            print(f"    fix available in the GUI: {r.fix_label}")


def cmd_train(args) -> None:
    """Train the FlickEncoder on the trace library under ~/.kovadapt.

    Uses Settings.load() rather than _settings(): training reads only
    <profile_dir>/traces and needs no KovaaK's install.
    """
    from . import ml

    if not ml.ML_AVAILABLE:
        sys.exit(
            "PyTorch is not installed — the neural workstream needs it.\n"
            "Install it with: pip install kovadapt[ml]"
        )
    from .ml.train import train as train_model

    s = Settings.load()
    try:
        result = train_model(
            s.profile_path / "traces", s.profile_path / "ml",
            epochs=args.epochs, seed=args.seed,
        )
    except RuntimeError as exc:  # not enough flick data
        sys.exit(str(exc))
    if result is None:  # unreachable with ML_AVAILABLE, kept for safety
        sys.exit("training unavailable (torch import failed)")
    heads = " | ".join(f"{k} {v:.4f}" for k, v in result.val_loss_per_head.items())
    print(f"dataset:     {result.train_size + result.val_size} flicks from "
          f"{result.n_traces} traces (train {result.train_size} / val {result.val_size})")
    print(f"device:      {result.device}")
    print(f"model:       {result.params:,} parameters")
    print(f"epochs:      {result.epochs_run} run (best val at epoch {result.best_epoch})")
    print(f"train loss:  {result.train_loss:.4f}  (MSE, standardized targets)")
    print(f"val loss:    {result.val_loss:.4f}  [{heads}]")
    print(f"checkpoint:  {result.checkpoint}")


def cmd_gui(args) -> None:
    try:
        from .gui.app import main as gui_main
    except ImportError as exc:
        sys.exit(
            f"GUI dependencies missing ({exc.name or exc}).\n"
            "Install them with: pip install kovadapt[gui]"
        )
    sys.exit(gui_main())


def _utf8_stdio() -> None:
    # Piped/redirected output on Windows defaults to the legacy codepage, which
    # mangles the dashes and glyphs in checkup/status output. Streams can be
    # absent or non-reconfigurable in a frozen GUI launch, hence the guard.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> None:
    _utf8_stdio()
    p = argparse.ArgumentParser(prog="kovadapt", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # There was no way to ask a running install what it is. `--version` exited
    # 2 with an argparse usage error, and the frozen exe carries no version
    # resource either — so a bug report could not name a build.
    from . import __version__
    p.add_argument("-V", "--version", action="version",
                   version=f"kovadapt {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scenarios", help="list installed scenarios")
    sc.add_argument("filter", nargs="?", default="")
    sc.set_defaults(fn=cmd_scenarios)

    g = sub.add_parser("gui", help="launch the desktop app")
    g.set_defaults(fn=cmd_gui)

    wd = sub.add_parser("watchdog", help="headless auto-tune on every game launch")
    wd.set_defaults(fn=cmd_watchdog)

    ck = sub.add_parser("checkup", help="print the system optimization checkup")
    ck.set_defaults(fn=cmd_checkup)

    tr = sub.add_parser("train", help="train the neural flick model on recorded traces")
    tr.add_argument("--epochs", type=int, default=60, help="max training epochs (default 60)")
    tr.add_argument("--seed", type=int, default=0, help="training seed (default 0)")
    tr.set_defaults(fn=cmd_train)

    for name, fn, hlp in (
        ("play", cmd_play, "jump KovaaK's into the adaptive variant (Steam deep link)"),
        ("watch", cmd_watch, "run the live adaptation loop"),
        ("generate", cmd_generate, "one-shot generate the adaptive variant"),
        ("status", cmd_status, "show learned player profile"),
        ("replay", cmd_replay, "rebuild profile from historical stats"),
    ):
        sp = sub.add_parser(name, help=hlp)
        sp.add_argument("scenario", help="base scenario name (as shown in KovaaK's)")
        sp.set_defaults(fn=fn)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
