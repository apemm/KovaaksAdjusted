"""Session watcher: the closed adaptation loop.

Polls the stats folder; when a new run of the tracked adaptive scenario (or
its base) lands, it: parses -> analyzes telemetry -> updates profile -> plans
-> regenerates the adaptive .sce. Next time the scenario is loaded in
KovaaK's, the new variant is live. Polling (1s) is plenty — runs end at human
timescales.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .adapt.archetype import detect_archetype
from .adapt.engine import AdaptationEngine
from .analysis.fatigue import SessionFatigueTracker
from .analysis.report import RunReport, build_report, run_time_window
from .config import ADAPTIVE_SUFFIX, Settings
from .profile.player import PlayerProfile
from .scenario.generator import generate_adaptive_variant
from .stats.parser import parse_stats_csv, parse_stats_filename
from .telemetry.trace import MouseTrace, TraceStore


class SessionWatcher:
    def __init__(
        self,
        settings: Settings,
        base_scenario: str,
        on_update: Callable[[str], None] = print,
        on_report: Callable[[RunReport], None] | None = None,
    ) -> None:
        self.s = settings
        self.base = base_scenario
        self.adaptive_name = base_scenario + ADAPTIVE_SUFFIX
        self.engine = AdaptationEngine(settings)
        self.log = on_update
        self.on_report = on_report
        self.stop_requested = False
        self._seen: set[str] = set()
        self.traces = TraceStore(settings.profile_path)
        self.recorder = None       # MouseRecorder while watching (if enabled)
        self.clip_recorder = None  # ClipRecorder while watching (if enabled)
        self.last_report: RunReport | None = None
        self.fatigue = SessionFatigueTracker(
            settings.fatigue_min_runs, settings.fatigue_sensitivity
        )
        self._fatigue_level_logged = "fresh"

    # ----------------------------------------------------------- telemetry
    def _start_capture(self) -> None:
        if self.s.telemetry_enabled:
            from .telemetry.raw_input import RAW_INPUT_AVAILABLE, MouseRecorder

            if RAW_INPUT_AVAILABLE:
                self.recorder = MouseRecorder()
                self.recorder.start()
                self.log("mouse telemetry: recording (Raw Input)")
            else:
                self.log("mouse telemetry: unavailable on this OS — skipping")
        if self.s.clips_enabled:
            from .capture.clips import CLIPS_AVAILABLE, ClipRecorder

            if CLIPS_AVAILABLE:
                self.clip_recorder = ClipRecorder(
                    fps=self.s.clip_fps,
                    buffer_seconds=self.s.clip_buffer_seconds,
                    scale=self.s.clip_scale,
                )
                self.clip_recorder.start()
                self.log("clip capture: recording (ring buffer)")
            else:
                self.log("clip capture: dxcam/opencv missing — pip install kovadapt[clips]")

    def _stop_capture(self) -> None:
        if self.recorder is not None:
            self.recorder.stop()
            self.recorder = None
        if self.clip_recorder is not None:
            self.clip_recorder.stop()
            self.clip_recorder = None

    def _run_trace(self, run) -> MouseTrace | None:
        """Slice the live recording down to this run's time window."""
        if self.recorder is None:
            return None
        trace = self.recorder.snapshot()
        win = run_time_window(run)
        return trace.window(*win) if win else trace

    def _report_path(self, run) -> Path:
        p = self.traces.path_for(run.scenario, run.started.isoformat())
        return p.parent.parent.parent / "reports" / p.parent.name / (p.stem + ".json")

    def _analyze(self, run) -> RunReport:
        """Build + persist the post-run report (trace, clips, JSON)."""
        trace = self._run_trace(run)
        rep, flicks, _ = build_report(run, trace)
        if self.s.fatigue_detection_enabled:
            state = self.fatigue.add_run(rep.n_flicks, rep.overshoot_rate, rep.mean_flick_ms)
            rep.fatigue = state.as_dict()
            if state.message and state.level != self._fatigue_level_logged:
                self._fatigue_level_logged = state.level
                self.log(f"  fatigue: {state.message}")
        if trace is not None and len(trace) > 10:
            rep.trace_file = str(self.traces.save(trace, run.scenario, run.started.isoformat()))
        if self.clip_recorder is not None and rep.notable:
            clip_dir = self._report_path(run).parent / "clips"
            for i, m in enumerate(rep.notable):
                out = self.clip_recorder.save_clip(
                    m["t_start"], m["t_end"],
                    clip_dir / f"{run.started:%Y%m%dT%H%M%S}_{i}_{m['kind']}.mp4",
                )
                if out is not None:
                    rep.clip_files[str(i)] = str(out)
        rep.save(self._report_path(run))
        self.last_report = rep
        if self.on_report is not None:
            self.on_report(rep)
        return rep

    # ------------------------------------------------------------------
    def base_sce_path(self) -> Path:
        return self.s.scenarios_dir / f"{self.base}.sce"

    def adaptive_sce_path(self) -> Path:
        return self.s.scenarios_dir / f"{self.adaptive_name}.sce"

    def _relevant(self, name: str) -> bool:
        meta = parse_stats_filename(name)
        return meta is not None and meta[0] in (self.base, self.adaptive_name)

    def _pending_files(self) -> list[Path]:
        out = [
            p for p in self.s.stats_dir.glob("*.csv")
            if p.name not in self._seen and self._relevant(p.name)
        ]
        return sorted(out, key=lambda p: p.stat().st_mtime)

    # ------------------------------------------------------------------
    def process_run(self, csv_path: Path) -> Path:
        """Fold one run into the model and regenerate the adaptive .sce."""
        run = parse_stats_csv(csv_path)
        rep = self._analyze(run)
        profile = PlayerProfile.load(self.adaptive_name, self.s.profile_path)
        profile.scenario = self.adaptive_name
        if not profile.archetype:
            profile.archetype = detect_archetype(self.base, run)
            self.log(f"  archetype: {profile.archetype}")
        # Bias needs both sides sampled; skip low-telemetry runs entirely.
        bias = rep.bias.get("bias_score") if rep.n_flicks >= 8 else None
        self.engine.observe(
            profile, run,
            region_deficits=rep.region_deficits or None,
            bias_score=bias,
        )
        fatigue = rep.fatigue.get("score", 0.0) if self.s.fatigue_easing else 0.0
        plan = self.engine.plan(profile, run, fatigue=fatigue)
        out = generate_adaptive_variant(
            self.base_sce_path(), plan, self.s, self.adaptive_sce_path()
        )
        profile.save(self.s.profile_path)
        self.log(
            f"[{datetime.now():%H:%M:%S}] run #{profile.run_count} "
            f"acc={run.accuracy:.1%} score={run.score:.0f} -> {plan.describe()}"
        )
        if rep.summary_text:
            self.log(f"  analysis: {rep.summary_text}")
        return out

    def bootstrap(self) -> Path:
        """Create the initial adaptive variant (neutral plan) if missing."""
        profile = PlayerProfile.load(self.adaptive_name, self.s.profile_path)
        profile.scenario = self.adaptive_name
        if not profile.archetype:
            profile.archetype = detect_archetype(self.base)
        plan = self.engine.plan(profile, None)
        out = generate_adaptive_variant(
            self.base_sce_path(), plan, self.s, self.adaptive_sce_path()
        )
        profile.save(self.s.profile_path)
        self.log(f"created {out.name} — play it in KovaaK's; I'll adapt after each run")
        return out

    def request_stop(self) -> None:
        """Ask a running watch() loop to exit (takes <= poll_interval)."""
        self.stop_requested = True

    def watch(self, poll_interval: float = 1.0, settle: float = 1.5) -> None:
        """Block until request_stop(), processing new stats files as they appear."""
        # Fatigue is session-scoped: start each watch with a fresh trend.
        self.fatigue = SessionFatigueTracker(
            self.s.fatigue_min_runs, self.s.fatigue_sensitivity
        )
        self._fatigue_level_logged = "fresh"
        # Ignore history that predates the watcher.
        self._seen = {p.name for p in self.s.stats_dir.glob("*.csv")}
        if not self.adaptive_sce_path().is_file():
            self.bootstrap()
        self._start_capture()
        self.stop_requested = False
        self.log(f"watching {self.s.stats_dir} for '{self.base}' runs (ctrl-c to stop)")
        try:
            while not self.stop_requested:
                for p in self._pending_files():
                    # Let KovaaK's finish writing.
                    while time.time() - p.stat().st_mtime < settle:
                        time.sleep(settle)
                    try:
                        self.process_run(p)
                    except Exception as exc:  # keep the loop alive
                        self.log(f"error processing {p.name}: {exc}")
                    self._seen.add(p.name)
                time.sleep(poll_interval)
        finally:
            self._stop_capture()
