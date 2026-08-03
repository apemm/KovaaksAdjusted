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

from .adapt.archetype import stamp_archetype
from .adapt.engine import AdaptationEngine, settle_focus
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
        # Neural flick scorer ([ml] extra): loaded lazily on first use, once
        # per watcher — None when torch or the checkpoint is missing.
        self._ml_scorer = None
        self._ml_tried = False
        self._shadow_warned = False

    # ----------------------------------------------------------- telemetry
    def _start_capture(self) -> None:
        if self.s.telemetry_enabled:
            from .telemetry.raw_input import RAW_INPUT_AVAILABLE, MouseRecorder

            if RAW_INPUT_AVAILABLE:
                keep_min = self.s.telemetry_retention_min
                self.recorder = MouseRecorder(
                    retention_s=keep_min * 60.0 if keep_min > 0 else None
                )
                self.recorder.start()
                self.log("mouse telemetry: recording (Raw Input)")
            else:
                self.log("mouse telemetry: unavailable on this OS — skipping")
        if self.s.clips_enabled:
            from .capture.clips import CLIPS_AVAILABLE, ClipRecorder

            if CLIPS_AVAILABLE:
                # Clips are a nice-to-have; dxcam raises plain Exceptions on
                # unsupported GPUs/outputs. Losing the ring buffer must not
                # cost the run its adaptation or its mouse telemetry.
                try:
                    self.clip_recorder = ClipRecorder(
                        fps=self.s.clip_fps,
                        buffer_seconds=self.s.clip_buffer_seconds,
                        scale=self.s.clip_scale,
                    )
                    self.clip_recorder.start()
                    self.log("clip capture: recording (ring buffer)")
                except Exception as exc:
                    self.clip_recorder = None
                    self.log(f"clip capture: unavailable ({exc}) — continuing without clips")
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
        win = run_time_window(run)
        if win:
            return self.recorder.snapshot(*win).window(*win)
        # No reconstructable window (e.g. a zero-kill run): analyzing the
        # whole live recording as if it were this run would poison the
        # profile with session-wide telemetry.
        return None

    def _report_path(self, run) -> Path:
        p = self.traces.path_for(run.scenario, run.started.isoformat())
        return p.parent.parent.parent / "reports" / p.parent.name / (p.stem + ".json")

    # -------------------------------------------------------------- neural
    def _scorer(self):
        """Lazily load the neural flick scorer (ml/infer.py), once per
        watcher. Returns None when torch or the checkpoint is missing — a
        `kovadapt train` finishing mid-session applies on the next watch."""
        if not self._ml_tried:
            self._ml_tried = True
            try:
                from .ml.infer import load_scorer

                self._ml_scorer = load_scorer(self.s.profile_path)
            except Exception:
                self._ml_scorer = None
            if self._ml_scorer is not None:
                self.log("neural scorer: checkpoint loaded — stamping flick "
                         "quality into reports")
        return self._ml_scorer

    def _profile_state(self, profile, rep) -> dict:
        """Plain-dict profile snapshot for the shadow log, keyed by
        ml/shadow.py:PROFILE_STATE_KEYS (missing attrs -> None; the schema
        tolerates absent keys)."""
        try:
            from .ml.shadow import PROFILE_STATE_KEYS
        except Exception:
            return {}
        state = {k: getattr(profile, k, None) for k in PROFILE_STATE_KEYS if k != "fatigue"}
        state["fatigue"] = dict(rep.fatigue)
        return state

    def _log_shadow(self, state: dict, plan, run) -> None:
        """Append this run's (state, plan) transition to the shadow-policy
        JSONL log (ml/shadow.py) — the future training set for the neural
        difficulty policy. Torch-free and best-effort: the untrained policy
        never influences the emitted plan, and a logging failure must never
        touch the adaptation loop."""
        try:
            import dataclasses
            import json

            from .ml.shadow import DifficultyShadowPolicy

            policy = DifficultyShadowPolicy(self.s.profile_path)
            try:  # default=float coerces numpy scalars along the way
                plan_d = json.loads(json.dumps(dataclasses.asdict(plan), default=float))
            except Exception:
                plan_d = {"describe": plan.describe()}
            sug = policy.propose(state)  # None until a policy is trained
            policy.log_transition({
                "ts": run.started.isoformat(),
                "profile_state": json.loads(json.dumps(state, default=float)),
                "plan": plan_d,
                "suggestion": None if sug is None else dataclasses.asdict(sug),
                # THIS run's measured result. Schema v1 had an `outcome` key
                # documented as "next-run outcome when known" and hard-coded
                # to None, so every transition ever logged carried a state and
                # an action with NO REWARD — nothing an off-policy learner can
                # use. Fixed forward rather than back-filled, because the log
                # is append-only: record[i]'s plan is rewarded by
                # record[i+1]["run_outcome"] for the same scenario.
                "run_outcome": {
                    "accuracy": float(run.accuracy),
                    "score": float(run.score),
                    # NULL, not 0.0, when the run has no measurable pace.
                    # kills_per_second needs two kill rows to have a span, and
                    # invincible-target scenarios report none at all — 162 of
                    # 398 real stats files on this machine. Writing 0.0 puts a
                    # structural fake into an APPEND-ONLY training log, where
                    # it cannot be corrected later and reads as "this plan
                    # produced no pace" rather than "pace is not measurable
                    # here". It is the same zero the PACE tile was fixed to
                    # stop printing.
                    "kps": (float(run.kills_per_second())
                            if len(run.kills) >= 2 else None),
                },
            })
        except Exception as exc:
            # Never let this touch the adaptation loop — but say so ONCE.
            # A bare `pass` here is why a broken training log is invisible:
            # `run.kills_per_second` is a method, not a property, so
            # `float(run.kills_per_second or 0)` raised and every transition
            # silently stopped being recorded. The suite did not catch it
            # either, because the assertion on the old always-null field
            # passed whether or not the record was ever written.
            if not self._shadow_warned:
                self._shadow_warned = True
                self.log(f"  note: shadow transition not logged ({exc}) — "
                         "adaptation is unaffected, but the ML training set "
                         "is not accumulating")

    def _analyze(self, run) -> RunReport:
        """Build + persist the post-run report (trace, clips, JSON)."""
        trace = self._run_trace(run)
        rep, flicks, _ = build_report(run, trace, region_cols=self.s.region_cols,
                                      region_rows=self.s.region_rows)
        if self.s.fatigue_detection_enabled:
            state = self.fatigue.add_run(rep.n_flicks, rep.overshoot_rate, rep.mean_flick_ms)
            rep.fatigue = state.as_dict()
            if state.message and state.level != self._fatigue_level_logged:
                self._fatigue_level_logged = state.level
                self.log(f"  fatigue: {state.message}")
        if flicks and trace is not None:
            scorer = self._scorer()
            if scorer is not None:
                try:
                    from .ml.infer import summarize

                    rep.ml = summarize(scorer.score(trace, flicks))
                except Exception as exc:  # best-effort; never break the loop
                    self.log(f"  neural scorer error: {exc}")
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
        def mtime(p: Path) -> float:
            try:
                return p.stat().st_mtime
            except OSError:      # vanished between glob and stat
                return 0.0

        out = [
            p for p in self.s.stats_dir.glob("*.csv")
            if p.name not in self._seen and self._relevant(p.name)
        ]
        return sorted(out, key=mtime)

    # ------------------------------------------------------------------
    def process_run(self, csv_path: Path) -> Path:
        """Fold one run into the model and regenerate the adaptive .sce."""
        run = parse_stats_csv(csv_path)
        rep = self._analyze(run)
        profile = PlayerProfile.load(self.adaptive_name, self.s.profile_path)
        profile.scenario = self.adaptive_name
        # A run is the strongest evidence there is, so it may correct a stamp
        # made from the name alone — or from nothing, which is what the
        # browser and `kovadapt generate` leave behind when they are used
        # before the scenario has ever been played. Only ever upgrades, so a
        # scenario is re-classified at most once and then holds.
        changed = stamp_archetype(profile, self.base, run)
        if changed:
            self.log(f"  archetype: {changed[0]} -> {changed[1]} — this run "
                     "disagrees with the guess made before it was played")
        elif profile.archetype_source == "stats" and profile.run_count == 0:
            self.log(f"  archetype: {profile.archetype}")
        # Shadow-policy schema: profile state is captured BEFORE observe()
        # folds this run in (ml/shadow.py:SHADOW_LOG_SCHEMA).
        shadow_state = self._profile_state(profile, rep)
        # Bias needs both sides sampled: directional_bias returns 0.0 for
        # "no evidence" too, and a vertical-heavy run must not decay a
        # learned skew toward balanced.
        both_sides = (rep.bias.get("left", {}).get("n", 0) >= 3
                      and rep.bias.get("right", {}).get("n", 0) >= 3)
        bias = rep.bias.get("bias_score") if rep.n_flicks >= 8 and both_sides else None
        self.engine.observe(
            profile, run,
            region_deficits=rep.region_deficits or None,
            bias_score=bias,
            fitts_slope_ms=rep.fitts_slope_ms or None,
        )
        fatigue = rep.fatigue.get("score", 0.0) if self.s.fatigue_easing else 0.0
        plan = self.engine.plan(profile, run, fatigue=fatigue)
        out = generate_adaptive_variant(
            self.base_sce_path(), plan, self.s, self.adaptive_sce_path()
        )
        if settle_focus(profile, plan):
            self.log(f"  note: region {plan.focus_region} has no spawns here — "
                     "focus not applied, arm not credited")
        profile.save(self.s.profile_path)
        self._log_shadow(shadow_state, plan, run)
        self.log(
            f"[{datetime.now():%H:%M:%S}] run #{profile.run_count} "
            f"acc={run.accuracy:.1%} score={run.score:.0f} -> {plan.describe()}"
        )
        if rep.summary_text:
            self.log(f"  analysis: {rep.summary_text}")
        # Emit only after the profile is saved: GUI handlers reload the
        # profile from disk, so an earlier emit shows last run's state.
        if self.on_report is not None:
            self.on_report(rep)
        return out

    def bootstrap(self) -> Path:
        """Create the initial adaptive variant (neutral plan) if missing."""
        profile = PlayerProfile.load(self.adaptive_name, self.s.profile_path)
        profile.scenario = self.adaptive_name
        if not profile.archetype:
            stamp_archetype(profile, self.base)
        plan = self.engine.plan(profile, None)
        out = generate_adaptive_variant(
            self.base_sce_path(), plan, self.s, self.adaptive_sce_path()
        )
        # Harmless on a genuinely fresh profile (credit_focus_region
        # early-returns at run_count 0), but bootstrap also runs when a
        # profile WITH runs finds its [Adaptive] file deleted.
        settle_focus(profile, plan)
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
        # Never reset stop_requested here: a request_stop() that lands while
        # watch() is still starting up (GUI Stop right after Start) must win.
        self.log(f"watching {self.s.stats_dir} for '{self.base}' runs (ctrl-c to stop)")
        try:
            # INSIDE the try: capture start is what most often fails (dxcam
            # raises on unsupported hardware), and if the mouse recorder is
            # already up when the clip recorder throws, only the finally
            # below can stop its Raw Input pump. Starting outside meant the
            # pump leaked, and the next watch() overwrote self.recorder and
            # orphaned the thread — with it, the message-only window and its
            # registered window class.
            self._start_capture()
            while not self.stop_requested:
                for p in self._pending_files():
                    try:
                        # Let KovaaK's finish writing; stay stoppable while
                        # waiting (a file whose mtime keeps advancing must
                        # not pin the loop past request_stop()).
                        while (not self.stop_requested
                               and time.time() - p.stat().st_mtime < settle):
                            time.sleep(min(settle, 0.5))
                    except OSError:       # vanished between glob and settle
                        self._seen.add(p.name)
                        continue
                    if self.stop_requested:
                        break
                    try:
                        self.process_run(p)
                    except Exception as exc:  # keep the loop alive
                        self.log(f"error processing {p.name}: {exc}")
                    self._seen.add(p.name)
                time.sleep(poll_interval)
        finally:
            self._stop_capture()
