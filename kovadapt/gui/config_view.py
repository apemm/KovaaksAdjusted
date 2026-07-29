"""Adaptability configuration tab: the full Settings surface, saved to
~/.kovadapt/settings.json. More knobs than KovaaK's official offering.

Sections: mouse & sensitivity first (feeds the model's per-task sensitivity
reasoning), then the everyday controls, the advanced engine internals
(exposed for power users; defaults reproduce the shipped behavior), the
trace-informed dodge/fatigue features, and per-archetype overrides. The
group boxes stack in one editorial column and flow in the page-space —
no nested scroll of their own.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings
from .onboarding import HintBar

# Override keys editable per archetype (subset of Settings fields that map
# cleanly to "how should adaptation differ for this task type").
_ARCH_KEYS = (
    ("target_accuracy_low", "Accuracy low"),
    ("target_accuracy_high", "Accuracy high"),
    ("size_learning_rate", "Size learning rate"),
    ("min_movement", "Min movement"),
    ("focus_weight", "Focus weight"),
)
_ARCH_EDITABLE = ("tracking", "switching")   # clicking is the baseline


def _dspin(val: float, lo: float, hi: float, step: float = 0.05, dec: int = 2,
           tip: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    w.setMinimumWidth(160)      # wide inputs for the editorial column
    if tip:
        w.setToolTip(tip)
    return w


def _ispin(val: int, lo: int, hi: int, tip: str = "") -> QSpinBox:
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setMinimumWidth(160)
    if tip:
        w.setToolTip(tip)
    return w


def _form(box: QGroupBox) -> QFormLayout:
    """Single-column form with the column's generous padding."""
    f = QFormLayout(box)
    f.setContentsMargins(16, 12, 16, 16)
    f.setHorizontalSpacing(28)
    f.setVerticalSpacing(12)
    return f


class ConfigView(QWidget):
    settings_changed = Signal(object)   # emits the saved Settings

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        s = settings

        # mouse & sensitivity — feeds the model's per-task sensitivity
        # reasoning. getattr-guarded: the fields may land in a later build.
        self.dpi = _dspin(
            float(getattr(s, "mouse_dpi", 0) or 800.0), 100.0, 32000.0, 50.0, 0,
            "Your mouse's hardware DPI (CPI) as set in its software.")
        self.sens = _dspin(
            float(getattr(s, "game_sens", 0) or 1.0), 0.01, 20.0, 0.01, 2,
            "Your in-game sensitivity in KovaaK's.")
        self.cm360 = QLabel("")
        self.cm360.setProperty("stat", True)
        cm_cap = QLabel(
            "The adaptive model uses this to reason about per-task sensitivity "
            "— cm/360 is derived from KovaaK's yaw of 0.022° per count.")
        cm_cap.setProperty("dim", True)
        cm_cap.setWordWrap(True)
        # "&&" is a literal ampersand. A bare "&" marks the next character as a
        # keyboard mnemonic, which is why this rendered as "Mouse _sensitivity".
        mouse = QGroupBox("Mouse && sensitivity")
        f = _form(mouse)
        f.addRow("Mouse DPI", self.dpi)
        f.addRow("In-game sensitivity", self.sens)
        f.addRow("cm per 360°", self.cm360)
        f.addRow(cm_cap)
        self.dpi.valueChanged.connect(self._update_cm360)
        self.sens.valueChanged.connect(self._update_cm360)
        self._update_cm360()

        # difficulty
        self.acc_lo = _dspin(s.target_accuracy_low, 0.1, 0.95)
        self.acc_hi = _dspin(s.target_accuracy_high, 0.15, 0.99)
        self.lr = _dspin(s.size_learning_rate, 0.1, 3.0, 0.1)
        self.scale_min = _dspin(s.min_target_scale, 0.1, 1.0)
        self.scale_max = _dspin(s.max_target_scale, 1.0, 5.0)
        diff = QGroupBox("Difficulty controller")
        f = _form(diff)
        f.addRow("Accuracy sweet spot — low", self.acc_lo)
        f.addRow("Accuracy sweet spot — high", self.acc_hi)
        f.addRow("Size learning rate", self.lr)
        f.addRow("Min target scale", self.scale_min)
        f.addRow("Max target scale", self.scale_max)

        # weakness targeting
        self.cols = _ispin(s.region_cols, 2, 5)
        self.rows = _ispin(s.region_rows, 1, 5)
        self.focus = _dspin(s.focus_weight, 0.0, 0.9)
        self.blend = _dspin(s.telemetry_blend, 0.0, 1.0)
        reg = QGroupBox("Weak-region targeting (bandit)")
        f = _form(reg)
        f.addRow("Grid columns", self.cols)
        f.addRow("Grid rows", self.rows)
        f.addRow("Focus weight (spawn mass on weak region)", self.focus)
        f.addRow("Telemetry blend (flick evidence weight)", self.blend)

        # stochastic movement
        self.theta = _dspin(s.ou_theta, 0.05, 2.0)
        self.sigma = _dspin(s.ou_sigma, 0.0, 1.5)
        self.mov_min = _dspin(s.min_movement, 0.0, 1.0)
        self.mov_max = _dspin(s.max_movement, 0.0, 1.0)
        mov = QGroupBox("Anti-autopilot movement (Ornstein-Uhlenbeck)")
        f = _form(mov)
        f.addRow("Mean reversion θ", self.theta)
        f.addRow("Diffusion σ", self.sigma)
        f.addRow("Min movement intensity", self.mov_min)
        f.addRow("Max movement intensity", self.mov_max)

        # telemetry / clips
        self.skip_splash_cb = QCheckBox(
            "Skip the loading screen (jump straight to the window)")
        self.skip_splash_cb.setChecked(bool(getattr(s, "skip_splash", False)))
        self.telemetry = QCheckBox("Record raw mouse telemetry while watching")
        self.telemetry.setChecked(s.telemetry_enabled)
        self.clips = QCheckBox("Capture video clips of notable moments (needs kovadapt[clips])")
        self.clips.setChecked(s.clips_enabled)
        self.clip_fps = _ispin(s.clip_fps, 10, 60)
        self.clip_buf = _dspin(s.clip_buffer_seconds, 30.0, 300.0, 10.0, 0)
        tel = QGroupBox("Telemetry && clips")
        f = _form(tel)
        f.addRow(self.skip_splash_cb)
        f.addRow(self.telemetry)
        f.addRow(self.clips)
        f.addRow("Clip FPS", self.clip_fps)
        f.addRow("Clip ring buffer (s)", self.clip_buf)

        # advanced engine internals
        self.half_life = _dspin(
            s.ewma_half_life, 1.0, 50.0, 1.0, 1,
            "Runs until an old run's influence on your averages halves.")
        self.coupling = _dspin(
            s.size_speed_coupling, 0.0, 1.0, 0.05, 2,
            "How much faster targets are enlarged to compensate (fairness floor).")
        self.pace_gain = _dspin(
            s.pace_coupling_gain, 0.0, 2.0, 0.05, 2,
            "How hard running above your normal pace pushes movement up next run.")
        self.min_shots = _ispin(
            s.min_shots_for_size, 0, 100,
            "Runs with fewer shots than this never move the size controller.")
        self.obs_noise = _dspin(
            s.bandit_obs_noise, 0.05, 2.0, 0.05, 2,
            "Observation noise of region evidence: lower = each run moves beliefs more.")
        self.prior_var = _dspin(
            s.bandit_prior_var, 0.1, 5.0, 0.1, 2,
            "Prior variance of unexplored regions: higher = more early exploration.")
        self.decay = _dspin(
            s.bandit_posterior_decay, 0.0, 0.5, 0.01, 2,
            "Per-run forgetting toward the prior so fixed weaknesses re-open (0 = never forget).")
        adv = QGroupBox("Advanced engine internals")
        f = _form(adv)
        f.addRow("EWMA half-life (runs)", self.half_life)
        f.addRow("Size–speed coupling", self.coupling)
        f.addRow("Pace coupling gain", self.pace_gain)
        f.addRow("Min shots for size control", self.min_shots)
        f.addRow("Bandit observation noise", self.obs_noise)
        f.addRow("Bandit prior variance", self.prior_var)
        f.addRow("Bandit posterior decay", self.decay)

        # trace-informed dodge
        self.dodge_en = QCheckBox("Targets strafe longer toward your weak flick side")
        self.dodge_en.setChecked(s.dodge_bias_enabled)
        self.dodge_gain = _dspin(
            s.dodge_bias_gain, 0.0, 2.0, 0.1, 1,
            "Scales measured left/right bias into strafe asymmetry.")
        dodge = QGroupBox("Trace-informed dodge direction")
        f = _form(dodge)
        f.addRow(self.dodge_en)
        f.addRow("Bias gain", self.dodge_gain)

        # fatigue
        self.fat_en = QCheckBox("Detect flick-quality decay and suggest breaks")
        self.fat_en.setChecked(s.fatigue_detection_enabled)
        self.fat_ease = QCheckBox("Ease difficulty while fatigued (bigger, calmer targets)")
        self.fat_ease.setChecked(s.fatigue_easing)
        self.fat_sens = _dspin(
            s.fatigue_sensitivity, 0.1, 3.0, 0.1, 1,
            "Above 1 flags fatigue sooner; below 1, later.")
        self.fat_runs = _ispin(
            s.fatigue_min_runs, 2, 20,
            "Runs with telemetry needed before the trend is trusted.")
        fat = QGroupBox("Session fatigue")
        f = _form(fat)
        f.addRow(self.fat_en)
        f.addRow(self.fat_ease)
        f.addRow("Sensitivity", self.fat_sens)
        f.addRow("Min runs", self.fat_runs)

        # per-archetype overrides
        self.arch_en = QCheckBox(
            "Adapt differently per task type (auto-detected: clicking / tracking / switching)")
        self.arch_en.setChecked(s.archetype_enabled)
        self.arch_spins: dict[str, dict[str, QDoubleSpinBox]] = {}
        # A spin with no override shows the global value it inherits, which is
        # indistinguishable from a deliberate per-archetype choice. Remember
        # which keys were real overrides and what each spin was loaded with, so
        # _save() can tell "inherited" from "chosen".
        self._arch_explicit: dict[str, set[str]] = {}
        self._arch_base: dict[str, dict[str, float]] = {}
        arch = QGroupBox("Per-archetype overrides (clicking is the baseline)")
        av = QVBoxLayout(arch)
        av.setContentsMargins(16, 12, 16, 16)
        av.setSpacing(14)
        av.addWidget(self.arch_en)
        for name in _ARCH_EDITABLE:
            ov = (s.archetype_overrides or {}).get(name) or {}
            spins: dict[str, QDoubleSpinBox] = {}
            for key, cap in _ARCH_KEYS:
                spins[key] = _dspin(float(ov.get(key, getattr(s, key))), 0.0, 3.0)
            self.arch_spins[name] = spins
            self._remember_arch(name, ov)
            box = QGroupBox(name)
            row = _form(box)
            for key, cap in _ARCH_KEYS:
                row.addRow(cap, spins[key])
            av.addWidget(box)

        save = QPushButton("Save settings")
        save.setProperty("accent", True)
        save.clicked.connect(self._save)
        reset = QPushButton("Reset to defaults")
        reset.setToolTip("Restore every knob to the shipped defaults (paths are kept). "
                         "Takes effect after Save.")
        reset.clicked.connect(self._reset)
        self.status = QLabel("")
        self.status.setProperty("dim", True)
        bar = QHBoxLayout()
        bar.setSpacing(12)
        bar.addWidget(save)
        bar.addWidget(reset)
        bar.addWidget(self.status)
        bar.addStretch(1)

        # group boxes stack in one column and flow in the page-space —
        # the outer shell scrolls, so no nested scroll area here.
        lay = QVBoxLayout(self)
        lay.setSpacing(18)
        lay.addWidget(HintBar(settings, (
            "Every knob has a tooltip — hover it. The defaults reproduce the "
            "shipped behavior, <b>Reset to defaults</b> gets you back, and "
            "nothing applies until <b>Save settings</b>.")))
        for box in (mouse, diff, reg, mov, tel, adv, dodge, fat, arch):
            lay.addWidget(box)
        lay.addLayout(bar)
        lay.addStretch(1)

    # ------------------------------------------------------------------
    def _remember_arch(self, name: str, ov: dict) -> None:
        """Snapshot an archetype's baseline: which knobs are real overrides
        and what the spins currently read. Anything loaded (init, reset, the
        readout refresh after a save) is by definition not a user choice."""
        self._arch_explicit[name] = {k for k, _ in _ARCH_KEYS if k in ov}
        self._arch_base[name] = {k: self.arch_spins[name][k].value() for k, _ in _ARCH_KEYS}

    def _update_cm360(self) -> None:
        """Live cm/360 readout (KovaaK's/Quake yaw: 0.022° per count)."""
        counts_per_deg = self.dpi.value() * self.sens.value() * 0.022
        if counts_per_deg <= 0:
            self.cm360.setText("—")
            return
        cm = 2.54 * 360.0 / counts_per_deg
        self.cm360.setText(f"{cm:.1f} cm / 360°")

    def _reset(self) -> None:
        """Load shipped defaults into every widget (Save still required)."""
        d = Settings(kovaaks_root=self.s.kovaaks_root, profile_dir=self.s.profile_dir)
        self.dpi.setValue(float(getattr(d, "mouse_dpi", 0) or 800.0))
        self.sens.setValue(float(getattr(d, "game_sens", 0) or 1.0))
        self.acc_lo.setValue(d.target_accuracy_low)
        self.acc_hi.setValue(d.target_accuracy_high)
        self.lr.setValue(d.size_learning_rate)
        self.scale_min.setValue(d.min_target_scale)
        self.scale_max.setValue(d.max_target_scale)
        self.cols.setValue(d.region_cols)
        self.rows.setValue(d.region_rows)
        self.focus.setValue(d.focus_weight)
        self.blend.setValue(d.telemetry_blend)
        self.theta.setValue(d.ou_theta)
        self.sigma.setValue(d.ou_sigma)
        self.mov_min.setValue(d.min_movement)
        self.mov_max.setValue(d.max_movement)
        self.skip_splash_cb.setChecked(bool(getattr(d, "skip_splash", False)))
        self.telemetry.setChecked(d.telemetry_enabled)
        self.clips.setChecked(d.clips_enabled)
        self.clip_fps.setValue(d.clip_fps)
        self.clip_buf.setValue(d.clip_buffer_seconds)
        self.half_life.setValue(d.ewma_half_life)
        self.coupling.setValue(d.size_speed_coupling)
        self.pace_gain.setValue(d.pace_coupling_gain)
        self.min_shots.setValue(d.min_shots_for_size)
        self.obs_noise.setValue(d.bandit_obs_noise)
        self.prior_var.setValue(d.bandit_prior_var)
        self.decay.setValue(d.bandit_posterior_decay)
        self.dodge_en.setChecked(d.dodge_bias_enabled)
        self.dodge_gain.setValue(d.dodge_bias_gain)
        self.fat_en.setChecked(d.fatigue_detection_enabled)
        self.fat_ease.setChecked(d.fatigue_easing)
        self.fat_sens.setValue(d.fatigue_sensitivity)
        self.fat_runs.setValue(d.fatigue_min_runs)
        self.arch_en.setChecked(d.archetype_enabled)
        for name, spins in self.arch_spins.items():
            ov = (d.archetype_overrides or {}).get(name) or {}
            for key, _ in _ARCH_KEYS:
                spins[key].setValue(float(ov.get(key, getattr(d, key))))
            self._remember_arch(name, ov)
        self.status.setText("defaults loaded — click Save settings to apply")

    def _save(self) -> None:
        s = self.s
        # plain attribute set is safe even before the fields land on Settings
        s.mouse_dpi = self.dpi.value()
        s.game_sens = self.sens.value()
        s.target_accuracy_low = min(self.acc_lo.value(), self.acc_hi.value() - 0.01)
        s.target_accuracy_high = self.acc_hi.value()
        s.size_learning_rate = self.lr.value()
        s.min_target_scale = self.scale_min.value()
        s.max_target_scale = self.scale_max.value()
        s.region_cols = self.cols.value()
        s.region_rows = self.rows.value()
        s.focus_weight = self.focus.value()
        s.telemetry_blend = self.blend.value()
        s.ou_theta = self.theta.value()
        s.ou_sigma = self.sigma.value()
        s.min_movement = min(self.mov_min.value(), self.mov_max.value())
        s.max_movement = self.mov_max.value()
        s.skip_splash = self.skip_splash_cb.isChecked()
        s.telemetry_enabled = self.telemetry.isChecked()
        s.clips_enabled = self.clips.isChecked()
        s.clip_fps = self.clip_fps.value()
        s.clip_buffer_seconds = self.clip_buf.value()
        s.ewma_half_life = self.half_life.value()
        s.size_speed_coupling = self.coupling.value()
        s.pace_coupling_gain = self.pace_gain.value()
        s.min_shots_for_size = self.min_shots.value()
        s.bandit_obs_noise = self.obs_noise.value()
        s.bandit_prior_var = self.prior_var.value()
        s.bandit_posterior_decay = self.decay.value()
        s.dodge_bias_enabled = self.dodge_en.isChecked()
        s.dodge_bias_gain = self.dodge_gain.value()
        s.fatigue_detection_enabled = self.fat_en.isChecked()
        s.fatigue_easing = self.fat_ease.isChecked()
        s.fatigue_sensitivity = self.fat_sens.value()
        s.fatigue_min_runs = self.fat_runs.value()
        s.archetype_enabled = self.arch_en.isChecked()
        overrides = dict(s.archetype_overrides or {})
        for name, spins in self.arch_spins.items():
            ov = dict(overrides.get(name) or {})
            base, explicit = self._arch_base[name], self._arch_explicit[name]
            for key, _ in _ARCH_KEYS:
                val = round(spins[key].value(), 4)
                glob = round(float(getattr(s, key)), 4)
                # An override may only exist where the user chose to differ.
                # Persisting an inherited value would freeze this archetype at
                # today's global and silently decouple it from every later edit
                # — and since the spin still holds the pre-edit global, the very
                # edit made above would never reach tracking/switching.
                chosen = key in explicit or abs(val - base[key]) > 1e-9
                if chosen and abs(val - glob) > 1e-9:
                    ov[key] = val
                else:
                    ov.pop(key, None)
                    spins[key].setValue(glob)   # keep the inherited readout honest
            # Keep the accuracy band ordered against what the archetype really
            # resolves to: a bound left inherited falls through to the global.
            lo = ov.get("target_accuracy_low", s.target_accuracy_low)
            hi = ov.get("target_accuracy_high", s.target_accuracy_high)
            if lo > hi - 0.01:
                lo = round(hi - 0.01, 4)
                ov["target_accuracy_low"] = lo
                spins["target_accuracy_low"].setValue(lo)
            overrides[name] = ov
            self._remember_arch(name, ov)
        s.archetype_overrides = overrides
        path = s.save()
        self.status.setText(
            f"saved to {path} — restart the watch session to apply everything")
        self.settings_changed.emit(s)
