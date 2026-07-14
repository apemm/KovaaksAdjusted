"""Adaptability configuration tab: the full Settings surface, saved to
~/.kovadapt/settings.json. More knobs than KovaaK's official offering."""

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
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..config import Settings


def _dspin(val: float, lo: float, hi: float, step: float = 0.05, dec: int = 2) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    return w


class ConfigView(QWidget):
    settings_changed = Signal(object)   # emits the saved Settings

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self.s = settings
        s = settings

        # difficulty
        self.acc_lo = _dspin(s.target_accuracy_low, 0.1, 0.95)
        self.acc_hi = _dspin(s.target_accuracy_high, 0.15, 0.99)
        self.lr = _dspin(s.size_learning_rate, 0.1, 3.0, 0.1)
        self.scale_min = _dspin(s.min_target_scale, 0.1, 1.0)
        self.scale_max = _dspin(s.max_target_scale, 1.0, 5.0)
        diff = QGroupBox("Difficulty controller")
        f = QFormLayout(diff)
        f.addRow("Accuracy sweet spot — low", self.acc_lo)
        f.addRow("Accuracy sweet spot — high", self.acc_hi)
        f.addRow("Size learning rate", self.lr)
        f.addRow("Min target scale", self.scale_min)
        f.addRow("Max target scale", self.scale_max)

        # weakness targeting
        self.cols = QSpinBox(); self.cols.setRange(2, 5); self.cols.setValue(s.region_cols)
        self.rows = QSpinBox(); self.rows.setRange(1, 5); self.rows.setValue(s.region_rows)
        self.focus = _dspin(s.focus_weight, 0.0, 0.9)
        self.blend = _dspin(s.telemetry_blend, 0.0, 1.0)
        reg = QGroupBox("Weak-region targeting (bandit)")
        f = QFormLayout(reg)
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
        f = QFormLayout(mov)
        f.addRow("Mean reversion θ", self.theta)
        f.addRow("Diffusion σ", self.sigma)
        f.addRow("Min movement intensity", self.mov_min)
        f.addRow("Max movement intensity", self.mov_max)

        # telemetry / clips
        self.telemetry = QCheckBox("Record raw mouse telemetry while watching")
        self.telemetry.setChecked(s.telemetry_enabled)
        self.clips = QCheckBox("Capture video clips of notable moments (needs kovadapt[clips])")
        self.clips.setChecked(s.clips_enabled)
        self.clip_fps = QSpinBox(); self.clip_fps.setRange(10, 60); self.clip_fps.setValue(s.clip_fps)
        self.clip_buf = _dspin(s.clip_buffer_seconds, 30.0, 300.0, 10.0, 0)
        tel = QGroupBox("Telemetry & clips")
        f = QFormLayout(tel)
        f.addRow(self.telemetry)
        f.addRow(self.clips)
        f.addRow("Clip FPS", self.clip_fps)
        f.addRow("Clip ring buffer (s)", self.clip_buf)

        save = QPushButton("Save settings")
        save.setProperty("accent", True)
        save.clicked.connect(self._save)
        self.status = QLabel("")
        self.status.setProperty("dim", True)
        bar = QHBoxLayout()
        bar.addWidget(save)
        bar.addWidget(self.status)
        bar.addStretch(1)

        inner = QWidget()
        col = QVBoxLayout(inner)
        for box in (diff, reg, mov, tel):
            col.addWidget(box)
        col.addLayout(bar)
        col.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        lay = QVBoxLayout(self)
        lay.addWidget(scroll)

    # ------------------------------------------------------------------
    def _save(self) -> None:
        s = self.s
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
        s.telemetry_enabled = self.telemetry.isChecked()
        s.clips_enabled = self.clips.isChecked()
        s.clip_fps = self.clip_fps.value()
        s.clip_buffer_seconds = self.clip_buf.value()
        path = s.save()
        self.status.setText(f"saved to {path} (applies to the next watch session)")
        self.settings_changed.emit(s)
