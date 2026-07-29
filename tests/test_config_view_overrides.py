"""Adaptability page: per-archetype overrides must stay *overrides*.

Saving the page used to freeze the current global tunables into every
archetype, so a later edit of a global knob silently stopped reaching
tracking/switching scenarios. These pin the inheritance rule: an override
exists only for a field the user actually chose to differ.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import Settings, default_archetype_overrides  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """ConfigView._save() calls Settings.save(), which defaults to
    Path.home()/.kovadapt/settings.json — never the developer's real file."""
    from pathlib import Path

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    yield app


@pytest.fixture()
def settings(tmp_path):
    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    return Settings(
        kovaaks_root=str(root),
        profile_dir=str(tmp_path / "prof"),
        telemetry_enabled=False,
        onboarding_done=True,
    )


def _view(settings):
    from kovadapt.gui.config_view import ConfigView

    return ConfigView(settings)


def test_save_does_not_invent_overrides(qapp, settings):
    """An untouched page saves the shipped override set — no extra keys."""
    view = _view(settings)
    view._save()
    assert settings.archetype_overrides == default_archetype_overrides()
    view.deleteLater()


def test_global_edit_still_reaches_inheriting_archetypes(qapp, settings):
    """focus_weight is inherited by tracking (only switching overrides it).

    Editing the global knob must keep propagating; freezing the spin's
    pre-edit value as an override would decouple tracking forever.
    """
    view = _view(settings)
    view.focus.setValue(0.80)
    view._save()
    assert settings.focus_weight == pytest.approx(0.80)
    assert settings.for_archetype("tracking").focus_weight == pytest.approx(0.80)
    assert "focus_weight" not in settings.archetype_overrides["tracking"]
    # switching genuinely differs, so its override survives untouched
    assert settings.for_archetype("switching").focus_weight == pytest.approx(0.60)
    view.deleteLater()


def test_global_edit_reaches_archetypes_after_repeated_saves(qapp, settings):
    """The view outlives a save (app.py builds it once), so the second save
    must not persist the stale value its spins were built with."""
    view = _view(settings)
    view._save()
    view.mov_min.setValue(0.20)          # tracking overrides min_movement...
    view.lr.setValue(0.30)               # ...but switching inherits both
    view._save()
    sw = settings.for_archetype("switching")
    assert sw.min_movement == pytest.approx(0.20)
    assert sw.size_learning_rate == pytest.approx(0.30)
    assert settings.for_archetype("tracking").min_movement == pytest.approx(0.35)
    view.deleteLater()


def test_edited_archetype_spin_becomes_an_override(qapp, settings):
    """The other half of the rule: a knob the user does move is persisted."""
    view = _view(settings)
    view.arch_spins["switching"]["size_learning_rate"].setValue(0.25)
    view._save()
    assert settings.archetype_overrides["switching"]["size_learning_rate"] == 0.25
    assert settings.size_learning_rate == pytest.approx(0.9)   # global untouched
    assert settings.for_archetype("switching").size_learning_rate == pytest.approx(0.25)
    view.deleteLater()


def test_bare_archetype_survives_accuracy_band_clamp(qapp, settings):
    """A settings.json with no accuracy override must not KeyError in the
    per-archetype band clamp, nor grow one out of thin air."""
    settings.archetype_overrides = {"clicking": {}, "tracking": {}, "switching": {}}
    view = _view(settings)
    view.acc_lo.setValue(0.50)
    view.acc_hi.setValue(0.60)
    view._save()
    assert settings.archetype_overrides["tracking"] == {}
    band = settings.for_archetype("tracking")
    assert (band.target_accuracy_low, band.target_accuracy_high) == (0.50, 0.60)
    view.deleteLater()


def test_reset_then_save_restores_shipped_overrides(qapp, settings):
    """Reset re-establishes the baseline: the values it loads are defaults,
    not user choices, so saving after it reproduces the shipped set."""
    view = _view(settings)
    view.arch_spins["tracking"]["min_movement"].setValue(0.90)
    view._save()
    assert settings.archetype_overrides["tracking"]["min_movement"] == 0.90
    view._reset()
    view._save()
    assert settings.archetype_overrides == default_archetype_overrides()
    view.deleteLater()
