"""Offscreen tests for the "How it learns" page (gui/ml_page.py): it
constructs, the sections and live diagrams exist, every sources line cites
real analysis/kb.py ids, the prose is real (no placeholders), and restyle
survives theme switches. Skipped wholesale without PySide6."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

from PySide6.QtWidgets import QApplication  # noqa: E402

from kovadapt.config import Settings  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Settings.save() defaults to Path.home()/.kovadapt/settings.json — these
    tests construct HintBars whose dismissal path saves, and must never write
    the developer's real settings file."""
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
    (root / "Saved" / "SaveGames" / "Scenarios").mkdir(parents=True)
    return Settings(
        kovaaks_root=str(root),
        profile_dir=str(tmp_path / "prof"),
        telemetry_enabled=False,
        onboarding_done=True,
    )


def test_ml_page_constructs_with_sections_and_diagrams(qapp, settings):
    from kovadapt.gui.ml_page import MLPage

    page = MLPage(settings)
    assert page.objectName() == "tabPage"       # backdrop-transparency contract
    assert len(page.section_titles) >= 6
    assert len(page.diagrams) == 4
    names = {type(d).__name__ for d in page.diagrams}
    assert names == {"ZoneGridDiagram", "DeadbandDiagram",
                     "OUTraceDiagram", "FittsDiagram"}
    page.deleteLater()


def test_prose_is_real_and_placeholder_free(qapp, settings):
    from kovadapt.gui.ml_page import MLPage

    page = MLPage(settings)
    assert len(page.prose) >= 8                 # lede + six sections + closing
    for lab in page.prose:
        text = lab.text()
        assert text.strip()
        assert len(text) > 200                  # paragraphs, not stubs
        low = text.lower()
        for marker in ("todo", "fixme", "xxx", "lorem", "placeholder", "tbd"):
            assert marker not in low
        assert lab.wordWrap()
    page.deleteLater()


def test_source_lines_cite_real_kb_ids(qapp, settings):
    from kovadapt.analysis import kb
    from kovadapt.gui.ml_page import MLPage

    page = MLPage(settings)
    assert page.cited_ids
    for kid in page.cited_ids:                  # nothing cited that kb lacks
        assert kid in kb.PRINCIPLES or kid in kb.DIAGNOSTICS
    for expected in (                           # the load-bearing doctrine ids
        "p-challenge-point", "p-speed-accuracy-governor", "p-speed-is-growth-axis",
        "dx-acc-above-band", "dx-acc-below-band", "p-two-phase-flick", "p-swipiness",
        "p-rest-position", "dx-region-deficit", "p-fitts-throughput",
        "dx-fitts-progress", "p-sensitivity-doctrine", "dx-input-health",
    ):
        assert expected in page.cited_ids
    assert len(page.source_lines) >= 7          # one line per cited part
    for lab in page.source_lines:
        assert lab.text().startswith("sources:")
        assert lab.toolTip().strip()            # citations live in the tooltip
        assert lab.property("dim") is True
    page.deleteLater()


def test_diagrams_animate_only_while_visible_and_paint(qapp, settings):
    from kovadapt.gui.ml_page import MLPage

    page = MLPage(settings)
    page.show()
    for d in page.diagrams:
        assert d._timer.isActive()              # ~12 fps only while visible
    page.hide()
    for d in page.diagrams:
        assert not d._timer.isActive()
    # painting at many phases must not raise; grab() drives paintEvent
    for d in page.diagrams:
        for _ in range(75):                     # crosses round/segment bounds
            d._advance()
        assert not d.grab().isNull()
    page.deleteLater()


def test_restyle_across_theme_switches(qapp, settings):
    from kovadapt.gui import theme
    from kovadapt.gui.ml_page import MLPage
    from kovadapt.gui.theme import ThemeManager

    themes = ThemeManager(qapp, settings)
    page = MLPage(settings)
    themes.set_mode("light")
    page.restyle(theme.current())
    assert not theme.current().is_dark
    themes.set_mode("dark")
    page.restyle(theme.current())
    assert theme.current().is_dark
    assert not page.grab().isNull()             # paints fine after switches
    page.deleteLater()
