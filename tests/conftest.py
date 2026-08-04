import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# Resolved at IMPORT, before any test can monkeypatch Path.home — this is the
# one path the suite must never write. See _never_write_the_real_settings.
_REAL_SETTINGS = (Path.home() / ".kovadapt" / "settings.json").resolve()

# Real KovaaK's install, if present (skipped on CI / other machines).
_ROOT = os.environ.get(
    "KOVAAKS_ROOT",
    r"C:\Program Files (x86)\Steam\steamapps\common\FPSAimTrainer\FPSAimTrainer",
)


@pytest.fixture(autouse=True)
def _never_write_the_real_settings(monkeypatch):
    """Make it IMPOSSIBLE for the suite to write the developer's settings.json.

    `Settings.save()` and `load()` deliberately default to the canonical
    ~/.kovadapt/settings.json regardless of a customized `profile_dir` — the
    bootstrap location has to be knowable (documented in ARCHITECTURE.md). The
    consequence is that a test which builds a Settings with a temp
    `profile_dir` and then calls `save()` with NO argument writes the real
    file, and monkeypatching `Path.home` does not help because the class-level
    default was already evaluated at import time.

    That has now happened twice on this machine, both times repointing
    `kovaaks_root`/`profile_dir` at a scratch directory and silently breaking
    the developer's install. A comment was not enough, so this is a wall.

    The rule is exactly "never write the REAL file", not "never save without a
    path": `save()` resolves its default through `Path.home()` at CALL time, so
    a test with proper home isolation is already safe and stays allowed. Only a
    write that would land on the genuine path — captured here at import, before
    any test can patch it — fails.
    """
    from kovadapt.config import Settings

    real = Settings.save

    def guarded(self, path=None):
        target = Path(path) if path is not None else \
            Path.home() / ".kovadapt" / "settings.json"
        if target.resolve() == _REAL_SETTINGS:
            raise AssertionError(
                f"this test would overwrite the REAL {_REAL_SETTINGS}. Either "
                "monkeypatch Path.home to tmp_path, or pass an explicit path: "
                "s.save(tmp_path / 'settings.json')")
        return real(self, target)

    monkeypatch.setattr(Settings, "save", guarded)


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def kovaaks_root() -> Path:
    p = Path(_ROOT)
    if not (p / "stats").is_dir():
        pytest.skip("KovaaK's install not available")
    return p


@pytest.fixture(autouse=True)
def _drain_deferred_deletes():
    """Flush Qt's DeferredDelete queue after every test.

    `close()` only HIDES a window, so a MainWindow that a test closed stays
    live — measured, 526 widgets leaked per window and never reclaimed. Six of
    them across the smoke tests put ~3100 permanently-live widgets in the
    QApplication, and `theme._apply` re-polishes EVERY live widget on every
    call (superlinear, about n^1.2: 0.7ms at 0 widgets, 150ms at 850, 2069ms
    at 6800). So the rendered theme tests were paying for windows that six
    earlier tests had "closed", and 72% of the suite's wall clock was that
    one effect rather than any work.

    Explicitly NOT gc.collect(): measured at +118s for nothing, because the
    windows are strongly held and the collector has nothing to reclaim. What
    they need is the deleteLater() their tests now call, and a loop turn for
    Qt to act on it.
    """
    yield
    try:
        from PySide6.QtCore import QCoreApplication, QEvent
    except Exception:
        return
    if QCoreApplication.instance() is not None:
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
