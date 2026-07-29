"""PlaylistInProgress staging: it must never eat a real in-progress playlist.

Everything runs against a fake Steam tree under tmp_path, with Path.home
redirected too, so nothing here can reach the real install or ~/.kovadapt.
_open_steam_url is monkeypatched in the play_adaptive tests (on Windows it
would genuinely start the game).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kovadapt import launcher
from kovadapt.config import ADAPTIVE_SUFFIX, Settings

# Stand-ins for a playlist the user was actually part-way through.
THEIRS = b'{"playlistName": "Viscose Benchmarks", "currentIndex": 3}'
THEIRS2 = b'{"playlistName": "VALORANT", "currentIndex": 1}'


@pytest.fixture
def fake_game(tmp_path, monkeypatch):
    """Fake <lib>/steamapps/common/FPSAimTrainer/FPSAimTrainer + a staged playlist."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.delenv(launcher.STAGE_ENV, raising=False)
    root = tmp_path / "lib" / "steamapps" / "common" / "FPSAimTrainer" / "FPSAimTrainer"
    (root / "stats").mkdir(parents=True)
    (root / "Saved" / "SaveGames" / "Scenarios").mkdir(parents=True)
    s = Settings(kovaaks_root=str(root), profile_dir=str(tmp_path / "prof"))
    launcher.write_playlist(s, [("X" + ADAPTIVE_SUFFIX, 2)])
    return s, root / "Saved" / "SaveGames" / "PlaylistInProgress.json"


def _backups(target: Path) -> set[bytes]:
    return {p.read_bytes() for p in target.parent.glob("PlaylistInProgress.json.*")}


# ------------------------------------------------------------------ backups
def test_empty_original_never_latches_as_the_backup(fake_game):
    """The game leaves the file empty between playlists; 0 bytes must never be
    frozen in as "the original" — that is what made every later staging
    unrecoverable."""
    s, target = fake_game
    target.write_bytes(b"")
    assert launcher._stage_playlist_in_progress(s)
    bak = target.with_suffix(".json.kovadapt.bak")
    assert not bak.exists()
    # The genuine playlist that turns up later is the one worth keeping.
    target.write_bytes(THEIRS)
    assert launcher._stage_playlist_in_progress(s)
    assert bak.read_bytes() == THEIRS


def test_replaced_content_is_always_recoverable(fake_game):
    s, target = fake_game
    target.write_bytes(b"")
    assert launcher._stage_playlist_in_progress(s)
    for genuine in (THEIRS, THEIRS2):
        target.write_bytes(genuine)
        assert launcher._stage_playlist_in_progress(s)
        assert genuine in _backups(target)


def test_restaging_our_own_bytes_keeps_the_genuine_prev_backup(fake_game):
    """Play clicked twice: the second staging sees our own file and must not
    rotate it over the copy of what we replaced the first time."""
    s, target = fake_game
    target.write_bytes(THEIRS)
    assert launcher._stage_playlist_in_progress(s)
    prev = target.with_suffix(".json.kovadapt.prev.bak")
    assert prev.read_bytes() == THEIRS
    assert launcher._stage_playlist_in_progress(s)
    assert prev.read_bytes() == THEIRS


# ------------------------------------------------------------------- opt-in
def test_play_adaptive_leaves_the_game_save_alone_by_default(fake_game, monkeypatch):
    s, target = fake_game
    (s.scenarios_dir / f"Foo{ADAPTIVE_SUFFIX}.sce").write_text("[Scenario]\n")
    target.write_bytes(THEIRS)
    monkeypatch.setattr(launcher, "game_is_running", lambda: False)
    monkeypatch.setattr(launcher, "_open_steam_url", lambda url: ("", True))
    msg, ok = launcher.play_adaptive(s, "Foo")
    assert ok, msg
    assert target.read_bytes() == THEIRS        # the game's file is untouched
    assert not _backups(target)                 # and nothing was littered beside it
    assert "staged to resume" not in msg
    # The supported path still happened.
    assert launcher._playlist_path(s).is_file()


def test_play_adaptive_stages_when_opted_in(fake_game, monkeypatch):
    s, target = fake_game
    (s.scenarios_dir / f"Foo{ADAPTIVE_SUFFIX}.sce").write_text("[Scenario]\n")
    target.write_bytes(THEIRS)
    monkeypatch.setenv(launcher.STAGE_ENV, "1")
    monkeypatch.setattr(launcher, "game_is_running", lambda: False)
    monkeypatch.setattr(launcher, "_open_steam_url", lambda url: ("", True))
    msg, ok = launcher.play_adaptive(s, "Foo")
    assert ok and "staged to resume" in msg
    assert b"kovadapt adaptive" in target.read_bytes()
    assert target.with_suffix(".json.kovadapt.bak").read_bytes() == THEIRS
