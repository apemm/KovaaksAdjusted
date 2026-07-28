"""Launch integration: start KovaaK's and jump into adaptive tasks from the app.

kovadapt never modifies or redistributes the game — integration goes through
surfaces Valve and the game officially sanction for external tools:

  * **Deep links** (KovaaK's 3.0.0+, verified present in the 3.9.x binary):
    ``steam://run/824270//?action=jump-to-scenario&name=<encoded>&mode=challenge``
    loads the game directly into a scenario. Steam delivers the query params
    via the Steamworks API, so the same URL works cold (Steam boots the game
    into the scenario) and warm (the running instance jumps in place).
  * **Steam's browser protocol** launches the game exactly like the library
    button does. Steam itself enforces ownership and DRM, so this doubles as
    the "you actually own KovaaK's" check: without an owning, logged-in
    account the game simply won't start.
  * **The game's user-editable save files** (``Saved/SaveGames/Playlists``)
    queue scenarios as a local playlist — the same JSON the in-game playlist
    editor writes (picked up at next game start).

Everything here is stdlib (+ optional psutil for the running-game probe) and
imports cleanly on any OS; probes degrade to False off-Windows.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .config import ADAPTIVE_SUFFIX, Settings

STEAM_APP_ID = 824270
GAME_PROCESS = "FPSAimTrainer"          # matches optimize.watchdog.GAME_PROCESS
PLAYLIST_NAME = "kovadapt adaptive"     # one well-known playlist, overwritten per task

WINDOWS = sys.platform == "win32"


# --------------------------------------------------------------------- status
@dataclass
class InstallStatus:
    """What we can verify about the local KovaaK's + Steam install.

    `manifest_found` is the tModLoader-style installation proof: the Steam
    library that contains the game also holds appmanifest_824270.acf, which
    only exists for apps installed by an owning account. Actual ownership is
    re-checked by Steam itself on every ``steam://`` launch.
    """

    root_found: bool = False        # kovaaks_root exists and looks like the game
    manifest_found: bool = False    # appmanifest_824270.acf in the owning library
    steam_found: bool = False       # Steam client locatable (registry)
    game_running: bool = False      # FPSAimTrainer process alive right now

    @property
    def ok(self) -> bool:
        return self.root_found and self.steam_found

    def describe(self) -> str:
        if not self.root_found:
            return "KovaaK's install not found — set KOVAAKS_ROOT or check settings"
        bits = ["KovaaK's found"]
        bits.append("Steam manifest OK" if self.manifest_found else "Steam manifest missing")
        if not self.steam_found:
            bits.append("Steam client not found")
        if self.game_running:
            bits.append("game running")
        return " · ".join(bits)


def steam_client_path() -> Path | None:
    """Steam install dir from the registry (None off-Windows / not installed)."""
    if not WINDOWS:
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as k:
            path, _ = winreg.QueryValueEx(k, "SteamPath")
        p = Path(path)
        return p if p.is_dir() else None
    except OSError:
        return None


def _steamapps_dir(root: Path) -> Path | None:
    """The steamapps directory that contains this KovaaK's install (walk up
    from <library>/steamapps/common/FPSAimTrainer/FPSAimTrainer)."""
    for parent in root.parents:
        if parent.name.lower() == "steamapps":
            return parent
    return None


def game_is_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["name"]):
        if GAME_PROCESS.lower() in (p.info["name"] or "").lower():
            return True
    return False


def check_install(settings: Settings) -> InstallStatus:
    st = InstallStatus()
    root = settings.root
    st.root_found = bool(settings.kovaaks_root) and (root / "stats").is_dir()
    if st.root_found:
        apps = _steamapps_dir(root)
        if apps is None:
            # kovaaks_root may be a junction/symlink whose textual path has
            # no "steamapps" component — the resolved path usually does.
            try:
                apps = _steamapps_dir(root.resolve())
            except OSError:
                apps = None
        st.manifest_found = (
            apps is not None and (apps / f"appmanifest_{STEAM_APP_ID}.acf").is_file()
        )
    st.steam_found = steam_client_path() is not None
    st.game_running = game_is_running()
    return st


# --------------------------------------------------------------------- launch
def _open_steam_url(url: str) -> tuple[str, bool]:
    """ShellExecute the URL (os.startfile) — never a shell, so '&' is safe."""
    if not WINDOWS:
        return "launching KovaaK's requires Windows", False
    try:
        import os

        os.startfile(url)
        return "", True
    except OSError as exc:
        return f"could not reach Steam ({exc}) — is it installed?", False


def launch_game() -> tuple[str, bool]:
    """Start KovaaK's through Steam (ownership enforced by Steam itself)."""
    if game_is_running():
        return "KovaaK's is already running", True
    err, ok = _open_steam_url(f"steam://run/{STEAM_APP_ID}")
    return (err, False) if not ok else ("launching KovaaK's through Steam…", True)


def scenario_url(scenario: str, mode: str = "challenge") -> str:
    """Deep-link URL loading the game straight into `scenario` (display name,
    not a path). This is the exact template the 3.9.x binary itself emits."""
    return (
        f"steam://run/{STEAM_APP_ID}//?action=jump-to-scenario"
        f"&name={urllib.parse.quote(scenario, safe='')}&mode={mode}"
    )


def launch_scenario(scenario: str, mode: str = "challenge") -> tuple[str, bool]:
    """Jump into `scenario` — cold or warm (a running game jumps in place)."""
    err, ok = _open_steam_url(scenario_url(scenario, mode))
    if not ok:
        return err, False
    verb = "jumping to" if game_is_running() else "launching KovaaK's into"
    return f"{verb} {scenario!r}…", True


# ------------------------------------------------------------------ playlists
def _playlist_path(settings: Settings, name: str = PLAYLIST_NAME) -> Path:
    safe = re.sub(r'[<>:"/\\|?*]+', "_", name).strip() or "kovadapt"
    return settings.playlists_dir / f"{safe}.json"


def write_playlist(
    settings: Settings,
    scenarios: list[tuple[str, int]],
    name: str = PLAYLIST_NAME,
    description: str = "",
) -> Path:
    """Write a local playlist in the game's own JSON schema. Key casing is
    load-bearing (`scenario_name`, `play_Count`); UTF-8 without BOM and tab
    indent match what the game itself writes. Picked up at next game start.
    Overwrites the previous kovadapt playlist — one well-known slot."""
    path = _playlist_path(settings, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "playlistName": name,
        "playlistId": 0,
        "authorSteamId": "",
        "authorName": "kovadapt",
        "scenarioList": [
            {"scenario_name": s, "play_Count": max(int(c), 1)} for s, c in scenarios
        ],
        "description": description,
        "hasOfflineScenarios": True,
        "hasEdited": False,
        "shareCode": "",
        "version": 31,
        "updated": int(time.time()),
        "isPrivate": True,
    }
    with path.open("w", encoding="utf-8", newline="\r\n") as f:
        json.dump(doc, f, indent="\t")
    return path


def play_adaptive(settings: Settings, base_scenario: str, runs: int = 5) -> tuple[str, bool]:
    """The Dashboard's Play action: queue the adaptive playlist (next-start
    pickup) and deep-link straight into the adaptive scenario (instant).
    The adaptive .sce must already exist (the watcher bootstraps it)."""
    adaptive = base_scenario + ADAPTIVE_SUFFIX
    if not (settings.scenarios_dir / f"{adaptive}.sce").is_file():
        return f"no adaptive variant yet for {base_scenario!r} — start adapting first", False
    try:
        write_playlist(
            settings,
            [(adaptive, runs)],
            description=f"kovadapt adaptive training for {base_scenario}",
        )
    except OSError as exc:
        return f"could not write playlist: {exc}", False
    # Deep links cannot open locally-generated scenarios (verified in-game),
    # so the playlist is the primary path; the deep-link URL still boots the
    # game and would start working if KovaaK's ever resolves local names.
    err, ok = _open_steam_url(scenario_url(adaptive))
    if not ok:
        return err, False
    return ("KovaaK's launching — in-game, open Playlists → kovadapt adaptive "
            "(or browse local scenarios) to start the task", True)
