"""Frozen-exe entry point.

kovadapt.exe             -> the GUI
kovadapt.exe --watchdog  -> headless watchdog (used by the startup entry)
Any other args           -> full CLI (kovadapt.exe checkup, status, ...)
"""

import sys


def _dead(stream) -> bool:
    if stream is None:
        return True
    try:
        stream.fileno()
        return False
    except Exception:
        return True  # PyInstaller NullWriter in a windowed build


def _attach_console() -> None:
    """The exe is a windowed (console=False) build so launching the GUI never
    flashes a console box — but that leaves CLI runs mute. When invoked from
    an existing console with no redirection, attach to it and rebind the dead
    streams; piped/redirected streams are real files and are left alone."""
    if sys.platform != "win32":
        return
    import ctypes

    if not ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
        return  # no parent console (double-click, Run key): nothing to print to
    for name in ("stdout", "stderr"):
        if _dead(getattr(sys, name)):
            try:
                setattr(sys, name, open("CONOUT$", "w", buffering=1,
                                        encoding="utf-8", errors="replace"))
            except OSError:
                pass


def main() -> None:
    args = sys.argv[1:]
    if args:
        _attach_console()
        from kovadapt.cli import main as cli_main

        cli_main(["watchdog"] if args == ["--watchdog"] else args)
    else:
        from kovadapt.gui.app import main as gui_main

        sys.exit(gui_main())


if __name__ == "__main__":
    main()
