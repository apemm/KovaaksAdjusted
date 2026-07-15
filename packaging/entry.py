"""Frozen-exe entry point.

kovadapt.exe             -> the GUI
kovadapt.exe --watchdog  -> headless watchdog (used by the startup entry)
Any other args           -> full CLI (kovadapt.exe checkup, status, ...)
"""

import sys


def main() -> None:
    args = sys.argv[1:]
    if args == ["--watchdog"]:
        from kovadapt.cli import main as cli_main

        cli_main(["watchdog"])
    elif args:
        from kovadapt.cli import main as cli_main

        cli_main(args)
    else:
        from kovadapt.gui.app import main as gui_main

        sys.exit(gui_main())


if __name__ == "__main__":
    main()
