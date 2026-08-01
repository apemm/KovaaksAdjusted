# PyInstaller spec for the kovadapt desktop app.
#
# Build (from the repo root):
#     pip install .[gui] pyinstaller
#     pyinstaller packaging/kovadapt.spec --noconfirm
#
# One-DIR build by design: one-file exes unpack to %TEMP% on every launch
# (seconds of startup lag) and trip antivirus heuristics far more often.
# dist/kovadapt/kovadapt.exe runs directly; zip the folder to distribute.
#
# The [clips] extra (dxcam/opencv) is intentionally excluded — it adds
# ~80 MB; users who want video clips should run from pip instead.

from PySide6 import __file__ as pyside_file  # noqa: F401 (assert import works)

# --- version resource -------------------------------------------------------
# The shipped exe carried NO version metadata: Windows reported 0.0.0.0, so a
# user could not tell which build they were running and a bug report could not
# name one. Read from the package so it can never drift from `kovadapt -V`.
import sys as _sys
_sys.path.insert(0, "..")
from kovadapt import __version__ as _v  # noqa: E402
# These are NOT injected into the spec namespace — import them.
from PyInstaller.utils.win32.versioninfo import (  # noqa: E402
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo,
    VarStruct, VSVersionInfo,
)

_parts = tuple(int(x) for x in _v.split(".")[:3]) + (0,)
_vs = VSVersionInfo(
    ffi=FixedFileInfo(filevers=_parts, prodvers=_parts, mask=0x3F, flags=0x0,
                      OS=0x40004, fileType=0x1, subtype=0x0),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", "kovadapt"),
            StringStruct("FileDescription", "kovadapt - adaptive KovaaK's"),
            StringStruct("FileVersion", _v),
            StringStruct("InternalName", "kovadapt"),
            StringStruct("OriginalFilename", "kovadapt.exe"),
            StringStruct("ProductName", "kovadapt"),
            StringStruct("ProductVersion", _v),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x409, 1200])]),
    ],
)

a = Analysis(
    ["../packaging/entry.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=[
        "kovadapt.gui.app",
        "kovadapt.gui.optimizer_window",
        "kovadapt.optimize.watchdog",
        "kovadapt.launcher",
        # The two optional pages are reached through `try: ... except
        # ImportError` in gui/app.py, which is exactly the shape where a
        # packager's silence costs a whole section with no error: the import
        # fails, the name becomes None, and the section is simply absent from
        # the shipped app. Named explicitly so the build cannot lose them.
        "kovadapt.gui.ml_page",
        "kovadapt.gui.changes_view",
        "pyqtgraph",
        "psutil",
    ],
    excludes=[
        "dxcam", "cv2",                       # [clips] extra: pip install only
        # [ml] extra: torch is 2.6 GB of a 2.9 GB build. kovadapt/ml/model.py
        # imports it at MODULE level and watcher.py/cli.py reach kovadapt.ml,
        # so PyInstaller follows it in regardless of the runtime guards — the
        # first build after torch was installed on a dev box would ship a
        # 29x-larger release than v0.3 without anyone deciding to. The
        # FlickEncoder is opt-in by design; users who want it run from pip.
        "torch", "torchvision", "torchaudio", "functorch",
        # ...and torch's own dependency tree, none of which anything outside
        # kovadapt/ml/ imports (verified by grep before excluding).
        "scipy", "PIL", "lxml", "sympy", "networkx", "jinja2",
        "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets",
        "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtQuick3D",
        "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
        "PySide6.QtDesigner", "PySide6.QtTest",
        "tkinter", "matplotlib", "IPython", "pandas",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="kovadapt",
    console=False,          # GUI app; the watchdog entry re-uses this exe
    icon=None,
    version=_vs,            # so Windows and bug reports can name the build
)
coll = COLLECT(exe, a.binaries, a.datas, name="kovadapt")
