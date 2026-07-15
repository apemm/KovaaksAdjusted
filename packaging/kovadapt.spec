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

a = Analysis(
    ["../packaging/entry.py"],
    pathex=[".."],
    binaries=[],
    datas=[],
    hiddenimports=[
        "kovadapt.gui.app",
        "kovadapt.gui.optimizer_window",
        "kovadapt.optimize.watchdog",
        "pyqtgraph",
        "psutil",
    ],
    excludes=[
        "dxcam", "cv2",                       # [clips] extra: pip install only
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
)
coll = COLLECT(exe, a.binaries, a.datas, name="kovadapt")
