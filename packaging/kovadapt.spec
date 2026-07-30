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
)
coll = COLLECT(exe, a.binaries, a.datas, name="kovadapt")
