"""Theming: dark/light palettes, OS sync, one modern QSS sheet.

Two `Palette`s share the same semantic roles; `build_qss()` renders either
into the app-wide stylesheet. `ThemeManager` owns the active palette: it
resolves the configured mode ("auto" follows the Windows light/dark setting
live via QStyleHints.colorSchemeChanged), applies QSS + pyqtgraph defaults,
and emits `changed` so views can restyle pens/labels that were baked at
construction. Views read colors through `current()` — never cache a palette
across theme switches.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    is_dark: bool
    bg: str            # window background
    bg_alt: str        # inputs, panels, plots
    bg_raised: str     # buttons, tooltips, cards
    border: str
    fg: str
    fg_dim: str
    accent: str
    accent_hover: str  # accent-button hover
    accent_fg: str     # text on accent backgrounds
    selection: str     # selection / focused-item background
    good: str
    warn: str
    bad: str


DARK = Palette(
    name="dark", is_dark=True,
    bg="#0f1117", bg_alt="#161a22", bg_raised="#1f2430",
    border="#2a3040", fg="#d9dde6", fg_dim="#8d95a6",
    accent="#4f9dff", accent_hover="#75b3ff", accent_fg="#0d1420",
    selection="#2e5f9e",
    good="#4fc17c", warn="#e0b45f", bad="#e06c5f",
)

LIGHT = Palette(
    name="light", is_dark=False,
    bg="#f2f4f7", bg_alt="#ffffff", bg_raised="#e7eaf0",
    border="#d3d8e0", fg="#22262e", fg_dim="#68707f",
    accent="#2f7de1", accent_hover="#1e63bd", accent_fg="#ffffff",
    selection="#cfe1f8",
    good="#1f9d55", warn="#b07d1f", bad="#cc4a3a",
)

_current: Palette = DARK


def current() -> Palette:
    """The active palette. Read at use time; never cache across switches."""
    return _current


def build_qss(p: Palette) -> str:
    return f"""
* {{
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    color: {p.fg};
}}
QMainWindow, QDialog, QWidget {{ background: {p.bg}; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {p.border}; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent; color: {p.fg_dim};
    padding: 8px 20px; margin-right: 2px;
    border: none; border-bottom: 2px solid transparent;
}}
QTabBar::tab:hover {{ color: {p.fg}; }}
QTabBar::tab:selected {{ color: {p.fg}; border-bottom: 2px solid {p.accent}; }}

QGroupBox {{
    border: 1px solid {p.border}; border-radius: 8px;
    margin-top: 14px; padding-top: 12px; background: {p.bg_alt};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; color: {p.fg_dim}; }}

QPushButton {{
    background: {p.bg_raised}; border: 1px solid {p.border};
    border-radius: 6px; padding: 7px 18px;
}}
QPushButton:hover {{ border-color: {p.accent}; }}
QPushButton:pressed {{ background: {p.border}; }}
QPushButton:disabled {{ color: {p.fg_dim}; background: {p.bg_alt}; }}
QPushButton[accent="true"] {{
    background: {p.accent}; border-color: {p.accent}; color: {p.accent_fg};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[accent="true"]:disabled {{ background: {p.bg_raised}; color: {p.fg_dim}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; padding: 4px 8px; }}
QPushButton[flat="true"]:hover {{ color: {p.accent}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {{
    background: {p.bg_alt}; border: 1px solid {p.border}; border-radius: 6px;
    padding: 5px 8px; selection-background-color: {p.selection};
    selection-color: {p.fg};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QPlainTextEdit:focus {{ border-color: {p.accent}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{
    image: none; width: 0; height: 0;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid {p.fg_dim}; margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background: {p.bg_raised}; border: 1px solid {p.border}; border-radius: 6px;
    selection-background-color: {p.selection}; outline: none;
}}

QListWidget::item {{ padding: 6px; border-bottom: 1px solid {p.border}; }}
QListWidget::item:selected {{ background: {p.selection}; color: {p.fg}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {p.border};
    border-radius: 4px; background: {p.bg_alt};
}}
QCheckBox::indicator:hover {{ border-color: {p.accent}; }}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}

QLabel {{ background: transparent; }}
QLabel[dim="true"] {{ color: {p.fg_dim}; }}
QLabel[headline="true"] {{ font-size: 17px; font-weight: 600; }}
QLabel[stat="true"] {{ color: {p.accent}; font-weight: 600; }}

QSlider::groove:horizontal {{
    height: 4px; background: {p.border}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: {p.accent}; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {p.fg_dim}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {p.border}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{
    background: {p.bg_alt}; border: 1px solid {p.border}; border-radius: 6px;
    text-align: center; color: {p.fg}; height: 16px;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 5px; }}

QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:hover {{ background: {p.accent}; }}

QMenu {{
    background: {p.bg_raised}; border: 1px solid {p.border}; border-radius: 6px;
    padding: 4px;
}}
QMenu::item {{ padding: 6px 24px 6px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background: {p.selection}; }}
QMenu::separator {{ height: 1px; background: {p.border}; margin: 4px 8px; }}

QStatusBar {{ background: {p.bg_alt}; color: {p.fg_dim}; }}
QToolTip {{
    background: {p.bg_raised}; color: {p.fg};
    border: 1px solid {p.border}; padding: 4px 8px;
}}

QFrame[hint="true"] {{
    background: {p.bg_alt}; border: 1px solid {p.border};
    border-left: 3px solid {p.accent}; border-radius: 6px;
}}
QFrame[card="true"] {{
    background: {p.bg_alt}; border: 1px solid {p.border}; border-radius: 8px;
}}
"""


def _apply(app, pal: Palette) -> None:
    global _current
    _current = pal
    app.setStyleSheet(build_qss(pal))
    import pyqtgraph as pg

    pg.setConfigOptions(
        background=pal.bg_alt,
        foreground=pal.fg_dim,
        antialias=True,
        imageAxisOrder="row-major",
    )


class ThemeManager:
    """Resolves settings.theme ("auto" | "dark" | "light") into the active
    palette and keeps the app in sync with the OS scheme while in auto.

    Qt-signal plumbing lives on an internal QObject so this module stays
    importable without Qt only when nothing instantiates the manager.
    """

    MODES = ("auto", "dark", "light")

    def __init__(self, app, settings) -> None:
        from PySide6.QtCore import QObject, Signal

        class _Notifier(QObject):
            changed = Signal(object)   # Palette

        self._app = app
        self.s = settings
        self._notifier = _Notifier()
        self.changed = self._notifier.changed
        self.mode = settings.theme if settings.theme in self.MODES else "auto"
        app.styleHints().colorSchemeChanged.connect(self._os_scheme_changed)
        self.apply()

    # ------------------------------------------------------------------
    @property
    def palette(self) -> Palette:
        return _current

    def _resolve(self) -> Palette:
        if self.mode == "dark":
            return DARK
        if self.mode == "light":
            return LIGHT
        from PySide6.QtCore import Qt

        scheme = self._app.styleHints().colorScheme()
        return LIGHT if scheme == Qt.ColorScheme.Light else DARK

    def set_mode(self, mode: str) -> None:
        """Switch + persist. No-op on unknown modes."""
        if mode not in self.MODES or mode == self.mode:
            return
        self.mode = mode
        self.s.theme = mode
        try:
            self.s.save()
        except OSError:
            pass  # theme still applies for this session
        self.apply()

    def _os_scheme_changed(self, *_args) -> None:
        if self.mode == "auto":
            self.apply()

    def apply(self) -> None:
        _apply(self._app, self._resolve())
        self.changed.emit(_current)
