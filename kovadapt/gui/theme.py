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
    rgb: bool = False    # RGB gamer mode: animated rainbow elements opt in


# Accent presets: (accent, hover, on-accent text, selection wash) per mode.
ACCENTS: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "indigo": {  # the Colicit-vivid default
        "light": ("#5b50e8", "#443ad1", "#ffffff", "#e3e0fb"),
        "dark": ("#8f84ff", "#a9a0ff", "#10111a", "#37317d"),
    },
    "ocean": {   # the v0.4 look
        "light": ("#2f7de1", "#1e63bd", "#ffffff", "#d8e7fa"),
        "dark": ("#4f9dff", "#75b3ff", "#0d1420", "#2e5f9e"),
    },
    "mint": {
        "light": ("#0f9d6b", "#0b7a53", "#ffffff", "#d3f0e4"),
        "dark": ("#3ecf9a", "#63dcb0", "#0a1410", "#1c5c46"),
    },
    "rose": {
        "light": ("#d64072", "#b32c5c", "#ffffff", "#f8dde8"),
        "dark": ("#ff7aa5", "#ff9bbb", "#1a0d12", "#7c2c49"),
    },
}


def build_palette(dark: bool, accent: str = "indigo",
                  midnight: bool = False, rgb: bool = False) -> Palette:
    """Assemble a palette: cream-editorial light / warm-tinted dark base
    (Colicit-style: paper, ink, one vivid accent) + the chosen accent.
    `midnight` drops the dark base to near-black; `rgb` is midnight with an
    electric accent and the animated-rainbow flag set (nyan bar, iris)."""
    if rgb:
        return Palette(
            name="rgb", is_dark=True, rgb=True,
            bg="#050507", bg_alt="#0a0b0f", bg_raised="#101117",
            border="#1b1d26", fg="#d5d9e6", fg_dim="#747a8c",
            accent="#00e5ff", accent_hover="#5df0ff", accent_fg="#03151a",
            selection="#093d47",
            good="#39ff8c", warn="#ffd23e", bad="#ff5470",
        )
    if midnight:
        acc, hover, on_acc, sel = ACCENTS.get(accent, ACCENTS["indigo"])["dark"]
        return Palette(
            name="midnight", is_dark=True,
            bg="#050507", bg_alt="#0b0c11", bg_raised="#12131a",
            border="#1c1e28", fg="#c9cdd9", fg_dim="#767c8d",
            accent=acc, accent_hover=hover, accent_fg=on_acc, selection=sel,
            good="#4fc17c", warn="#e0b45f", bad="#e06c5f",
        )
    acc, hover, on_acc, sel = ACCENTS.get(accent, ACCENTS["indigo"])[
        "dark" if dark else "light"]
    if dark:
        return Palette(
            name="dark", is_dark=True,
            bg="#101116", bg_alt="#171922", bg_raised="#1f2230",
            border="#2b2f3d", fg="#dcdee6", fg_dim="#8e94a3",
            accent=acc, accent_hover=hover, accent_fg=on_acc, selection=sel,
            good="#4fc17c", warn="#e0b45f", bad="#e06c5f",
        )
    return Palette(
        name="light", is_dark=False,
        bg="#f6f4ee", bg_alt="#fdfcf8", bg_raised="#edeae1",
        border="#ddd8cb", fg="#191b1f", fg_dim="#706d63",
        accent=acc, accent_hover=hover, accent_fg=on_acc, selection=sel,
        good="#1f9d55", warn="#a87b18", bad="#c94f3d",
    )


DARK = build_palette(dark=True)
LIGHT = build_palette(dark=False)

_current: Palette = DARK


def _rgba(hex_color: str, alpha: int) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


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
QMainWindow {{
    background: qradialgradient(cx: 0.9, cy: 0.05, radius: 1.4,
        fx: 0.9, fy: 0.05,
        stop: 0 {_rgba(p.accent, 26 if p.is_dark else 18)},
        stop: 0.45 {p.bg}, stop: 1 {p.bg});
}}

QTabWidget {{ background: transparent; }}
QTabWidget::pane {{ border: none; border-top: 1px solid {p.border}; background: transparent; }}
QWidget#tabPage {{ background: transparent; }}
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
    margin-top: 14px; padding-top: 12px; background: {_rgba(p.bg_alt, 214)};
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
QPlainTextEdit {{ background: {_rgba(p.bg_alt, 208)}; }}
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

    MODES = ("auto", "dark", "light", "midnight", "rgb")

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
        accent = getattr(self.s, "accent", "indigo")
        if self.mode == "rgb":
            return build_palette(dark=True, accent=accent, rgb=True)
        if self.mode == "midnight":
            return build_palette(dark=True, accent=accent, midnight=True)
        if self.mode == "dark":
            return build_palette(dark=True, accent=accent)
        if self.mode == "light":
            return build_palette(dark=False, accent=accent)
        from PySide6.QtCore import Qt

        scheme = self._app.styleHints().colorScheme()
        return build_palette(dark=scheme != Qt.ColorScheme.Light, accent=accent)

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

    def set_accent(self, accent: str) -> None:
        if accent not in ACCENTS or accent == getattr(self.s, "accent", "indigo"):
            return
        self.s.accent = accent
        try:
            self.s.save()
        except OSError:
            pass
        self.apply()

    def _os_scheme_changed(self, *_args) -> None:
        if self.mode == "auto":
            self.apply()

    def apply(self) -> None:
        _apply(self._app, self._resolve())
        self.changed.emit(_current)
