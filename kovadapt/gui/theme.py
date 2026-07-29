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

from . import color


@dataclass(frozen=True)
class Palette:
    name: str
    is_dark: bool
    bg: str            # window background
    bg_alt: str        # inputs, panels, plots
    bg_raised: str     # buttons, tooltips, cards
    border: str        # HAIRLINE rule between sections — may be low contrast
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
    # Control outlines are NOT rules: an input's edge is the affordance, so it
    # carries a real >= 3:1 contrast floor while a section divider does not.
    # One `border` token served both, which is why every input on cream was
    # outlined at 1.29:1.
    border_control: str = ""
    accent_name: str = "indigo"   # the preset key, so nothing reverse-maps hex


# Accent presets as OKLCH (hue degrees, chroma). Lightness is NOT fixed here:
# it is fitted per theme so the accent clears its contrast floor on that
# theme's background. Hand-picked hex could not do that, which is how mint,
# ocean and rose all shipped under 4.5:1 on cream.
ACCENTS: dict[str, tuple[float, float]] = {
    "indigo": (277.0, 0.185),    # the Colicit-vivid default
    "ocean": (248.0, 0.150),
    "mint": (162.0, 0.130),
    "rose": (8.0, 0.170),
    "ember": (55.0, 0.140),      # warm — pairs with the cream paper
}

# Contrast floors (WCAG 2.x). Text pairs 4.5:1, control edges 3:1.
TEXT_CONTRAST = 4.5
DIM_CONTRAST = 4.5
CONTROL_CONTRAST = 3.0

# (base lightness, chroma, hue) per theme — the whole palette derives from it.
_BASES = {
    "light": (0.967, 0.024, 90.0),     # warm paper; 0.024 chroma is the whole
    "dark": (0.205, 0.012, 280.0),     # difference between paper and warm grey
    "midnight": (0.115, 0.010, 280.0),
    "rgb": (0.115, 0.010, 280.0),
}


# Preferred (most vivid) lightness for a saturated hue, per mode. The fitter
# only walks away from these as far as the contrast floor demands.
_VIVID_L_DARK = 0.78
_VIVID_L_LIGHT = 0.66


def _fit_text(hue: float, chroma: float, bg: str, dark: bool,
              target: float) -> str:
    """An ink/accent color at this hue that clears `target` on `bg`."""
    return color.fit_contrast(_VIVID_L_DARK if dark else _VIVID_L_LIGHT,
                              chroma, hue,
                              against=bg, target=target, prefer_lighter=dark)


def build_palette(dark: bool, accent: str = "indigo",
                  midnight: bool = False, rgb: bool = False) -> Palette:
    """Derive a whole palette from a base (L, C, H) plus an accent preset.

    Every color is computed, not chosen, so `tests/test_theme_contrast.py`
    can walk every role pair and fail when one drops under its floor. The
    accent's lightness is fitted against this theme's background rather than
    fixed, which is what keeps a saturated preset legible on cream without
    hand-tuning it.
    """
    key = "rgb" if rgb else "midnight" if midnight else "dark" if dark else "light"
    base_l, base_c, base_h = _BASES[key]
    is_dark = key != "light"
    bg = color.oklch_to_hex(base_l, base_c, base_h)

    # Surfaces step AWAY from the page in dark themes and barely move in
    # light: near-white panels on cream is what made the editorial light mode
    # read as a generic white app, since the panels cover most of the page.
    if is_dark:
        bg_alt = color.oklch_to_hex(base_l + 0.030, base_c, base_h)
        bg_raised = color.oklch_to_hex(base_l + 0.055, base_c, base_h)
        rule = color.oklch_to_hex(base_l + 0.075, base_c, base_h)
        fg = color.oklch_to_hex(0.900, base_c * 0.5, base_h)
    else:
        bg_alt = color.oklch_to_hex(base_l + 0.012, base_c * 0.8, base_h)
        bg_raised = color.oklch_to_hex(base_l - 0.030, base_c, base_h)
        rule = color.oklch_to_hex(base_l - 0.090, base_c, base_h)
        fg = color.oklch_to_hex(0.220, base_c * 0.6, base_h)

    fg_dim = color.fit_contrast(0.62 if is_dark else 0.52, base_c * 0.7, base_h,
                                against=bg, target=DIM_CONTRAST,
                                prefer_lighter=is_dark)
    border_control = color.fit_contrast(0.50, base_c, base_h, against=bg,
                                        target=CONTROL_CONTRAST,
                                        prefer_lighter=is_dark)

    if rgb:
        acc_h, acc_c, name = 195.0, 0.150, "rgb"      # electric cyan
    else:
        acc_h, acc_c = ACCENTS.get(accent, ACCENTS["indigo"])
        name = accent if accent in ACCENTS else "indigo"
    acc = _fit_text(acc_h, acc_c, bg, is_dark, TEXT_CONTRAST)
    # Hover moves further from the page, so it never loses contrast.
    acc_l = 0.80 if is_dark else 0.38
    hover = color.fit_contrast(acc_l, acc_c, acc_h, against=bg,
                               target=TEXT_CONTRAST + 1.0,
                               prefer_lighter=is_dark)
    # Text ON the accent fill: whichever of paper/ink actually contrasts.
    ink = color.oklch_to_hex(0.16, base_c, base_h)
    paper = color.oklch_to_hex(0.98, base_c * 0.4, base_h)
    accent_fg = paper if color.contrast_ratio(paper, acc) >= \
        color.contrast_ratio(ink, acc) else ink
    selection = color.mix(bg, acc, 0.30 if is_dark else 0.18)

    good = _fit_text(150.0, 0.130, bg, is_dark, TEXT_CONTRAST)
    warn = _fit_text(85.0, 0.130, bg, is_dark, TEXT_CONTRAST)
    bad = _fit_text(27.0, 0.150, bg, is_dark, TEXT_CONTRAST)

    return Palette(
        name=key, is_dark=is_dark, rgb=rgb,
        bg=bg, bg_alt=bg_alt, bg_raised=bg_raised, border=rule,
        border_control=border_control, fg=fg, fg_dim=fg_dim,
        accent=acc, accent_hover=hover, accent_fg=accent_fg,
        selection=selection, good=good, warn=warn, bad=bad,
        accent_name=name,
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


# Only these pixel sizes put BOTH the advance width and the row pitch on
# integer pixels for Cascadia Mono (measured: 12->7/14, 14->8/16, 20->12/24,
# 24->14/28). Off-grid sizes — 13, 16, 18 — land glyph origins on half
# pixels, and Windows hinting plus ClearType then fringe each row differently,
# which is what makes a static character grid look like it is shimmering.
CELL_SIZES = (12, 14, 20, 24)

_MONO_FAMILY: str | None = None


def mono_family() -> str:
    """The app's monospace family, probed once.

    QFontMetrics answers AFTER Qt's font substitution, so it reports a happy
    non-zero advance for families that are not installed — a fallback then
    silently changes the character cell's aspect ratio and distorts every
    piece of art drawn on it. QFontDatabase.families() is the honest check.
    """
    global _MONO_FAMILY
    if _MONO_FAMILY is None:
        from PySide6.QtGui import QFontDatabase

        installed = set(QFontDatabase.families())
        for name in ("Cascadia Mono", "Cascadia Code", "Consolas",
                     "DejaVu Sans Mono", "Courier New"):
            if name in installed:
                _MONO_FAMILY = name
                break
        else:
            _MONO_FAMILY = "monospace"
    return _MONO_FAMILY


def mono(size: int = 14, *, bold: bool = False):
    """A grid-snapped monospace QFont for anything numeric or structural.

    `size` is snapped to the nearest CELL_SIZES entry rather than honoured
    exactly — a value that renders crisply matters more here than one that
    matches a spec sheet.
    """
    from PySide6.QtGui import QFont

    px = min(CELL_SIZES, key=lambda s: abs(s - size))
    f = QFont(mono_family())
    f.setPixelSize(px)
    f.setStyleHint(QFont.Monospace)
    if bold:
        f.setBold(True)
    return f


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

/* single-page shell (gui/shell.py): nav bar + stacked sections */
QFrame#navBar {{
    background: transparent; border: none;
    border-bottom: 1px solid {p.border};
}}
QPushButton[navLink="true"] {{
    background: transparent; border: none; border-radius: 0;
    padding: 8px 14px; color: {p.fg_dim}; font-weight: 600;
}}
QPushButton[navLink="true"]:hover {{ background: transparent; color: {p.fg}; }}
QPushButton[navLink="true"]:pressed {{ background: transparent; color: {p.accent}; }}
QPushButton[navLink="true"][active="true"] {{
    color: {p.fg}; border-bottom: 2px solid {p.accent};
}}
QLabel[sectionTitle="true"] {{
    /* Tracking tightens as type grows: at display sizes the gaps between
       letterforms grow with the glyphs, so positive tracking makes a
       headline look loose and unset. Small uppercase eyebrows are the only
       place it should go positive. */
    font-size: 21px; font-weight: 700; letter-spacing: -0.4px;
}}
QFrame[sectionDivider="true"] {{ background: {p.border}; border: none; }}

QGroupBox {{
    border: 1px solid {p.border}; border-radius: 8px;
    margin-top: 16px; padding-top: 14px; background: {_rgba(p.bg_alt, 214)};
}}
/* A panel's label outranks its contents: it was set in the DIM role, which
   inverted the hierarchy on every section of the app. Small uppercase with
   positive tracking is the one place tracking should open up. */
QGroupBox::title {{
    subcontrol-origin: margin; left: 12px; padding: 0 4px;
    color: {p.fg}; font-size: 11px; font-weight: 700;
    letter-spacing: 0.8px; text-transform: uppercase;
}}

QPushButton {{
    background: {p.bg_raised}; border: 1px solid {p.border_control};
    border-radius: 6px; padding: 7px 18px;
}}
QPushButton:hover {{ border-color: {p.accent}; }}
QPushButton:pressed {{ background: {p.selection}; }}
QPushButton:disabled {{ color: {p.fg_dim}; background: {p.bg_alt}; }}
/* Checkable buttons (the overlay toggle) had NO checked state at all, so
   "on" and "off" were pixel-identical and the only way to know was to look
   at the overlay itself. */
QPushButton:checked {{
    background: {p.selection}; border-color: {p.accent}; color: {p.fg};
    font-weight: 600;
}}
QPushButton:checked:hover {{ border-color: {p.accent_hover}; }}
QPushButton[accent="true"] {{
    background: {p.accent}; border-color: {p.accent}; color: {p.accent_fg};
    font-weight: 600;
}}
QPushButton[accent="true"]:hover {{ background: {p.accent_hover}; border-color: {p.accent_hover}; }}
QPushButton[accent="true"]:disabled {{ background: {p.bg_raised}; color: {p.fg_dim}; }}
QPushButton[flat="true"] {{ background: transparent; border: none; padding: 4px 8px; }}
QPushButton[flat="true"]:hover {{ color: {p.accent}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {{
    background: {p.bg_alt}; border: 1px solid {p.border_control}; border-radius: 6px;
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
    background: {p.bg_raised}; border: 1px solid {p.border_control}; border-radius: 6px;
    selection-background-color: {p.selection}; outline: none;
}}

QListWidget::item {{ padding: 6px; border-bottom: 1px solid {p.border}; }}
QListWidget::item:selected {{ background: {p.selection}; color: {p.fg}; }}

/* Spin buttons: unstyled, these render as native Fusion arrows on 25+
   controls in Adaptability — the one place the app looked like a stock
   Qt dialog dropped onto the page. */
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background: transparent; border: none; width: 18px;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none; width: 0; height: 0; margin-right: 6px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid {p.fg_dim};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none; width: 0; height: 0; margin-right: 6px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {p.fg_dim};
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    border-bottom-color: {p.accent};
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    border-top-color: {p.accent};
}}

/* Tables (the Scenarios browser) carried NO rules at all, so Qt fell back
   to the native Fusion header and the Windows system highlight — a blue
   selection bar on warm paper, the single loudest wrong pixel in the app.
   Rows are data, so they read in the mono face and align on their digits. */
QTableView, QTableWidget {{
    background: {_rgba(p.bg_alt, 200)}; alternate-background-color: transparent;
    border: 1px solid {p.border}; border-radius: 6px;
    gridline-color: transparent; outline: none;
    selection-background-color: {p.selection}; selection-color: {p.fg};
}}
QTableView::item, QTableWidget::item {{
    padding: 7px 10px; border: none;
    border-bottom: 1px solid {p.border};
}}
QTableView::item:selected, QTableWidget::item:selected {{
    background: {p.selection}; color: {p.fg};
}}
QTableView::item:hover, QTableWidget::item:hover {{ background: {_rgba(p.fg, 12)}; }}
QHeaderView {{ background: transparent; border: none; }}
QHeaderView::section {{
    background: transparent; color: {p.fg_dim};
    padding: 6px 10px; border: none;
    border-bottom: 1px solid {p.border_control};
    font-size: 11px; font-weight: 700; letter-spacing: 0.7px;
    text-transform: uppercase;
}}
QHeaderView::section:hover {{ color: {p.fg}; }}
QTableCornerButton::section {{ background: transparent; border: none; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 16px; height: 16px; border: 1px solid {p.border_control};
    border-radius: 4px; background: {p.bg_alt};
}}
QCheckBox::indicator:hover {{ border-color: {p.accent}; }}
QCheckBox::indicator:checked {{ background: {p.accent}; border-color: {p.accent}; }}

QLabel {{ background: transparent; }}
QLabel[dim="true"] {{ color: {p.fg_dim}; }}
QLabel[headline="true"] {{ font-size: 17px; font-weight: 600; }}
QLabel[stat="true"] {{ color: {p.accent}; font-weight: 600; }}

/* Tracks, handles and chunks are CONTROLS, not rules: drawn in the hairline
   `border` role they sat at ~1.2:1 and effectively vanished. */
QSlider::groove:horizontal {{
    height: 4px; background: {p.border_control}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px; margin: -5px 0;
    background: {p.accent}; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {p.accent}; border-radius: 2px; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {p.border_control}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {p.fg_dim}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {p.border_control}; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{
    background: {p.bg_alt}; border: 1px solid {p.border_control}; border-radius: 6px;
    text-align: center; color: {p.fg}; height: 16px;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 5px; }}

QSplitter::handle {{ background: {p.border}; }}
QSplitter::handle:horizontal {{ width: 5px; }}
QSplitter::handle:vertical {{ height: 5px; }}
QSplitter::handle:hover {{ background: {p.accent}; }}

QMenu {{
    background: {p.bg_raised}; border: 1px solid {p.border_control}; border-radius: 6px;
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
