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
    # The primary button's PRESSED fill. It needs its own token because the
    # generic `QPushButton:pressed` rule cannot reach an accent button: an
    # attribute selector and a pseudo-state carry equal CSS2 specificity, so
    # the later `QPushButton[accent="true"]` declaration wins and the fill
    # never changed. Measured: byte-identical up and down, all 20 theme x
    # accent combos.
    accent_press: str = ""


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
# A selected row must separate from the surface it sits on WITHOUT pushing the
# text on top of it under the 4.5:1 floor — at 3.0 the fit drives `fg` on
# selection down to 4.08:1 (measured, every theme x accent). See `selection`
# in build_palette.
SELECTION_CONTRAST = 2.6

# The body size the app-wide sheet sets. Named because align_baselines
# has to reproduce it: `widget.font()` cannot be trusted for it (that
# sheet outranks setFont), so the value has to exist in one place.
BODY_PX = 13

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
    # Pressed is a proportional darkening of the ACCENT AS FITTED, and that
    # is the whole subtlety: `acc_l` above is only the PREFERRED lightness,
    # and _fit_text walks away from it as far as the contrast floor demands.
    # Deriving the press from acc_l instead produced a 2.4-luminance step on
    # mint, whose accent is fitted all the way down to #008057 — invisible.
    # Scaling the fitted colour cannot drift from it. It also always darkens,
    # on light and dark alike, so a press reads as a press rather than as
    # more hover: on dark, accent -> accent_hover is only 7-13 per channel.
    press = "#{:02x}{:02x}{:02x}".format(
        *(int(round(int(acc[i:i + 2], 16) * 0.78)) for i in (1, 3, 5)))
    # Text ON the accent fill: whichever of paper/ink actually contrasts.
    ink = color.oklch_to_hex(0.16, base_c, base_h)
    paper = color.oklch_to_hex(0.98, base_c * 0.4, base_h)
    accent_fg = paper if color.contrast_ratio(paper, acc) >= \
        color.contrast_ratio(ink, acc) else ink
    # Selection is DERIVED IN OKLCH at the accent's hue, and then FITTED
    # against the surface it is actually painted on.
    #
    # It used to be mix(bg, accent) in linear light, which on warm cream
    # landed at #e8e2e1 — a cold neutral that read as the Windows system
    # highlight the table rules were written to kill. Moving to a stated
    # hue fixed that, but a stated LIGHTNESS then broke the other end: a
    # fixed +0.095 offset put dark's selection 1.21:1 from bg_alt where the
    # old mix had 3.11:1, so a selected row stopped being distinguishable
    # from an unselected one on three of the four themes. Getting the hue
    # right is not worth losing the separation.
    #
    # Fitting solves both at once, and against `bg_alt` rather than `bg`
    # because that is the surface selections are drawn on — table rows, list
    # items, the combo popup. SELECTION_CONTRAST is 2.6 rather than the 3.0
    # control floor for a measured reason: at 3.0 the fit drives the
    # selection far enough from the page that `fg` on top of it falls to
    # 4.08:1, under the 4.5:1 text floor. 2.6 holds text at 4.75:1 worst
    # case across every theme x accent while keeping the highlight obvious.
    sel_l = base_l + (0.095 if is_dark else -0.055)
    selection = color.fit_contrast(
        sel_l, 0.055 if is_dark else 0.045, acc_h,
        against=bg_alt, target=SELECTION_CONTRAST, prefer_lighter=is_dark)

    good = _fit_text(150.0, 0.130, bg, is_dark, TEXT_CONTRAST)
    warn = _fit_text(85.0, 0.130, bg, is_dark, TEXT_CONTRAST)
    bad = _fit_text(27.0, 0.150, bg, is_dark, TEXT_CONTRAST)

    return Palette(
        name=key, is_dark=is_dark, rgb=rgb,
        bg=bg, bg_alt=bg_alt, bg_raised=bg_raised, border=rule,
        border_control=border_control, fg=fg, fg_dim=fg_dim,
        accent=acc, accent_hover=hover, accent_fg=accent_fg,
        selection=selection, good=good, warn=warn, bad=bad,
        accent_name=name, accent_press=press,
    )


DARK = build_palette(dark=True)
LIGHT = build_palette(dark=False)

_current: Palette = DARK


def _rgba(hex_color: str, alpha: int) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {alpha})"


# Weight of the accent in the window's corner glow. It reads as an alpha and
# it used to BE one; it is now a mix ratio, and _glow() is why.
_GLOW_T = {True: 26 / 255.0, False: 18 / 255.0}     # keyed on is_dark


def _png(width: int, height: int, rows: list[bytes]) -> bytes:
    """A minimal RGBA PNG. Deliberately hand-rolled rather than QPixmap:
    build_qss() must stay callable with no QGuiApplication (the contrast
    tests do exactly that), and this needs no Qt at all."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + r for r in rows)      # filter byte 0 per scanline

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6,
                                         0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9))
            + chunk(b"IEND", b""))


_ARROWS: dict[tuple[str, bool], str] = {}
_ARROW_W, _ARROW_H, _ARROW_SS = 11, 6, 4


def _arrow_url(hex_color: str, *, up: bool) -> str:
    """Path to a caret image, as a QSS url().

    QT STYLE SHEETS CANNOT DRAW A TRIANGLE. The CSS trick — zero width and
    height, transparent left/right borders, a solid border on one side — is
    a browser idiom that Qt parses and gets wrong: it drew a SOLID
    RECTANGLE in fg_dim. Every combo box and all 25+ spin controls in
    Adaptability were rendering a filled block where their caret should be,
    including the spin rule whose own comment said it existed to stop the
    app looking like a stock Qt dialog. Removing the rule is not a fix
    either — once a widget is styled, Qt draws no arrow at all.

    So the caret is a real image, generated once per colour and cached on
    disk under the system temp dir (never the user's data directory, which
    is why this can run in tests). It is supersampled and box-filtered
    because an 11x6 hard-edged triangle is visibly jagged.
    """
    import tempfile
    from pathlib import Path

    key = (hex_color, up)
    cached = _ARROWS.get(key)
    if cached is not None:
        return cached

    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    w, h, ss = _ARROW_W, _ARROW_H, _ARROW_SS
    rows = []
    for y in range(h):
        row = bytearray()
        for x in range(w):
            hits = 0
            for sy in range(ss):
                yy = (y * ss + sy + 0.5) / (h * ss)          # 0..1 down
                v = yy if not up else 1.0 - yy
                # half-width of the triangle at this height, in cells
                half = (1.0 - v) * (w / 2.0)
                for sx in range(ss):
                    xx = (x * ss + sx + 0.5) / (w * ss) * w - w / 2.0
                    if abs(xx) <= half:
                        hits += 1
            a = int(round(255 * hits / (ss * ss)))
            row += bytes((r, g, b, a))
        rows.append(bytes(row))

    # The whole filesystem interaction is inside the guard, not just the
    # write. gettempdir() consults TMP/TEMP and raises if the environment
    # points at something unusable, and mkdir() raises on a read-only or
    # full volume — both sat OUTSIDE the try, so on a locked-down machine
    # build_qss() would raise and take the app's startup with it, for a
    # decoration. A missing caret is a cosmetic loss; a crash is not.
    data = _png(w, h, rows)
    try:
        path = Path(tempfile.gettempdir()) / "kovadapt-ui"
        path.mkdir(parents=True, exist_ok=True)
        f = path / f"caret-{'up' if up else 'dn'}-{hex_color.lstrip('#')}.png"
        if not f.is_file() or f.read_bytes() != data:
            f.write_bytes(data)
    except (OSError, ValueError):
        _ARROWS[key] = ""    # cache the failure: retrying per rule is pointless
        return ""
    url = str(f).replace("\\", "/")
    _ARROWS[key] = url
    return url


def _glow(p: Palette) -> str:
    """The corner glow, ALREADY COMPOSITED onto the page — never a
    translucent gradient stop.

    A QMainWindow's background brush is the bottom of the paint stack: there
    is nothing behind it to blend into, so Qt fills a translucent stop from
    the cleared backing store, which is black. The 7% accent stop this
    replaces therefore drew a near-black blob in the top-right corner at
    every theme. On dark it landed within a few luminance points of the page
    and went unseen for two versions; on cream it measured luminance 20
    against the page's 244 — a charcoal box, which is how it was finally
    caught, by looking. Compositing here means the stop states its result
    instead of asking the compositor for one it cannot supply.

    The blend is a straight sRGB lerp and deliberately NOT `color.mix`,
    which interpolates in linear light. Linear light is the better model of
    how light adds, and it is wrong for this one job: the number being
    replaced is a Qt ALPHA, and Qt composites 8-bit sRGB. Mixing in linear
    light honoured the ratio but not the operation — it favours the brighter
    colour, and a 10% accent came out at luminance 76 on a page of 23, four
    times the glow the alpha asked for. Reproducing SourceOver keeps the
    dark themes looking exactly as they always have, so this fix changes
    only the theme that was broken.
    """
    t = _GLOW_T[p.is_dark]
    top = (int(p.accent[i:i + 2], 16) for i in (1, 3, 5))
    bot = [int(p.bg[i:i + 2], 16) for i in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(
        *(max(0, min(255, round(b + (a - b) * t))) for a, b in zip(top, bot)))


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


def body_font():
    """The font the app-wide sheet actually gives ordinary widgets."""
    from PySide6.QtGui import QFont

    f = QFont("Segoe UI")
    f.setPixelSize(BODY_PX)
    return f


def align_baselines(small_widget, big_font, small_font) -> None:
    """Sit `small_widget`'s text on the same baseline as `big_font`'s.

    A layout's Qt.AlignBottom aligns bottom EDGES, not baselines, and a
    label's bottom edge sits one DESCENT below its baseline. Pair a 48px
    hero numeral with a 13px state word and the descents differ by 8px, so
    the word hangs 8px below the number it qualifies; a 24px KPI value beside
    its 12px unit differs by 3px. Qt's layouts do not implement baseline
    alignment, so the correction is a bottom content margin.

    THE FONTS ARE PASSED IN, not read off the widgets, and that is the whole
    subtlety. theme.py's app-wide `* { font-size: 13px }` outranks
    setFont(), so `widget.font()` reports 13px for BOTH labels and the drop
    computes to zero — which is exactly what my first version did, silently
    and only on the tiles whose stylesheet had not resolved yet. The caller
    knows which faces it means; ask it.
    """
    from PySide6.QtGui import QFontMetricsF

    drop = round(QFontMetricsF(big_font).descent()
                 - QFontMetricsF(small_font).descent())
    m = small_widget.contentsMargins()
    small_widget.setContentsMargins(m.left(), m.top(), m.right(), max(drop, 0))


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
    font-size: {BODY_PX}px;
    color: {p.fg};
}}
QMainWindow, QDialog, QWidget {{ background: {p.bg}; }}
QMainWindow {{
    background: qradialgradient(cx: 0.9, cy: 0.05, radius: 1.4,
        fx: 0.9, fy: 0.05,
        stop: 0 {_glow(p)},
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
    /* The title lives in this margin (subcontrol-origin: margin), so the
       margin IS the clearance between the label and the box's top border.
       At 16px an 11px label sat right on the line. */
    /* padding-left/right match the 14px every card and panel in the app
       uses for its own contents (dashboard.py, analysis_view.py,
       overlay.py). Only padding-top was stated, so the sides fell back to
       Qt's default and group-box contents sat ~4px closer to the border
       than identical content in a card. */
    margin-top: 26px; padding: 12px 14px 12px 14px;
    background: {_rgba(p.bg_alt, 214)};
}}
/* A panel's label outranks its contents: it was set in the DIM role, which
   inverted the hierarchy on every section of the app. Small and tracked is
   the one place letter-spacing should open up rather than tighten.
   NOTE: no text-transform here — Qt Style Sheets do not implement it, so the
   rule silently did nothing; titles are uppercased at the call site instead. */
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px 8px 6px;
    color: {p.fg}; font-size: 11px; font-weight: 700;
    letter-spacing: 0.8px;
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
/* The generic QPushButton:pressed rule above CANNOT reach this button: an
   attribute selector and a pseudo-state carry equal CSS2 specificity, so the
   later QPushButton[accent="true"] declaration wins and the accent fill was
   never replaced. Measured byte-identical up and down across all 20 theme x
   accent combos — the app's primary call to action, which writes a playlist
   and launches Steam, acknowledged a click with nothing at all. */
QPushButton[accent="true"]:pressed {{
    background: {p.accent_press}; border-color: {p.accent_press};
}}
/* A dead button has to look dead. This set background and color but not
   border-color, so the accent ring from the rule above SURVIVED — measured
   byte-identical to the enabled ring in 20/20 theme x accent combos. On
   midnight and rgb that ring reads ~10:1 against its own fill while its
   greyed label reads 5.26:1, so the disabled primary was the most saturated
   thing in an otherwise greyed row. Its fill was bg_raised too — the fill an
   ENABLED plain button gets — so it read as a live control wearing the
   primary marker. bg_alt + border_control is exactly what the disabled plain
   button beside it already uses. */
QPushButton[accent="true"]:disabled {{
    background: {p.bg_alt}; border-color: {p.border_control}; color: {p.fg_dim};
}}
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
    image: url({_arrow_url(p.fg_dim, up=False)});
    width: {_ARROW_W}px; height: {_ARROW_H}px; margin-right: 6px;
}}
QComboBox::down-arrow:hover, QComboBox::down-arrow:on {{
    image: url({_arrow_url(p.accent, up=False)});
}}
/* bg_alt, matching the CLOSED box above: the popup is that control opening,
   not a menu arriving from somewhere else. On bg_raised the plate landed
   14.0 lum away from the box it dropped out of on light, 5.1 on dark.

   NOTE this rule does not decide what you see. Editing it changes nothing on
   screen — verified by mutation. The list is inside a
   QComboBoxPrivateContainer, and the instance stylesheet install_popup_style
   puts on that container reaches the list through palette inheritance and
   wins. The colour is stated here because this is where a reader looks for
   it; it is PAINTED there. Change both or neither. */
QComboBox QAbstractItemView {{
    background: {p.bg_alt}; border: 1px solid {p.border_control}; border-radius: 6px;
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
    image: url({_arrow_url(p.fg_dim, up=True)});
    width: {_ARROW_W}px; height: {_ARROW_H}px; margin-right: 6px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url({_arrow_url(p.fg_dim, up=False)});
    width: {_ARROW_W}px; height: {_ARROW_H}px; margin-right: 6px;
}}
QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{
    image: url({_arrow_url(p.accent, up=True)});
}}
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    image: url({_arrow_url(p.accent, up=False)});
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

/* Transparent, all three, for one reason: `QMainWindow, QDialog, QWidget`
   above sets the PAGE colour, and a Qt type selector matches subclasses — so
   every one of these, sitting inside a group box, painted the page opaquely
   ON TOP of that box's plate. A checkbox came out as a 350x18 page-coloured
   stripe across a panel; a scroll area's viewport as a full rectangle of it.
   QLabel was already exempted here for exactly this, and shell.py works
   around the same thing with objectName("tabPage"); this is that fix applied
   to the rest of what it bites.

   The viewport needs the child-of-child selector: `QScrollArea` alone styles
   the frame and leaves the viewport painting page colour underneath it. */
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: transparent; }}
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


_popup_style_installed = False


def install_popup_style(app) -> None:
    """Take the native menu chrome off combo-box popups. Idempotent.

    QSS styles `QComboBox QAbstractItemView`, which is the LIST. The list
    lives inside a `QComboBoxPrivateContainer`, and no global selector can
    name that class — which is what the audit meant by "a QSS rule on the
    container does nothing". Fusion paints its own PE_PanelMenu under it:
    measured on cream, a SQUARE #969288 outer line around a 6px-rounded
    list, plus a 5px #fffaeb band above and below, wrapping every drop-down
    in the app. The container also painted bg_raised while the box the popup
    drops out of is bg_alt — 14.0 luminance apart on light, 5.1 on dark.

    A global selector cannot reach the container. An INSTANCE stylesheet can,
    and it is the entire fix: `border: none` takes the panel off, and the
    plate colour makes the inset it holds the list in disappear into the
    list.

    Two other approaches are deliberately absent, both tried and both
    measured byte-identical with and without on every theme:

    * a QProxyStyle no-opping PE_PanelMenu and zeroing PM_MenuVMargin. Once
      a stylesheet is set Qt inserts QStyleSheetStyle above any proxy and
      answers the metric itself without delegating down — a spy on the proxy
      records PM_MenuVMargin never being asked for at all. It also replaces
      the application's style, which is a large side effect for no pixels.
    * zeroing the container's layout margins on Show. The 10px inset stays
      either way; it just no longer shows, because it is the plate's colour.
    """
    global _popup_style_installed
    if _popup_style_installed and getattr(app, "_kovadapt_popup_margins", None):
        return
    from PySide6.QtCore import QEvent, QObject

    class _PopupPlate(QObject):
        """Give the combo container the plate's own colour and no border.

        The `_kovadapt_styled` mark is load-bearing: setStyleSheet posts a
        StyleChange, which re-enters this filter, which sets the stylesheet.
        Without the guard that recursion runs until the interpreter dies with
        no traceback at all.
        """

        def eventFilter(self, obj, event):
            if (event.type() in (QEvent.Show, QEvent.StyleChange)
                    and obj.metaObject().className() == "QComboBoxPrivateContainer"
                    and not obj.property("_kovadapt_styled")):
                obj.setProperty("_kovadapt_styled", True)
                # SCOPED to the container's own class: a bare
                # "background: ..." here cascades onto the list inside it.
                obj.setStyleSheet("QComboBoxPrivateContainer { background: %s; "
                                  "border: none; }" % current().bg_alt)
            return False

    # Take the previous one off first. An app-wide event filter sees EVERY
    # event for every object, so a second copy doubles that cost and an Nth
    # multiplies it: a test that re-installed per case stacked twenty of them
    # and put four minutes on the suite, all of it in the tests that ran
    # after it.
    old_filter = getattr(app, "_kovadapt_popup_margins", None)
    if old_filter is not None:
        app.removeEventFilter(old_filter)
    app._kovadapt_popup_margins = _PopupPlate(app)   # keep it alive
    app.installEventFilter(app._kovadapt_popup_margins)
    _popup_style_installed = True


def _apply(app, pal: Palette) -> None:
    global _current
    _current = pal
    install_popup_style(app)
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
