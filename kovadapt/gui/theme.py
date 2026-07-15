"""Dark theme: one QSS sheet + pyqtgraph defaults. Simple, low-contrast-noise."""

from __future__ import annotations

BG = "#111318"
BG_ALT = "#181b22"
BG_RAISED = "#1f232d"
BORDER = "#2a2f3a"
FG = "#d6dae3"
FG_DIM = "#8b93a3"
ACCENT = "#4f9dff"
ACCENT_DIM = "#2e5f9e"
GOOD = "#4fc17c"
WARN = "#e0b45f"
BAD = "#e06c5f"

QSS = f"""
* {{
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
    color: {FG};
}}
QMainWindow, QDialog, QWidget {{ background: {BG}; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-top: none; }}
QTabBar::tab {{
    background: {BG_ALT}; color: {FG_DIM};
    padding: 7px 18px; border: 1px solid {BORDER}; border-bottom: none;
}}
QTabBar::tab:selected {{ background: {BG}; color: {FG}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{
    border: 1px solid {BORDER}; border-radius: 4px;
    margin-top: 12px; padding-top: 10px; background: {BG_ALT};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; color: {FG_DIM}; }}
QPushButton {{
    background: {BG_RAISED}; border: 1px solid {BORDER};
    border-radius: 4px; padding: 6px 16px;
}}
QPushButton:hover {{ border-color: {ACCENT_DIM}; }}
QPushButton:pressed {{ background: {BORDER}; }}
QPushButton:disabled {{ color: {FG_DIM}; background: {BG_ALT}; }}
QPushButton[accent="true"] {{ background: {ACCENT_DIM}; border-color: {ACCENT_DIM}; }}
QPushButton[accent="true"]:hover {{ background: {ACCENT}; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget {{
    background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 6px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {ACCENT}; }}
QListWidget::item {{ padding: 6px; border-bottom: 1px solid {BORDER}; }}
QListWidget::item:selected {{ background: {ACCENT_DIM}; }}
QCheckBox::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER}; border-radius: 3px; background: {BG_ALT};
}}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QLabel[dim="true"] {{ color: {FG_DIM}; }}
QLabel[headline="true"] {{ font-size: 16px; font-weight: 600; }}
QScrollBar:vertical {{ background: {BG}; width: 10px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QProgressBar {{
    background: {BG_ALT}; border: 1px solid {BORDER}; border-radius: 4px;
    text-align: center; color: {FG}; height: 16px;
}}
QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: 3px; }}
QStatusBar {{ background: {BG_ALT}; color: {FG_DIM}; }}
QToolTip {{ background: {BG_RAISED}; color: {FG}; border: 1px solid {BORDER}; }}
"""


def apply_theme(app) -> None:
    """Apply QSS + pyqtgraph global options. Call once at startup."""
    import pyqtgraph as pg

    app.setStyleSheet(QSS)
    pg.setConfigOptions(
        background=BG_ALT,
        foreground=FG_DIM,
        antialias=True,
        imageAxisOrder="row-major",
    )
