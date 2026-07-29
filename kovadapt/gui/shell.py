"""Single-page shell: one scrollable space of sections instead of tabs.

`PageSpace` is a vertical QScrollArea whose content is the five page widgets
stacked as full-width sections, each under a big title + hairline divider, so
the whole app reads as one continuous surface over the parallax backdrop.
`NavBar` is the slim bar above it: flat link buttons that smooth-scroll to
their section, the active section highlighted, the window's corner controls
docked on the right.

Everything here stays background-transparent via the `tabPage` objectName
QSS contract in theme.py — the backdrop must show between and through
sections, so no wrapper may paint an opaque background.
"""

from __future__ import annotations

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QObject,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_SCROLL_MS = 350


class WheelGuard(QObject):
    """Everything is clickable first: inner scrollables (plots, logs, tables,
    combos, spinboxes, nested scroll areas) only consume the wheel AFTER
    being clicked — until then the wheel always scrolls the page, so hovering
    a panel never traps the scroll."""

    def __init__(self, space: "PageSpace") -> None:
        super().__init__(space)
        self._space = space

    def eventFilter(self, obj, ev) -> bool:
        if ev.type() != QEvent.Wheel:
            return False
        w = obj if hasattr(obj, "hasFocus") else None
        if w is None:
            return False
        owner = w
        if not owner.hasFocus() and hasattr(owner, "parentWidget"):
            parent = owner.parentWidget()      # viewports focus their area
            if parent is not None and parent.hasFocus():
                owner = parent
        if owner.hasFocus():
            return False                       # clicked-in: let it scroll
        self._space.wheelEvent(ev)             # hand the wheel to the page
        return True

    def guard(self, root: QWidget) -> None:
        for w in root.findChildren(QWidget):
            if isinstance(w, (QAbstractScrollArea, QComboBox, QAbstractSpinBox)):
                w.setFocusPolicy(Qt.ClickFocus)
                w.installEventFilter(self)
                if isinstance(w, QAbstractScrollArea):
                    w.viewport().installEventFilter(self)


class _Section(QWidget):
    """One full-width section: header row (title + divider) over the page."""

    def __init__(self, title: str, page: QWidget, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("tabPage")       # transparent per theme.py contract
        self.page = page
        page.setObjectName("tabPage")

        head = QLabel(title)
        head.setProperty("sectionTitle", True)
        divider = QFrame()
        divider.setProperty("sectionDivider", True)
        divider.setFixedHeight(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 28, 22, 40)
        lay.setSpacing(14)
        lay.addWidget(head)
        lay.addWidget(divider)
        lay.addWidget(page, 1)


class PageSpace(QScrollArea):
    """The scrollable page-space. Sections are added top to bottom; every
    section is kept at least one viewport tall so each reads as a page and
    the last one can still reach the top of the view."""

    current_changed = Signal(int)   # nearest-section index under viewport top

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("tabPage")
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.viewport().setObjectName("tabPage")

        self._content = QWidget()
        self._content.setObjectName("tabPage")
        self._col = QVBoxLayout(self._content)
        self._col.setContentsMargins(0, 0, 0, 0)
        self._col.setSpacing(0)
        self.setWidget(self._content)

        self._sections: list[_Section] = []
        self._names: list[str] = []
        self._current = 0

        bar = self.verticalScrollBar()
        bar.setSingleStep(24)               # fine pixel steps on keys
        bar.valueChanged.connect(self._track)
        self._anim = QPropertyAnimation(bar, b"value", self)
        self._anim.setDuration(_SCROLL_MS)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        # kinetic wheel: ticks accumulate into one short eased glide
        self._wheel_anim = QPropertyAnimation(bar, b"value", self)
        self._wheel_anim.setDuration(220)
        self._wheel_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._wheel_target: int | None = None

    def wheelEvent(self, event) -> None:
        """Animated wheel scrolling: successive ticks glide, never step."""
        delta = event.angleDelta().y()
        if delta == 0 or not self.isVisible():
            super().wheelEvent(event)
            return
        bar = self.verticalScrollBar()
        running = self._wheel_anim.state() == QPropertyAnimation.Running
        base = self._wheel_target if running and self._wheel_target is not None \
            else bar.value()
        self._wheel_target = int(max(0, min(bar.maximum(), base - delta * 1.1)))
        self._anim.stop()
        self._wheel_anim.stop()
        self._wheel_anim.setStartValue(bar.value())
        self._wheel_anim.setEndValue(self._wheel_target)
        self._wheel_anim.start()
        event.accept()

    # ------------------------------------------------------------ sections
    def add_section(self, name: str, page: QWidget) -> QWidget:
        sec = _Section(name, page)
        self._col.addWidget(sec)
        self._sections.append(sec)
        self._names.append(name)
        return sec

    def count(self) -> int:
        return len(self._sections)

    def names(self) -> list[str]:
        return list(self._names)

    def section_at(self, index: int) -> QWidget:
        return self._sections[index]

    def index_of(self, page: QWidget) -> int:
        for i, sec in enumerate(self._sections):
            if sec.page is page:
                return i
        return -1

    def current_index(self) -> int:
        return self._current

    # ----------------------------------------------------------- scrolling
    def scroll_to(self, index: int, animated: bool = True) -> None:
        """Smooth-scroll a section's header to the top of the viewport."""
        if not 0 <= index < len(self._sections):
            return
        bar = self.verticalScrollBar()
        target = min(self._sections[index].y(), bar.maximum())
        self._anim.stop()
        if not animated or not self.isVisible():
            bar.setValue(target)            # offscreen/hidden: land instantly
            return
        self._anim.setStartValue(bar.value())
        self._anim.setEndValue(target)
        self._anim.start()

    def _track(self, value: int) -> None:
        if not self._sections:
            return
        idx = min(range(len(self._sections)),
                  key=lambda i: abs(self._sections[i].y() - value))
        if idx != self._current:
            self._current = idx
            self.current_changed.emit(idx)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        h = self.viewport().height()
        for sec in self._sections:
            sec.setMinimumHeight(h)


class NavBar(QFrame):
    """Slim top bar: one flat link per section (click smooth-scrolls there,
    the section under the viewport top stays highlighted) + the corner
    controls (theme/accent pickers, help menu) docked on the right."""

    def __init__(self, space: PageSpace, corner: QWidget | None = None,
                 parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("navBar")
        self.space = space
        self._links: list[QPushButton] = []
        self._names: list[str] = space.names()

        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 0, 6, 0)
        lay.setSpacing(2)
        for i, name in enumerate(self._names):
            btn = QPushButton(name)
            btn.setProperty("navLink", True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(lambda _c=False, i=i: space.scroll_to(i))
            lay.addWidget(btn)
            self._links.append(btn)
        lay.addStretch(1)
        if corner is not None:
            lay.addWidget(corner)

        space.current_changed.connect(self._set_active)
        self._set_active(space.current_index())

    def links(self) -> list[QPushButton]:
        return list(self._links)

    def set_badge(self, index: int, on: bool) -> None:
        """Unread dot on a nav link (new report landed in Analysis)."""
        if 0 <= index < len(self._links):
            name = self._names[index]
            self._links[index].setText(f"{name} •" if on else name)

    def _set_active(self, index: int) -> None:
        for i, btn in enumerate(self._links):
            active = i == index
            if btn.property("active") == active:
                continue
            btn.setProperty("active", active)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            btn.update()

    def restyle(self, *_pal) -> None:
        """Re-assert the active highlight after a theme swap repolishes."""
        self._set_active(self.space.current_index())
