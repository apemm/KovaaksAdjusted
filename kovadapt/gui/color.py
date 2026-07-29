"""OKLCH color math for the theme system — pure functions, no Qt.

Palettes are *derived*, not hand-picked. Four themes used to be 56 literal
hex values that nobody could check; two accents shipped below the WCAG 4.5:1
floor on the primary button because there was no way to notice. Here a
palette is a handful of parameters and every color falls out of them, so
`tests/test_theme_contrast.py` can walk every role pair and fail the build
when one drops under its floor.

OKLCH is the working space because it is perceptually uniform: rotating hue
at fixed L and C keeps every step the same apparent brightness. HSV does
not — `QColor.fromHsvF` swings luminance about 4:1 between yellow and blue,
which is exactly why the rainbow iris reads lopsided and the cat bar pulses.

Kept deliberately dependency-free (numpy only) so it imports on any OS and
runs in the test suite without a display.
"""

from __future__ import annotations

import math

# OKLab <-> linear sRGB (Björn Ottosson, https://bottosson.github.io/posts/oklab/)
_LMS_FROM_OKLAB = (
    (1.0, +0.3963377774, +0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_RGB_FROM_LMS = (
    (+4.0767416621, -3.3077115913, +0.2309699292),
    (-1.2684380046, +2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, +1.7076147010),
)


def _srgb_encode(x: float) -> float:
    """Linear light -> sRGB, the standard piecewise transfer function."""
    if x <= 0.0031308:
        return 12.92 * x
    return 1.055 * (x ** (1.0 / 2.4)) - 0.055


def _srgb_decode(x: float) -> float:
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


def oklch_to_linear(lightness: float, chroma: float, hue_deg: float):
    """OKLCH -> linear sRGB, unclamped (may fall outside the gamut)."""
    h = math.radians(hue_deg)
    a = chroma * math.cos(h)
    b = chroma * math.sin(h)
    lms = []
    for c0, c1, c2 in _LMS_FROM_OKLAB:
        lms.append((c0 * lightness + c1 * a + c2 * b) ** 3)
    return tuple(sum(m * v for m, v in zip(row, lms)) for row in _RGB_FROM_LMS)


def in_gamut(rgb, eps: float = 1e-4) -> bool:
    return all(-eps <= c <= 1.0 + eps for c in rgb)


def oklch_to_hex(lightness: float, chroma: float, hue_deg: float) -> str:
    """OKLCH -> '#rrggbb', reducing chroma until the color fits in sRGB.

    Clipping channels instead would shift hue AND lightness; walking chroma
    down keeps both, which is what makes a generated ramp stay coherent at
    the edges of the gamut.
    """
    lo, hi = 0.0, max(chroma, 0.0)
    rgb = oklch_to_linear(lightness, hi, hue_deg)
    if not in_gamut(rgb):
        for _ in range(24):                     # bisect to ~1e-7 chroma
            mid = 0.5 * (lo + hi)
            if in_gamut(oklch_to_linear(lightness, mid, hue_deg)):
                lo = mid
            else:
                hi = mid
        rgb = oklch_to_linear(lightness, lo, hue_deg)
    out = []
    for c in rgb:
        v = _srgb_encode(min(max(c, 0.0), 1.0))
        out.append(max(0, min(255, int(round(v * 255.0)))))
    return "#{:02x}{:02x}{:02x}".format(*out)


def hex_to_rgb(value: str) -> tuple[float, float, float]:
    v = value.lstrip("#")
    if len(v) == 3:
        v = "".join(ch * 2 for ch in v)
    return tuple(int(v[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def relative_luminance(value: str) -> float:
    """WCAG 2.x relative luminance of a '#rrggbb' color."""
    r, g, b = (_srgb_decode(c) for c in hex_to_rgb(value))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG contrast ratio, 1.0 (identical) .. 21.0 (black on white)."""
    a, b = relative_luminance(fg), relative_luminance(bg)
    lo, hi = min(a, b), max(a, b)
    return (hi + 0.05) / (lo + 0.05)


def fit_contrast(lightness: float, chroma: float, hue_deg: float, *,
                 against: str, target: float,
                 prefer_lighter: bool | None = None) -> str:
    """The color CLOSEST to `lightness` that still clears `target` on `against`.

    Hue and chroma are held — the accent stays recognisably itself; only
    lightness moves, which is the axis contrast actually depends on.

    It bisects to the contrast boundary rather than returning the first
    passing step, and that distinction is the whole point: a linear walk
    overshoots, and an accent dragged further from the page than the floor
    requires is a duller accent for no accessibility gain. `lightness` is
    read as the preferred, most-vivid value; the search gives away as
    little of it as the target allows.

    Returns the best achievable color when the target is unreachable in
    gamut (black or white on mid grey cannot reach 4.5:1 in either
    direction).
    """
    if prefer_lighter is None:
        prefer_lighter = relative_luminance(against) < 0.18
    preferred = oklch_to_hex(lightness, chroma, hue_deg)
    if contrast_ratio(preferred, against) >= target:
        return preferred
    # extreme end is guaranteed to be the highest contrast available
    far = 1.0 if prefer_lighter else 0.0
    if contrast_ratio(oklch_to_hex(far, chroma, hue_deg), against) < target:
        return oklch_to_hex(far, chroma, hue_deg)      # unreachable: best effort
    lo, hi = lightness, far                            # lo fails, hi passes
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if contrast_ratio(oklch_to_hex(mid, chroma, hue_deg), against) >= target:
            hi = mid
        else:
            lo = mid
    return oklch_to_hex(hi, chroma, hue_deg)


def mix(a: str, b: str, t: float) -> str:
    """Blend two hex colors in linear light (t=0 -> a, t=1 -> b)."""
    ra, ga, ba = (_srgb_decode(c) for c in hex_to_rgb(a))
    rb, gb, bb = (_srgb_decode(c) for c in hex_to_rgb(b))
    out = []
    for lo, hi in ((ra, rb), (ga, gb), (ba, bb)):
        v = _srgb_encode(lo + (hi - lo) * t)
        out.append(max(0, min(255, int(round(v * 255.0)))))
    return "#{:02x}{:02x}{:02x}".format(*out)


def rainbow_hex(t: float, *, lightness: float = 0.72,
                chroma: float = 0.12) -> str:
    """A rainbow stop at fraction t, at CONSTANT perceptual lightness.

    Used by the iris and the progress cat. HSV rainbows are ~4:1 brighter at
    yellow than at blue, which makes a rotating iris look lopsided and a
    scrolling bar look like it is pulsing; holding L and C fixed makes the
    hue the only thing that changes.
    """
    return oklch_to_hex(lightness, chroma, (t % 1.0) * 360.0)
