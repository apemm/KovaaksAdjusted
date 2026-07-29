"""Movement analysis: flick segmentation and aggregate metrics.

All computation is vectorized numpy over uniform-grid velocities; a full
run's trace (~10^5 packets) analyzes in a few milliseconds.

Model: aimed movements are click-anchored. For each left click we look back
up to `lookback` seconds, find the movement onset (speed rising through a
fraction of the segment peak), and characterize the flick:

  amplitude      net displacement (counts) from onset to click
  angle          direction of net displacement (radians, aim convention: +y up)
  peak_speed     max speed on the segment (counts/s)
  time_to_peak   onset -> peak speed (s)
  overshoot      how far past the endpoint the path traveled along the flick
                 axis, as a fraction of amplitude (classic flick overshoot)
  corrections    number of corrective submovements after the peak (speed
                 valleys followed by re-acceleration)
  duration       onset -> click (s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..telemetry.trace import MouseTrace, ResampleCache


@dataclass(slots=True)
class Flick:
    t_click: float
    t_onset: float
    duration: float
    amplitude: float
    angle: float            # radians, 0 = right, pi/2 = up
    peak_speed: float
    time_to_peak: float
    overshoot: float        # fraction of amplitude, >= 0
    corrections: int

    @property
    def horizontal(self) -> str:
        c = np.cos(self.angle)
        return "right" if c > 0.25 else ("left" if c < -0.25 else "vertical")


def _smooth(v: np.ndarray, rate: float, tau_ms: float = 8.0) -> np.ndarray:
    """Causal exponential moving average (cheap, phase-lag ~tau).

    Vectorized as a truncated exponential FIR plus the acc=v[0] initial-
    condition term; the kernel is cut where its weight reaches 1e-9, so the
    deviation from the exact recurrence is far below the onset/hysteresis
    thresholds downstream.
    """
    if v.size == 0:
        return v
    a = 1.0 - np.exp(-1.0 / (rate * tau_ms / 1000.0))
    n = v.size
    klen = min(n, int(np.ceil(np.log(1e-9) / np.log(1.0 - a))))
    k = a * (1.0 - a) ** np.arange(klen)
    decay = (1.0 - a) ** (np.arange(n, dtype=np.float64) + 1.0)
    return np.convolve(np.asarray(v, dtype=np.float64), k)[:n] + v[0] * decay


def segment_flicks(
    trace: MouseTrace,
    rate: float = 500.0,
    lookback: float = 0.8,
    onset_frac: float = 0.08,
    min_amplitude: float = 15.0,
    *,
    grid: ResampleCache | None = None,
) -> list[Flick]:
    # `grid` (a ResampleCache over `trace`) lets build_report share one
    # resample across analysis passes; bare-trace calls behave identically.
    tg, vx, vy = (grid if grid is not None else trace).resample(rate)
    if tg.size == 0 or trace.clicks.size == 0:
        return []
    vy = -vy  # aim convention: +y up
    vxs, vys = _smooth(vx, rate), _smooth(vy, rate)
    speed = np.hypot(vxs, vys)
    x = np.concatenate([[0.0], np.cumsum(vxs)]) / rate  # position (counts)
    y = np.concatenate([[0.0], np.cumsum(vys)]) / rate

    flicks: list[Flick] = []
    n = tg.size
    for k, tc in enumerate(trace.clicks):
        ic = int(np.searchsorted(tg, tc))
        if ic <= 1:
            continue
        ic = min(ic, n)
        # Never look back past the previous click: at real kill pacing
        # (<0.8s between kills) the window would swallow the previous flick.
        lo = tc - lookback
        if k > 0:
            lo = max(lo, float(trace.clicks[k - 1]))
        i0 = max(int(np.searchsorted(tg, lo)), 0)
        seg = speed[i0:ic]
        if seg.size < 4:
            continue
        ipk_rel = int(np.argmax(seg))
        pk = seg[ipk_rel]
        if pk <= 0:
            continue
        # onset: last sample before the peak with speed below onset_frac*peak
        below = np.nonzero(seg[: ipk_rel + 1] < onset_frac * pk)[0]
        ion = i0 + (int(below[-1]) if below.size else 0)
        ipk = i0 + ipk_rel

        dx_net = x[ic] - x[ion]
        dy_net = y[ic] - y[ion]
        amplitude = float(np.hypot(dx_net, dy_net))
        if amplitude < min_amplitude:
            continue
        ux, uy = dx_net / amplitude, dy_net / amplitude

        # overshoot: max projection along flick axis beyond the click point
        proj = (x[ion:ic + 1] - x[ion]) * ux + (y[ion:ic + 1] - y[ion]) * uy
        overshoot = float(max(proj.max() - amplitude, 0.0) / amplitude)

        # corrective submovements after the peak: speed collapses (below 15%
        # of peak), then re-accelerates (above 35%). Hysteresis rejects the
        # sample-level jitter of integer mouse counts.
        post = speed[ipk:ic]
        corrections = 0
        if post.size >= 3:
            m = np.where(post < 0.15 * pk, -1, np.where(post > 0.35 * pk, 1, 0))
            m = m[m != 0]
            if m.size >= 2:
                corrections = int(np.sum((m[:-1] == -1) & (m[1:] == 1)))

        flicks.append(
            Flick(
                t_click=float(tc),
                t_onset=float(tg[ion]),
                duration=float(tc - tg[ion]),
                amplitude=amplitude,
                angle=float(np.arctan2(dy_net, dx_net)),
                peak_speed=float(pk),
                time_to_peak=float(tg[ipk] - tg[ion]),
                overshoot=overshoot,
                corrections=corrections,
            )
        )
    return flicks


# ---------------------------------------------------------------- aggregates
def directional_bias(flicks: list[Flick]) -> dict:
    """Left/right (and vertical) split of flick quality.

    bias_score in [-1, 1]: positive = *left* flicks are worse (higher
    overshoot + more corrections), i.e. left is the weak side.
    """
    def agg(fs: list[Flick]) -> dict:
        if not fs:
            return {"n": 0, "overshoot": 0.0, "corrections": 0.0,
                    "peak_speed": 0.0, "duration": 0.0}
        return {
            "n": len(fs),
            "overshoot": float(np.mean([f.overshoot for f in fs])),
            "corrections": float(np.mean([f.corrections for f in fs])),
            "peak_speed": float(np.mean([f.peak_speed for f in fs])),
            "duration": float(np.mean([f.duration for f in fs])),
        }

    left = agg([f for f in flicks if f.horizontal == "left"])
    right = agg([f for f in flicks if f.horizontal == "right"])
    vertical = agg([f for f in flicks if f.horizontal == "vertical"])

    def badness(a: dict) -> float:
        return a["overshoot"] + 0.15 * a["corrections"]

    score = 0.0
    if left["n"] >= 3 and right["n"] >= 3:
        b_l, b_r = badness(left), badness(right)
        denom = b_l + b_r
        score = float((b_l - b_r) / denom) if denom > 0 else 0.0
    return {"left": left, "right": right, "vertical": vertical, "bias_score": score}


def region_deficits(flicks: list[Flick], cols: int = 3, rows: int = 3) -> dict[str, float]:
    """Per-region weakness signal from real flick data.

    Direction AND amplitude aware: each flick credits the wall region it was
    aimed toward, at a ring distance from the grid center set by its
    amplitude relative to the run's own flick-amplitude distribution
    (amplitude / p90, clipped to 1). Short flicks credit the inner cells,
    long flicks the edges — skill decays with distance from the mouse rest
    position, so large-angle weakness lives at the extremes (analysis/kb.py:
    p-rest-position, dx-region-deficit). The unit direction is stretched
    onto the square (divided by max(|ux|, |uy|)) so diagonal flicks can
    reach the corner cells at any grid size.

    Deficit per region = mean (overshoot + correction penalty + slowness
    penalty) over regions with >= 2 observations, z-scored across regions.
    Feeds the bandit as *observed* rewards, replacing (or blending with)
    run-level attribution. Keys follow the r{row}c{col} cross-module
    contract in aim convention (+y up: an upward flick credits a higher row).
    """
    if not flicks:
        return {}
    center_r, center_c = (rows - 1) / 2.0, (cols - 1) / 2.0
    buckets: dict[str, list[float]] = {}
    durs = np.array([f.duration for f in flicks])
    med_dur = float(np.median(durs))
    ref_amp = max(float(np.percentile([f.amplitude for f in flicks], 90)), 1e-9)
    for f in flicks:
        # direction stretched onto the square, scaled by the amplitude ring
        ux, uy = np.cos(f.angle), np.sin(f.angle)
        m = max(abs(ux), abs(uy), 1e-9)
        ring = min(f.amplitude / ref_amp, 1.0)
        c = int(np.clip(round(center_c + (ux / m) * ring * center_c), 0, cols - 1))
        r = int(np.clip(round(center_r + (uy / m) * ring * center_r), 0, rows - 1))
        slowness = max(f.duration / med_dur - 1.0, 0.0) if med_dur > 0 else 0.0
        buckets.setdefault(f"r{r}c{c}", []).append(
            f.overshoot + 0.15 * f.corrections + 0.25 * slowness
        )
    means = {k: float(np.mean(v)) for k, v in buckets.items() if len(v) >= 2}
    if not means:
        return {}
    vals = np.array(list(means.values()))
    mu, sd = vals.mean(), vals.std() or 1.0
    return {k: float((v - mu) / sd) for k, v in means.items()}


def movement_heatmap(
    trace: MouseTrace,
    bins: int = 64,
    rate: float = 250.0,
    *,
    grid: ResampleCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D occupancy histogram of crosshair displacement (recentered every
    click, so it shows where your aim travels relative to each engagement)."""
    tg, vx, vy = (grid if grid is not None else trace).resample(rate)
    if tg.size == 0:
        z = np.zeros((bins, bins))
        e = np.linspace(-1, 1, bins + 1)
        return z, e, e
    x = np.cumsum(vx) / rate
    y = np.cumsum(-vy) / rate
    # recenter at each click
    anchors = np.searchsorted(tg, trace.clicks)
    offset_x, offset_y = np.zeros_like(x), np.zeros_like(y)
    last = 0
    ox = oy = 0.0
    for a in anchors:
        a = min(int(a), x.size - 1)
        offset_x[last:a] = ox
        offset_y[last:a] = oy
        ox, oy = x[a], y[a]
        last = a
    offset_x[last:] = ox
    offset_y[last:] = oy
    rx, ry = x - offset_x, y - offset_y
    lim = max(float(np.percentile(np.abs(np.concatenate([rx, ry])), 99)), 1.0)
    hist, xe, ye = np.histogram2d(
        rx, ry, bins=bins, range=[[-lim, lim], [-lim, lim]]
    )
    return hist, xe, ye
