"""Instant-rate instruments: evaluate everything AT THE ASKED MOMENT.

The diagnosis (HANDOFF 6n tutorial, ping pong at 1.4s): the pipeline answered an
instant question with three AVERAGES -- an interval-average radial speed, a
track-median pixel ruler, and a fit window that straddled a regime change. Each
function here replaces one average with the value at the instant.

THE WINDOW IS NOT A PARAMETER. It is discovered: a regime break is a frame
where the velocity CHANGE is several times larger than the typical change in
its own neighbourhood. Steady fall is not a break -- each falling frame
resembles the next -- only the transition is. The fit then uses the frames
between the breaks that surround the asked instant, so it can never mix
rolling with falling no matter how the constant is tuned.
"""
from __future__ import annotations
import numpy as np


def _series(g, fps):
    d = g.sort_values("f")
    t = d["f"].to_numpy(float) / max(fps, 1e-9)
    return t, d["u"].to_numpy(float), d["v"].to_numpy(float), d["equiv"].to_numpy(float)


def breakpoints(t, u, v, ratio=4.0, floor=25.0, hood=5):
    """Indices where a NEW motion regime starts.

    dv_i is compared to the median dv over the TRAILING `hood` frames -- the
    past only. A symmetric neighbourhood was tried first and MISSED the table
    edge: the fall's own huge jumps sat inside the reference window, so the
    transition did not look special against them. "Is this jump big compared to
    what the ball was JUST doing" is the question; the future must not vote.
    Steady acceleration still passes (each fall frame resembles the previous
    fall frame once the fall is a few frames old), and consecutive flagged
    frames merge into the one event that started them.
    `floor` (px/s) keeps mask jitter on a still object from producing breaks.
    """
    if len(t) < 6:
        return []
    vu, vv = np.gradient(u, t), np.gradient(v, t)
    dv = np.hypot(np.diff(vu), np.diff(vv))
    out = []
    for i, x in enumerate(dv):
        past = dv[max(0, i - hood):i]
        if not len(past):
            continue
        if x > floor and x > ratio * max(float(np.median(past)), 1e-9):
            out.append(i + 1)              # the regime starts AT frame i+1
    merged = []
    for b in out:                          # a transition flags 2-3 frames; one event
        if not merged or b - merged[-1] > 2:
            merged.append(b)
    return merged


def _slope_at(t, y, m, at_time):
    """d(y)/dt AT at_time: quadratic when the points allow it, else linear.

    A straight line over a window reports the window's AVERAGE slope; when the
    asked instant sits at the window's edge of a curving motion that average
    answers a different question -- the very mistake this module exists to fix.
    """
    deg = 2 if m.sum() >= 6 else 1
    c = np.polyfit(t[m] - at_time, y[m], deg)
    return float(c[-2])                    # d/dt at t=at_time


def segment_of(t, breaks, at_time):
    """[start,end) frame indices of the regime containing `at_time`."""
    edges = [0] + list(breaks) + [len(t)]
    k = int(np.searchsorted(t, at_time, side="right")) - 1
    k = min(max(k, 0), len(t) - 1)
    for a, b in zip(edges[:-1], edges[1:]):
        if a <= k < b:
            return a, b
    return 0, len(t)


def speed_at(g, fps, at_time, window=0.35):
    """(px_per_s, n_used, (seg_t0, seg_t1)) -- linear fit, never across a break."""
    t, u, v, _ = _series(g, fps)
    if len(t) < 3:
        return None
    a, b = segment_of(t, breakpoints(t, u, v), at_time)
    m = np.zeros(len(t), bool)
    m[a:b] = True
    m &= np.abs(t - at_time) <= window
    if m.sum() < 3:                        # widen inside the segment only
        idx = np.argsort(np.abs(t - at_time))
        for i in idx:
            if a <= i < b:
                m[i] = True
            if m.sum() >= 3:
                break
    if m.sum() < 3:
        return None
    du = _slope_at(t, u, m, at_time)
    dv = _slope_at(t, v, m, at_time)
    return float(np.hypot(du, dv)), int(m.sum()), (float(t[a]), float(t[b - 1]))


def size_at(g, fps, at_time, stat="equiv"):
    """Apparent size (px) at the instant: rolling-median-3, interpolated.

    `stat` must be the axis THE PRIOR NAMES. The gate caught this: with equiv
    hard-coded, a prior "height of the astronaut = 2.2 m" was divided by the
    area-equivalent diameter (~65 px) instead of the height (~131 px), and every
    answer on that video came out a consistent 2x high -- the signature of a
    ruler error. Balls hide the bug because all their axes coincide.
    """
    d = g.sort_values("f")
    t = d["f"].to_numpy(float) / max(fps, 1e-9)
    s = d[stat].to_numpy(float)
    if len(t) < 3:
        return None
    sm = np.array([np.median(s[max(0, i - 1):i + 2]) for i in range(len(s))])
    return float(np.interp(at_time, t, sm))


def radial_at(g, fps, at_time, z_anchor_t, z_anchor, window=0.35, min_trend=0.02):
    """Depth speed (m/s, + = approaching) at the instant, from the size series.

    Z(t) = Z(t_a) * s(t_a) / s(t): a ball twice as big is half as far. Dense in
    time (every frame has a size), unlike depth_info's two samples, so it can be
    differentiated AT the instant. Same no-cliff rule as speed_at. Refuses when
    the size trend inside the window is under `min_trend` -- differentiating a
    flat noisy series manufactures a rate out of jitter.
    """
    t, u, v, s = _series(g, fps)
    if len(t) < 5:
        return None
    sm = np.array([np.median(s[max(0, i - 1):i + 2]) for i in range(len(s))])
    s_a = np.interp(z_anchor_t, t, sm)
    if s_a <= 0:
        return None
    z = z_anchor * s_a / np.maximum(sm, 1e-9)
    a, b = segment_of(t, breakpoints(t, u, v), at_time)
    m = np.zeros(len(t), bool)
    m[a:b] = True
    m &= np.abs(t - at_time) <= window
    if m.sum() < 4:
        return None
    if (sm[m].max() - sm[m].min()) / max(sm[m].mean(), 1e-9) < min_trend:
        return None
    dz = _slope_at(t, z, m, at_time)
    return float(-dz)                      # Z shrinking = approaching = positive
