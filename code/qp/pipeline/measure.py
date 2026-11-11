"""MEASURE stage: an object's track -> a length in pixels.

TWO PROBLEMS, and the second is the one that actually bites.

1. WHICH DIMENSION. "height" is the bounding-box height; "diameter" is the
   equal-area circle; "length" is the principal axis. Trivial once the plan
   says which.

2. WHICH FRAMES. This is where the errors live:
     * bowling pins measured across the whole clip -> 44% error, because half
       the frames are AFTER the ball scatters them. Measured while upright -> 0.1%.
     * tennis balls swell 44.5 -> 55 px exactly while moving fast (motion blur).
     * a ball held by keyed-out fingers is partly erased.
   All three are the same shape: a rigid object's apparent size should be STABLE
   (or vary smoothly with depth). Frames where it is not are contaminated.

   The fix has TWO parts, and the second was learned the hard way:
     (a) find runs of consecutive frames where the measurement is self-consistent
     (b) among those, prefer the run where the object is MOVING SLOWEST.

   (a) alone is NOT enough. On the tennis ball the longest stable run is the
   ROLLING phase (26 frames, 45.4 px) rather than the at-rest phase (22 frames,
   40.5 px) — it picked the blur-inflated window purely because it was longer.
   Motion blur scales with speed, so (b) is the criterion that actually tracks
   the physics rather than an accident of clip length.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .types import Axis, PixelMeasure, Result
from .strategy import Strategy, run_chain

_AXIS_COL = {"height": "bh", "width": "bw", "length": "major",
             "diameter": "equiv", "any": "equiv", "separation": "equiv"}


def _gap_limit(frames: np.ndarray) -> float:
    """How large a frame-number jump breaks a run.

    Absolute thresholds break the moment the video is SUBSAMPLED: at stride 9,
    consecutive kept frames are 9 apart, so a fixed "gap > 2" shatters every run
    into singletons and the stable-run logic silently stops working. Scale the
    limit to the spacing actually present instead.
    """
    if len(frames) < 3:
        return 2.5
    d = np.diff(np.asarray(frames, float))
    step = np.median(d[d > 0]) if (d > 0).any() else 1.0
    return max(2.5, 2.5 * step)


def stable_run(values: np.ndarray, frames: np.ndarray, tol: float = 0.08,
               min_len: int = 3) -> np.ndarray:
    """Indices of the longest consecutive run whose values stay within `tol`
    of that run's own median. This is the automatic version of 'measure the
    object in its canonical state'."""
    n = len(values)
    if n == 0:
        return np.array([], int)
    gap = _gap_limit(frames)
    best = (0, 0)
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            seg = values[i:j + 1]
            med = np.median(seg)
            if med <= 0 or np.max(np.abs(seg - med) / med) > tol:
                break
            if frames[j] - frames[j - 1] > gap:        # a gap breaks the run
                break
            j += 1
        if (j - i) > (best[1] - best[0]):
            best = (i, j)
        i = max(i + 1, j)
    lo, hi = best
    return np.arange(lo, hi) if (hi - lo) >= min_len else np.arange(n)


def slowest_stable_run(vals, frames, u, v, tol=0.08, min_len=3):
    """All stable runs, then the one with the lowest median pixel speed.

    Motion blur inflates a mask in proportion to how far the object travels
    during the exposure, so the slowest window is the least contaminated.
    """
    runs, n, i = [], len(vals), 0
    gap = _gap_limit(frames)
    while i < n:
        j = i + 1
        while j < n:
            seg = vals[i:j + 1]; med = np.median(seg)
            if med <= 0 or np.max(np.abs(seg - med) / med) > tol: break
            if frames[j] - frames[j - 1] > gap: break
            j += 1
        if (j - i) >= min_len:
            runs.append((i, j))
        i = max(i + 1, j)
    if not runs:
        return np.arange(n)
    speed = np.hypot(np.gradient(u), np.gradient(v))
    best = min(runs, key=lambda r: np.median(speed[r[0]:r[1]]))
    return np.arange(*best)


def _from_masks(df: pd.DataFrame, axis: Axis, method_col: str, label: str) -> Result:
    if not len(df):
        return Result.bad("empty track", label)
    d = df.sort_values("f")
    vals = d[method_col].to_numpy(float)
    frames = d["f"].to_numpy(int)
    keep = slowest_stable_run(vals, frames, d["u"].to_numpy(float), d["v"].to_numpy(float))
    v = vals[keep]
    if len(v) == 0 or np.median(v) <= 0:
        return Result.bad("no usable frames", label)
    med = float(np.median(v))
    spread = float(np.std(v) / med)
    return Result.good(
        PixelMeasure(pixels=med, axis=axis, n_frames=len(v), spread=spread,
                     method=label, frames_used=frames[keep].tolist()),
        label, confidence=float(np.clip(1.0 - spread * 4, 0.1, 1.0)))


def measure(track_df: pd.DataFrame, axis: Axis, frames_bgr=None) -> tuple[Result, list]:
    """Measure `axis` on a track. Strategies ordered most- to least-accurate.

    The gradient method is the most ACCURATE where it applies but fails silently
    where it does not:
      * patterned objects  — rays stop at an internal edge (soccer ball: 48.8 px
        against a true 95.4, i.e. HALF, reported with 1.4% spread)
      * non-circular objects — a radial diameter is not a height (bowling pin:
        18.3 px against 55.0)
    Both failures are CONFIDENT: low spread, so a stability check cannot catch
    them. What does catch them is CORROBORATION — the gradient is accepted only
    when it agrees with an independent mask measurement. Two methods disagreeing
    is itself the evidence.
    """
    col = _AXIS_COL.get(axis, "equiv")

    def gradient_corroborated():
        if frames_bgr is None or axis not in ("diameter", "any"):
            return Result.bad("gradient only applies to round objects", "gradient")
        g = measure_gradient(track_df, frames_bgr, axis)
        if not g.ok:
            return g
        m = _from_masks(track_df.sort_values("f"), axis, "equiv", "mask:equiv")
        if not m.ok:
            return Result.bad("no mask measure to corroborate against", "gradient")
        ratio = g.value.pixels / m.value.pixels
        if not (0.8 <= ratio <= 1.25):
            return Result.bad(
                f"gradient {g.value.pixels:.1f} px disagrees with mask "
                f"{m.value.pixels:.1f} px ({ratio:.2f}x) — likely a patterned or "
                f"non-circular object", "gradient")
        return g

    chain = [
        Strategy("gradient", gradient_corroborated,
                 applies_if=lambda: frames_bgr is not None and axis in ("diameter", "any")),
        Strategy(f"mask:{col}",  lambda: _from_masks(track_df, axis, col, f"mask:{col}")),
        Strategy("mask:equiv",   lambda: _from_masks(track_df, axis, "equiv", "mask:equiv"),
                 applies_if=lambda: col != "equiv"),
        Strategy("mask:major",   lambda: _from_masks(track_df, axis, "major", "mask:major"),
                 applies_if=lambda: col != "major"),
    ]
    def validate(pm: PixelMeasure) -> str | None:
        if pm.pixels < 4:
            return f"only {pm.pixels:.1f} px — too small to measure to 5%"
        if pm.spread > 0.25:
            return f"unstable: {pm.spread*100:.0f}% spread across frames"
        return None
    return run_chain(chain, validate=validate)


# ─────────────────── sub-pixel gradient measurement ───────────────────
def radial_diameter(img, cu, cv, rmax=None, nray=180) -> float:
    """True object diameter from the image gradient, sub-pixel.

    Walks rays out from the centroid and finds where brightness crosses halfway
    between object and background. Independent of the mask, so it is immune to
    the mask's own bias — which on a tennis ball was -7.5% (minor axis) and
    +1.4% (mask area) against a gradient truth of 43.9 px.
    """
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(float)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    H, W = g.shape
    rmax = rmax or min(H, W) // 4
    rs = []
    for a in np.linspace(0, 2 * np.pi, nray, endpoint=False):
        rr = np.arange(2, rmax, 0.25)
        xs = cu + rr * np.cos(a); ys = cv + rr * np.sin(a)
        ok = (xs > 0) & (xs < W - 1) & (ys > 0) & (ys < H - 1)
        if ok.sum() < 10:
            continue
        rr2 = rr[ok]
        v = cv2.remap(g, xs[ok].astype(np.float32).reshape(-1, 1),
                      ys[ok].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
        inner, outer = np.median(v[:8]), np.median(v[-8:])
        if abs(inner - outer) < 12:
            continue
        half = (inner + outer) / 2
        cross = np.where((v[:-1] - half) * (v[1:] - half) < 0)[0]
        if not len(cross):
            continue
        i = cross[0]
        frac = (half - v[i]) / (v[i + 1] - v[i] + 1e-9)
        rs.append(rr2[i] + frac * (rr2[i + 1] - rr2[i]))
    return 2 * float(np.median(rs)) if len(rs) > 20 else float("nan")


def measure_gradient(track_df, frames_bgr, axis: Axis) -> Result:
    """Gradient measurement on the frames chosen by slowest_stable_run."""
    if not len(track_df) or frames_bgr is None:
        return Result.bad("no frames supplied", "gradient")
    d = track_df.sort_values("f")
    keep = slowest_stable_run(d["equiv"].to_numpy(float), d["f"].to_numpy(int),
                              d["u"].to_numpy(float), d["v"].to_numpy(float))
    sub = d.iloc[keep]
    vals, used = [], []
    for _, r in sub.iterrows():
        fi = int(r["f"])
        if fi >= len(frames_bgr):
            continue
        # Crop to a window scaled by the object's own size, and cap the ray
        # length just past its edge. Without this the rays run off the object
        # into the background and the profile becomes meaningless (27% spread).
        rad = max(6.0, float(r["equiv"]) / 2)
        H = int(np.ceil(rad * 2.2))
        u0, v0 = float(r["u"]), float(r["v"])
        img = frames_bgr[fi]
        pad = np.zeros((img.shape[0] + 2*H, img.shape[1] + 2*H, 3), img.dtype)
        pad[H:H+img.shape[0], H:H+img.shape[1]] = img
        crop = pad[int(v0):int(v0)+2*H, int(u0):int(u0)+2*H]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        val = radial_diameter(crop, H, H, rmax=rad * 1.8)
        if np.isfinite(val) and val > 3:
            vals.append(val); used.append(fi)
    if len(vals) < 3:
        return Result.bad("gradient profile inconsistent (occlusion?)", "gradient")
    med = float(np.median(vals)); spread = float(np.std(vals) / med)
    if spread > 0.20:
        return Result.bad(f"gradient unstable ({spread*100:.0f}% spread)", "gradient")
    return Result.good(
        PixelMeasure(pixels=med, axis=axis, n_frames=len(vals), spread=spread,
                     method="gradient", frames_used=used),
        "gradient", confidence=float(np.clip(1.0 - spread * 4, 0.2, 1.0)))
