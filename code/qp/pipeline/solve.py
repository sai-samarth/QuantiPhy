"""SOLVE stage: pixel observables + the prior -> a number in the requested unit.

THE ONE IDEA THAT MAKES THIS SMALL.

You would expect four different formulas, one per inference_type (SS/SD/DS/DD):
size-from-size, speed-from-size, size-from-speed, speed-from-speed. There is
only one. Both a length and a speed are "pixels times a scale":

        metres        = px      * (m/px)
        metres/second = px/s    * (m/px)

so whatever the prior describes, it pins down the SAME unknown:

        m_per_px = (the prior, in SI) / (that same object, measured in pixels)

and whatever the question asks for is then that scale times the target's own
pixel observable. The prior being a speed instead of a size changes only WHICH
pixel number goes in the denominator. That is why this module has one code path
and not four.

WHERE fps ENTERS, AND WHERE IT CANCELS.
A pixel speed is measured in px/frame; converting to px/s multiplies by fps.
Writing V for an SI speed, p/q for reference/target pixel speeds and R/T for
reference/target pixel lengths:

    SS  size from size    T * (R_m / R)                        no fps at all
    SD  speed from size   q*fps * (R_m / R)                     proportional to fps
    DS  size from speed   T * (V / (p*fps))                     proportional to 1/fps
    DD  speed from speed  q*fps * (V / (p*fps)) = q * (V/p)     fps CANCELS

Only the two mixed types are exposed to the fps disagreement between container
and metadata (91 test videos, up to 1.5x). SS and DD are immune by construction.
This falls out of the algebra rather than being special-cased, which is the
point of having one formula.

DEPTH IS A MULTIPLIER ON BOTH.
m_per_px is a property of a DISTANCE, not of a scene (see geometry.py). The
target's scale is the reference's scale times Z_target/Z_ref. A transverse speed
scales with depth exactly as a length does, so the same correction applies to
both, unmodified.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .geometry import Scale, to_metres
from .types import Answer, Plan, PixelMeasure, Result

# SI factor per unit token: multiply a value in that unit by this to get SI.
UNIT_SI = {
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "cm": 0.01, "mm": 0.001, "km": 1000.0,
    "m/s": 1.0, "cm/s": 0.01, "km/s": 1000.0,
    "m/s^2": 1.0, "m/s2": 1.0, "cm/s^2": 0.01, "cm/s2": 0.01,
}


def si_factor(unit: str) -> float:
    """SI multiplier for a unit token; 1.0 (metres) if unrecognised."""
    return UNIT_SI.get(str(unit).strip().lower().replace(" ", ""), 1.0)


@dataclass
class Observable:
    """A pixel quantity plus what it is a quantity OF.

    kind is 'length' (px), 'rate' (px/s) or 'accel' (px/s^2). Keeping the kind
    attached is what lets solve() refuse to divide a length by a speed: a prior
    that describes a size cannot calibrate a pipeline that measured a speed, and
    that mismatch is otherwise a silent factor-of-fps error rather than a crash.
    """
    pixels: float
    kind: str = "length"
    spread: float = 0.0
    method: str = ""
    n_frames: int = 0

    @staticmethod
    def from_measure(pm: PixelMeasure) -> "Observable":
        return Observable(pm.pixels, "length", pm.spread, pm.method, pm.n_frames)


QUANTITY_KIND = {"size": "length", "distance": "length",
                 "speed": "rate", "acceleration": "accel"}


def pixel_accel(df, fps: float, mode: str = "magnitude",
                at_time: float | None = None) -> Result:
    """Pixel acceleration (px/s^2) of a track, by QUADRATIC FIT.

    A second derivative taken as a difference of differences multiplies mask
    noise by 1/dt^2 and is unusable at these frame rates. Fitting

        u(t) = u0 + vu.t + 0.5.au.t^2

    over the whole track uses every frame to constrain three numbers, so the
    noise averages down instead of compounding. `2 * the quadratic coefficient`
    is the acceleration.

    MODE exists because 'ball acceleration = 3.0 m/s^2' does not say WHICH
    acceleration, and the benchmark has no stated convention:
        magnitude  |a| of the 2-D acceleration vector
        vertical   the image-y component alone (right for free fall / gravity)
        along      the component along the direction of travel (right for
                   a body speeding up or braking in a straight line)
    Which one the annotators used is an empirical question, so it is a
    parameter to be fitted, not a constant to be guessed.
    """
    d = df.sort_values("f")
    t = d["f"].to_numpy(float) / max(fps, 1e-9)
    u, v = d["u"].to_numpy(float), d["v"].to_numpy(float)
    if len(t) < 5:
        return Result.bad("need >=5 frames to fit an acceleration", "accel")
    if at_time is not None:
        m = np.abs(t - at_time) <= 0.5
        if m.sum() >= 5:
            t, u, v = t[m], u[m], v[m]
    t = t - t.mean()
    try:
        cu = np.polyfit(t, u, 2)
        cv = np.polyfit(t, v, 2)
    except Exception as e:
        return Result.bad(f"quadratic fit failed: {e}", "accel")
    au, av = 2.0 * cu[0], 2.0 * cv[0]

    if mode == "vertical":
        a = abs(av)
    elif mode == "along":
        # project the acceleration onto the mean direction of travel
        du, dv = np.polyval(np.polyder(cu), 0.0), np.polyval(np.polyder(cv), 0.0)
        n = np.hypot(du, dv)
        a = abs((au * du + av * dv) / n) if n > 1e-9 else float(np.hypot(au, av))
    else:
        a = float(np.hypot(au, av))

    if not np.isfinite(a) or a <= 0:
        return Result.bad("degenerate acceleration fit", "accel")
    # residual of the fit, relative to the motion it explains, as a spread proxy
    res = np.hypot(u - np.polyval(cu, t), v - np.polyval(cv, t))
    span = max(np.hypot(u.max() - u.min(), v.max() - v.min()), 1e-6)
    spread = float(np.median(res) / span)
    return Result.good(
        Observable(a, "accel", spread, f"quadfit:{mode}", len(t)),
        "accel", confidence=float(np.clip(1 - spread * 3, 0.1, 1.0)))


def solve(plan: Plan, reference: Observable, target: Observable,
          scale: Scale) -> Result:
    """The whole conversion. Returns an Answer in `plan.unit`.

    Deliberately pure: no video, no models, no I/O. Every failure here is an
    arithmetic impossibility (a zero denominator, a unit mismatch), never a
    perception failure -- those were already turned into Results upstream.
    """
    want = QUANTITY_KIND.get(plan.quantity, "length")
    have = getattr(plan, "reference_kind", None) or (
        "rate" if plan.reference_is_motion else "length")

    if reference.kind != have:
        return Result.bad(
            f"prior describes a {have} but the reference was measured as a "
            f"{reference.kind}", "solve")
    if target.kind != want:
        return Result.bad(
            f"question asks for a {want} but the target was measured as a "
            f"{target.kind}", "solve")
    if not np.isfinite(reference.pixels) or reference.pixels <= 0:
        return Result.bad("reference pixel measurement is zero or non-finite", "solve")
    if not np.isfinite(target.pixels):
        return Result.bad("target pixel measurement is non-finite", "solve")
    if plan.reference_value <= 0:
        return Result.bad("prior value is zero or negative", "solve")

    # 1. the scale, from the prior and the reference's own pixels
    m_per_px = plan.reference_value / reference.pixels

    # 2. the target, at the target's depth
    si = target.pixels * m_per_px * scale.depth_ratio

    # 3. an acceleration prior calibrates px/s^2 -> m/s^2 identically; no extra
    #    time factor, because both sides carry the same /s^2.

    value = si / si_factor(plan.unit)
    if not np.isfinite(value):
        return Result.bad("non-finite result", "solve")

    # confidence: the product of what each stage independently knows.
    # Spread is relative std across frames; two measurements each contribute.
    meas_conf = float(np.clip(1.0 - 4.0 * (reference.spread + target.spread), 0.1, 1.0))
    conf = float(np.clip(scale.confidence * meas_conf, 0.0, 1.0))

    return Result.good(
        Answer(value=value, unit=plan.unit, confidence=conf, provenance={
            "quantity": plan.quantity, "axis": plan.axis,
            "prior_si": plan.reference_value,
            "reference_px": reference.pixels, "reference_method": reference.method,
            "target_px": target.pixels, "target_method": target.method,
            "m_per_px": m_per_px,
            "depth_ratio": scale.depth_ratio, "depth_status": scale.status,
            "depth_note": scale.note,
            "si_value": si,
            "ref_spread": reference.spread, "tgt_spread": target.spread,
        }),
        "solve", confidence=conf)


def pixel_separation(df_a, df_b, frames=None) -> Result:
    """Centroid separation in pixels between two tracks, for 'distance between'.

    Measured only on frames where BOTH objects are present -- otherwise the
    separation is computed against a position the object no longer holds, which
    reads as a plausible number rather than as a failure.
    """
    a = df_a.set_index("f")[["u", "v"]]
    b = df_b.set_index("f")[["u", "v"]]
    common = a.index.intersection(b.index)
    if frames is not None:
        common = common.intersection(list(frames))
    if len(common) == 0:
        return Result.bad("the two objects are never both visible", "separation")
    d = np.hypot(a.loc[common, "u"].to_numpy() - b.loc[common, "u"].to_numpy(),
                 a.loc[common, "v"].to_numpy() - b.loc[common, "v"].to_numpy())
    med = float(np.median(d))
    if med <= 0:
        return Result.bad("zero separation (same object matched twice)", "separation")
    spread = float(np.std(d) / med)
    return Result.good(
        Observable(med, "length", spread, "centroid-separation", len(common)),
        "separation", confidence=float(np.clip(1 - spread * 2, 0.1, 1.0)))


def pixel_speed(df, fps: float, at_time: float | None = None,
                window: float = 0.25, average: bool = False) -> Result:
    """Pixel speed (px/s) of a track: instantaneous near `at_time`, or mean.

    Instantaneous speed is a LOCAL LINEAR FIT, not a two-frame difference. A
    difference of two centroids divides mask noise by a very small dt and
    amplifies it; a fit over a window averages that noise down while still
    resolving the instant the question asks about.
    """
    d = df.sort_values("f")
    t = d["f"].to_numpy(float) / max(fps, 1e-9)
    u, v = d["u"].to_numpy(float), d["v"].to_numpy(float)
    if len(t) < 3:
        return Result.bad("track too short to differentiate", "speed")

    if average or at_time is None:
        # total path length over elapsed time: robust to endpoint noise
        dist = float(np.sum(np.hypot(np.diff(u), np.diff(v))))
        span = float(t[-1] - t[0])
        if span <= 0:
            return Result.bad("zero time span", "speed")
        return Result.good(Observable(dist / span, "rate", 0.0, "path/time", len(t)),
                           "speed", confidence=0.8)

    m = np.abs(t - at_time) <= window
    if m.sum() < 3:
        m = np.argsort(np.abs(t - at_time))[:5]
        m = np.isin(np.arange(len(t)), m)
    if m.sum() < 3:
        return Result.bad(f"only {int(m.sum())} frames near t={at_time}", "speed")
    du = np.polyfit(t[m] - at_time, u[m], 1)[0]
    dv = np.polyfit(t[m] - at_time, v[m], 1)[0]
    sp = float(np.hypot(du, dv))
    if not np.isfinite(sp):
        return Result.bad("fit did not converge", "speed")
    return Result.good(Observable(sp, "rate", 0.0, f"linfit@{at_time}s", int(m.sum())),
                       "speed", confidence=0.85)
