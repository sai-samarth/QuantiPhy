"""Geometry: converting pixel measurements into metres.

THE CENTRAL FACT: metres-per-pixel is not a property of a scene, it is a property
of a DISTANCE.

        pixels = f * real_size / distance     =>     m_per_px = distance / f

So the scale is LINEAR in depth. A target 5% further away than the ruler yields a
5% size error — our entire full-credit budget — with a perfect mask and a perfect
edge. It is a pure multiplier: it does not average out over frames and no amount
of sub-pixel work can recover it.

MEASURED EXPOSURE on the validation split:
    69 items  2D scene, coplanarity given by the benchmark      -> safe
    45 items  3D, both depths known                             -> correctable
    45 items  3D, one or both depths MISSING                    -> silent risk
Among the correctable ones, ignoring depth would cost >5% on 22/45 and >25% on
15/45. Worst case: ruler (boat) at 13.15 m vs target (pier) at 18.92 m = +44%.

This module makes all three cases EXPLICIT and refuses to guess silently.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

import numpy as np

from .types import Result


@dataclass
class Scale:
    """How to turn pixels into metres, and how much to trust it."""
    m_per_px_at_ref: float
    depth_ratio: float          # Z_target / Z_ref; 1.0 when coplanar
    status: str                 # see DepthStatus below
    confidence: float
    note: str = ""


# what we know about the two depths
COPLANAR_2D   = "2d-coplanar"        # scene declared 2D: ratio is 1.0 by construction
BOTH_KNOWN    = "both-known"         # depth_info names ruler AND target
REF_ONLY      = "ruler-only"         # target depth unknown
TARGET_ONLY   = "target-only"        # ruler depth unknown
NEITHER       = "neither-known"


def parse_depth_info(text) -> dict[str, float]:
    """{object token: metres} from the depth_info field.

    Handles 'distance_boat_camera = 7.452 m' and 'distance_pingpong_camera=1.511 m'.
    """
    if text is None:
        return {}
    return {_key(m.group(1) or m.group(2)): float(m.group(3))
            for m in _DIST.finditer(str(text))}


# "t=0s, distance_boat_camera = 7.452 m" then "t=1s, ... = 6.655m".
# A timestamp applies to every distance that FOLLOWS it until the next one.
_TSTAMP = re.compile(r"\bt\s*=\s*(\d+(?:\.\d+)?)\s*s", re.I)
# Three more spellings the dataset uses, each of which used to lose its object:
# a space inside the key ('distance_center_of _the_truck_camera', a truck
# closing at 6 m/s), no '_camera' suffix
# ('distance_the_nearest_freestanding_art_display_panel'), and the order
# reversed ('distance_camera_B1sign'). Every entry in both splits has an '='.
_DIST = re.compile(r"distance[_ ](?:camera[_ ]([a-z0-9][a-z_0-9 ]*?)"
                   r"|([a-z0-9][a-z_0-9 ]*?)(?:[_ ]camera)?)\s*=\s*([\d.]+)", re.I)


def _key(raw: str) -> str:
    """'yellowCarLeftFrontTire' -> 'yellow_car_left_front_tire'; spaces -> '_'.

    The camelCase split has to happen here, before lower-casing: lower-cased,
    the key is one 22-letter word that shares nothing with 'tire'.
    """
    k = re.sub(r"([a-z])([A-Z])", r"\1_\2", raw.strip(" _"))
    return re.sub(r"[\s_]+", "_", k).lower()


def parse_depth_series(text) -> dict[str, list]:
    """{object token: [(t, metres), ...]} sorted by t.

    parse_depth_info() keys a dict on the object name, so when the SAME object
    is given at two times the second overwrites the first and the motion is
    destroyed at parse time. That motion is the single largest error source in
    3D: an object moving along the camera axis produces almost no pixel
    displacement, so tracking under-measures its speed by 3x on average, always
    downward. 1,144 test rows state it explicitly and we were discarding it.
    """
    out: dict[str, list] = {}
    tokens = re.split(r"[,\n;]", str(text))
    t_now = None
    for tok in tokens:
        m = _TSTAMP.search(tok)
        if m:
            t_now = float(m.group(1))
        d = _DIST.search(tok)
        if d:
            out.setdefault(_key(d.group(1) or d.group(2)), []).append((t_now, float(d.group(3))))
    return {k: sorted(v, key=lambda p: (p[0] is None, p[0])) for k, v in out.items()}


def depth_at(series, t):
    """Distance at time t from a stated series, or None if t or the times are missing.

    The scale is measured at one instant (the prior's 'at 0.5s') and the target
    at another, and an object moving in depth is at a different distance at
    each. Reading the LAST sample for both is the parse_depth_info overwrite
    again: a ruler measured at 0.5s, 2.36 m away, was paired with its 1.0s
    distance of 1.81 m, and a target 3.07 m away came out x1.70 instead of
    x1.30. Linear between samples, held flat beyond them -- never extrapolated.
    """
    pts = [(a, z) for a, z in (series or []) if a is not None]
    if t is None or not pts:
        return None
    return float(np.interp(t, [a for a, _ in pts], [z for _, z in pts]))


def radial_velocity(series, at_time=None, tol_frac: float = 0.25):
    """dZ/dt in metres per second, or None.

    Positive means RECEDING. Only the magnitude matters for a speed, but the
    sign tells the model which way the object is going.

    EXTRAPOLATION GUARD. depth_info typically states two distances, at t=0s and
    t=1s, so this is an AVERAGE over that window and nothing more. Applying it
    to an instant outside the window is extrapolation, and it measurably hurts:
    a ping-pong ball asked about at 1.4s -- 0.4s past the last depth sample --
    was pushed from 78 (truth 87.9) to 123, because a depth rate that had
    already stopped applying was still being added in. When the question names
    an instant well outside the window, we return None rather than guess.
    """
    pts = [(t, z) for t, z in (series or []) if t is not None]
    if len(pts) < 2:
        return None
    (t0, z0), (t1, z1) = pts[0], pts[-1]
    if t1 == t0:
        return None
    if at_time is not None:
        span = t1 - t0
        pad = tol_frac * span
        if not (t0 - pad <= at_time <= t1 + pad):
            return None
    return (z1 - z0) / (t1 - t0)


def radial_from_size(times, sizes, z_anchor: float, t_anchor=None,
                     at_time=None, min_frac: float = 0.04):
    """dZ/dt (m/s) inferred from how the object's APPARENT SIZE changes.

    Apparent size is inversely proportional to distance, so if the distance is
    known at ONE moment the object's own growth or shrinkage gives it at every
    moment:

        Z(t) = z_anchor * size(t_anchor) / size(t)

    This is the only route to radial motion when depth_info states just one
    distance, and it is the ONLY route at all when an object moves straight at
    the camera -- there, pixel displacement is ~0 and every tracker reports a
    speed of zero. Validated earlier on captured_0021x (ball moving ~98% along
    the camera axis): reconstruction within 1-12% where trackers scored 0.

    Returns None when the size trend is too small to be trusted. Mask noise is
    a few percent, so a size change under `min_frac` over the whole clip is
    indistinguishable from jitter and would manufacture motion out of nothing.
    """
    t = np.asarray(times, float); s_ = np.asarray(sizes, float)
    ok = np.isfinite(t) & np.isfinite(s_) & (s_ > 0)
    t, s_ = t[ok], s_[ok]
    if len(t) < 5 or z_anchor is None or z_anchor <= 0:
        return None
    # robust linear trend in size; a straight fit is enough over these spans
    try:
        b, a = np.polyfit(t, s_, 1)
    except Exception:
        return None
    s_mid = a + b * float(np.median(t))
    if s_mid <= 0:
        return None
    span = abs(b * (t[-1] - t[0]))
    if span / s_mid < min_frac:            # trend smaller than mask noise
        return None
    ta = float(np.median(t)) if t_anchor is None else float(t_anchor)
    s_at_anchor = a + b * ta
    if s_at_anchor <= 0:
        return None
    # Z(t) = C / s(t), so dZ/dt = -C * s'(t) / s(t)^2
    C = z_anchor * s_at_anchor
    t_eval = float(np.median(t)) if at_time is None else float(at_time)
    s_eval = a + b * t_eval
    if s_eval <= 0:
        return None
    return float(-C * b / (s_eval ** 2))


_SYN = {"human": "person", "pingpong": "ping pong", "tennis": "tennis ball",
        # the question and depth_info name these differently: the walking
        # pedestrian is 'distance_human_camera', the bicycle 'distance_bike_camera'
        "pedestrian": "person", "man": "person", "woman": "person", "bicycle": "bike"}
# words that say nothing about WHICH object: 'the' alone matched 'bench seat in
# the atrium' to 'the_central_sculpture'
_STOP = {"the", "and", "its", "itself", "each", "one", "with", "from", "that", "this"}
_COLOURS = {"white", "black", "red", "orange", "yellow", "green", "blue", "purple",
            "pink", "brown", "grey", "gray"}
# head nouns that, shared ALONE, do not say which object is meant
_GENERIC = {"ball", "car", "person", "box", "cup", "bag", "bottle", "toy"}

def _expand(words: set) -> set:
    for a, b in _SYN.items():
        if a in words:
            words = words | set(b.split())
    return words


def _tokens(text) -> list:
    """Words of a phrase or key (keys arrive camelCase-split from _key), stopwords
    dropped. A letter+digit token survives ('B1') so "the 'B1' lettering" can
    find 'distance_camera_B1sign'."""
    return [w for w in re.findall(r"[a-z]+\d+|[a-z]{3,}", str(text).lower())
            if w not in _STOP]


def _words(tokens) -> set:
    # a 'basketball' is a ball: depth_info calls it 'distance_ball_camera'
    return _expand(set(tokens)) | {g for t in tokens for g in _GENERIC
                                   if t.endswith(g) and len(t) >= len(g) + 3}


def _different(pt, kt, pw, kw, common) -> bool:
    """True when a match is only a shared head noun that each side qualifies
    differently -- 'white ball' / 'tennis_ball', 'left person' / 'near_person' --
    or when both name a colour and the colours differ ('purple car' /
    'yellow_car'). Both were stated as a DEPTH ratio, with the wrong object's
    distance and radial speed."""
    pc, kc = pw & _COLOURS, kw & _COLOURS
    if pc and kc and not pc & kc:
        return True
    if not common <= _GENERIC:
        return False

    def extra(toks):
        return [t for t in toks if t not in common
                and not set(_SYN.get(t, "~").split()) <= common]
    return bool(extra(pt)) and bool(extra(kt))


def depth_key_of(phrase: str, depths) -> str | None:
    """Which depth_info entry describes this phrase. Returns the KEY.

    Split out of depth_of so the radial-velocity code matches phrases to depth
    entries by exactly the same rule. It matters: the depth key is
    'distance_human_camera' while the plan's phrase is 'person', which share no
    words at all -- only the synonym table connects them, and a second,
    simpler matcher silently found nothing.
    """
    if not depths or not phrase:
        return None
    pt = _tokens(phrase)
    best, score = None, 0
    for key in depths:
        kt = _tokens(key)
        pw, kw = _words(pt), _words(kt)
        # a compound on one side is two words on the other: whiteboard / white_board
        pw |= {w for a, b in zip(kt, kt[1:]) if a + b in pt for w in (a, b)}
        kw |= {w for a, b in zip(pt, pt[1:]) if a + b in kt for w in (a, b)}
        common = kw & pw
        if not common or _different(pt, kt, pw, kw, common):
            continue
        if len(common) > score:
            best, score = key, len(common)
    return best


def depth_of(phrase: str, depths: dict[str, float]) -> float | None:
    """Best-matching depth VALUE for an object phrase, by shared words."""
    k = depth_key_of(phrase, depths)
    return depths[k] if k is not None else None


def resolve_scale(ref_px: float, ref_m: float, ref_phrase: str, tgt_phrase: str,
                  depth_info=None, is_3d: bool = False) -> Result:
    """Metres-per-pixel for the TARGET, with an explicit depth verdict.

    We never silently assume coplanarity in a 3D scene. When the target's depth is
    unknown we still return an answer (a blank scores 0 under MRA, so a hedge always
    beats abstaining outright) but we mark the confidence down so the caller can
    widen or defer.
    """
    if ref_px <= 0 or ref_m <= 0:
        return Result.bad("reference measurement missing", "geometry")
    base = ref_m / ref_px                      # metres per pixel AT THE RULER'S DEPTH

    if not is_3d:
        return Result.good(
            Scale(base, 1.0, COPLANAR_2D, 1.0,
                  "2D scene: benchmark declares the objects coplanar"), "geometry")

    depths = parse_depth_info(depth_info)
    Zr, Zt = depth_of(ref_phrase, depths), depth_of(tgt_phrase, depths)

    if Zr and Zt:
        ratio = Zt / Zr
        # a large ratio is legitimate but worth flagging: it means the answer is
        # dominated by the depth term rather than by the pixel measurement
        conf = 1.0 if abs(ratio - 1) < 0.5 else 0.75
        return Result.good(
            Scale(base, ratio, BOTH_KNOWN, conf,
                  f"ruler @{Zr:.2f} m, target @{Zt:.2f} m (x{ratio:.3f})"), "geometry")

    if Zr and not Zt:
        return Result.good(
            Scale(base, 1.0, REF_ONLY, 0.45,
                  f"ruler @{Zr:.2f} m but TARGET DEPTH UNKNOWN — assuming coplanar; "
                  f"a 25% depth difference would be a 25% error"), "geometry")

    if Zt and not Zr:
        return Result.good(
            Scale(base, 1.0, TARGET_ONLY, 0.45,
                  f"target @{Zt:.2f} m but RULER DEPTH UNKNOWN — assuming coplanar"), "geometry")

    return Result.good(
        Scale(base, 1.0, NEITHER, 0.35,
              "3D scene but depth_info names neither object — assuming coplanar"), "geometry")


def to_metres(pixels: float, scale: Scale) -> float:
    """size_m = pixels * (m_per_px at the ruler) * (Z_target / Z_ref)."""
    return pixels * scale.m_per_px_at_ref * scale.depth_ratio
