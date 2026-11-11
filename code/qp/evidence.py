"""EVIDENCE: turn segmentation into something a VLM can reason with.

THE ARCHITECTURAL POINT. The measurement pipeline and the VLM fail at opposite
things, and measuring that is what motivated this module:

    the pipeline MEASURES well          bowling pin 0.2%, soccer ball 5.4%
    the pipeline DECIDES badly          17% of test questions it cannot even
                                        parse; 28/100 rows ground the wrong
                                        object; a bird's "width" is a WINGSPAN
                                        and no mask statistic knows that

    the VLM DECIDES well                it reads the question, knows which
                                        instance is "the orange one", knows a
                                        boat seen at an angle is foreshortened
    the VLM MEASURES badly              it cannot count pixels

So the division of labour is: SAM MEASURES, THE VLM DECIDES. This module
produces the measurements as evidence and refuses to draw the conclusion.

TWO RULES THAT MAKE IT WORK:

1. REPORT EVERY INSTANCE, NEVER PICK ONE. Choosing the longest track is how we
   measured the wrong sculpture (54% error). The VLM can see the frames and is
   far better placed to know which "ball" the question means -- but only if we
   hand it all of them, with positions, rather than a single number that has
   already thrown the alternatives away.

2. REPORT EVERY DIMENSION, NEVER JUST THE ASKED-FOR ONE. "width" resolved to a
   bounding-box width is what turned a 3.62x1.23 m boat into a 3.17 m-wide one.
   Giving bbox, principal axes and area-equivalent diameter together lets the
   model apply the semantics: a wingspan is a major axis, a thickness is a minor
   axis, and an obliquely-viewed hull is neither.

The block is deliberately compact -- it rides in every prompt, so tokens here
are paid 3,289 times.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from .pipeline.geometry import (depth_at, depth_key_of, depth_of, parse_depth_info,
                                parse_depth_series, radial_from_size,
                                radial_velocity)
from .pipeline import cache
from . import vlmplan
from .pipeline.ground import head_noun
from .pipeline.measure import slowest_stable_run
from .pipeline.plan import parse_prior, plan_by_rules
from .pipeline.solve import pixel_accel, pixel_speed


# Questions disambiguate instances the way people do -- "the ORANGE ball", "the
# LEFT tennis ball", "the LARGEST sculpture" -- so the evidence has to speak that
# language. Raw (x,y) pixel coordinates make the model do a conversion it has no
# reason to get right, and instance confusion is our single biggest error source.
# MEASURED, and it did not pay. Appearance descriptors ("orange bottom-left")
# instead of raw coordinates scored 0.4328 vs 0.4408 over 3 runs each -- a wash
# within noise, and slightly worse. S3 gained (+0.048) while S2 and D2 lost about
# as much, so n=159 cannot separate them. The likely cause is that a WRONG colour
# label misleads more than no label helps: the mask's median colour picks up
# background wherever the silhouette is loose.
# Left in place and off by default; re-enable only with a colour that is measured
# on the mask INTERIOR (eroded), which is the obvious next thing to try.
DESCRIBE_APPEARANCE = False

_COLOURS = [
    ("black", (30, 30, 30)), ("white", (235, 235, 235)), ("grey", (128, 128, 128)),
    ("red", (200, 40, 40)), ("orange", (230, 130, 30)), ("yellow", (235, 220, 60)),
    ("green", (60, 160, 70)), ("blue", (50, 90, 200)), ("purple", (130, 60, 170)),
    ("brown", (120, 80, 50)), ("pink", (235, 150, 180)),
]


def colour_name(rgb) -> str:
    r, g, b = [float(x) for x in rgb]
    mx, mn = max(r, g, b), min(r, g, b)
    if mx - mn < 25:                      # unsaturated: judge by lightness only
        return "black" if mx < 70 else ("white" if mx > 200 else "grey")
    return min(_COLOURS, key=lambda c: (r-c[1][0])**2 + (g-c[1][1])**2 + (b-c[1][2])**2)[0]


def _where(cx: float, cy: float, W: int, H: int) -> str:
    h = "left" if cx < W / 3 else ("right" if cx > 2 * W / 3 else "centre")
    v = "top" if cy < H / 3 else ("bottom" if cy > 2 * H / 3 else "middle")
    return f"{v}-{h}"


def _instances(df: pd.DataFrame, n_total: int, fps: float, max_report: int = 4,
               at_time: float | None = None, want_accel: bool = False,
               per_frame=None, frames=None, interval=None,
               z_anchor=None):
    """Per-object summary statistics, biggest track first."""
    if df is None or not len(df):
        return []
    out = []
    for oid, g in df.groupby("obj_id"):
        g = g.sort_values("f")
        if len(g) < 3:
            continue
        keep = slowest_stable_run(g["equiv"].to_numpy(float), g["f"].to_numpy(int),
                                  g["u"].to_numpy(float), g["v"].to_numpy(float))
        s = g.iloc[keep]
        t = g["f"].to_numpy(float) / max(fps, 1e-9)
        u, v = g["u"].to_numpy(float), g["v"].to_numpy(float)
        # If the question names an INTERVAL, the motion statistics must describe
        # THAT interval. A whole-clip average answers a different question: a ball
        # that rolls for 1 s of a 2.5 s clip averages far below its speed while
        # moving, and 193 test rows ask for exactly such a window.
        iv_ok = False
        if interval is not None:
            sel = (t >= interval[0]) & (t <= interval[1])
            if sel.sum() >= 2:
                ti, ui, vi, iv_ok = t[sel], u[sel], v[sel], True
        if not iv_ok:
            ti, ui, vi = t, u, v
        span = float(ti[-1] - ti[0])
        path = float(np.sum(np.hypot(np.diff(ui), np.diff(vi))))
        net = float(np.hypot(ui[-1] - ui[0], vi[-1] - vi[0]))   # straight-line
        equiv = float(s["equiv"].median())
        # An "average speed over the clip" answers a different question from
        # "speed at 1.2 s". A ball that rolls then stops has an average far below
        # its speed while moving, and most velocity questions name an instant.
        v_at = np.nan
        if at_time is not None:
            r = pixel_speed(g, fps, at_time)
            if r.ok:
                v_at = r.value.pixels
        # RADIAL motion, inferred from this instance's own apparent size.
        # Pixel tracking sees only motion ACROSS the image; an object moving
        # along the camera axis produces almost no pixel displacement, which is
        # why 3D speed measurements come out ~3x low and never high.
        v_rad_size = np.nan
        if z_anchor:
            r = radial_from_size(t, g["equiv"].to_numpy(float), z_anchor,
                                 at_time=at_time)
            if r is not None:
                v_rad_size = r
        a_px = np.nan
        if want_accel:
            r = pixel_accel(g, fps, at_time=at_time)
            if r.ok:
                a_px = r.value.pixels
        # median colour inside the mask, on one representative frame
        col = ""
        if per_frame is not None and frames is not None and len(s):
            fi = int(s["f"].iloc[len(s) // 2])
            m = (per_frame.get(fi) or {}).get(int(oid))
            if m is not None and fi < len(frames) and m.any():
                px = frames[fi][m]                       # BGR
                col = colour_name(np.median(px, axis=0)[::-1])
        out.append(dict(
            obj_id=int(oid), n=len(g), frac=len(g) / max(n_total, 1),
            v_at=v_at, a_px=a_px, colour=col, path=path, net=net, span=span,
            v_rad_size=v_rad_size,
            iv=(interval if iv_ok else None),
            cx=float(s["u"].median()), cy=float(s["v"].median()),
            bw=float(s["bw"].median()), bh=float(s["bh"].median()),
            major=float(s["major"].median()), minor=float(s["minor"].median()),
            equiv=equiv,
            spread=float(s["equiv"].std() / equiv) if equiv > 0 else np.nan,
            speed=(path / span) if span > 0 else np.nan,
        ))
    out.sort(key=lambda d: -d["n"])
    return out[:max_report]


# FULL RATE. Subsampling was measured and REJECTED: capping at 96 frames scored
# 0.4211 against 0.4408 over three runs each (-0.0197, non-overlapping ranges),
# because SAM 3 is a video TRACKER -- dropping frames changes the temporal
# dynamics and marginally-detected objects vanish (internet_0004's ruler is found
# in 53 of 428 frames at stride 1, 3 at stride 2, and 0 at stride 5).
# It also buys almost nothing here: test videos are SHORT (median 60 frames, p90
# 100), so a 96-frame cap touches only 75 of 568 videos and saves 16% of the
# encoding. The speed comes from parallelism and from not writing masks to disk.
# None = full rate, which is what quality wants (a 96-frame cap costs 0.0197
# MRA). Overridable per run with QP_EVIDENCE_MAX_FRAMES, because a few very long
# or very crowded clips push SAM's tracking state past 32 GB, and PyTorch's
# allocator then FREES CACHED BLOCKS AND RETRIES INTERNALLY rather than raising
# -- so no except clause can catch it and the worker crawls at ~1 row/min
# instead of failing fast. A cap is the only lever that actually bounds it.
MAX_FRAMES = (int(os.environ["QP_EVIDENCE_MAX_FRAMES"])
              if os.environ.get("QP_EVIDENCE_MAX_FRAMES") else None)


# VLM-in-the-loop grounding. Off by default: it costs a VLM call per question
# and the two halves are measured separately.
#   QP_GROUND_LOOP=1    ask the VLM for other names when SAM finds NOTHING
#   QP_GROUND_VERIFY=1  additionally check every found track IS the right object
AXIS_MAP_V2 = os.environ.get("QP_AXIS_MAP_V2") == "1"
# Instant-rate instruments (HANDOFF 6n tutorial): evaluate the pixel speed, the
# ruler and the radial component AT THE ASKED INSTANT instead of as averages.
# Gated k=1 on the 28 validation speed-at-instant rows: 0.675 vs 0.590 shipped.
INSTANT_RATE = os.environ.get("QP_INSTANT_RATE") == "1"
GROUND_LOOP = os.environ.get("QP_GROUND_LOOP") == "1"
GROUND_VERIFY = os.environ.get("QP_GROUND_VERIFY") == "1"
VLM_PORT = int(os.environ.get("QP_VLM_PORT", "8004"))
VLM_MODEL = os.environ.get("QP_VLM_MODEL", "qwen3vl-32b")


def _frames_b64(row, n=6):
    from qp.data import sample_for_row, encode
    try:
        return encode(sample_for_row(row, n=n), max_side=768)
    except Exception:
        return []


# Moondream boxes, precomputed by scripts/build_detections.py (separate venv).
# SAM 3 tracks well and disambiguates badly -- "left tennis ball" returns BOTH
# balls, "big bird" returns nothing. Moondream is the mirror image: it boxes the
# right one and cannot track. So SAM supplies the tracks and Moondream says
# which track the phrase means.
_DET = {}
_DET_LOADED = False


def detections():
    global _DET, _DET_LOADED
    if not _DET_LOADED:
        _DET_LOADED = True
        path = os.environ.get("QP_DETECTIONS", "")
        if path and os.path.exists(path):
            import pickle
            try:
                with open(path, "rb") as f:
                    _DET = pickle.load(f)
            except Exception:
                _DET = {}
    return _DET


def match_instance(video_id, phrase, inst):
    """Index of the instance Moondream's box points at, or None.

    Matched on centre distance normalised by the box size: an IoU would punish
    SAM and Moondream disagreeing about extent, which is exactly the thing we do
    NOT want to arbitrate here -- SAM's mask is the better extent, Moondream's
    box is only being used to say WHICH object.
    """
    d = detections()
    boxes = d.get((video_id, phrase)) or d.get((video_id, str(phrase).lower()))
    if not boxes or not inst:
        return None
    b = boxes[0]
    bcx, bcy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
    scale = max(b["x1"] - b["x0"], b["y1"] - b["y0"], 8.0)
    best, bestd = None, 1e18
    for k, i in enumerate(inst):
        dd = np.hypot(i["cx"] - bcx, i["cy"] - bcy) / scale
        if dd < bestd:
            best, bestd = k, dd
    # a box three object-widths away is not the same object
    return best if bestd <= 3.0 else None


_BOXM = {}
_BOXM_LOADED = False


def boxmasks():
    """Masks SAM produced when PROMPTED WITH MOONDREAM'S BOX rather than text.

    Identity from the detector that can read "left tennis ball"; extent from the
    segmenter that produces real silhouettes. Text-prompted SAM finds the target
    on 77% of validation rows; box-prompted it is 101/101 on the pairs Moondream
    boxed.
    """
    global _BOXM, _BOXM_LOADED
    if not _BOXM_LOADED:
        _BOXM_LOADED = True
        path = os.environ.get("QP_BOXMASKS", "")
        if path and os.path.exists(path):
            import pickle
            try:
                _BOXM = pickle.load(open(path, "rb"))
            except Exception:
                _BOXM = {}
    return _BOXM


def _thr_key():
    """SAM's detection thresholds, as part of the cache key.

    The cached VALUE depends on them, so leaving them out of the key means a
    re-run at a new threshold silently replays the old detections and the
    experiment measures nothing.
    """
    from qp.segment import SCORE_THR, NEWDET_THR
    return f"{SCORE_THR:.3f}/{NEWDET_THR:.3f}"


def segment_all(video_path: str, phrases, fps: float, max_frames=MAX_FRAMES):
    """{phrase: {df, n_sampled}} for every phrase.

    WHAT IS NOT CACHED, and why it matters more than it sounds: the per-frame
    MASKS. They were ~25 MB per (video, phrase) -- 27 GB of cache for 812
    entries -- and across the test split that is ~68 GB of pickling, with every
    worker contending for the same disk. The evidence block only ever reads the
    per-frame STATISTICS, which are already columns in the DataFrame, so the
    masks were pure write amplification. Dropping them shrinks an entry roughly
    500x.
    """
    want = [p for p in dict.fromkeys(phrases) if p and len(p) >= 3]
    out, missing = {}, []
    for ph in want:
        c = cache.get("sam3f", video=video_path, phrase=ph, fps=fps,
                      max_frames=max_frames, thr=_thr_key())
        if c is None:
            missing.append(ph)
        else:
            out[ph] = c
    if missing:
        from qp.segment import track_phrases
        res, _bgr, n_s = track_phrases(video_path, missing, fps, max_frames=max_frames)
        for ph, (df, _pf) in res.items():
            val = {"df": df, "n_sampled": n_s}          # masks deliberately dropped
            cache.put("sam3f", val, video=video_path, phrase=ph, fps=fps,
                      max_frames=max_frames, thr=_thr_key())
            out[ph] = val
    return out


_FRAMES: dict[str, list] = {}


def _video_frames(path: str):
    """Decoded frames, one video cached at a time (they are large)."""
    if path not in _FRAMES:
        from qp.segment import read_video
        _FRAMES.clear()
        try:
            _FRAMES[path] = read_video(path)
        except Exception:
            _FRAMES[path] = []
    return _FRAMES[path] or None


def look_for(video_path: str, phrase: str, fps: float, n_total: int,
             at_time: float | None = None, want_accel: bool = False,
             segs: dict | None = None, interval=None, z_anchor=None,
             fallback: str | None = None):
    """Instances for a phrase, falling back to a plainer name for it.

    `n_total` is the number of frames actually SAMPLED, not the video length --
    "seen in 60% of frames" has to be a fraction of what was looked at.

    `fallback` must be the SAME string the caller pre-segmented into `segs`.
    This used to be hard-coded to head_noun(phrase), which silently disagreed
    with the caller as soon as the fallback came from anywhere else: the
    language planner pre-segmented "light" while this looked up "woman's head",
    found neither, and grounding fell from 123 rows to 26.
    """
    segs = segs or {}
    for ph in [phrase, (fallback if fallback is not None else head_noun(phrase))]:
        if not ph or len(ph) < 3 or ph not in segs:
            continue
        v = segs[ph]
        inst = _instances(v["df"], v.get("n_sampled") or n_total, fps,
                          at_time=at_time, want_accel=want_accel,
                          per_frame=v.get("per_frame"),
                          frames=_video_frames(video_path) if DESCRIBE_APPEARANCE else None,
                          interval=interval)
        if inst:
            return ph, inst
    return phrase, []


def _reorder(video_id, phrase, ev, key) -> bool:
    """Move the Moondream-matched instance to the front. True if one matched.

    Front, because the SCALE line uses instance [0] and the block lists them in
    order -- so the disambiguation has to change what is read, not merely be
    mentioned. All instances are still reported; nothing is discarded.
    """
    inst = ev.get(key) or []
    if len(inst) < 2:
        return False
    k = match_instance(video_id, phrase, inst)
    if k is None or k == 0:
        return k == 0
    ev[key] = [inst[k]] + [x for j, x in enumerate(inst) if j != k]
    return True


def _depth_then(phrase, ser, d, t):
    """The phrase's stated distance at time t; without a t, the last one stated."""
    k = depth_key_of(phrase, ser) if ser else None
    if k is None:
        return None
    z = depth_at(ser[k], t)
    return z if z is not None else d.get(k)


def gather(row, n_total: int, frame_hw) -> dict:
    """All the measurements relevant to one question. Draws no conclusion."""
    fps = float(row["fps"]) if pd.notna(row.get("fps")) else 24.0
    video = row["video_path"]
    H, W = frame_hw
    ev = {"fps": fps, "W": W, "H": H, "n_frames": n_total,
          "is_3d": str(row.get("video_type", ""))[1:2] == "3"}

    pr = plan_by_rules(str(row["question"]), str(row["prior"]), row.get("depth_info"))
    if pr:
        p = pr.value
        ref_phrase, tgt_phrase = p.reference.phrase, p.target.phrase
        ev.update(axis=p.axis, quantity=p.quantity,
                  ref_kind=getattr(p, "reference_kind", "length"),
                  ref_value=p.reference_value, ref_axis=p.reference_axis)
    else:
        ref_phrase, _, _, _, _ = parse_prior(str(row["prior"]))
        tgt_phrase = ""
        ev.update(axis="any", quantity="size", ref_kind="length", ref_value=0.0)

    # LANGUAGE PLAN OVERRIDE (QP_VLM_PLAN=1, off by default).
    # It replaces the OBJECT, AXIS and QUANTITY parsing only. Numbers, units and
    # timestamps stay with the regex: "t=1.5, ball acceleration = 3.0 m/s^2" is
    # exactly what a regex is reliable at, there is no evidence it fails there,
    # and a model that miscopies a digit would be believed downstream.
    # A refusal falls through to the rules, so the worst case is today's
    # behaviour rather than a blank.
    ev["ref_fallback"] = head_noun(ref_phrase) if ref_phrase else ""
    ev["tgt_fallback"] = head_noun(tgt_phrase) if tgt_phrase else ""
    if vlmplan.USE:
        g = vlmplan.parse(row["question"], row.get("prior"))
        if g:
            if g["target"]:
                tgt_phrase = g["target"]
                ev["tgt_fallback"] = g["target_noun"] or head_noun(tgt_phrase)
            # THE REGEX KEEPS THE WHOLE PRIOR CLAUSE, including the object it
            # names. Measured: overriding it cost the reference on 10 of 159
            # rows and recovered it on 0, because the model copies the prior's
            # WORDING rather than naming the thing -- "walking velocity",
            # "speed of the bubble in the drop", and "gravity" where a gravity
            # prior must leave the phrase EMPTY so the reference falls back to
            # the target (HANDOFF 6d, worth 285 test rows). The prior clause is
            # where the number lives and the number stays with the rules, so the
            # object in that clause stays with them too.
            ev["quantity"] = g["quantity"]
            if g["target_axis"] and g["target_axis"] != "none":
                ev["axis"] = g["target_axis"]
            # only for a LENGTH prior: which dimension of the reference the
            # prior names. For a rate or an acceleration the unit already
            # settles it and the regex read the unit.
            if ev.get("ref_kind", "length") == "length" and \
                    g["reference_axis"] in ("length", "width", "height", "diameter",
                                            "thickness", "wingspan"):
                ev["ref_axis"] = g["reference_axis"]
            ev["plan_source"] = "vlm"

    at_time = pr.value.at_time if pr else None
    # The PRIOR's instant is not the QUESTION's instant. "t=1.5, ball
    # acceleration = 3.0 m/s^2" pins the scale to t=1.5 no matter what the
    # question asks about; measuring it at the question's instant gave two
    # scales 22x apart for the same video and prior. Falls back to the
    # question's time only when the prior names none.
    prior_t = (pr.value.prior_at_time if pr else None)
    ref_t = prior_t if prior_t is not None else at_time
    interval = ((pr.value.t_from, pr.value.t_to)
                if (pr and pr.value.t_from is not None) else None)
    # A question "wants motion" if its ANSWER is a rate or a travelled
    # distance. Note quantity=="distance" covers both "displacement of the
    # ball" (motion) and "distance between the two lamps" (a static
    # separation, which has a target_b) -- only the former qualifies.
    # Nor does a missing target_b make it motion: "the eagle's wingspan (the
    # distance from the tip ...)", "the height of the bicycle (from the handlebar
    # to the ground)" and "the basketball's distance from the camera at 1.2s" all
    # parse as distance, and depth-motion lines on a size question cost -0.124
    # over 38 rows (HANDOFF 1c). A distance is travelled only if it says so.
    q = ev.get("quantity", "size")
    travelled = re.search(r"placement\b|\btravell?ed\b|\bwalked\b|\bmoved\b"
                          r"|\bmoving distance\b", str(row["question"]), re.I)
    ev["wants_motion"] = (q in ("speed", "acceleration")
                          or (q == "distance" and bool(travelled)
                              and (pr and pr.value.target_b is None)))
    ev["at_time"] = at_time
    ev["prior_at_time"] = prior_t
    ev["ref_at_time"] = ref_t
    ev["interval"] = interval
    # every phrase we might need, including the head-noun fallbacks, segmented
    # in ONE pass -- the fallback rungs are then free rather than another
    # full traversal of the video
    cands = [ref_phrase] + ([tgt_phrase] if tgt_phrase else [])
    segs = segment_all(video, cands, fps)
    # head-noun fallbacks cost a whole extra pass each, so only pay for the ones
    # actually needed: 2,130 passes instead of 2,731 across the test split
    fb = {ref_phrase: ev.get("ref_fallback") or head_noun(ref_phrase)}
    if tgt_phrase:
        fb[tgt_phrase] = ev.get("tgt_fallback") or head_noun(tgt_phrase)
    need = [fb.get(x) or head_noun(x) for x in cands
            if x in segs and not len(segs[x]["df"])]
    need = [x for x in need if x and x not in segs]
    if need:
        segs.update(segment_all(video, need, fps))
    ev["n_sampled"] = next((v.get("n_sampled") for v in segs.values()
                            if v.get("n_sampled")), n_total)

    # Depth is parsed BEFORE grounding because the size-based radial estimate
    # needs a distance anchor while the instances are being built. Matched on
    # the phrase the QUESTION used, since look_for may fall back to a head noun
    # and a head noun is only ever a subset of those words.
    d = parse_depth_info(row.get("depth_info"))
    z_ref0 = depth_of(ref_phrase, d) if d else None
    z_tgt0 = depth_of(tgt_phrase, d) if (d and tgt_phrase) else None
    anchor_ref = z_ref0 if ev["is_3d"] else None
    anchor_tgt = z_tgt0 if ev["is_3d"] else None

    ev["ref_phrase"], ev["ref_inst"] = look_for(
        video, ref_phrase, fps, ev["n_sampled"], ref_t,
        want_accel=(ev.get("ref_kind") == "accel"), segs=segs,
        z_anchor=anchor_ref, fallback=ev.get("ref_fallback"))
    ev["ref_boxes"] = detections().get((row.get("video_id"), ref_phrase)) or []
    ev["ref_bm"] = boxmasks().get((row.get("video_id"), ref_phrase)) or []
    if tgt_phrase:
        ev["tgt_phrase"], ev["tgt_inst"] = look_for(
            video, tgt_phrase, fps, ev["n_sampled"], at_time,
            want_accel=(ev.get("quantity") == "acceleration"), segs=segs,
            interval=interval, z_anchor=anchor_tgt,
            fallback=ev.get("tgt_fallback"))
        ev["tgt_boxes"] = detections().get((row.get("video_id"), tgt_phrase)) or []
        ev["tgt_bm"] = boxmasks().get((row.get("video_id"), tgt_phrase)) or []
    else:
        ev["tgt_phrase"], ev["tgt_inst"] = "", []
        ev["tgt_boxes"] = []
        ev["tgt_bm"] = []

    # ---- VLM in the loop ----
    # Only reached when the cheap ladder has already failed, so the VLM sees the
    # residual rather than every question.
    ev["loop"] = ""
    if (GROUND_LOOP or GROUND_VERIFY) and tgt_phrase:
        need_loop = (not ev["tgt_inst"]) if GROUND_LOOP else False
        if need_loop or GROUND_VERIFY:
            from qp.vlmground import ground_loop
            imgs = _frames_b64(row)
            if imgs:
                def seg(ph, _v=video, _f=fps):
                    return segment_all(_v, [ph], _f).get(ph, {}).get("df")
                used, df, trail = ground_loop(
                    video, str(row["question"]), tgt_phrase, fps, imgs, seg,
                    tried=[tgt_phrase, head_noun(tgt_phrase)] if not need_loop else None,
                    do_verify=GROUND_VERIFY,
                    model=VLM_MODEL, port=VLM_PORT)
                ev["loop"] = " | ".join(trail)
                if df is not None and len(df):
                    inst = _instances(df, ev["n_sampled"], fps, at_time=at_time,
                                      want_accel=(ev.get("quantity") == "acceleration"),
                                      interval=interval, z_anchor=anchor_tgt)
                    if inst:
                        ev["tgt_phrase"], ev["tgt_inst"] = used, inst

    # Each distance AT THE MOMENT ITS PIXELS WERE MEASURED. A speed or an
    # acceleration ruler is read at the prior's instant; the target at the asked
    # instant, or mid-way through the asked interval. With no instant, the last
    # stated distance, as before. parse_depth_info alone keeps only the last
    # sample, and a ruler read at 0.5s was paired with its distance at 1.0s.
    ser = parse_depth_series(row.get("depth_info")) if d else {}
    t_ref = ref_t if ev.get("ref_kind") in ("rate", "accel") else None
    t_tgt = at_time if at_time is not None else (sum(interval) / 2 if interval else None)
    ev["Z_ref"] = _depth_then(ev["ref_phrase"], ser, d, t_ref)
    ev["Z_tgt"] = _depth_then(ev["tgt_phrase"], ser, d, t_tgt)
    ev["depth_keys"] = sorted(d) if d else []

    # ---- radial motion, 3D scenes only ----
    # 2D scenes declare the objects coplanar, so there is no depth motion by
    # construction -- and measured, our 2D speeds are already at 0.995 of truth
    # while 3D sits at 0.326. This whole block is the 3D correction.
    ev["v_rad_tgt"] = ev["v_rad_ref"] = None
    if ev["is_3d"] and d:
        ser = parse_depth_series(row.get("depth_info"))
        for role, phrase in (("tgt", ev["tgt_phrase"]), ("ref", ev["ref_phrase"])):
            key = depth_key_of(phrase, ser)
            stated = radial_velocity(ser[key], at_time=at_time) if key else None

            # depth_info gives only two samples, so radial_velocity can only
            # return a CONSTANT average over the clip. The true rate varies:
            # the same ping-pong ball at 1.4s wanted across-only while at
            # another instant it wanted the full combination -- a
            # time-resolution problem, not a convention one. The size-based
            # estimator IS time-resolved (Z = C/s(t), so dZ/dt varies with t),
            # so for a question naming an INSTANT prefer it, anchored on the
            # stated distance and sanity-checked against the stated average.
            timed = np.nan
            if at_time is not None:
                for i in (ev.get(f"{role}_inst") or []):
                    r = i.get("v_rad_size", np.nan)
                    if np.isfinite(r):
                        timed = r
                        break

            pick = None
            if np.isfinite(timed) and at_time is not None:
                agrees = (stated is None or abs(stated) < 1e-6
                          or 0.25 <= abs(timed / stated) <= 4.0)
                sign_ok = (stated is None or abs(stated) < 1e-6
                           or (timed * stated) > 0)
                if agrees and sign_ok:
                    pick = (float(timed),
                            f"from its apparent size at {at_time:g}s, "
                            f"anchored on depth_info")
            if pick is None and stated is not None:
                pick = (stated, "stated in depth_info, averaged over the clip")
            if pick is not None:
                ev[f"v_rad_{role}"] = pick

    if (INSTANT_RATE and at_time is not None
            and ev.get("quantity") == "speed"):
        _instant_upgrade(ev, segs, row, fps, at_time)
    return ev


def _instant_upgrade(ev, segs, row, fps, at_time):
    """Swap the three averaged numbers for instant ones, on speed-at-t rows.

    Scope is exactly what the k=1 gate measured -- the TARGET's v_at, the
    TARGET's radial, and the self-calibrating ruler. The reference side of a
    rate-prior scale is untouched: it was not gated, and an ungated change is
    how five ideas died.
    """
    from .pipeline.instant import speed_at, size_at, radial_at, _series
    ph = ev.get("tgt_phrase")
    seg = segs.get(ph) or {}
    df = seg.get("df")
    if df is None or not len(df) or not ev.get("tgt_inst"):
        return
    tracks = {int(o): g for o, g in df.groupby("obj_id")}

    # blindness refusal, 3D only: a strong size trend with no depth anchor for
    # the target means the radial component exists and cannot be measured; the
    # across value alone would be a confidently wrong answer. Keep the old lines.
    ser = parse_depth_series(row.get("depth_info")) if ev.get("is_3d") else {}
    key = depth_key_of(ph, ser) if ser else None
    g0 = tracks.get(ev["tgt_inst"][0]["obj_id"])
    if ev.get("is_3d") and key is None and g0 is not None and len(g0) >= 3:
        t_, _, _, s_ = _series(g0, fps)
        m_ = np.abs(t_ - at_time) <= 0.35
        if m_.sum() >= 3:
            trend = (s_[m_].max() - s_[m_].min()) / max(float(s_[m_].mean()), 1e-9)
            if trend > 0.15:
                return

    for inst in ev["tgt_inst"]:
        g = tracks.get(inst["obj_id"])
        if g is None:
            continue
        sp = speed_at(g, fps, at_time)
        if sp is not None:
            inst["v_at"] = sp[0]

    if key:
        samples = [(t0, z) for t0, z in ser[key] if t0 is not None]
        if samples and g0 is not None:
            t_a, z_a = min(samples, key=lambda p: abs(p[0] - at_time))
            vr = radial_at(g0, fps, at_time, z_anchor_t=t_a, z_anchor=z_a)
            # the same sanity check gather applies to its size-based rate: within
            # 4x of the stated depth change and the same sign, else keep the
            # stated value gather already set
            stated = radial_velocity(ser[key], at_time=at_time)
            if vr is not None and not (stated is None or abs(stated) < 1e-6
                                       or (0.25 <= abs(vr / stated) <= 4.0
                                           and -vr * stated > 0)):
                vr = None
            if vr is not None:
                # v_rad convention: positive = AWAY; radial_at: positive = approaching
                ev["v_rad_tgt"] = (float(-vr),
                                   f"from its apparent size at {at_time:g}s, "
                                   f"anchored on depth_info")

    # self-calibrating ruler: the prior names THIS object's own length, so the
    # scale belongs at the asked instant, on the axis the prior names.
    if (ev.get("ref_kind") == "length"
            and str(ev.get("ref_phrase", "")).lower() == str(ph).lower()):
        stat = {"height": "bh", "width": "bw", "length": "major",
                "diameter": "equiv"}.get(ev.get("ref_axis", "any"), "equiv")
        sz = size_at(g0, fps, at_time, stat=stat) if g0 is not None else None
        if sz and sz > 2:
            ev["ref_px_instant"] = float(sz)


def _fmt_inst(i: dict, tag: str, at_time=None, W=None, H=None) -> str:
    m = []
    if np.isfinite(i.get("speed", np.nan)):
        m.append(f"avg speed {i['speed']:.0f} px/s")
    if at_time is not None and np.isfinite(i.get("v_at", np.nan)):
        m.append(f"speed at {at_time:g}s = {i['v_at']:.0f} px/s")
    if np.isfinite(i.get("a_px", np.nan)):
        m.append(f"accel {i['a_px']:.0f} px/s^2")
    # HOW FAR IT MOVED, which is a different question from how big it is.
    # Without these the model has no way to answer "what is the displacement of
    # the toy car" except by guessing, and the programmatic path answered such
    # questions with the car's own SIZE.
    if np.isfinite(i.get("path", np.nan)) and i.get("path", 0) > 1:
        iv = i.get("iv")
        when = (f"between {iv[0]:g}s and {iv[1]:g}s" if iv
                else f"over {i['span']:.2f}s")
        m.append(f"travelled {i['path']:.0f} px total, {i['net']:.0f} px net {when}")
    m.append(f"seen in {100*i['frac']:.0f}% of frames")
    who = []
    if DESCRIBE_APPEARANCE:
        if i.get("colour"):
            who.append(i["colour"])
        if W and H:
            who.append(_where(i["cx"], i["cy"], W, H))
    desc = " ".join(who) + " " if who else ""
    return (f"    {tag} {desc}at ({i['cx']:.0f},{i['cy']:.0f}): "
            f"box {i['bw']:.0f}x{i['bh']:.0f}, long axis {i['major']:.0f}, "
            f"short axis {i['minor']:.0f}, area-equiv diameter {i['equiv']:.1f} px"
            f" [{', '.join(m)}]")


def _scale_lines(ev: dict) -> list[str]:
    """The SCALE and DEPTH statements. Kept separate from the instance list so
    they can never be the part that gets truncated -- they were, and the block
    ended '...so 1 px = 0.0010137 m AT THE REFERE'."""
    L = []
    kind = ev.get("ref_kind", "length")
    inst = ev.get("ref_inst") or []
    val = ev.get("ref_value", 0) or 0
    if not inst or val <= 0:
        return L
    which, multi = None, []

    if kind == "length":
        r = inst[0]
        if ev.get("ref_px_instant"):
            px = ev["ref_px_instant"]
            mpp = val / px
            ev["mpp"] = mpp
            L.append(f"  SCALE [HIGH confidence (from a directly measured size)]: "
                     f"the prior gives this object's {ev.get('ref_axis','size')} "
                     f"as {val:.4g} m and it measures {px:.1f} px AT THE ASKED INSTANT "
                     f"(it moves in depth, so the ruler is taken at that moment), "
                     f"so 1 px = {mpp:.5g} m THERE.")
            return L
        # AXIS -> STATISTIC. Audited against ground truth on 55 validation size
        # questions by a test that needs no scale at all: on one video the RATIO
        # of two true sizes must equal the ratio of their pixel measurements.
        # Our map agrees on 45% of questions; the best statistic of the SAME mask
        # agrees on 56%, so a tenth of the error is reading the wrong number off
        # a correct outline. An exhaustive search over all 625 maps picks two
        # changes, and both have a mechanism rather than just a fit:
        #   width    -> equiv, not bw.    A bounding box's x-extent mixes in the
        #                                 object's LENGTH whenever it is oblique.
        #   diameter -> minor, not equiv. Area-equivalent diameter equals the
        #                                 diameter only for a circle; for a tube
        #                                 it is 3x too big (plunge tube: equiv
        #                                 141 px, true diameter 47 px).
        # 45->56% overall, 55->64% on 2D. Off by default, not yet A/B'd.
        wide = "equiv" if AXIS_MAP_V2 else "bw"
        dia = "minor" if AXIS_MAP_V2 else "equiv"
        stat = {"height": "bh", "width": wide, "length": "major",
                "diameter": dia}.get(ev.get("ref_axis", "any"), "equiv")
        px = r[stat]
        what = f"{ev.get('ref_axis','size')}"
        # WHICH INSTANCE IS THE RULER. Instance #1 is only the longest track, and
        # the block never said it was the one used: on captured_0032 "tennis ball"
        # #1 is the 18.8 px ping-pong ball (the target "white ball" claims the same
        # mask) and the 31.9 px tennis ball is #2, so a HIGH-confidence 6.7 cm
        # ruler sat on a 4 cm ball. Name it; and when persistent instances (the
        # persistence guard below) disagree by more than 25%, give one scale per
        # instance rather than pick -- rule 1 of this module.
        per = [i for i in inst if i.get("frac", 0) >= 0.25 and i[stat] > 0]
        if (len(per) > 1 and r in per and px > 0
                and max(abs(np.log(i[stat] / px)) for i in per) > np.log(1.25)):
            multi = [(k, i) for k, i in enumerate(inst, 1) if i in per]
            which = (f"  (the reference phrase matched {len(multi)} objects whose sizes "
                     f"differ by more than 25%, so there is one scale per instance -- use "
                     f"the one that is the object the prior describes)")
        elif len(inst) > 1:
            which = (f"  (the scale below uses reference instance #1 at "
                     f"({r['cx']:.0f},{r['cy']:.0f}))")
    else:
        key = "a_px" if kind == "accel" else "v_at"
        movers = [i for i in inst if np.isfinite(i.get(key, np.nan)) and i.get(key, 0) > 0]
        if not movers and kind != "accel":
            movers = [i for i in inst if np.isfinite(i.get("speed", np.nan))]
            key = "speed"
        if not movers:
            L.append(f"  NOTE: the prior describes the reference's "
                     f"{'acceleration' if kind == 'accel' else 'speed'}, but that "
                     f"could not be measured reliably, so no scale is offered.")
            return L
        # persistence guard: a 4-frame flicker with a huge apparent speed must
        # not be allowed to calibrate the entire answer
        solid = [i for i in movers if i.get("frac", 0) >= 0.25] or movers
        r = max(solid, key=lambda i: i[key])
        px = r[key]
        what = "acceleration" if kind == "accel" else "speed"
        if len(inst) > 1:
            L.append(f"  (the prior describes motion, so the scale below uses the "
                     f"MOVING instance at ({r['cx']:.0f},{r['cy']:.0f}))")

    if not (np.isfinite(px) and px > 0):
        return L
    mpp = val / px
    ev["mpp"] = mpp
    unit = {"length": "m", "rate": "m/s", "accel": "m/s^2"}[kind]
    at = ev.get("ref_at_time")
    when = f" at {at:g}s" if (at is not None and kind != "length") else ""
    px_unit = {"length": "px", "rate": "px/s", "accel": "px/s^2"}[kind]
    # CONFIDENCE, stamped by HOW the ruler was made -- the one fact the model
    # cannot see from the numbers. Measured (HANDOFF 6h + the D3 dissection):
    # size-derived scales are usually right; speed-derived drift; acceleration-
    # derived were 3-6x off on three of four audited videos. The thinking model
    # executes whatever scale it is handed with perfect arithmetic, so the
    # trust decision must ride WITH the scale. This tag replaced a prompt rule
    # keyed on "simulation scenes", which backfired: D3 simulations are exactly
    # where motion-derived scales are worst, and that rule told the model to
    # trust them there.
    conf = {"length": "HIGH confidence (from a directly measured size)",
            "rate": "MEDIUM confidence (derived from a pixel speed; can drift ~1.5-2x)",
            "accel": "LOW confidence (derived from a fitted pixel acceleration; "
                     "typical error 2-3x, sometimes worse)"}[kind]
    if which:
        L.append(which)
    if multi:
        # the first line in full, the rest short: four full lines crowded the
        # instance list out of the block (max_chars)
        ev["mpp_by_ref"] = [(k, val / i[stat]) for k, i in multi]
        for n, ((k, i), (_, m)) in enumerate(zip(multi, ev["mpp_by_ref"])):
            head = (f"{conf} IF #{k} at ({i['cx']:.0f},{i['cy']:.0f}) is that object]: "
                    f"the prior gives the reference's {what} as {val:.4g} {unit} and it"
                    if n == 0 else
                    f"same IF #{k} at ({i['cx']:.0f},{i['cy']:.0f}) is that object]: it")
            L.append(f"  SCALE [{head} measures {i[stat]:.1f} {px_unit}, so 1 px = {m:.5g} m "
                     f"AT THE REFERENCE'S DEPTH.")
    else:
        L.append(f"  SCALE [{conf}]: the prior gives the reference's {what}{when} as {val:.4g} {unit} "
                 f"and it measures {px:.1f} {px_unit}, so 1 px = {mpp:.5g} m "
                 f"AT THE REFERENCE'S DEPTH.")

    if ev.get("is_3d"):
        Zr, Zt = ev.get("Z_ref"), ev.get("Z_tgt")
        if Zr and Zt:
            k = Zt / Zr
            # A dataset typo once produced a target 8420 m away in a tabletop
            # clip -- "multiply by 12202.90". State the distances, but do not
            # assert a correction the scene cannot support.
            if 0.05 <= k <= 20:
                ev["depth_ratio"] = k
                L.append(f"  DEPTH: reference {Zr:.2f} m away, target {Zt:.2f} m away "
                         f"-> multiply the target's pixel size by {k:.2f}.")
            else:
                L.append(f"  DEPTH: the given distances ({Zr:.2f} m and {Zt:.2f} m) imply a "
                         f"{k:.0f}x correction, which is implausible for this scene -- "
                         f"treat the depth information as unreliable here.")
        else:
            L.append("  DEPTH: 3D scene and the two depths are not both given, so the "
                     "pixel scale above may not apply to the target.")
    return L


def _depth_motion_lines(ev: dict) -> list[str]:
    """State motion ALONG the camera axis, and DO the combination.

    Pixels cannot show motion along the camera axis, which is why 3D speeds
    measure 0.326 of truth while 2D speeds measure 0.995 -- always low, the
    signature of a missing component.

    We combine the two components ourselves rather than describing the formula.
    Measured reason: given the components and sqrt(a^2+b^2) in words, the model
    sometimes answered with the DEPTH COMPONENT ALONE -- a ping-pong ball went
    from 78 (truth 87.9) to 12. A sum of squares can only ever increase a value,
    so 78 -> 12 was not a combination at all. Combining is arithmetic, not
    judgement, so it belongs here next to the scale conversion; WHICH number the
    question wants is still the model's call.

    Gated on the QUESTION, not the scene: the same lines on SIZE questions cost
    -0.124 MRA over 38 rows (one "width of the purple car" went from 1.81
    against a truth of 1.86 to 67). Irrelevant information is not neutral.
    """
    if not (ev.get("is_3d") and ev.get("wants_motion")):
        return []
    rv = None
    stated = ev.get("v_rad_tgt")
    if stated:
        rv, src = stated
    else:
        for i in (ev.get("tgt_inst") or []):
            r = i.get("v_rad_size", np.nan)
            if np.isfinite(r) and abs(r) > 1e-3:
                rv, src = r, "inferred from its changing apparent size"
                break
    if rv is None or abs(rv) < 1e-3:
        return []

    way = "AWAY FROM" if rv > 0 else "TOWARD"
    out = [f"  DEPTH MOTION: the asked-about object is ALSO moving "
           f"{abs(rv):.3g} m/s {way} the camera ({src}). Pixel speeds measure "
           f"only motion ACROSS the image, so they MISS this entirely."]

    # combine, per instance, where we can convert px/s to m/s
    mpp = ev.get("mpp")
    if mpp:
        # one row per scale when _scale_lines gave one per reference instance:
        # a single combination would quietly re-pick the ruler it declined to pick
        scales = ev.get("mpp_by_ref") or [(None, mpp)]
        rows = []
        for k, i in enumerate(ev.get("tgt_inst") or [], 1):
            v_px = i.get("v_at", np.nan)
            if not np.isfinite(v_px):
                v_px = i.get("speed", np.nan)
            if not np.isfinite(v_px):
                continue
            for ref_k, m in scales:
                across = v_px * m * (ev.get("depth_ratio") or 1.0)
                total = float(np.hypot(across, rv))
                tag = f"#{k}" if ref_k is None else f"#{k} (scale of reference #{ref_k})"
                rows.append(f"    {tag}: across {across:.3g} m/s, depth {abs(rv):.3g} m/s"
                            f"  ->  FULL 3-D SPEED {total:.3g} m/s")
        if rows:
            out.append("  COMBINED SPEED (across and depth together):")
            out.extend(rows[:3 * len(scales)])
    return out


def _second_opinion(ev: dict) -> list[str]:
    """A different detector's boxes for the same phrases.

    Reported, never arbitrated. The two disagree usefully: SAM 3 tracks across
    frames and cannot disambiguate ("left tennis ball" returns BOTH balls, "big
    bird" returns nothing at all), while this one boxes the right object and
    cannot track. Measured on the same clips -- "big bird" 0 vs 1 (the big one),
    "left tennis ball" both vs the one at x=173, "pier" 0 vs 1.

    Deciding between them here would be exactly the mistake this architecture
    exists to avoid, and every attempt to arbitrate automatically has cost
    score. So both go in the block and the model picks.
    """
    # ONLY for static extents. This detector has no tracking, so on a motion
    # question its boxes add nothing and cost budget -- and measured, they cost
    # score: S2 (static prior, 2D) went 0.545 -> 0.569 while every motion-bearing
    # category fell. Report it where it is meaningful and stay silent elsewhere.
    if ev.get("wants_motion") or ev.get("quantity") in ("speed", "acceleration"):
        return []
    out = []
    for key, label in (("tgt_bm", "asked-about object"), ("ref_bm", "reference object")):
        st = ev.get(key) or []
        if not st:
            continue
        # median across frames, same convention as the tracked measurements
        med = {k: float(np.median([x[k] for x in st]))
               for k in ("bw", "bh", "major", "minor", "equiv")}
        out.append(f"  SECOND OPINION, {label} (a detector that reads the phrase "
                   f"literally -- it can tell 'left' from 'right' and 'big' from "
                   f"'small' -- then segmented): box {med['bw']:.0f}x{med['bh']:.0f}, "
                   f"long axis {med['major']:.0f}, short axis {med['minor']:.0f}, "
                   f"area-equiv diameter {med['equiv']:.1f} px "
                   f"(over {len(st)} frames)")
    if out:
        out.append("  Where the two disagree, this one is more likely to have the "
                   "RIGHT OBJECT; the tracked measurements above cover more frames.")
    return out


def render(ev: dict, max_chars: int = 2300) -> str:
    """The evidence block that goes into the prompt.

    Assembled so that the SCALE and DEPTH lines survive truncation: the instance
    list is what gets shortened, because a missing instance costs the model one
    option while a half-written scale actively misinforms it.
    """
    head = [f"PIXEL MEASUREMENTS from automatic segmentation "
            f"(frame {ev['W']}x{ev['H']}, {ev['fps']:.0f} fps):"]

    ref_hdr, ref_lines = None, []
    if ev["ref_inst"]:
        ref_hdr = f"  Reference object \"{ev['ref_phrase']}\" (what the given prior describes):"
        ref_lines = [_fmt_inst(i, f"#{k}", ev.get("ref_at_time"), ev["W"], ev["H"])
                     for k, i in enumerate(ev["ref_inst"], 1)]
    else:
        ref_hdr = f"  Reference object \"{ev['ref_phrase']}\": NOT FOUND by segmentation."

    tgt_hdr, tgt_lines = None, []
    if ev["tgt_phrase"]:
        if ev["tgt_inst"]:
            n = len(ev["tgt_inst"])
            tgt_hdr = (f"  Asked-about object \"{ev['tgt_phrase']}\": "
                       f"{n} instance{'s' if n > 1 else ''} found"
                       + (" - decide from the frames which one the question means:"
                          if n > 1 else ":"))
            tgt_lines = [_fmt_inst(i, f"#{k}", ev.get("at_time"), ev["W"], ev["H"])
                         for k, i in enumerate(ev["tgt_inst"], 1)]
        else:
            tgt_hdr = f"  Asked-about object \"{ev['tgt_phrase']}\": NOT FOUND by segmentation."

    tail = _scale_lines(ev) + _depth_motion_lines(ev) + _second_opinion(ev)
    tail.append("  These pixel numbers are measured, not estimated. The object identity, "
                "which instance, and which dimension the question means are for you to judge.")

    # Instances are taken TARGET FIRST, alternating, so the object the question
    # actually asks about can never be the part that gets dropped. On a 1080p
    # clip with four large reference instances, a naive in-order fill emitted
    # "(further instances omitted)" before a single target instance appeared.
    # The "(further instances omitted)" note was never budgeted, so a block
    # could end up to 31 chars over max_chars; refill with room for it if so.
    omit = "    (further instances omitted)"
    for budget in (max_chars, max_chars - len(omit) - 1):
        used = len("\n".join(head + [x for x in (ref_hdr, tgt_hdr) if x] + tail)) + 1
        keep_t, keep_r = [], []
        dropped = False
        for k in range(max(len(tgt_lines), len(ref_lines))):
            for src, dst in ((tgt_lines, keep_t), (ref_lines, keep_r)):
                if k < len(src):
                    if used + len(src[k]) + 1 <= budget:
                        dst.append(src[k]); used += len(src[k]) + 1
                    else:
                        dropped = True
        body = []
        if ref_hdr:
            body.append(ref_hdr); body += keep_r
        if tgt_hdr:
            body.append(tgt_hdr); body += keep_t
        if dropped:
            body.append(omit)
        out = "\n".join(head + body + tail)
        if len(out) <= max_chars:
            break
    return out
