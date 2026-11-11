"""RUN stage: one dataset row -> one Answer, with a complete audit trail.

This is the only module that knows the pipeline has an ORDER. Every other stage
is independently testable because it never imports a sibling; RUN is where they
are composed, and it is deliberately the thinnest interesting file in the
package -- if orchestration starts growing logic, that logic belongs in a stage.

WHAT RUN IS RESPONSIBLE FOR, that no single stage can be:

  * choosing WHICH INSTANCE when grounding returns several. SAM tracks every
    "sculpture" in the clip; the question means one of them. This has cost us
    54% on a single item, so the choice is made explicitly, recorded, and
    counted -- never left as an implicit "take the biggest".

  * deciding whether the reference should be measured as a SIZE or as a SPEED.
    That depends on the prior, which only the plan knows, and on the track,
    which only grounding produces.

  * carrying provenance. Every number in the output can be traced to the
    strategy that produced it and the frames it was measured on. This is not
    bookkeeping for its own sake: the confidence gate is built entirely out of
    these fields, and a field nobody records is a gate nobody can build.

RUN NEVER DECIDES WHETHER TO TRUST THE ANSWER. It reports what happened; gate.py
turns that into an accept/reject. Keeping the two apart means the trust policy
can be retuned against validation without touching perception.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd

from .geometry import resolve_scale
from .ground import ground_one
from .measure import measure
from .plan import plan_by_rules, validate_plan
from .solve import Observable, pixel_accel, pixel_separation, pixel_speed, solve
from .types import Answer, Plan, Result, Target


# ─────────────────────── instance selection ───────────────────────
# Positional words in the question that name WHICH instance is meant.
_HINTS = [
    (r"\bleft(?:most|-hand)?\b|\bleft side\b", "leftmost"),
    (r"\bright(?:most|-hand)?\b|\bright side\b", "rightmost"),
    (r"\b(top|upper|highest)\b", "topmost"),
    (r"\b(bottom|lower|lowest)\b", "bottommost"),
    (r"\b(nearest|closest|front)\b", "largest"),
    (r"\b(farthest|furthest|far)\b", "smallest"),
    (r"\b(central|middle|centre|center)\b", "central"),
]


def instance_hint(phrase: str, question: str = "") -> str | None:
    """A positional word that disambiguates which instance is meant, if any."""
    text = f"{phrase} {question}"
    for pat, rule in _HINTS:
        if re.search(pat, text, re.I):
            return rule
    return None


def pick_instance(df: pd.DataFrame, hint: str | None, frame_w: int, frame_h: int):
    """(obj_id, n_instances, how). Longest track unless the phrase says otherwise.

    'Longest track' is the right DEFAULT (a real object persists; a spurious
    detection flickers) but it is a wrong ANSWER whenever the clip genuinely
    contains several of the thing. Returning n_instances alongside is what lets
    the gate treat those two situations differently.
    """
    if not len(df):
        return None, 0, "empty"
    counts = df.groupby("obj_id").size()
    n = int(len(counts))
    # only instances that persist are real candidates
    solid = counts[counts >= max(3, 0.25 * counts.max())]
    cands = list(solid.index) if len(solid) else list(counts.index)
    if hint is None or len(cands) == 1:
        return int(counts.idxmax()), n, "longest-track"
    g = df[df.obj_id.isin(cands)].groupby("obj_id")
    u, v, area = g.u.median(), g.v.median(), g.area.median()
    chooser = {
        "leftmost": lambda: u.idxmin(), "rightmost": lambda: u.idxmax(),
        "topmost": lambda: v.idxmin(), "bottommost": lambda: v.idxmax(),
        "largest": lambda: area.idxmax(), "smallest": lambda: area.idxmin(),
        "central": lambda: (np.hypot(u - frame_w / 2, v - frame_h / 2)).idxmin(),
    }.get(hint)
    if chooser is None:
        return int(counts.idxmax()), n, "longest-track"
    return int(chooser()), n, f"hint:{hint}"


# ─────────────────────── outcome record ───────────────────────
@dataclass
class Outcome:
    """Everything that happened, whether or not it worked."""
    ok: bool
    value: float = float("nan")
    unit: str = "m"
    confidence: float = 0.0
    stage_failed: str = ""
    reason: str = ""
    plan: Plan | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    trail: dict[str, list] = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {"ok": self.ok, "value": self.value, "unit": self.unit,
             "confidence": self.confidence, "stage_failed": self.stage_failed,
             "reason": self.reason}
        d.update(self.facts)
        return d


def _video_shape(path: str):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return n, (h, w)


# which acceleration a prior like '3.0 m/s^2' refers to. Fitted, not guessed --
# see scripts/fit_accel_prior.py.
ACCEL_MODE = "magnitude"

_FRAME_CACHE: dict[str, list] = {}


def _frames(path: str, enable: bool):
    """Decoded frames, memoised. Only for the gradient measurement, which is the
    only consumer that needs pixels rather than masks."""
    if not enable:
        return None
    if path not in _FRAME_CACHE:
        from qp.segment import read_video
        _FRAME_CACHE.clear()          # one video's frames at a time; these are large
        _FRAME_CACHE[path] = read_video(path)
    return _FRAME_CACHE[path]


def run_one(row, use_gradient: bool = False, verbose: bool = False) -> Outcome:
    """PLAN -> GROUND -> MEASURE -> SOLVE for a single question row."""
    video, question, prior = row["video_path"], str(row["question"]), str(row["prior"])
    fps = float(row["fps"]) if pd.notna(row.get("fps")) else 24.0
    is_3d = str(row.get("video_type", ""))[1:2] == "3"
    trail: dict[str, list] = {}
    facts: dict[str, Any] = {"video_id": row.get("video_id"), "fps": fps, "is_3d": is_3d}

    # ── PLAN ──────────────────────────────────────────────────────
    pr = plan_by_rules(question, prior, row.get("depth_info"))
    if not pr:
        return Outcome(False, stage_failed="plan", reason=pr.error, facts=facts)
    plan: Plan = pr.value
    why = validate_plan(plan)
    if why:
        return Outcome(False, stage_failed="plan", reason=why, plan=plan, facts=facts)
    facts.update(quantity=plan.quantity, axis=plan.axis, unit=plan.unit,
                 ref_phrase=plan.reference.phrase, tgt_phrase=plan.target.phrase,
                 prior_si=plan.reference_value, ref_is_motion=plan.reference_is_motion)

    n_frames, (H, W) = _video_shape(video)

    # ── GROUND ────────────────────────────────────────────────────
    def ground(t: Target, axis: str):
        # the axis matters: it decides which frame borders can invalidate the mask
        r, tr = ground_one(video, t, fps, n_frames, (H, W), axis)
        trail[f"ground:{t.role}"] = tr
        return r

    # a motion prior is read off centroids, so extents (and their truncation)
    # do not constrain it
    ref_axis = "separation" if plan.reference_is_motion else plan.reference_axis
    tgt_axis = "separation" if plan.quantity in ("speed", "acceleration") else plan.axis

    gr = ground(plan.reference, ref_axis)
    if not gr:
        return Outcome(False, stage_failed="ground", reason=f"reference: {gr.error}",
                       plan=plan, facts=facts, trail=trail)
    gt_ = ground(plan.target, tgt_axis)
    if not gt_:
        return Outcome(False, stage_failed="ground", reason=f"target: {gt_.error}",
                       plan=plan, facts=facts, trail=trail)

    ref_df_all, tgt_df_all = gr.value["df"], gt_.value["df"]
    ref_id, ref_n, ref_how = pick_instance(
        ref_df_all, instance_hint(plan.reference.phrase), W, H)
    tgt_id, tgt_n, tgt_how = pick_instance(
        tgt_df_all, instance_hint(plan.target.phrase, question), W, H)
    ref_df = ref_df_all[ref_df_all.obj_id == ref_id]
    tgt_df = tgt_df_all[tgt_df_all.obj_id == tgt_id]
    facts.update(ref_instances=ref_n, tgt_instances=tgt_n,
                 ref_pick=ref_how, tgt_pick=tgt_how,
                 ref_track_len=len(ref_df), tgt_track_len=len(tgt_df),
                 ref_strategy=gr.strategy, tgt_strategy=gt_.strategy)

    frames_bgr = _frames(video, use_gradient)

    # ── MEASURE the reference ─────────────────────────────────────
    # A speed prior calibrates against the reference's MOTION; a size prior
    # against its EXTENT. Same object, different observable.
    ref_kind = getattr(plan, "reference_kind", None) or (
        "rate" if plan.reference_is_motion else "length")
    facts["ref_kind"] = ref_kind
    if ref_kind == "accel":
        # 'ball acceleration = 3.0 m/s^2' calibrates against px/s^2, NOT px/s.
        # Dividing m/s^2 by px/s produced 115,000% errors before this split.
        rr = pixel_accel(ref_df, fps, mode=ACCEL_MODE, at_time=plan.at_time)
        if not rr:
            return Outcome(False, stage_failed="measure", reason=f"reference accel: {rr.error}",
                           plan=plan, facts=facts, trail=trail)
        ref_obs = rr.value
    elif ref_kind == "rate":
        rr = pixel_speed(ref_df, fps, plan.at_time,
                         average=bool(row.get("wants_average", False)))
        if not rr:
            return Outcome(False, stage_failed="measure", reason=f"reference speed: {rr.error}",
                           plan=plan, facts=facts, trail=trail)
        ref_obs = rr.value
    else:
        mr, tr = measure(ref_df, plan.reference_axis, frames_bgr)
        trail["measure:reference"] = tr
        if not mr:
            return Outcome(False, stage_failed="measure", reason=f"reference: {mr.error}",
                           plan=plan, facts=facts, trail=trail)
        ref_obs = Observable.from_measure(mr.value)
    facts.update(ref_px=ref_obs.pixels, ref_method=ref_obs.method,
                 ref_spread=ref_obs.spread)

    # ── MEASURE the target ────────────────────────────────────────
    if plan.quantity in ("speed", "acceleration"):
        tr_ = pixel_speed(tgt_df, fps, plan.at_time,
                          average=bool(row.get("wants_average", False)))
        if not tr_:
            return Outcome(False, stage_failed="measure", reason=f"target speed: {tr_.error}",
                           plan=plan, facts=facts, trail=trail)
        tgt_obs = tr_.value
        if plan.quantity == "acceleration":
            tgt_obs.kind = "accel"          # measured as a rate; see gate/limits
    elif plan.target_b is not None:
        gb = ground(plan.target_b, tgt_axis)
        if not gb:
            return Outcome(False, stage_failed="ground", reason=f"target_b: {gb.error}",
                           plan=plan, facts=facts, trail=trail)
        b_all = gb.value["df"]
        b_id, b_n, b_how = pick_instance(
            b_all, instance_hint(plan.target_b.phrase, question), W, H)
        if b_id == tgt_id and b_all is tgt_df_all:
            # "distance between the two X" grounded to a single instance
            ids = [i for i in b_all.obj_id.unique() if i != tgt_id]
            if not ids:
                return Outcome(False, stage_failed="ground",
                               reason="only one instance found for a two-object distance",
                               plan=plan, facts=facts, trail=trail)
            b_id = int(b_all[b_all.obj_id.isin(ids)].groupby("obj_id").size().idxmax())
        sep = pixel_separation(tgt_df, b_all[b_all.obj_id == b_id])
        if not sep:
            return Outcome(False, stage_failed="measure", reason=sep.error,
                           plan=plan, facts=facts, trail=trail)
        tgt_obs = sep.value
        facts.update(target_b_instances=b_n, target_b_pick=b_how)
    else:
        mt, tt = measure(tgt_df, plan.axis, frames_bgr)
        trail["measure:target"] = tt
        if not mt:
            return Outcome(False, stage_failed="measure", reason=f"target: {mt.error}",
                           plan=plan, facts=facts, trail=trail)
        tgt_obs = Observable.from_measure(mt.value)
    facts.update(tgt_px=tgt_obs.pixels, tgt_method=tgt_obs.method,
                 tgt_spread=tgt_obs.spread)

    # ── GEOMETRY + SOLVE ──────────────────────────────────────────
    sc = resolve_scale(ref_obs.pixels, plan.reference_value,
                       plan.reference.phrase, plan.target.phrase,
                       row.get("depth_info"), is_3d)
    if not sc:
        return Outcome(False, stage_failed="geometry", reason=sc.error,
                       plan=plan, facts=facts, trail=trail)
    scale = sc.value
    facts.update(depth_status=scale.status, depth_ratio=scale.depth_ratio,
                 depth_conf=scale.confidence, depth_note=scale.note)

    ans = solve(plan, ref_obs, tgt_obs, scale)
    if not ans:
        return Outcome(False, stage_failed="solve", reason=ans.error,
                       plan=plan, facts=facts, trail=trail)
    a: Answer = ans.value
    facts.update(a.provenance)
    return Outcome(True, value=a.value, unit=a.unit, confidence=a.confidence,
                   plan=plan, facts=facts, trail=trail)


def run_many(df, use_gradient: bool = False, progress=True) -> pd.DataFrame:
    """run_one over a dataframe, ordered by video so the frame cache pays off."""
    out = []
    order = df.sort_values("video_id") if "video_id" in df.columns else df
    for k, (idx, row) in enumerate(order.iterrows()):
        try:
            o = run_one(row, use_gradient=use_gradient)
        except Exception as e:
            o = Outcome(False, stage_failed="exception",
                        reason=f"{type(e).__name__}: {e}",
                        facts={"video_id": row.get("video_id")})
        rec = {"row": idx, "id": row.get("id"), "question": row.get("question"),
               "prior": row.get("prior"), "category": row.get("category")}
        if "ground_truth_posterior" in row:
            rec["gt"] = row["ground_truth_posterior"]
        rec.update(o.as_row())
        out.append(rec)
        if progress and (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(order)}", flush=True)
    return pd.DataFrame(out)
