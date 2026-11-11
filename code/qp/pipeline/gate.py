"""GATE: decide which pipeline answers to trust.

This is the module that makes a hybrid possible, and it rests on one observation:

    THE PIPELINE DOES NOT HAVE TO WORK EVERYWHERE.
    IT HAS TO WORK WHERE IT WORKS, AND KNOW WHEN IT DOES NOT.

A pipeline answering 40% of rows at MRA 0.75 beats one answering 100% at 0.40,
because the other 60% can fall back to the VLM. So the useful question is never
"is the pipeline good?" but "on which rows is it good, and can that be
determined WITHOUT the ground truth?" Every rule here is computable at test time.

WHY RULES AND NOT A LEARNED CLASSIFIER: with ~100 validation rows, a classifier
over a dozen features memorises. These rules encode failures we have actually
diagnosed and can each be counted, so a rule that never fires -- or fires on
good rows -- is visible and gets deleted. Each carries its own threshold so the
whole policy can be refit as data, and `explain()` prints what fired and why.

CALIBRATION IS NOT OPTIONAL. Thresholds here are fitted by scripts/fit_gate.py
against the validation run, not guessed. A gate that rejects good rows costs
exactly as much as one that accepts bad ones.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Rule:
    name: str
    test: Callable[[dict], bool]      # True == this row is RISKY
    why: str
    weight: float = 1.0               # how much risk it contributes


def _f(row, key, default=np.nan):
    v = row.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _s(row, key, default=""):
    v = row.get(key, default)
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


# ───────────────────────── the rules ─────────────────────────
# Each is a failure mode we have diagnosed on real items, not a guess.

DEFAULTS = dict(
    min_px=20.0,          # below this, one pixel of edge error is >5% of the answer
    max_spread=0.15,      # relative std across frames
    max_depth_ratio=2.5,  # a huge depth correction dominates the pixel measurement
    min_track=6,          # frames
)


def build_rules(p: dict | None = None) -> list[Rule]:
    p = {**DEFAULTS, **(p or {})}
    return [
        # ---- depth: the single largest measured separator we have ----
        # 5.4% median error when both depths are known vs 53.9% when the
        # target's is not. This is a pure multiplier on the answer.
        Rule("depth-unknown",
             lambda r: _s(r, "depth_status") in ("ruler-only", "target-only", "neither-known"),
             "3D scene but a depth is missing; coplanarity is being assumed", 1.0),
        Rule("depth-extreme",
             lambda r: np.isfinite(_f(r, "depth_ratio")) and (
                 _f(r, "depth_ratio") > p["max_depth_ratio"]
                 or _f(r, "depth_ratio") < 1 / p["max_depth_ratio"]),
             "the depth correction, not the pixels, dominates the answer", 0.5),

        # ---- grounding: the dominant failure mode ----
        # 4 'central sculpture' instances, measured the wrong one: 54% error.
        Rule("ambiguous-instance",
             lambda r: _f(r, "tgt_instances", 1) > 1 and _s(r, "tgt_pick") == "longest-track",
             "several instances matched and nothing in the question chose between them", 1.0),
        Rule("ambiguous-reference",
             lambda r: _f(r, "ref_instances", 1) > 1 and _s(r, "ref_pick") == "longest-track",
             "the reference object is ambiguous, so the scale is ambiguous", 0.7),
        # falling back to a generic noun means we measured 'block', not 'wood block'
        Rule("phrase-degraded",
             lambda r: "generic" in _s(r, "tgt_strategy") or "head_noun" in _s(r, "tgt_strategy"),
             "grounding only worked after weakening the phrase", 0.7),
        Rule("reference-degraded",
             lambda r: "generic" in _s(r, "ref_strategy") or "head_noun" in _s(r, "ref_strategy"),
             "the reference was found only after weakening the phrase", 0.5),

        # ---- measurement ----
        Rule("tiny-target",
             lambda r: _f(r, "tgt_px") < p["min_px"],
             "target too small for a 5%-accurate edge", 0.8),
        Rule("tiny-reference",
             lambda r: _f(r, "ref_px") < p["min_px"],
             "reference too small: it sets the scale for everything", 1.0),
        Rule("unstable-target",
             lambda r: _f(r, "tgt_spread", 0) > p["max_spread"],
             "target size varies across frames (blur, occlusion, or a bad track)", 0.6),
        Rule("unstable-reference",
             lambda r: _f(r, "ref_spread", 0) > p["max_spread"],
             "reference size varies across frames", 0.6),
        Rule("short-track",
             lambda r: min(_f(r, "tgt_track_len", 99), _f(r, "ref_track_len", 99)) < p["min_track"],
             "too few frames to average away mask noise", 0.5),

        # ---- semantics we know we get wrong ----
        # MEASURED on the validation run, by axis:
        #   length  median error  ~48%   (mask MAJOR axis -- survives rotation)
        #   width   median error  ~89%   (bounding-box width)
        #   height  median error  ~55%   (bounding-box height)
        # An obliquely-viewed object's BOUNDING BOX mixes its length into its
        # width: simulation_0009's boat is 64x52 px but physically 3.62x1.23 m.
        # The major axis is rotation-invariant, so 'length' is the one axis that
        # works; the bounding-box extents are the ones that do not.
        Rule("bbox-axis",
             lambda r: _s(r, "axis") in ("width", "height"),
             "a bounding-box extent is not a physical extent once the object is "
             "rotated relative to the camera", 0.6),
        Rule("bbox-axis-3d",
             lambda r: _s(r, "axis") in ("width", "height") and _s(r, "is_3d") == "True",
             "...and in a 3D scene it is also foreshortened", 0.4),
    ]


@dataclass
class Verdict:
    accept: bool
    risk: float
    fired: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)


def judge(row: dict, threshold: float = 1.0, params: dict | None = None) -> Verdict:
    """Total risk for one pipeline row; accept when it stays under `threshold`."""
    if not row.get("ok", False):
        return Verdict(False, 99.0, ["no-answer"], ["the pipeline did not produce a value"])
    risk, fired, detail = 0.0, [], []
    for r in build_rules(params):
        try:
            hit = r.test(row)
        except Exception:
            hit = False
        if hit:
            risk += r.weight
            fired.append(r.name)
            detail.append(f"{r.name}: {r.why}")
    return Verdict(risk < threshold, risk, fired, detail)


def judge_frame(df: pd.DataFrame, threshold: float = 1.0,
                params: dict | None = None) -> pd.DataFrame:
    """Add accept/risk/fired columns to a pipeline result frame."""
    v = [judge(r, threshold, params) for r in df.to_dict("records")]
    out = df.copy()
    out["accept"] = [x.accept for x in v]
    out["risk"] = [x.risk for x in v]
    out["fired"] = ["|".join(x.fired) for x in v]
    return out


def explain(row: dict, threshold: float = 1.0, params: dict | None = None) -> str:
    v = judge(row, threshold, params)
    head = (f"{'ACCEPT' if v.accept else 'REJECT'}  risk={v.risk:.2f} "
            f"(threshold {threshold})")
    if not v.detail:
        return head + "\n  nothing fired: every check passed"
    return head + "\n  " + "\n  ".join(v.detail)
