"""GROUND stage: Plan + video -> Tracks (which pixels are the object).

This is where every serious failure has happened, so the chain encodes the
lessons directly:

  sam3:text        the phrase as written                      (cheapest)
  sam3:alternates  plan-supplied rephrasings
  sam3:head_noun   strip modifiers: "ball return opening" -> "opening"
  sam3:generic     fall back to a bare category noun
                   (a VLM pointing strategy slots in here later)

VALIDATION is separate from the strategies, because a strategy can SUCCEED and
be WRONG. Rejected by the validator:
  * full-frame slivers      (an 11x480 px strip SAM called a "billiard ball")
  * masks touching a border (truncated -> under-measured; cost us a car's size)
  * masks found in too few frames to be a real object
A rejected result falls through to the next strategy exactly like an error.
"""
from __future__ import annotations
import os
import re

import numpy as np

from .cache import get_or_compute
from .types import Plan, Result, Target, Track
from .strategy import Strategy, run_chain

_STOP = {"the", "a", "an", "of", "in", "on", "at", "left", "right", "upper", "lower",
         "middle", "centre", "center", "top", "bottom", "two", "small", "big", "large"}


# Where a post-modifier starts. English puts the head of a noun phrase at the
# FRONT when a modifier follows it, so everything from here on describes the
# head rather than being it.
_POST = re.compile(r"\b(of|above|below|beneath|under|underneath|on|in|at|near|"
                   r"behind|beside|between|among|with|inside|outside|next|"
                   r"that|which|who|holding|attached|placed|resting|sitting|"
                   r"standing|hanging|mounted|located)\b", re.I)

# Off by default; the tail rule is what shipped 0.545 and the replacement has
# not been A/B'd yet.
HEAD_FRONT = os.environ.get("QP_HEAD_FRONT") == "1"


def head_noun(phrase: str) -> str:
    # normalise the curly apostrophe first: the dataset writes "eagle's" with
    # U+2019, which [a-z0-9']+ splits into ["eagle", "s"], so the head noun came
    # out as "s wingspan" and grounded nothing
    phrase = str(phrase).replace("\u2019", "'").replace("\u02bc", "'")
    # TAKING THE TAIL IS WRONG WHENEVER A MODIFIER FOLLOWS THE HEAD. It is right
    # for a compound ("trash bags" -> trash bags) and wrong for every
    # post-modified phrase, and the fallback exists precisely for the
    # compositional phrases SAM cannot ground directly, so this fires on exactly
    # the cases it gets wrong:
    #     "light right above the woman's head"   -> woman's head   (SAM found one)
    #     "pedestal of the central sculpture"    -> central sculpture
    #     "road that the bike is crossing through" -> crossing through
    # 293 test rows / 8.2 macro points ground a different object under the two
    # rules, counting both the target and the reference.
    if HEAD_FRONT:
        m = _POST.search(phrase)
        if m and m.start() > 0:
            head = phrase[:m.start()].strip()
            if [w for w in re.findall(r"[a-z0-9']+", head.lower()) if w not in _STOP]:
                phrase = head
    words = [w for w in re.findall(r"[a-z0-9']+", phrase.lower()) if w not in _STOP]
    return " ".join(words[-2:]) if len(words) >= 2 else (words[-1] if words else phrase)


def generic_noun(phrase: str) -> str:
    for g in ["ball", "car", "person", "bird", "boat", "bag", "pin", "table", "sign",
              "block", "bottle", "cup", "box", "wheel", "lane", "panel", "bench"]:
        if re.search(rf"\b{g}s?\b", phrase, re.I):
            return g
    return head_noun(phrase).split()[-1]


def _sam_track(video_path: str, phrase: str, fps: float) -> Result:
    """One SAM 3 call, cached on (video, phrase)."""
    from qp.segment import track_multi
    def compute():
        df, per_frame, _ = track_multi(video_path, phrase, fps)
        return {"df": df, "per_frame": per_frame}
    try:
        out = get_or_compute("sam3", compute, video=video_path, phrase=phrase, fps=fps)
    except Exception as e:
        return Result.bad(f"sam3 raised: {type(e).__name__}: {e}")
    df = out["df"]
    if not len(df):
        return Result.bad(f"no detections for {phrase!r}")
    return Result.good(out, f"sam3:{phrase}")


# which frame borders can corrupt which measurement.
# Truncation only matters ALONG THE AXIS BEING MEASURED. A ruler that runs off
# the left and right edges is still a perfectly good ruler to measure the
# CALIBRE of -- and rejecting it there cost us every 'ruler calibre' item in the
# benchmark, because the fallback chain then ground a worse phrase instead.
_AXIS_BORDERS = {
    "height": ("top", "bottom"),
    "width": ("left", "right"),
    "diameter": ("left", "right", "top", "bottom"),   # area-based: any edge biases it
    "any": ("left", "right", "top", "bottom"),
    "separation": (),                                  # centroids, not extents
}


def make_validator(n_frames: int, frame_hw, axis: str = "any"):
    """Reject masks that are successfully-segmented but WRONG.

    `axis` is what the MEASURE stage will read off this track. It is required
    because two of the three checks are only meaningful relative to a specific
    dimension: a truncated mask is unusable along the truncated axis and fine
    along the other, and a 'sliver' is only implausible if we are about to treat
    it as a compact object.
    """
    H, W = frame_hw

    def validate(out) -> str | None:
        df = out["df"]
        if not len(df):
            return "empty"
        counts = df.groupby("obj_id").size()
        best = counts.max()
        if best < max(3, 0.10 * n_frames):
            return f"best track covers only {best}/{n_frames} frames"
        g = df[df.obj_id == counts.idxmax()]
        if (g.bh.median() > 0.9 * H) and (g.bw.median() > 0.9 * W):
            return "mask spans the whole frame"

        if axis == "length":
            # the major axis: whichever direction it actually points
            sides = ("left", "right") if g.bw.median() >= g.bh.median() else ("top", "bottom")
        else:
            sides = _AXIS_BORDERS.get(axis, _AXIS_BORDERS["any"])
        edge = {
            "left":   (g.u - g.bw / 2) < 2,
            "right":  (g.u + g.bw / 2) > W - 2,
            "top":    (g.v - g.bh / 2) < 2,
            "bottom": (g.v + g.bh / 2) > H - 2,
        }
        for side in sides:
            frac = edge[side].mean()
            if frac > 0.8:
                return (f"mask leaves the {side} of the frame in {frac*100:.0f}% of "
                        f"frames, which truncates the {axis} we need")
        return None
    return validate


def ground_one(video_path: str, target: Target, fps: float,
               n_frames: int, frame_hw, axis: str = "any") -> tuple[Result, list]:
    chain = [
        Strategy("sam3:text",       lambda: _sam_track(video_path, target.phrase, fps), cost=1),
        *[Strategy(f"sam3:alt[{a}]", (lambda a=a: _sam_track(video_path, a, fps)), cost=1)
          for a in target.alternates],
        Strategy("sam3:head_noun",  lambda: _sam_track(video_path, head_noun(target.phrase), fps),
                 applies_if=lambda: head_noun(target.phrase) != target.phrase, cost=1),
        Strategy("sam3:generic",    lambda: _sam_track(video_path, generic_noun(target.phrase), fps),
                 applies_if=lambda: generic_noun(target.phrase) != target.phrase, cost=1),
    ]
    return run_chain(chain, validate=make_validator(n_frames, frame_hw, axis))
