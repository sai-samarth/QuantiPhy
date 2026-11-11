"""Declarative fallback chains.

THE PROBLEM this solves: without it, fallback logic metastasises into nested
try/except inside every stage, and you can no longer see what the system will
actually do. We hit three real cases already:
  * SAM found nothing for the phrase "pencil bag"  -> need a rephrase, then pointing
  * a track was a full-frame sliver                -> need to reject and retry
  * an object was measured while motion-blurred    -> need a different frame subset

THE SHAPE: a stage declares an ORDERED list of Strategies. The runner tries each
until one returns an ok Result, recording which one worked and why the earlier
ones didn't. Business logic never contains the word "fallback".

WHY IT MATTERS BEYOND TIDINESS: because the chain is data, we can log it, count
how often each strategy fires across the dataset, and delete the ones that never
win. That is how the fallback list stays short instead of growing forever.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable, Any

from .types import Result


@dataclass
class Strategy:
    name: str
    fn: Callable[..., Result]
    applies_if: Callable[..., bool] | None = None   # skip when not relevant
    cost: float = 1.0                               # relative expense, for ordering


@dataclass
class Attempt:
    strategy: str
    ok: bool
    reason: str = ""
    seconds: float = 0.0


def run_chain(strategies: list[Strategy], *args, validate: Callable[[Any], str | None] = None,
              **kwargs) -> tuple[Result, list[Attempt]]:
    """Try strategies in order; return the first ok Result plus the full trail.

    `validate` is the crucial part: a strategy can SUCCEED and still be WRONG.
    A sliver mask is a successful segmentation of the wrong thing. So every
    result is checked by an independent validator before being accepted, and a
    validation failure falls through to the next strategy exactly like an error.
    """
    trail: list[Attempt] = []
    for s in strategies:
        if s.applies_if is not None:
            try:
                if not s.applies_if(*args, **kwargs):
                    trail.append(Attempt(s.name, False, "not applicable"))
                    continue
            except Exception as e:
                trail.append(Attempt(s.name, False, f"applies_if raised: {e}"))
                continue
        t0 = time.time()
        try:
            r = s.fn(*args, **kwargs)
        except Exception as e:
            trail.append(Attempt(s.name, False, f"{type(e).__name__}: {e}", time.time()-t0))
            continue
        dt = time.time() - t0
        if not r.ok:
            trail.append(Attempt(s.name, False, r.error or "no value", dt))
            continue
        if validate is not None:
            why = validate(r.value)
            if why:
                trail.append(Attempt(s.name, False, f"rejected by validator: {why}", dt))
                continue
        r.strategy = r.strategy or s.name
        r.notes = list(r.notes) + [f"chain: {len(trail)+1} attempt(s)"]
        trail.append(Attempt(s.name, True, "", dt))
        return r, trail
    return Result.bad("all strategies exhausted",
                      notes=[f"{a.strategy}: {a.reason}" for a in trail]), trail
