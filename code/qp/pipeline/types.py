"""Data contracts between pipeline stages.

DESIGN RULE: stages communicate ONLY through these types. No stage may import
another stage. That is what lets us swap SAM for something else, unit-test the
geometry with no video files, and reorder fallbacks without touching logic.

DESIGN RULE 2: failure is a value, not an exception. Every stage returns a
Result that is either ok(value) or fail(reason). A stage that cannot do its job
says so; it never invents a number. This matters because MRA gives partial
credit for a hedged guess and zero for confident nonsense.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import numpy as np

T = TypeVar("T")


# ───────────────────────── Result ─────────────────────────
@dataclass
class Result(Generic[T]):
    """Either a value or a reason it could not be produced."""
    value: T | None = None
    error: str | None = None
    strategy: str | None = None          # which approach produced this
    confidence: float = 1.0              # 0..1, used to decide hedging
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.error is None and self.value is not None

    def __bool__(self) -> bool:
        return self.ok

    @staticmethod
    def good(value: T, strategy: str = "", confidence: float = 1.0, notes=None) -> "Result[T]":
        return Result(value=value, strategy=strategy, confidence=confidence, notes=notes or [])

    @staticmethod
    def bad(reason: str, strategy: str = "", notes=None) -> "Result[T]":
        return Result(error=reason, strategy=strategy, confidence=0.0, notes=notes or [])


# ───────────────────────── PLAN ─────────────────────────
Quantity = Literal["size", "distance", "speed", "acceleration"]
Axis = Literal["height", "width", "length", "diameter", "any", "separation"]


@dataclass
class Target:
    """One thing we need to find in the video."""
    phrase: str                          # what to hand a grounding model
    alternates: list[str] = field(default_factory=list)   # fallback phrasings
    role: Literal["reference", "target", "target_b"] = "target"


@dataclass
class Plan:
    """How to answer one question. Produced by the PLAN stage."""
    quantity: Quantity
    axis: Axis
    unit: str                            # the unit the answer must be in
    reference: Target                    # object the prior describes
    target: Target                       # object the question asks about
    target_b: Target | None = None       # second object for "distance between X and Y"
    at_time: float | None = None         # timestamp the QUESTION names, if any
    t_from: float | None = None          # question names an INTERVAL, e.g.
    t_to: float | None = None            # "displacement from 0.5s to 1.0s". A
                                         # whole-clip average answers a different
                                         # question: 193 test rows (6.2 macro
                                         # points) were getting one.
    prior_at_time: float | None = None   # timestamp the PRIOR names, if any.
                                         # SEPARATE ON PURPOSE: 368 test rows carry
                                         # a prior like "velocity of the ball at 1.5s
                                         # = 5.21 m/s" while the question asks about a
                                         # different instant. Calibrating the scale at
                                         # the question's instant instead of the
                                         # prior's gave two scales 22x apart for the
                                         # same video.
    reference_is_motion: bool = False    # prior is a speed/accel, not a size
    reference_kind: str = "length"       # 'length' | 'rate' | 'accel' -- the ORDER of
                                         # the prior. 'ball acceleration = 3 m/s^2' and
                                         # 'ball speed = 3 m/s' are BOTH motion priors
                                         # but calibrate against different pixel
                                         # observables; conflating them divides m/s^2
                                         # by px/s and yields 100,000% errors.
    reference_value: float = 0.0         # the prior, in SI
    reference_axis: Axis = "any"         # which dimension the prior describes
    notes: list[str] = field(default_factory=list)


# ───────────────────────── GROUND ─────────────────────────
@dataclass
class Track:
    """Where an object is, frame by frame. Produced by the GROUND stage."""
    role: str
    phrase: str
    frames: np.ndarray                   # frame indices where found
    times: np.ndarray                    # seconds
    masks: dict[int, np.ndarray]         # frame index -> boolean mask
    obj_id: int | None = None
    score: float = 0.0

    def __len__(self) -> int:
        return len(self.frames)


# ───────────────────────── MEASURE ─────────────────────────
@dataclass
class PixelMeasure:
    """A length in pixels, with the evidence behind it."""
    pixels: float
    axis: Axis
    n_frames: int
    spread: float                        # relative std across frames used
    method: str                          # 'gradient' | 'mask-minor' | 'mask-bbox' ...
    frames_used: list[int] = field(default_factory=list)


# ───────────────────────── SOLVE ─────────────────────────
@dataclass
class Answer:
    """The final number plus a full audit trail."""
    value: float
    unit: str
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Answer({self.value:.4g} {self.unit}, conf={self.confidence:.2f})"
