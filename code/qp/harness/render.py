"""Synthetic shapes with EXACT known extents.

Draw at SS-times resolution, then area-average down. Area-averaging is what makes
a boundary pixel take a fractional value equal to its coverage — the same partial-
area effect real optics produce — so the result is a genuine anti-aliased image
whose true extent we nonetheless know to arbitrary precision.

Everything returns float32 in [0,1]; degradations and encoding happen downstream.
"""
from __future__ import annotations
import numpy as np
import cv2

SS = 16          # supersampling factor


def _down(hi: np.ndarray, ss: int = SS) -> np.ndarray:
    h, w = hi.shape[:2]
    return cv2.resize(hi, (w // ss, h // ss), interpolation=cv2.INTER_AREA)


def disc(diameter: float, canvas: int = 256, fg: float = 0.85, bg: float = 0.15,
         ss: int = SS) -> tuple[np.ndarray, dict]:
    """A disc of EXACTLY `diameter` source pixels."""
    N = canvas * ss
    hi = np.full((N, N), bg, np.float32)
    c = N / 2.0 - 0.5
    r = diameter * ss / 2.0
    y, x = np.mgrid[0:N, 0:N].astype(np.float32)
    hi[((x - c) ** 2 + (y - c) ** 2) <= r * r] = fg
    return _down(hi, ss), {"shape": "disc", "diameter": diameter,
                           "height": diameter, "width": diameter}


def pin(height: float, canvas: int = 256, fg: float = 0.85, bg: float = 0.15,
        ss: int = SS) -> tuple[np.ndarray, dict]:
    """A bowling-pin-like NON-CONVEX silhouette of exact `height`.

    Non-convex on purpose: it is the shape class that broke the radial-ray method,
    and the one where 'area-equivalent diameter' is meaningless.
    """
    N = canvas * ss
    hi = np.full((N, N), bg, np.float32)
    H = height * ss
    cy0 = N / 2.0 - H / 2.0
    cx = N / 2.0
    # profile: neck + bulb, as a half-width function of normalised height t
    t = np.linspace(0, 1, int(H))
    halfw = (0.10 + 0.16 * np.exp(-((t - 0.82) ** 2) / 0.012)
             + 0.055 * np.exp(-((t - 0.30) ** 2) / 0.03)) * H
    for i, hw in enumerate(halfw):
        yy = int(cy0 + i)
        if 0 <= yy < N:
            hi[yy, max(0, int(cx - hw)):min(N, int(cx + hw))] = fg
    w = 2 * halfw.max() / ss
    return _down(hi, ss), {"shape": "pin", "height": height, "width": w,
                           "diameter": None}


def rect(width: float, height: float, canvas: int = 256, fg: float = 0.85,
         bg: float = 0.15, ss: int = SS) -> tuple[np.ndarray, dict]:
    """An axis-aligned rectangle — the easiest possible extent, as a control."""
    N = canvas * ss
    hi = np.full((N, N), bg, np.float32)
    cy, cx = N / 2.0, N / 2.0
    hi[int(cy - height * ss / 2):int(cy + height * ss / 2),
       int(cx - width * ss / 2):int(cx + width * ss / 2)] = fg
    return _down(hi, ss), {"shape": "rect", "height": height, "width": width,
                           "diameter": None}
