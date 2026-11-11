"""Six ways to measure an object's extent, for head-to-head scoring.

Two families:

  THRESHOLD-BASED — binarise first, then measure the blob. This is what an ordinary
  segmentation mask gives you, and what we have been doing.
      bbox      bounding-box extent
      area_eq   diameter of the equal-area circle
      moments   principal-axis lengths from image moments

  CONTINUOUS — never binarise; work on the soft field directly.
      iso       0.5 level set by marching squares, then a Taubin circle fit
      coverage  sum of alpha, converted to an equivalent diameter
      iso_extent  raw extent of the 0.5 level set (no shape assumption)

Predicted by the survey and testable here: under blur, MOMENT axes inflate by
+sigma^2 while the ISO contour shrinks by -sigma^2/(2R). Opposite signs on the same
field is exactly the +1.4% area vs -7.5% minor-axis split we saw on a real ball.
"""
from __future__ import annotations
import numpy as np
import cv2

try:
    from skimage import measure as _skmeasure
except Exception:
    _skmeasure = None


def _norm(img):
    """Map the field to [0,1] so that 0.5 means half-coverage.

    BUG THIS FIXES: using global percentiles fails whenever the object is small.
    A 12 px disc covers 0.17% of a 256x256 canvas, so even the 99th percentile is
    background — the normalisation then divides by ~0 and reports +144% error.
    Estimate the two levels from the histogram's two modes instead, split at the
    midpoint between the global extremes, which is scale-free.
    """
    lo_all, hi_all = float(np.min(img)), float(np.max(img))
    if hi_all - lo_all < 1e-6:
        return np.zeros_like(img)
    mid = (lo_all + hi_all) / 2
    dark, bright = img[img <= mid], img[img > mid]
    lo = float(np.median(dark)) if dark.size else lo_all
    hi = float(np.median(bright)) if bright.size else hi_all
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0, 1)


# ── threshold family ────────────────────────────────────────────────
def bbox(img, axis="diameter"):
    a = _norm(img) > 0.5
    if a.sum() < 4: return np.nan
    ys, xs = np.nonzero(a)
    h, w = ys.max()-ys.min()+1, xs.max()-xs.min()+1
    return {"height": h, "width": w}.get(axis, max(h, w))

def area_eq(img, axis="diameter"):
    a = _norm(img) > 0.5
    return 2*np.sqrt(a.sum()/np.pi) if a.sum() >= 4 else np.nan

def moments(img, axis="diameter"):
    a = (_norm(img) > 0.5).astype(np.uint8)
    if a.sum() < 4: return np.nan
    M = cv2.moments(a, binaryImage=True)
    mu20, mu02, mu11 = M["mu20"]/M["m00"], M["mu02"]/M["m00"], M["mu11"]/M["m00"]
    com = np.sqrt(max(0.0, 4*mu11**2 + (mu20-mu02)**2))
    l1, l2 = (mu20+mu02+com)/2, (mu20+mu02-com)/2
    maj, mnr = 4*np.sqrt(max(l1, 0)), 4*np.sqrt(max(l2, 0))
    return {"height": maj, "width": mnr}.get(axis, mnr)


# ── continuous family ───────────────────────────────────────────────
def _outer_contour(img):
    """Sub-pixel 0.5 level set, outermost closed loop only.

    'Outermost only' is what makes this immune to internal texture: a soccer
    ball's black patches create interior level sets that a naive extractor
    would happily return.
    """
    if _skmeasure is None: return None
    f = _norm(img)
    cs = _skmeasure.find_contours(f, 0.5)
    if not cs: return None
    def area(c):
        x, y = c[:,1], c[:,0]
        return abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))/2
    return max(cs, key=area)

def _taubin(x, y):
    """Taubin circle fit — unbiased on partial arcs, unlike an algebraic (Kasa) fit."""
    x, y = np.asarray(x,float), np.asarray(y,float)
    xm, ym = x.mean(), y.mean()
    u, v = x-xm, y-ym
    z = u*u + v*v
    zm = z.mean()
    Suu,Svv,Suv = (u*u).mean(),(v*v).mean(),(u*v).mean()
    Suz,Svz = (u*z).mean(),(v*z).mean()
    A = np.array([[Suu,Suv],[Suv,Svv]]); b = np.array([Suz,Svz])/2
    try: uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError: return np.nan
    r = np.sqrt(uc*uc + vc*vc + zm)
    return 2*r

def iso(img, axis="diameter"):
    c = _outer_contour(img)
    return _taubin(c[:,1], c[:,0]) if c is not None and len(c) >= 8 else np.nan

def iso_extent(img, axis="diameter"):
    c = _outer_contour(img)
    if c is None or len(c) < 8: return np.nan
    h = c[:,0].max()-c[:,0].min(); w = c[:,1].max()-c[:,1].min()
    return {"height": h, "width": w}.get(axis, max(h, w))

def coverage(img, axis="diameter"):
    """sum(alpha) -> equal-area diameter. Exactly blur-invariant (convolution
    conserves the integral) but sensitive to any gain error in the field."""
    f = _norm(img)
    a = f.sum()
    return 2*np.sqrt(a/np.pi) if a > 4 else np.nan


ALL = {"bbox": bbox, "area_eq": area_eq, "moments": moments,
       "iso": iso, "iso_extent": iso_extent, "coverage": coverage}
