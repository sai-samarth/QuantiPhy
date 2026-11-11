"""Boundary refinement: SAM says WHICH object, BiRefNet says WHERE THE EDGE IS.

WHY THIS EXISTS: SAM 3's encoder is fixed at 1008x1008 with patch 14 and its mask
decoder upsamples 3x, so masks are computed at reduced resolution and scaled back
up. It is optimised for concept recall, not contour fidelity. On a 4K frame that
is several native pixels of boundary error before we start — and our answers are
scored on 5% relative accuracy.

BiRefNet_HR-matting (MIT, ungated, 220M params, ~146 ms at 2048px on a 5090) emits
a CONTINUOUS alpha, not a binary mask. That is a sub-pixel edge by construction: a
boundary pixel with alpha 0.37 means the true edge sits 37% of the way through it.

THREE CONDITIONS, each learned by measuring:
1. BiRefNet is NOT promptable — it segments whatever is salient in the crop. On a
   tennis-ball crop it swallowed the whole TABLE (57.5 px vs a true 43.9); on a pin
   crop it took ALL FIVE pins. Its alpha MUST be intersected with a dilated SAM mask.
2. Once constrained, crop margin barely matters (0.4-1.9% error for pads 1.15-3.2),
   so use a generous margin — undercropping is the only real risk.
3. It can find NOTHING and still yield a number if you measure with a relative
   threshold. At pad 2.4 the pin crop gave alpha.max()=0.029 (no object) yet a naive
   extent function reported 61 px of pure noise. An ABSOLUTE confidence gate is
   required, not a plausibility check on the output.

NOT a universal replacement for the SAM mask: better on the tennis ball
(0.4% vs 1.3%) and far better on patterned objects (soccer ball 94.8 vs a
brightness-based 48.8), but WORSE on the bowling pin, where SAM is exact.
It is one more strategy in the chain, corroborated like the others.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import Result

_M = {}

# MEASURED complementarity — neither variant alone covers our objects:
#   bowling pin, ball, person, sculpture -> HR works, HR-matting returns ~0
#   wood block, ping pong ball           -> HR-matting works, HR returns ~0
# and the required crop margin differs per object (2x to 8x). So we sweep a
# ladder of (variant, pad, resolution) and take the first that passes the
# absolute confidence gate. Coverage went from 5/15 objects to near-complete.
VARIANTS = ["ZhengPeng7/BiRefNet_HR", "ZhengPeng7/BiRefNet_HR-matting"]
LADDER = [(2.0, 1024), (4.0, 1024), (8.0, 2048)]


def _model(name="ZhengPeng7/BiRefNet_HR-matting", device="cuda"):
    if name not in _M:
        import torch
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        _M[name] = AutoModelForImageSegmentation.from_pretrained(
            name, trust_remote_code=True).to(device).eval().half()
        _M["n"] = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        _M["torch"] = torch
    return _M[name], _M["n"], _M["torch"]


def alpha_map(bgr, res=1024, device="cuda", name="ZhengPeng7/BiRefNet_HR-matting") -> np.ndarray:
    m, norm, torch = _model(name, device)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    t = torch.from_numpy(cv2.resize(rgb, (res, res))).permute(2, 0, 1).float() / 255
    t = norm(t).unsqueeze(0).to(device).half()
    with torch.inference_mode():
        a = m(t)[-1].sigmoid()[0, 0].float().cpu().numpy()
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)


def refined_alpha_once(frame_bgr, sam_mask, pad, res, name, dilate_frac=0.12) -> Result:
    """One (variant, pad, res) attempt."""
    ys, xs = np.nonzero(sam_mask)
    if len(xs) < 8:
        return Result.bad("sam mask too small to refine", "birefnet")
    cu, cv_ = xs.mean(), ys.mean()
    obj = max(int(np.ptp(xs)), int(np.ptp(ys)), 8)
    R = int(obj * pad / 2) + 4
    H, W = frame_bgr.shape[:2]
    P = np.zeros((H + 2*R, W + 2*R, 3), frame_bgr.dtype); P[R:R+H, R:R+W] = frame_bgr
    S = np.zeros((H + 2*R, W + 2*R), np.uint8);           S[R:R+H, R:R+W] = sam_mask
    y0, x0 = int(cv_), int(cu)
    crop = P[y0:y0 + 2*R, x0:x0 + 2*R]
    samc = S[y0:y0 + 2*R, x0:x0 + 2*R]
    if crop.shape[0] < 16 or crop.shape[1] < 16:
        return Result.bad("crop degenerate", "birefnet")

    a = alpha_map(crop, res=res, name=name)
    # ABSOLUTE confidence gate — see condition 3 in the module docstring
    if a.max() < 0.6 or (a > 0.5).mean() < 0.005:
        return Result.bad(f"no object found (alpha max {a.max():.3f})", "birefnet")

    k = max(3, int(obj * dilate_frac) | 1)
    dil = cv2.dilate(samc, np.ones((k, k), np.uint8))
    a = np.where(dil > 0, a, 0.0)
    if a.sum() < 4:
        return Result.bad("alpha vanished after SAM constraint", "birefnet")
    return Result.good({"alpha": a, "origin": (y0 - R, x0 - R), "crop": crop,
                        "variant": name.split("/")[-1], "pad": pad, "res": res},
                       f"birefnet:{name.split('/')[-1]}@pad{pad}",
                       confidence=float(np.clip(a.max(), 0.2, 1.0)))


def refined_alpha(frame_bgr, sam_mask: np.ndarray, pad: float = None,
                  dilate_frac: float = 0.12, res: int = None) -> Result:
    """Sweep variants x crop margins; return the first that finds the object."""
    tried = []
    combos = ([(pad, res, n) for n in VARIANTS] if (pad and res)
              else [(p, r, n) for n in VARIANTS for p, r in LADDER])
    for p_, r_, name in combos:
        out = refined_alpha_once(frame_bgr, sam_mask, p_, r_, name, dilate_frac)
        if out.ok:
            out.notes = list(out.notes) + [f"after {len(tried)} failed combos"]
            return out
        tried.append(f"{name.split('/')[-1]}@pad{p_}: {out.error}")
    return Result.bad("no variant/margin found the object", "birefnet", notes=tried)


def alpha_extents(a: np.ndarray) -> dict:
    """Sub-pixel size from a continuous alpha map.

    `area` sums alpha, so a boundary pixel at 0.37 contributes 0.37 — this is the
    sub-pixel part. Extents use an ABSOLUTE alpha threshold (never a relative one,
    which measures noise when nothing was found).
    """
    area = float(a.sum())
    rows, cols = a.sum(1), a.sum(0)
    def ext(prof):
        nz = np.where(prof > 0.5)[0]
        return float(nz[-1] - nz[0] + 1) if len(nz) else 0.0
    return {"area": area, "equiv": 2 * np.sqrt(area / np.pi),
            "height": ext(rows), "width": ext(cols)}
