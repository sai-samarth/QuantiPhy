"""Crop-to-fill segmentation: re-run the segmenter with the object filling the canvas.

THE PROBLEM, measured not inferred. SAM 3's raw mask lattice is 288x288 for a
1008x1008 encoder input, i.e. one mask cell = 3.5 model pixels. What that is worth
in SOURCE pixels depends entirely on how much the frame was shrunk to fit:

    854-wide frame  -> resize 1.180x -> one mask cell =  2.97 source px
   1920-wide frame  -> resize 0.525x -> one mask cell =  6.67 source px
   3840-wide frame  -> resize 0.263x -> one mask cell = 13.33 source px

A 36 px ball on a 854-wide frame therefore sits on a lattice ~8% of its own size.
That is most of a 5% accuracy budget spent before the model makes any mistake.

THE FIX costs nothing and invents nothing: crop a window around the object we have
ALREADY located, and segment that. A 2x-margin crop of the same 36 px ball is a
72 px window resized 14x, so one mask cell = 0.25 source px — 12x finer.

TWO RULES that make the ratio safe:
  * Use the IDENTICAL crop procedure and interpolation kernel for the ruler and
    the target, so any residual resampling bias is common-mode and cancels when
    we divide one by the other.
  * Convert measurements back to SOURCE pixels with the exact inverse of the
    forward transform. A measurement of m pixels inside a crop of C source pixels
    rendered at R pixels corresponds to m * C / R source pixels.

CAUTION (from the survey, worth repeating): this removes the DECODER's quantisation
floor, not the sensor's. Expect a real improvement, not the full 12x.
"""
from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .types import Result


@dataclass
class CropView:
    """A crop plus everything needed to map measurements back to source pixels."""
    image: np.ndarray          # the rendered crop
    x0: float                  # source-pixel coords of the crop's top-left
    y0: float
    size_src: float            # crop side length in SOURCE pixels
    size_rendered: int         # crop side length as rendered

    @property
    def px_per_source(self) -> float:
        return self.size_rendered / self.size_src

    def to_source_px(self, measurement: float) -> float:
        """Convert a length measured in the crop back into source pixels."""
        return measurement / self.px_per_source

    def point_to_source(self, u, v):
        return (self.x0 + u / self.px_per_source, self.y0 + v / self.px_per_source)


def make_crop(frame_bgr: np.ndarray, cx: float, cy: float, obj_px: float,
              margin: float = 2.0, render: int = 1008,
              interp=cv2.INTER_LINEAR) -> CropView:
    """Square crop centred on (cx, cy), rendered to `render` px.

    `interp` is fixed deliberately: the SAME kernel must be used for ruler and
    target so resampling bias cancels in the ratio.
    """
    side = max(float(obj_px) * margin, 8.0)
    half = side / 2.0
    x0, y0 = cx - half, cy - half
    H, W = frame_bgr.shape[:2]
    # pad rather than clamp, so the object stays centred and the geometry stays exact
    pad = int(np.ceil(side))
    P = cv2.copyMakeBorder(frame_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    xi, yi = int(round(x0)) + pad, int(round(y0)) + pad
    s = int(round(side))
    crop = P[yi:yi + s, xi:xi + s]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        crop = np.zeros((8, 8, 3), frame_bgr.dtype)
        s = 8
    out = cv2.resize(crop, (render, render), interpolation=interp)
    return CropView(out, float(int(round(x0))), float(int(round(y0))), float(s), render)


def segment_crop(view: CropView, phrase: str, device="cuda",
                 score_thr: float = 0.35) -> Result:
    """Run SAM 3 (image mode) on a crop. Returns the best mask in CROP coordinates."""
    from PIL import Image
    import torch
    from transformers import Sam3Processor, Sam3Model
    global _P, _M
    try:
        _P
    except NameError:
        _P = Sam3Processor.from_pretrained("facebook/sam3")
        _M = Sam3Model.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to(device).eval()
    img = Image.fromarray(cv2.cvtColor(view.image, cv2.COLOR_BGR2RGB))
    inp = _P(images=img, text=phrase, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = _M(**inp)
    res = _P.post_process_instance_segmentation(
        out, threshold=score_thr, mask_threshold=0.5,
        target_sizes=inp["original_sizes"].tolist())[0]
    if not len(res["scores"]):
        return Result.bad(f"nothing found for {phrase!r} in crop", "cropseg")
    # prefer the instance nearest the crop centre — that is the object we cropped for
    c = view.size_rendered / 2
    best, bd = None, 1e18
    for i in range(len(res["scores"])):
        m = np.asarray(res["masks"][i].squeeze().cpu(), dtype=bool)
        if m.sum() < 16:
            continue
        ys, xs = np.nonzero(m)
        d = (xs.mean() - c) ** 2 + (ys.mean() - c) ** 2
        if d < bd:
            best, bd = m, d
    if best is None:
        return Result.bad("all crop masks too small", "cropseg")
    return Result.good({"mask": best, "score": float(res["scores"][0])}, "cropseg")
