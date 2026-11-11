"""Realistic degradations, applied one at a time so blame is attributable.

The survey's key distinction: BLUR and SMEAR produce BIAS (a consistent shift in
the measured edge position); NOISE produces VARIANCE (scatter that averages out
over frames). Our real errors are bias-dominated, so blur and smear are the ones
that matter, and the ones a cross-frame median cannot rescue.

Gamma note: optical blur is linear in RADIANCE, but 8-bit video stores gamma-
encoded values. We therefore blur in LINEAR light and re-encode afterwards, which
is what a real camera does. Getting this backwards introduces an edge-position
error that REVERSES SIGN with contrast polarity — one of the survey's findings.
"""
from __future__ import annotations
import numpy as np
import cv2

GAMMA = 2.2


def to_linear(x):  return np.power(np.clip(x, 0, 1), GAMMA)
def to_encoded(x): return np.power(np.clip(x, 0, 1), 1.0 / GAMMA)


def blur(img: np.ndarray, sigma: float, in_linear: bool = True) -> np.ndarray:
    """Gaussian optical blur. Performed in linear light by default."""
    if sigma <= 0:
        return img.copy()
    x = to_linear(img) if in_linear else img
    x = cv2.GaussianBlur(x, (0, 0), sigma)
    return to_encoded(x) if in_linear else x


def motion(img: np.ndarray, length: float, angle_deg: float = 0.0,
           in_linear: bool = True) -> np.ndarray:
    """Linear motion smear of `length` pixels at `angle_deg`."""
    if length <= 0:
        return img.copy()
    n = max(3, int(round(length)) | 1)
    k = np.zeros((n, n), np.float32)
    k[n // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle_deg, 1.0)
    k = cv2.warpAffine(k, M, (n, n))
    k /= k.sum()
    x = to_linear(img) if in_linear else img
    x = cv2.filter2D(x, -1, k)
    return to_encoded(x) if in_linear else x


def noise(img: np.ndarray, sigma_grey: float, seed: int = 0) -> np.ndarray:
    """Additive Gaussian sensor noise, sigma given in 8-bit grey levels."""
    if sigma_grey <= 0:
        return img.copy()
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sigma_grey / 255.0, img.shape), 0, 1).astype(np.float32)


def jpeg(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """Round-trip through JPEG at the given quality (a stand-in for codec ringing)."""
    u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        return img.copy()
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def h264(frames: list[np.ndarray], crf: int = 23, fps: int = 24) -> list[np.ndarray]:
    """Round-trip a sequence through H.264 at a given CRF, via OpenCV's writer."""
    import tempfile, os
    h, w = frames[0].shape[:2]
    tmp = tempfile.mktemp(suffix=".mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    for f in frames:
        u8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        vw.write(cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR))
    vw.release()
    out, cap = [], cv2.VideoCapture(tmp)
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    cap.release(); os.unlink(tmp)
    return out
