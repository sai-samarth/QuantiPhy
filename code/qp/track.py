"""Simple, verifiable object tracking for clean-background validation videos.

Deliberately NOT a neural network. For the convention-fitting experiment we need
a handful of trajectories we can trust completely, on videos where the object is
trivially separable by colour or brightness. Every tracker here is auditable and
its output is checked by drawing boxes back onto the frames.
"""
from __future__ import annotations
import cv2, numpy as np, pandas as pd


def _largest(mask, min_area=20):
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_area:
        return None
    x, y, w, h, a = stats[k]
    return dict(u=float(cent[k][0]), v=float(cent[k][1]), w=int(w), h=int(h),
                area=int(a), diag=float(np.hypot(w, h)))


def seg_dark(fr, thr=200):                       # dark object on white
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) < thr

def seg_bright(fr, thr=200):                     # bright object on dark
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) > thr

def seg_red(fr, sat=90, val=60):                 # red object on blue/neutral
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (((h < 12) | (h > 168)) & (s > sat) & (v > val))

def seg_bright_roi(fr, thr=200, top=0.0, bot=1.0):
    m = seg_bright(fr, thr); H = m.shape[0]
    out = np.zeros_like(m); out[int(top*H):int(bot*H)] = m[int(top*H):int(bot*H)]
    return out


def track(path, segfn, fps, min_area=20) -> pd.DataFrame:
    cap = cv2.VideoCapture(path); rows = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        r = _largest(segfn(fr), min_area)
        if r:
            r.update(f=i, t=i / fps); rows.append(r)
        i += 1
    cap.release()
    return pd.DataFrame(rows)


def overlay(path, segfn, fps, out_png, picks=(0.05, 0.35, 0.65, 0.95), min_area=20, width=300):
    """Draw the detected box on sampled frames so tracking can be eyeballed."""
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = {int(n * p) for p in picks}; got = {}; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            r = _largest(segfn(fr), min_area)
            if r:
                x = int(r['u'] - r['w'] / 2); y = int(r['v'] - r['h'] / 2)
                cv2.rectangle(fr, (x, y), (x + r['w'], y + r['h']), (0, 0, 255), 2)
                cv2.putText(fr, f"{r['diag']:.0f}px", (x, max(14, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            got[i] = fr
        i += 1
    cap.release()
    tiles = [cv2.resize(got[k], (width, int(width * got[k].shape[0] / got[k].shape[1])))
             for k in sorted(got)]
    cv2.imwrite(out_png, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_png


def seg_not_background(fr, bg, tol=28):
    """Anything differing from a uniform background colour. More robust than a
    colour mask when the object is multi-coloured (e.g. a car with dark glass)."""
    return np.linalg.norm(fr.astype(np.int16) - np.array(bg, np.int16), axis=2) > tol


def background_colour(path, n_sample=5):
    """Median colour of the frame borders — the uniform backdrop."""
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = {int(n * i / (n_sample + 1)) for i in range(1, n_sample + 1)}
    cols = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in picks:
            b = np.concatenate([fr[0:3].reshape(-1, 3), fr[-3:].reshape(-1, 3),
                                fr[:, 0:3].reshape(-1, 3), fr[:, -3:].reshape(-1, 3)])
            cols.append(np.median(b, axis=0))
        i += 1
    cap.release()
    return np.median(np.array(cols), axis=0)
