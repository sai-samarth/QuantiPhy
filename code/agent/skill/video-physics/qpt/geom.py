"""Shape statistics of a binary mask, shared by the qpt client and the SAM tool server.

Every length is in ORIGINAL video pixels. Several definitions of "size" are given
on purpose, because questions differ in which one they mean:
  bw, bh       axis-aligned bounding box width / height (x / y extent)
  rect_long/short  sides of the minimum-area rotated rectangle (object's own axes)
  feret        largest distance between two mask points (tip-to-tip extent)
  major/minor  lengths of the equivalent ellipse from image moments (robust to ragged edges)
  equiv        diameter of the circle with the mask's area
"""
import base64
import cv2
import numpy as np


def largest_component(mask):
    m = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    return m.astype(bool)


def mask_stats(mask, keep_largest=True):
    m = largest_component(mask) if keep_largest else mask.astype(bool)
    area = int(m.sum())
    if area < 3:
        return None
    ys, xs = np.nonzero(m)
    m8 = m.astype(np.uint8)
    M = cv2.moments(m8, binaryImage=True)
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    common = np.sqrt(max(0.0, 4 * mu11 ** 2 + (mu20 - mu02) ** 2))
    l1, l2 = (mu20 + mu02 + common) / 2, (mu20 + mu02 - common) / 2
    cs = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    pts = np.vstack([c.reshape(-1, 2) for c in cs]).astype(np.float32)
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) > 1:
        d = np.sqrt(((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1))
        i, j = np.unravel_index(int(np.argmax(d)), d.shape)
        feret, fa, fb = float(d[i, j]) + 1.0, hull[i].tolist(), hull[j].tolist()
    else:
        feret, fa, fb = 1.0, hull[0].tolist(), hull[0].tolist()
    return dict(cx=float(cx), cy=float(cy), area=area,
                x0=int(xs.min()), y0=int(ys.min()), x1=int(xs.max()), y1=int(ys.max()),
                bw=int(xs.max() - xs.min() + 1), bh=int(ys.max() - ys.min() + 1),
                rect_long=float(max(rw, rh)) + 1.0, rect_short=float(min(rw, rh)) + 1.0,
                rect_angle=float(ang), feret=feret, feret_a=fa, feret_b=fb,
                major=float(4 * np.sqrt(max(l1, 0))), minor=float(4 * np.sqrt(max(l2, 0))),
                equiv=float(2 * np.sqrt(area / np.pi)))


def pack_mask(mask):
    m = np.asarray(mask, bool)
    return {"shape": list(m.shape), "bits": base64.b64encode(np.packbits(m.ravel())).decode()}


def unpack_mask(d):
    h, w = d["shape"]
    bits = np.frombuffer(base64.b64decode(d["bits"]), np.uint8)
    return np.unpackbits(bits)[: h * w].reshape(h, w).astype(bool)
