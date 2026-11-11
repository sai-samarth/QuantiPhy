#!/usr/bin/env python
"""qpt -- measuring tools for one video task directory (video.mp4 + questions.json).

Conventions for every command:
  * coordinates are ORIGINAL video pixels: x to the right, y down, (0,0) = top-left
    corner of the frame -- also on zoomed views, whose grid labels are original pixels;
  * frames are ORIGINAL frame indices (frame 0 = first frame) and t = frame / fps with
    the fps in questions.json; --t T picks the nearest frame, round(T * fps);
  * images go to ./qpt_out/ and their paths are printed: open them with your image
    viewer (the read tool) to look at them.

Run `qpt <command> -h` for the options of one command.
"""
import argparse, csv, http.client, json, math, os, re, socket, sys, base64
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from geom import mask_stats, unpack_mask  # noqa: E402

OUT = Path("qpt_out")
MAX_SIDE = 1600          # longest side of any image we write (bigger ones get downscaled by the viewer anyway)
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(0, 0, 255), (0, 200, 0), (255, 128, 0), (255, 0, 255), (0, 220, 220), (255, 255, 0),
          (128, 0, 255), (0, 128, 255)]


# ───────────────────────── video access ─────────────────────────
LAB_F_FRAC = 0.58     # lab recordings: one documented camera rig (~82 deg horizontal FOV): f ~ 0.58 x image width
LAB_DROPS = (11, 35, 59)   # lab files: 25 fps capture made 24 fps by dropping one source frame per second


class Video:
    def __init__(self, path="video.mp4"):
        self.path = path
        if not Path(path).exists():
            sys.exit(f"qpt: {path} not found (run qpt inside the task directory)")
        cap = cv2.VideoCapture(path)
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); self.fps_container = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.q = json.load(open("questions.json")) if Path("questions.json").exists() else {}
        self.fps = float(self.q.get("video", {}).get("fps") or self.fps_container or 30.0)
        src = str(self.q.get("video", {}).get("source", "")).lower()
        self.source = "lab" if src.startswith("lab") else "simulation" if src.startswith("sim") else \
            "internet" if src.startswith("internet") else ""
        self.lab_f = LAB_F_FRAC * self.W if self.source == "lab" else None
        self._all = None

    def all(self):
        """Every decoded frame (BGR). Clips are short; decoding sequentially is exact."""
        if self._all is None:
            cap = cv2.VideoCapture(self.path); fr = []
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                fr.append(f)
            cap.release()
            self._all = fr
        return self._all

    @property
    def n(self):
        return len(self.all())

    def frame(self, i):
        fr = self.all()
        if not 0 <= i < len(fr):
            sys.exit(f"qpt: frame {i} out of range 0..{len(fr) - 1}")
        return fr[i]

    def f_of(self, t):
        return int(min(max(round(t * self.fps), 0), self.n - 1))

    def pick(self, a):
        """Frame index from --f / --t (default frame 0)."""
        if getattr(a, "f", None) is not None:
            return int(a.f)
        if getattr(a, "t", None) is not None:
            return self.f_of(float(a.t))
        return 0

    def span(self, a):
        """Frame range from --from/--to (seconds), default the whole clip."""
        f0 = self.f_of(a.t_from) if getattr(a, "t_from", None) is not None else 0
        f1 = self.f_of(a.t_to) if getattr(a, "t_to", None) is not None else self.n - 1
        return f0, f1


def save(img, name, jpg=True):
    OUT.mkdir(exist_ok=True)
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    p = OUT / (name + (".jpg" if jpg else ".png"))
    cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92] if jpg else [])
    return p


def label_band(text, width, h=22, scale=0.5):
    band = np.full((h, width, 3), 32, np.uint8)
    cv2.putText(band, text, (5, h - 7), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return band


def tile_sheet(tiles, labels, cols=None, tile_w=None, title=None):
    """Grid of images, each with a label band above it (labels never cover the picture)."""
    n = len(tiles)
    cols = cols or min(4, n)
    tile_w = tile_w or max(160, min(480, MAX_SIDE // cols))
    out = []
    for im, lab in zip(tiles, labels):
        h = max(1, int(round(tile_w * im.shape[0] / im.shape[1])))
        out.append(np.vstack([label_band(lab, tile_w), cv2.resize(im, (tile_w, h), interpolation=cv2.INTER_AREA)]))
    hmax = max(t.shape[0] for t in out)
    out = [np.vstack([t, np.zeros((hmax - t.shape[0], tile_w, 3), np.uint8)]) for t in out]
    while len(out) % cols:
        out.append(np.zeros_like(out[0]))
    rows = [np.hstack(out[i:i + cols] if cols == 1 else sum(([t, np.full((hmax, 4, 3), 255, np.uint8)] for t in out[i:i + cols]), [])[:-1])
            for i in range(0, len(out), cols)]
    sheet = np.vstack(sum(([r, np.full((4, rows[0].shape[1], 3), 255, np.uint8)] for r in rows), [])[:-1])
    if title:
        sheet = np.vstack([label_band(title, sheet.shape[1], 26, 0.55), sheet])
    return sheet


def flabel(v, i):
    return f"f{i}  t={i / v.fps:.3f}s"


# ───────────────────────── info ─────────────────────────
TIME_RX = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b", re.I)
PAIR_RX = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)\b", re.I)


def question_times(text):
    ts = [float(m.group(1)) for m in TIME_RX.finditer(text)]
    for m in PAIR_RX.finditer(text):
        ts += [float(m.group(1)), float(m.group(2))]
    return sorted(set(ts))


def cmd_info(a):
    v = Video()
    fr = v.all()
    meta = v.q.get("video", {})
    print(f"video: {v.W}x{v.H}, {len(fr)} decoded frames (questions.json says {meta.get('n_frames')}), "
          f"fps {v.fps:g} (container {v.fps_container:g}) -> t = frame / {v.fps:g}; "
          f"last frame at t = {(len(fr) - 1) / v.fps:.3f}s")
    if v.source == "lab":
        print(f"lab recording: one documented camera (~82 deg horizontal FOV): f ~ {v.lab_f:.0f} px (0.58 x width) -- plane/xyz use it "
              "by default; 25 fps capture made 24 fps by dropping a frame after f11, f35, f59 -- fit uses that by default")
    elif v.source == "simulation":
        print("simulation: the camera's field of view can be anything (about 8-103 deg); motion can be keyframed (need not "
              "follow 9.8 m/s^2); the scene can be scaled -- the prior and depth_info set the scale")
    elif v.source == "internet":
        print("internet footage: the camera may move (check a static object before measuring motion)")
    r = frame_jumps(v); P = int(round(v.fps))
    hi = [i for i, x in enumerate(r) if x is not None and x >= 1.4]
    if hi:
        dif = np.diff(hi) if len(hi) >= 3 else np.array([])
        k = int(np.bincount(dif).argmax()) if len(dif) else 0
        regular = len(dif) >= 2 and k < P - 2 and np.mean(np.abs(dif - k) <= 1) >= 0.6
        rates_differ = bool(v.fps_container and abs(v.fps_container / v.fps - 1) > 0.02)
        # once per second: sparse jumps, and one same-phase chain i, i+P, ... (each link >= 1.3x) holds 40%+ of them
        chains = {}
        for i in hi:
            chains.setdefault(i % P, []).append(i)
        best = max(chains.values(), key=len)
        sparse = len(hi) <= max(4, 3 * len(r) / P)
        paired = best if (sparse and len(best) >= 2 and len(best) >= 0.4 * len(hi)
                          and all(r[j] is not None and r[j] >= 1.3 for j in range(best[0], best[-1] + 1, P))) else []
        head = "whole-frame change jumps at " + ", ".join(f"f{i}->f{i + 1} ({r[i]:.1f}x)" for i in hi[:8]) + (" ..." if len(hi) > 8 else "")
        if regular and rates_differ:
            print(head + f"; regular, about every {k} frames: the clip was re-timed (frames dropped OR repeated; compare the stated "
                  "and file rates above). A repeated (held) frame is not a drop: give a step to `fit --gaps` only where a tracked "
                  "mover's step doubles there (fit's NOTE)")
        elif regular:
            print(head + f"; regular, about every {k} frames, but the stated and file rates agree: a periodic motion (gait, "
                  "wingbeat, spin, camera shake), not dropped frames")
        elif paired:
            print(head + f"; {', '.join(f'f{i}' for i in paired)} come {P} frames apart: a source frame dropped once per second? "
                  f"confirm on a mover tracked across one of them (`fit` NOTE), then `fit --gaps {paired[0]}`")
        else:
            print(head + " (isolated: a cut or a sudden motion, unless a tracked mover doubles there too)")
    if v.fps_container and abs(v.fps_container / v.fps - 1) > 0.02:
        print(f"the stated rate ({v.fps:g}) and the file's ({v.fps_container:.4g}) differ: the stated clock holds unless moving objects "
              f"show a regular doubled step that turns one rate into the other (then `fit --gaps` switches to the file's clock and "
              f"asked times are real times, frame ~ t x {v.fps_container:.4g})")
    # Changed pixels between consecutive frames, at FULL resolution (a mean over a shrunken
    # frame misses small moving objects and calls their frames "identical").
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.int16) for f in fr]
    n = np.array([int((np.abs(g[i + 1] - g[i]) > 20).sum()) for i in range(len(g) - 1)])
    if len(n):
        mov = n >= 10
        pattern = "".join("M" if m else "." for m in mov)
        print(f"frame-to-frame change (M = something moved, . = no visible change), pairs f0-f1 ... : "
              + (pattern if len(pattern) <= 120 else pattern[:120] + "..."))
        # a frame held WHILE things are moving: still pair with moving pairs on both sides
        held = [i for i in range(1, len(n) - 1) if not mov[i] and mov[i - 1] and mov[i + 1]]
        if mov.any() and len(held) >= max(3, 0.2 * int(mov.sum())):
            print(f"  {len(held)} frames are held (identical to the previous one) in the middle of motion: the video "
                  "repeats frames, so positions update in steps -- prefer fits over several frames to frame-to-frame differences")
        elif held:
            print(f"  isolated held frame(s) at pair(s) {', '.join(f'f{i}-f{i + 1}' for i in held[:6])}")
    for q in v.q.get("questions", []):
        print(f"\nQ{q['id']}: {q['question']}")
        print(f"  prior: {q.get('prior')}")
        if q.get("depth_info"):
            print(f"  depth_info: {q['depth_info']}")
        print(f"  scene: {q.get('scene')}")
        ts = question_times(q["question"])
        if ts:
            two = v.fps_container and abs(v.fps_container / v.fps - 1) > 0.02
            print("  times: " + ", ".join(f"{t:g}s -> frame {t * v.fps:g}" + ("" if abs(t * v.fps - round(t * v.fps)) < 1e-6 else f" (nearest {round(t * v.fps)})")
                                         + (f" (file's clock: frame {t * v.fps_container:.1f})" if two else "")
                                         + ("" if t * v.fps <= len(fr) - 1 + 1e-6 or (two and t * v.fps_container <= len(fr) - 1 + 1e-6)
                                            else "  [AFTER THE LAST FRAME]") for t in ts))


def fov_note(f, W, source=None):
    """'' inside a 30-90 deg horizontal field of view, else what to do -- real footage only: simulation cameras can have
    any field of view (about 8-103 deg)."""
    fov = 2 * math.degrees(math.atan(W / 2 / f))
    if 30 <= fov <= 90 or source == "simulation":
        return ""
    if source == "lab":
        return (f"  lab recording: the lab camera's f is ~{LAB_F_FRAC * W:.0f} px (~82 deg); {f:.0f} px ({fov:.0f} deg) does not fit -- "
                "the measurement or the distances behind it are wrong; use the lab camera")
    return (f"  OUTSIDE 30-90 deg ({fov:.0f}) for real footage: check the clock (dropped frames, `qpt info`), then the distances "
            "or sizes behind f (agreeing distances can come from another view of the scene) and drop what does not fit; if "
            f"nothing then gives f inside the range, use its nearest end (f = {W / 2:.0f} px at 90 deg, {W / 2 / math.tan(math.radians(15)):.0f} px at 30 deg)")


def gap_unit(gaps, fps, n_last, fps_container=None):
    """Source frame interval when a source frame was dropped at each gap step. Normally the clip keeps its duration
    (n_last / fps). But when the file's own frame rate differs from the stated one and the drops account for exactly
    that difference (container x (n_last + drops) / n_last = stated), the stated rate is the source's: interval 1/fps."""
    unit = (n_last / fps) / (n_last + len(gaps))
    if fps_container and abs(fps_container / fps - 1) > 0.02 and n_last > 0:
        if abs(fps_container * (n_last + len(gaps)) / n_last / fps - 1) < 0.015:
            return 1.0 / fps, (f"the drops turn the file's {fps_container:.3g} fps into the stated {fps:g} fps: source frames are "
                               f"1/{fps:g} s apart and the clip lasts {n_last / fps_container:.3f} s (the file's clock)")
    return unit, ""


def frame_jumps(v):
    """Whole-frame change per step k->k+1 (noise-floored, half size) over its neighbours' mean: a dropped source frame makes
    every moving thing jump at once (~2x), also where no track covers it. Cached in qpt_out/frame_jumps.json."""
    cache = Path("qpt_out/frame_jumps.json")
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    prev, d = None, []
    for f in v.all():
        g = cv2.cvtColor(cv2.resize(f, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            d.append(float(np.mean(np.maximum(np.abs(cv2.GaussianBlur(g - prev, (5, 5), 0)) - 6, 0))))
        prev = g
    n = len(d)
    r = [d[i] / max(1e-4, 0.5 * (d[i - 1] + d[i + 1])) if 0 < i < n - 1 and d[i - 1] > 1e-3 and d[i + 1] > 1e-3 else None
         for i in range(n)]
    cache.parent.mkdir(exist_ok=True); cache.write_text(json.dumps(r))
    return r


def gap_time(frames, gaps, fps, n_last, unit=None):
    """True time of each frame when the source dropped a frame at each gap step k->k+1: two intervals
    pass there. Default interval: the clip keeps its duration (n_last / fps); see gap_unit."""
    frames = np.asarray(frames, float)
    shift = np.zeros_like(frames)
    for k in gaps:
        shift += (frames >= k + 1)
    unit = unit or (n_last / fps) / (n_last + len(gaps))
    return (frames + shift) * unit


# ───────────────────────── frames / sheet ─────────────────────────
def cmd_frames(a):
    v = Video()
    if a.f:
        idx = [int(x) for x in a.f]
    elif a.t:
        idx = [v.f_of(float(x)) for x in a.t]
    else:
        f0, f1 = v.span(a)
        idx = sorted(set(np.linspace(f0, f1, min(a.n, f1 - f0 + 1)).round().astype(int).tolist())) if a.every is None \
            else list(range(f0, f1 + 1, a.every))
    frames = [v.frame(i) for i in idx]
    if a.save:
        for i, f in zip(idx, frames):
            print(save(f, f"frame_f{i:04d}"))
    cols = a.cols or min(4, len(idx))
    tag = f"f{idx[0]}-{idx[-1]}_{len(idx)}"
    p = save(tile_sheet(frames, [flabel(v, i) for i in idx], cols=cols,
                        title=f"{len(idx)} frames  ({v.W}x{v.H}, fps {v.fps:g})"), f"sheet_{tag}", jpg=True)
    print(p)
    print("frames:", " ".join(f"f{i}(t={i / v.fps:.3f})" for i in idx))


# ───────────────────────── view: crop + zoom + pixel grid ─────────────────────────
def nice_step(span, target=10):
    raw = span / target
    for s in [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]:
        if s >= raw:
            return s
    return 1000


def median_background(v, max_n=80):
    fr = v.all()
    idx = np.linspace(0, len(fr) - 1, min(max_n, len(fr))).round().astype(int)
    return np.median(np.stack([fr[i] for i in idx]), axis=0).astype(np.uint8)


def render_view(v, img, box, zoom, grid, lines=(), pts=(), rects=(), smooth=None, title=""):
    x0, y0, x1, y1 = box
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if zoom is None:   # fit about 1200 px: large crops are shrunk HERE (not later), so grid labels stay readable
        zoom = max(0.2, min(16.0, 1200 / max(cw, ch)))
    zw, zh = int(round(cw * zoom)), int(round(ch * zoom))
    if smooth is None:
        smooth = zoom < 3
    big = cv2.resize(crop, (zw, zh), interpolation=cv2.INTER_AREA if zoom < 1 else
                     (cv2.INTER_LANCZOS4 if smooth else cv2.INTER_NEAREST))
    ML, MT, MR, MB = 52, 24, 8, 24
    can = np.full((zh + MT + MB, zw + ML + MR, 3), 255, np.uint8)
    can[MT:MT + zh, ML:ML + zw] = big
    X = lambda x: int(round(ML + (x - x0) * zoom))       # pixel EDGE x -> canvas
    Y = lambda y: int(round(MT + (y - y0) * zoom))
    Xc = lambda x: int(round(ML + (x + 0.5 - x0) * zoom))  # pixel CENTRE -> canvas
    Yc = lambda y: int(round(MT + (y + 0.5 - y0) * zoom))
    step = grid or nice_step(max(cw, ch))
    ov = can.copy()
    for gx in range(int(math.ceil(x0 / step) * step), x1 + 1, step):
        cv2.line(ov, (X(gx), MT), (X(gx), MT + zh), (0, 255, 255), 1)
        cv2.putText(can, str(gx), (X(gx) - 10, MT - 7), FONT, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(can, str(gx), (X(gx) - 10, MT + zh + 16), FONT, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    for gy in range(int(math.ceil(y0 / step) * step), y1 + 1, step):
        cv2.line(ov, (ML, Y(gy)), (ML + zw, Y(gy)), (0, 255, 255), 1)
        cv2.putText(can, str(gy), (3, Y(gy) + 4), FONT, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
    can[MT:MT + zh, ML:ML + zw] = cv2.addWeighted(ov, 0.45, can, 0.55, 0)[MT:MT + zh, ML:ML + zw]
    for k, (ax, ay, bx, by) in enumerate(lines):
        c = COLORS[(k + 3) % len(COLORS)]
        cv2.line(can, (Xc(ax), Yc(ay)), (Xc(bx), Yc(by)), c, 1, cv2.LINE_AA)
        for px, py in ((ax, ay), (bx, by)):
            cv2.circle(can, (Xc(px), Yc(py)), 3, c, -1)
        L = math.hypot(bx - ax, by - ay)
        cv2.putText(can, f"L{k + 1}={L:.1f}px", (Xc((ax + bx) / 2) + 4, Yc((ay + by) / 2) - 4), FONT, 0.45, c, 1, cv2.LINE_AA)
    for k, (px, py) in enumerate(pts):
        cv2.drawMarker(can, (Xc(px), Yc(py)), COLORS[k % len(COLORS)], cv2.MARKER_CROSS, 14, 1)
    for k, (rx0, ry0, rx1, ry1) in enumerate(rects):
        cv2.rectangle(can, (X(rx0), Y(ry0)), (X(rx1 + 1), Y(ry1 + 1)), COLORS[k % len(COLORS)], 1)
    if title:
        can = np.vstack([label_band(title, can.shape[1]), can])
    return can, zoom, step


def parse_box(b, v):
    x0, y0, x1, y1 = [int(round(float(s))) for s in b]
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    return max(0, x0), max(0, y0), min(v.W, x1 + 1), min(v.H, y1 + 1)


def cmd_view(a):
    v = Video()
    i = v.pick(a)
    img = v.frame(i).copy()
    if a.diff:
        bg = median_background(v)
        d = cv2.absdiff(img, bg).max(axis=2)
        img = cv2.cvtColor(np.clip(d.astype(np.float32) * 4, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if a.enhance:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(3.0, (8, 8)).apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    box = parse_box(a.box, v) if a.box else (0, 0, v.W, v.H)
    lines = [tuple(map(float, l)) for l in (a.line or [])]
    pts = [tuple(map(float, p)) for p in (a.pt or [])]
    rects = [tuple(int(float(s)) for s in r) for r in (a.rect or [])]
    title = f"{flabel(v, i)}  crop x {box[0]}..{box[2] - 1}, y {box[1]}..{box[3] - 1}" + ("  [diff vs background]" if a.diff else "")
    can, zoom, step = render_view(v, img, box, a.zoom, a.grid, lines, pts, rects, a.smooth, title)
    name = a.name or f"view_f{i:04d}_{box[0]}_{box[1]}_{box[2] - 1}_{box[3] - 1}"
    print(save(can, name))
    print(f"zoom x{zoom:.2f} (1 original px = {zoom:.2f} image px), grid every {step} px, labels in original pixels"
          + ("; nearest-neighbour upscale shows true pixel edges" if not (a.smooth or zoom < 3) else ""))
    for k, (ax, ay, bx, by) in enumerate(lines):
        print(f"L{k + 1}: ({ax:g},{ay:g}) -> ({bx:g},{by:g})  length {math.hypot(bx - ax, by - ay):.2f} px  "
              f"(dx {bx - ax:+.1f}, dy {by - ay:+.1f})")


# ───────────────────────── edges along a line ─────────────────────────
def first_step(d, spp=4, frac=0.5):
    """Sub-sample index where d (colour difference from this end's background) crosses frac x the level of the FIRST
    step up from the background -- the object's outer boundary even when a darker or brighter interior part (window,
    shading, highlight) sets the line's maximum. None when there is no clear step."""
    if len(d) < 6 * spp or d.max() < 15:
        return None
    sm = np.convolve(d, np.ones(2 * spp + 1) / (2 * spp + 1), "same")
    dd = np.gradient(sm)
    if dd.max() <= 0:
        return None
    k = int(np.nonzero(dd > 0.15 * dd.max())[0][0])          # start of the first significant rise
    j, pk = k, dd[k]
    while j + 1 < len(dd) and dd[j + 1] > 0.25 * pk:          # follow the rise (a blur ramp keeps its slope)
        j += 1; pk = max(pk, dd[j])
    P = float(np.median(d[j:j + 2 * spp + 1]))                # level just inside the first step
    if P < max(15.0, 0.15 * d.max()):
        return None
    above = np.nonzero(d[k:] > frac * P)[0]
    if not len(above):
        return None
    i = k + int(above[0])
    return float(i) if i == 0 or d[i] == d[i - 1] else (i - 1) + (frac * P - d[i - 1]) / (d[i] - d[i - 1])


def line_extent(img, ax, ay, bx, by, frac=0.5, ref=3):
    """Half-contrast extent (px) of the object along one line, each end against its own background; nan if none."""
    L = math.hypot(bx - ax, by - ay); n = max(2, int(L * 4) + 1); s = np.linspace(0, 1, n)
    prof = cv2.remap(img, (ax + s * (bx - ax)).astype(np.float32)[None], (ay + s * (by - ay)).astype(np.float32)[None],
                     cv2.INTER_LINEAR)[0]
    w = max(1, int(ref * 4))
    dl = np.sqrt(((prof - np.median(prof[:w], axis=0)) ** 2).sum(1)); dr = np.sqrt(((prof - np.median(prof[-w:], axis=0)) ** 2).sum(1))
    if min(dl.max(), dr.max()) < 15:
        return float("nan")
    fl, fr = first_step(dl, frac=frac), first_step(dr[::-1], frac=frac)
    if fl is not None and fr is not None and (n - 1 - fr) > fl:
        return ((n - 1 - fr) - fl) / (n - 1) * L
    tl, tr = frac * dl.max(), frac * dr.max()
    ol, orr = np.nonzero(dl > tl)[0], np.nonzero(dr > tr)[0]
    if not len(ol) or not len(orr) or orr[-1] <= ol[0]:
        return float("nan")
    k0, k1 = ol[0], orr[-1]                       # sub-sample threshold crossings, as in cmd_edges
    s0 = k0 - (dl[k0] - tl) / (dl[k0] - dl[k0 - 1]) if k0 > 0 and dl[k0] != dl[k0 - 1] else k0
    s1 = k1 + (dr[k1] - tr) / (dr[k1] - dr[k1 + 1]) if k1 < n - 1 and dr[k1] != dr[k1 + 1] else k1
    return (s1 - s0) / (n - 1) * L


def cmd_edges(a):
    """Colour profile along a line; finds where the object starts and ends."""
    v = Video()
    i = v.pick(a)
    img = v.frame(i).astype(np.float32)
    ax, ay, bx, by = map(float, a.line)
    L = math.hypot(bx - ax, by - ay)
    n = max(2, int(L * 4) + 1)                    # 4 samples per pixel
    s = np.linspace(0, 1, n)
    xs, ys = ax + s * (bx - ax), ay + s * (by - ay)
    prof = cv2.remap(img, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
    w = max(1, int(a.ref * 4))
    if a.inside:
        dist = np.sqrt(((prof - prof[n // 2]) ** 2).sum(1)); dist = dist.max() - dist
        dl = dr = dist
    else:   # each end is compared with ITS OWN background colour (the two sides can differ)
        dl = np.sqrt(((prof - np.median(prof[:w], axis=0)) ** 2).sum(1))
        dr = np.sqrt(((prof - np.median(prof[-w:], axis=0)) ** 2).sum(1))
        dist = np.minimum(dl, dr)
    tl, tr = a.frac * dl.max(), a.frac * dr.max()
    onl, onr = np.nonzero(dl > tl)[0], np.nonzero(dr > tr)[0]
    on = np.nonzero(dist > a.frac * dist.max())[0]
    print(f"{flabel(v, i)}  line ({ax:g},{ay:g}) -> ({bx:g},{by:g}), {L:.2f} px long")
    if not len(onl) or not len(onr) or dist.max() < 15:
        print("no clear object along the line (profile nearly flat)"); return
    k0, k1 = onl[0], onr[-1]
    def cross(d, thr, k, kprev):   # sub-sample crossing of the threshold
        d0, d1 = d[kprev], d[k]
        return kprev + (thr - d0) / (d1 - d0) * (k - kprev) if d1 != d0 else k
    s0 = cross(dl, tl, k0, max(k0 - 1, 0)) / (n - 1); s1 = cross(dr, tr, k1, min(k1 + 1, n - 1)) / (n - 1)
    g_ext = (s1 - s0) * L; how = f"threshold {a.frac:.0%} of the max colour difference from {'the line centre' if a.inside else 'each end'}"
    if not a.inside:      # each end at half the height of its FIRST step from the background (outer boundary)
        fl, fr = first_step(dl, frac=a.frac), first_step(dr[::-1], frac=a.frac)
        if fl is not None and fr is not None and (n - 1 - fr) > fl:
            s0, s1 = fl / (n - 1), (n - 1 - fr) / (n - 1)
            how = f"each end at {a.frac:.0%} of its first step up from that end's background"
    if s1 <= s0:
        print("could not find two edges (the two ends see different backgrounds and the object is not distinct) -- "
              "move the line ends onto plain background just outside the object, or use qpt view / qpt seg"); return
    p0 = (ax + s0 * (bx - ax), ay + s0 * (by - ay)); p1 = (ax + s1 * (bx - ax), ay + s1 * (by - ay))
    print(f"object runs from ({p0[0]:.1f},{p0[1]:.1f}) to ({p1[0]:.1f},{p1[1]:.1f}): extent {(s1 - s0) * L:.2f} px ({how})")
    if abs(g_ext - (s1 - s0) * L) > 0.05 * (s1 - s0) * L:
        print(f"  (the line's global half-maximum gives {g_ext:.2f} px: an interior part -- window, shading, highlight -- or "
              "something beyond the object sets the maximum; check the picture to see which ends are the object's)")
    if not a.inside:      # how much the extent depends on where the edge is put
        ext = []
        for fr_ in (0.25, 0.5, 0.75):
            ol, orr = np.nonzero(dl > fr_ * dl.max())[0], np.nonzero(dr > fr_ * dr.max())[0]
            ext.append((orr[-1] - ol[0]) / (n - 1) * L if len(ol) and len(orr) and orr[-1] > ol[0] else float("nan"))
        print(f"extent at 25% / 50% / 75% contrast: {ext[0]:.2f} / {ext[1]:.2f} / {ext[2]:.2f} px  "
              "(the half-contrast value is the edge; a big spread means a blurred or soft edge)")
        if a.sweep:       # parallel copies of the line: a line off the object's centre measures a chord, not its width
            nx, ny = -(by - ay) / L, (bx - ax) / L
            offs = np.arange(-a.sweep, a.sweep + 0.01, 0.5)
            ex = np.array([line_extent(img, ax + o * nx, ay + o * ny, bx + o * nx, by + o * ny, a.frac, a.ref) for o in offs])
            if np.isfinite(ex).any():
                j = int(np.nanargmax(ex)); top = np.sort(ex[np.isfinite(ex)])[-3:]
                print(f"sweep: {len(offs)} parallel lines within +-{a.sweep:g} px: widest extent {ex[j]:.2f} px at offset "
                      f"{offs[j]:+.1f} px (mean of the 3 widest {top.mean():.2f} px) -- the object's extent along THIS "
                      f"direction only (your line gave {(s1 - s0) * L:.2f} px): point the line along the asked dimension "
                      "(a wheel seen at an angle: across its face, not along the axle)")
        elif (s1 - s0) * L < 20:
            print("small object (< 20 px): a line off its centre measures a shorter chord -- rerun with --sweep 4 (widest of "
                  "parallel lines), at several frames, and take the median")
    if k0 < 2 or k1 > n - 3:
        print("warning: the object touches an end of the line -- start and end the line on background")
    outside = on[(on < s0 * (n - 1) - 8) | (on > s1 * (n - 1) + 8)]
    if len(outside):
        print("warning: something else stands out on the line beyond the detected ends (another object, shadow or "
              "texture) -- check the picture that the ends are this object's")
    # compact profile: 40 bins
    bins = np.array_split(dist, min(40, n))
    print("profile (0-9 = colour difference, | = threshold crossings):")
    print("  " + "".join(str(min(9, int(10 * b.mean() / (dist.max() + 1e-9)))) for b in bins))
    pad = max(6, int(0.15 * L))
    x0, x1 = int(min(ax, bx) - pad), int(max(ax, bx) + pad)
    y0, y1 = int(min(ay, by) - pad), int(max(ay, by) + pad)
    box = parse_box([x0, y0, x1, y1], v)
    can, zoom, step = render_view(v, v.frame(i), box, None, None, [(ax, ay, bx, by), (p0[0], p0[1], p1[0], p1[1])],
                                  title=f"{flabel(v, i)}  L1 = probe line, L2 = detected extent")
    print(save(can, f"edges_f{i:04d}_{int(ax)}_{int(ay)}_{int(bx)}_{int(by)}"))


# ───────────────────────── motion: background subtraction ─────────────────────────
def camera_shift(v, k=None):
    fr = v.all()
    if len(fr) < 3:
        return 0.0
    k = k or max(1, len(fr) // 10)
    g = [np.float32(cv2.cvtColor(cv2.resize(fr[i], (320, max(1, int(320 * v.H / v.W)))), cv2.COLOR_BGR2GRAY))
         for i in range(0, len(fr), k)]
    sh = [math.hypot(*cv2.phaseCorrelate(g[i], g[i + 1])[0]) * v.W / 320 for i in range(len(g) - 1)]
    return float(np.median(sh)) if sh else 0.0


def foreground_blobs(v, bg, f, thr, min_area, box=None):
    d = cv2.absdiff(v.frame(f), bg).max(axis=2)
    m = (d > thr).astype(np.uint8)
    if box is not None:
        keep = np.zeros_like(m); x0, y0, x1, y1 = box; keep[y0:y1, x0:x1] = 1; m *= keep
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for j in range(1, n):
        if st[j, cv2.CC_STAT_AREA] >= min_area:
            x, y, w, h, ar = st[j]
            edge = int(x <= 0 or y <= 0 or x + w >= v.W or y + h >= v.H)   # cut by the frame border
            out.append(dict(f=f, cx=float(cen[j][0]), cy=float(cen[j][1]), x0=int(x), y0=int(y), w=int(w), h=int(h), area=int(ar), edge=edge))
    return out, m


def link(dets_by_frame, max_jump):
    """Greedy nearest-centroid linking of per-frame blobs into tracks."""
    tracks, nxt = {}, 0
    last = {}
    for f in sorted(dets_by_frame):
        used = set()
        pairs = sorted(((math.hypot(d["cx"] - last[t]["cx"], d["cy"] - last[t]["cy"]), t, j)
                        for t in last for j, d in enumerate(dets_by_frame[f])
                        if f - last[t]["f"] <= 3), key=lambda p: p[0])
        assigned = {}
        for dist, t, j in pairs:
            if dist > max_jump * (f - last[t]["f"]) or t in assigned or j in used:
                continue
            assigned[t] = j; used.add(j)
        for t, j in assigned.items():
            d = dict(dets_by_frame[f][j], track=t); tracks[t].append(d); last[t] = d
        for j, d in enumerate(dets_by_frame[f]):
            if j not in used:
                d = dict(d, track=nxt); tracks[nxt] = [d]; last[nxt] = d; nxt += 1
    return tracks


def cmd_motion(a):
    v = Video()
    fr = v.all()
    sh = camera_shift(v)
    print(f"camera motion check: median shift between sampled frames {sh:.1f} px"
          + ("  -> the CAMERA MOVES: background subtraction is unreliable; use qpt track / qpt segtrack" if sh > 2 else " (static camera)"))
    bg = median_background(v)
    f0, f1 = v.span(a)
    idx = list(range(f0, f1 + 1, a.every))
    min_area = a.min_area or (max(6, int(2e-5 * v.W * v.H)) if a.box else max(12, int(2e-4 * v.W * v.H)))
    dets, masks = {}, {}
    for f in idx:
        dets[f], masks[f] = foreground_blobs(v, bg, f, a.thr, min_area, parse_box(a.box, v) if a.box else None)
    tracks = link(dets, max_jump=a.max_jump or 0.08 * math.hypot(v.W, v.H))
    keep = {t: r for t, r in tracks.items() if len(r) >= max(3, int(0.2 * len(idx)))}
    keep = dict(sorted(keep.items(), key=lambda kv: -np.median([d["area"] for d in kv[1]]))[: a.top])
    OUT.mkdir(exist_ok=True)
    path = OUT / "motion.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["track", "f", "t", "cx", "cy", "x0", "y0", "w", "h", "area", "equiv", "edge"])
        for t, rows in keep.items():
            for d in rows:
                w.writerow([t, d["f"], f"{d['f'] / v.fps:.4f}", f"{d['cx']:.2f}", f"{d['cy']:.2f}", d["x0"], d["y0"], d["w"], d["h"], d["area"], f"{2 * math.sqrt(d['area'] / math.pi):.2f}", d["edge"]])
    print(f"background = median of the clip; foreground = colour change > {a.thr} grey levels, blobs >= {min_area} px; "
          f"{len(idx)} frames f{f0}..f{f1} every {a.every}")
    print(f"{len(keep)} moving blob track(s) (largest first) -> {path}  (rows: track,f,t,cx,cy,x0,y0,w,h,area,equiv,edge)")
    print("edge = 1: the blob touches the frame border, so it is cut off -- its centre and size are biased; leave those frames out of fits")
    print(f"{'track':>5} {'frames':>12} {'n':>4} {'med w':>6} {'med h':>6} {'med area':>8}  start (cx,cy) -> end (cx,cy)   px/frame  edge frames")
    for t, rows in keep.items():
        a0, a1 = rows[0], rows[-1]
        sp = math.hypot(a1["cx"] - a0["cx"], a1["cy"] - a0["cy"]) / max(1, a1["f"] - a0["f"])
        fr_ = "f%d-f%d" % (a0["f"], a1["f"])
        print(f"{t:>5} {fr_:>12} {len(rows):>4} {np.median([d['w'] for d in rows]):>6.0f} "
              f"{np.median([d['h'] for d in rows]):>6.0f} {np.median([d['area'] for d in rows]):>8.0f}  "
              f"({a0['cx']:.0f},{a0['cy']:.0f}) -> ({a1['cx']:.0f},{a1['cy']:.0f})   {sp:8.2f}  "
              + (",".join(f"f{d['f']}" for d in rows if d["edge"]) or "-"))
    show = np.linspace(0, len(idx) - 1, min(8, len(idx))).round().astype(int)
    tiles, labs = [], []
    for s in show:
        f = idx[s]
        im = v.frame(f).copy()
        cs = cv2.findContours(masks[f], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(im, cs, -1, (255, 0, 255), 1)
        for t, rows in keep.items():
            for d in rows:
                if d["f"] == f:
                    c = COLORS[t % len(COLORS)]
                    cv2.rectangle(im, (d["x0"], d["y0"]), (d["x0"] + d["w"] - 1, d["y0"] + d["h"] - 1), c, 2)
                    cv2.putText(im, f"#{t}", (d["x0"], max(12, d["y0"] - 4)), FONT, 0.6, c, 2, cv2.LINE_AA)
        tiles.append(im); labs.append(flabel(v, f))
    print(save(tile_sheet(tiles, labs, title="moving blobs (magenta outline = changed pixels, boxes = tracks; shadows and blur count as change)"),
               "motion_sheet", jpg=True))


# ───────────────────────── track: template matching ─────────────────────────
def cmd_track(a):
    v = Video()
    fr = v.all()
    fs = v.pick(a)
    x0, y0, x1, y1 = parse_box(a.box, v)
    tw, th = x1 - x0, y1 - y0
    tmpl0 = cv2.cvtColor(fr[fs][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if len(fr) >= 8:   # how much of the box is the mover? (vs the clip's median = static background)
        bg = np.median(np.stack([cv2.cvtColor(fr[i][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                                 for i in np.linspace(0, len(fr) - 1, min(15, len(fr))).astype(int)]), axis=0)
        fg = float(np.mean(np.abs(tmpl0.astype(np.float32) - bg) > 20))
        if fg < 0.3:
            print(f"WARNING: only {fg:.0%} of the box differs from the static background -- the template may follow the "
                  "background, not the object: tighten the box on the object, or use segtrack / motion (whole-object tracks)")
    f0, f1 = v.span(a)
    res = {fs: (x0 + tw / 2, y0 + th / 2, 1.0, 1.0)}
    scales = [1.0] if not a.scale_search else [0.9, 0.95, 1.0, 1.05, 1.1]
    stopped = []
    for direction in (+1, -1):
        cx, cy, sc = x0 + tw / 2, y0 + th / 2, 1.0
        vx = vy = 0.0                                   # px per frame, for the search centre
        f = fs + direction
        while f0 <= f <= f1:
            px_, py_ = cx + vx, cy + vy                 # constant-velocity prediction
            bw, bh = tw * sc, th * sc
            inside = (min(px_ + bw / 2, v.W) - max(px_ - bw / 2, 0)) * (min(py_ + bh / 2, v.H) - max(py_ - bh / 2, 0)) / (bw * bh)
            if inside < 0.7:
                stopped.append(f"object leaves the frame near f{f} -- tracking stopped there"); break
            g = cv2.cvtColor(fr[f], cv2.COLOR_BGR2GRAY)
            best = None
            for s_ in scales:
                s = sc * s_
                tm = cv2.resize(tmpl0, (max(4, int(round(tw * s))), max(4, int(round(th * s)))))
                r = int(a.search or max(12, 0.75 * max(tm.shape)))
                sx0, sy0 = int(max(0, px_ - tm.shape[1] / 2 - r)), int(max(0, py_ - tm.shape[0] / 2 - r))
                sx1, sy1 = int(min(v.W, px_ + tm.shape[1] / 2 + r)), int(min(v.H, py_ + tm.shape[0] / 2 + r))
                win = g[sy0:sy1, sx0:sx1]
                if win.shape[0] < tm.shape[0] or win.shape[1] < tm.shape[1]:
                    continue
                m = cv2.matchTemplate(win, tm, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(m)
                qx, qy = float(loc[0]), float(loc[1])
                if 0 < loc[0] < m.shape[1] - 1:          # parabolic sub-pixel refinement
                    l, c, rr = m[loc[1], loc[0] - 1], m[loc[1], loc[0]], m[loc[1], loc[0] + 1]
                    den = l - 2 * c + rr; qx += 0.5 * (l - rr) / den if den else 0
                if 0 < loc[1] < m.shape[0] - 1:
                    u, c, dd = m[loc[1] - 1, loc[0]], m[loc[1], loc[0]], m[loc[1] + 1, loc[0]]
                    den = u - 2 * c + dd; qy += 0.5 * (u - dd) / den if den else 0
                cand = (mx, sx0 + qx + tm.shape[1] / 2, sy0 + qy + tm.shape[0] / 2, s)
                if best is None or cand[0] > best[0]:
                    best = cand
            if best is None or best[0] < 0.3:
                stopped.append(f"lost the object at f{f} (match score {best[0] if best else 0:.2f}) -- tracking stopped there"); break
            score, nx, ny, sc = best
            n_done = abs(f - fs)
            vx, vy = (nx - cx, ny - cy) if n_done == 1 else (0.5 * vx + 0.5 * (nx - cx), 0.5 * vy + 0.5 * (ny - cy))
            cx, cy = nx, ny
            res[f] = (cx, cy, sc, score)
            f += direction
    OUT.mkdir(exist_ok=True)
    name = a.name or f"track_f{fs}_{x0}_{y0}"
    path = OUT / f"{name}.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["f", "t", "cx", "cy", "w", "h", "scale", "score"])
        for f in sorted(res):
            cx, cy, s, sc = res[f]
            w.writerow([f, f"{f / v.fps:.4f}", f"{cx:.2f}", f"{cy:.2f}", f"{tw * s:.1f}", f"{th * s:.1f}", f"{s:.3f}", f"{sc:.3f}"])
    lows = [f for f in res if res[f][3] < 0.6]
    for msg in stopped:
        print("  " + msg)
    print(f"tracked template {tw}x{th} px from f{fs} over f{min(res)}..f{max(res)} ({len(res)} frames) -> {path}")
    print("  columns: f,t,cx,cy (box centre, px), w,h (box size), scale, score (1 = perfect match)")
    if lows:
        print(f"  WARNING: weak match (score < 0.6) on {len(lows)} frame(s), e.g. {sorted(lows)[:8]} -- the track may have slipped")
    show = sorted(res)[:: max(1, len(res) // 8)][:8]
    tiles, labs = [], []
    for f in show:
        cx, cy, s, sc = res[f]
        im = fr[f].copy(); bw, bh = tw * s, th * s
        cv2.rectangle(im, (int(cx - bw / 2), int(cy - bh / 2)), (int(cx + bw / 2), int(cy + bh / 2)), (0, 0, 255), 2)
        cv2.drawMarker(im, (int(cx), int(cy)), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)
        tiles.append(im); labs.append(f"{flabel(v, f)} score {sc:.2f}")
    print(save(tile_sheet(tiles, labs, title=f"template track from f{fs}"), f"{name}_sheet", jpg=True))


# ───────────────────────── fit: speed / acceleration from a track ─────────────────────────
def load_track(a):
    """(t, P, names): times and positions (n x d, d = 2 or 3) from a csv, rows filtered by track/obj."""
    rows = list(csv.DictReader(open(a.csv)))
    if not rows:
        sys.exit("qpt: empty csv")
    for col, val in (("track", a.track), ("obj_id", a.obj)):
        if val is not None and col in rows[0]:
            rows = [r for r in rows if int(float(r[col])) == int(val)]
    if rows and "edge" in rows[0] and not a.keep_edge:
        n0 = len(rows); rows = [r for r in rows if r["edge"] in ("0", "")]
        if len(rows) < n0:
            print(f"(left out {n0 - len(rows)} sample(s) cut by the frame border; --keep-edge to use them)")
    if not rows:
        sys.exit("qpt: no rows left (check --track/--obj)")
    names = (a.cols.split(",") if a.cols else
             next((list(c) for c in (("X", "Y", "Z"), ("cx", "cy"), ("u", "v"), ("x", "y")) if all(k in rows[0] for k in c)), None))
    if not names:
        sys.exit("qpt: give --cols X,Y[,Z]")
    fps = Video().fps if Path("video.mp4").exists() else None
    t = np.array([float(r["t"]) if r.get("t") not in (None, "") else float(r["f"]) / fps for r in rows])
    P = np.array([[float(r[k]) if r[k] not in ("", "nan") else np.nan for k in names] for r in rows])
    scol = next((c for c in ("equiv", "size_px", "feret", "major", "w", "bw") if c in rows[0]), None)
    S_ = np.array([float(r[scol]) if r[scol] not in ("", "nan") else np.nan for r in rows]) if scol else None
    ok = np.isfinite(P).all(1)
    o = np.argsort(t[ok])
    load_track.size = (scol, S_[ok][o]) if scol else None
    return t[ok][o], P[ok][o], names


def poly_at(t, P, T, k, order):
    near = np.argsort(np.abs(t - T))[: 2 * k + 1]
    tt = t[near] - T
    if len(tt) < order + 2:
        return None
    co = [np.polyfit(tt, P[near, j], order) for j in range(P.shape[1])]
    vel = np.array([np.polyval(np.polyder(c), 0) for c in co])
    acc = np.array([np.polyval(np.polyder(c, 2), 0) if order >= 2 else 0.0 for c in co])
    res = np.sqrt(np.mean(sum((np.polyval(c, tt) - P[near, j]) ** 2 for j, c in enumerate(co))))
    sp = float(np.linalg.norm(vel))
    return dict(span=(t[near].min(), t[near].max()), n=len(near), vel=vel, v=sp,
                acc=acc, a=float(np.linalg.norm(acc)), a_tan=float(acc @ vel / sp) if sp > 0 else 0.0, rms=res)


def pos_at(t, P, T):
    """Measured position at T: the sample itself, or linear interpolation between the two
    neighbouring samples (no fitted curve: fits are biased at the ends of a track)."""
    if T <= t[0]:
        return P[0]
    if T >= t[-1]:
        return P[-1]
    return np.array([np.interp(T, t, P[:, j]) for j in range(P.shape[1])])


def step_check(t, P):
    """Frame steps that are irregular: a held frame (no motion) or a jump about twice the neighbours."""
    if len(t) < 5:
        return []
    d = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(np.diff(t), 1e-9)
    moving = np.median(d[d > 0]) if (d > 0).any() else 0.0
    if np.median(np.linalg.norm(np.diff(P, axis=0), axis=1)) < 1.0:
        return []                       # under a pixel per frame: steps are measurement noise
    out = []
    tm = (t[:-1] + t[1:]) / 2
    for i in range(1, len(d) - 1):
        nb = [j for j in range(max(0, i - 3), min(len(d), i + 4)) if j != i]
        if len(nb) >= 3:                # expected step from the neighbours' trend (handles acceleration)
            c = np.polyfit(tm[nb], d[nb], 1)
            ref = float(np.polyval(c, tm[i]))
        else:
            ref = float(np.median(d[nb])) if nb else 0.0
        if ref < 0.5 * moving:          # starting, stopping or resting: steps are noise, not frame timing
            continue
        if d[i] > 1.6 * ref or d[i] < 0.35 * ref:
            nbr = [j for j in (i - 1, i + 1) if 0 <= j < len(d)]
            if len(nb) >= 3 and len(nbr) == 2:       # clean doubling: both neighbours follow the trend (not a fall
                fit_ = np.polyval(np.polyfit(tm[nb], d[nb], 2 if len(nb) >= 5 else 1), tm[nbr])   # start or a bounce)
                clean = 1.7 <= d[i] / ref <= 2.3 and all(abs(d[j] / max(fj, 1e-9) - 1) < 0.25 for j, fj in zip(nbr, fit_))
            else:
                clean = False
            out.append((t[i], t[i + 1], d[i] / ref, clean))
    return out


GAPS_FILE = Path("qpt_out/gaps.txt")   # dropped-frame steps, once given, apply to every later fit


def full_gaps(gaps, fps, n, fps_container=None):
    """A frame-rate conversion that drops one source frame per second of output drops them every round(fps) frames:
    when the given steps share that period, return every step of the pattern inside the clip (where nothing moves the
    drop is invisible, but the time still passes). Steps written K! are kept exactly as given (no pattern)."""
    if any(str(x).endswith("!") for x in gaps):
        return sorted({int(str(x).rstrip("!")) for x in gaps}), False
    P = int(round(fps)); g = sorted({int(x) for x in gaps})
    if fps_container and abs(fps_container / fps - 1) > 0.02:     # another conversion: give every step found
        return g, False
    if g and P > 1 and all((x - g[0]) % P == 0 for x in g):
        return sorted(set(range(g[0] % P, n - 1, P))), True
    return g, False


def wobble(t, P, fps):
    """Smooth path (quadratic) plus one periodic swing (fundamental + 2nd harmonic): a flapping, walking or spinning
    body's swing dominates short-window accelerations. Returns the best period, the share of the quadratic's residual
    the swing explains, its amplitude, the smooth fit's coefficients per component, and whether it is a WOBBLE."""
    t = np.asarray(t, float); span = t.max() - t.min()
    if len(t) < 12 or span <= 0:
        return None
    t0 = t - t.min()
    design = lambda w: np.array([np.ones_like(t0), t0, 0.5 * t0 ** 2] + ([np.sin(w * t0), np.cos(w * t0), np.sin(2 * w * t0),
                                np.cos(2 * w * t0)] if w else [])).T
    A0 = design(0)
    var0 = sum(np.sum((P[:, j] - A0 @ np.linalg.lstsq(A0, P[:, j], rcond=None)[0]) ** 2) for j in range(P.shape[1]))
    best = None
    for per in np.arange(4 / fps, span / 2.5, 0.25 / fps):
        A = design(2 * np.pi / per); cs = [np.linalg.lstsq(A, P[:, j], rcond=None)[0] for j in range(P.shape[1])]
        res = sum(np.sum((P[:, j] - A @ c) ** 2) for j, c in enumerate(cs))
        if best is None or res < best[0]:
            best = (res, per, cs)
    if best is None or var0 <= 0:
        return None
    res, per, cs = best
    frac = 1 - res / var0
    amp = max(np.hypot(c[3], c[4]) + np.hypot(c[5], c[6]) for c in cs)
    travel = max(np.ptp(P[:, j]) for j in range(P.shape[1]))
    return dict(period=per, explained=frac, amp=amp, cs=cs, t0=t.min(),
                flag=frac >= 0.6 and amp >= 2.0 and amp <= 0.1 * travel)


def cmd_fit(a):
    t, P, names = load_track(a)
    vv = Video() if Path("video.mp4").exists() else None
    if a.gaps is None and not (GAPS_FILE.exists() and GAPS_FILE.read_text().strip()) and vv and vv.source == "lab" \
            and abs(vv.fps - 24) < 0.5:
        a.gaps = ",".join(str(k) for k in LAB_DROPS if k < vv.n - 1)
        print(f"lab recording: 25 fps capture made 24 fps by dropping one frame per second -> true time axis with --gaps "
              f"{a.gaps} by default (--gaps none to switch off)")
    if a.gaps is None and GAPS_FILE.exists() and GAPS_FILE.read_text().strip():
        a.gaps = GAPS_FILE.read_text().strip()
        print(f"using the dropped-frame steps in {GAPS_FILE} ({a.gaps}); --gaps none to fit on t = frame / fps")
    elif a.gaps and a.gaps not in ("auto", "none"):
        GAPS_FILE.parent.mkdir(exist_ok=True); GAPS_FILE.write_text(a.gaps)
        print(f"saved --gaps {a.gaps} to {GAPS_FILE}: every later fit uses it")
    if a.gaps == "none":
        a.gaps = None
    if a.gaps == "auto":
        v = Video()
        dbl = [b_ for (a_, b_, r_, c_) in step_check(t, P) if c_]
        a.gaps = ",".join(str(int(round(b_ * v.fps)) - 1) for b_ in dbl)
        print(f"--gaps auto: doubled steps on this track after frame(s) {a.gaps or 'none'}")
    if a.gaps:
        v = Video()
        given = a.gaps.split(",")
        gaps, ext = full_gaps(given, v.fps, v.n, v.fps_container)
        if ext and gaps != sorted({int(x) for x in given}):
            print(f"dropped steps repeat every {int(round(v.fps))} frames (a conversion from {int(round(v.fps)) + 1} fps drops one "
                  f"source frame per second): using the whole pattern {','.join(map(str, gaps))} (also where nothing moves); "
                  f"write --gaps {','.join(x.rstrip('!') + '!' for x in given)} to keep only the given steps")
            GAPS_FILE.parent.mkdir(exist_ok=True); GAPS_FILE.write_text(",".join(map(str, gaps)))
        fr_ = np.round(t * v.fps)
        unit, why = gap_unit(gaps, v.fps, v.n - 1, v.fps_container)
        t = gap_time(fr_, gaps, v.fps, v.n - 1, unit)
        if why:   # the file's clock is the real one: an asked time IS a true time (its frame is about t x file fps)
            print(why + "; asked times are real times on this clock (frame ~ t x " + f"{v.fps_container:.4g})")
            conv = lambda T: float(T)
        else:
            conv = lambda T: float(gap_time([T * v.fps], gaps, v.fps, v.n - 1, unit)[0])
        if a.at is not None:
            a.at = conv(a.at)
        if a.avg:
            a.avg = [conv(x) for x in a.avg]
        if a.accel:
            a.accel = [conv(x) for x in a.accel]
        ms = unit * 1000
        print(f"true time axis: source frames dropped after f{', f'.join(str(k) for k in gaps)} -- each of those steps "
              f"lasts two intervals of {ms:.2f} ms; true time of frame f = (f + dropped frames before f) x {ms:.2f} ms "
              "(use it in your own code too); " + ("" if why else "asked times are mapped frame = t x fps (the question "
              "convention), ") + "times printed below are true times")
    if a.axis != "all":
        keep = [names.index(c) for c in a.axis.split(",")] if all(c in names for c in a.axis.split(",")) else \
            {"x": [0], "y": [1], "xy": [0, 1]}[a.axis]
        P = P[:, keep]; names = [names[j] for j in keep]
    S = a.scale or 1.0
    U = a.unit if a.scale else ("m" if names[0] == "X" else "px")
    rU = "m" if names[0] == "X" else "px"   # unit of the fit residual
    if getattr(load_track, "size", None):
        scol, sz = load_track.size
        good = np.isfinite(sz) & (sz > 0)
        if good.sum() >= 4:
            sl = np.polyfit(t[good], np.log(sz[good]), 1)[0]
            print(f"apparent size ('{scol}') changes {sl * 100:+.1f} %/s over the track: the distance to the camera changes "
                  f"by about {-sl * 100:+.1f} %/s ({'approaching' if sl > 0 else 'receding'} -- part of the motion is along the "
                  f"line of sight)" if abs(sl) > 0.02 else
                  f"apparent size ('{scol}') is steady ({sl * 100:+.1f} %/s): little motion toward or away from the camera")
    odd = step_check(t, P)
    if odd and not a.gaps:
        print("NOTE: irregular frame steps (held or dropped frame, or a tracking jump): "
              + ", ".join(f"{a_:.3f}-{b_:.3f}s x{r_:.1f}" for a_, b_, r_, c_ in odd[:6])
              + " -- keep fit windows off them")
        v_ = Video() if Path("video.mp4").exists() else None
        dbl = [int(round(b_ * v_.fps)) - 1 for a_, b_, r_, c_ in odd if c_] if v_ else []
        if dbl:
            fj = frame_jumps(v_)
            jr = {k: fj[k] for k in dbl if k < len(fj) and fj[k] is not None}
            print(f"  a step about 2x its neighbours (after frame {', '.join(map(str, dbl))}); the whole frame changes "
                  + ", ".join(f"{jr[k]:.1f}x at f{k}" for k in jr) + " there. "
                  + ("The whole frame jumps too: a source frame was dropped, and EVERY interval of the clip is shorter than "
                     f"its mean step (speeds and accelerations low on t = frame / fps, even in windows that avoid the step): "
                     f"refit with --gaps {','.join(str(k) for k in jr if jr[k] >= 1.3)} (remembered for every later fit; use "
                     "the true times it prints in your own code too)" if any(x >= 1.3 for x in jr.values()) else
                     "The rest of the frame does not jump: a tracking jump, unless another mover doubles at the same step"))
    wb = wobble(t, P, Video().fps if Path("video.mp4").exists() else 1 / np.median(np.diff(t))) if names[0] != "X" else None
    if wb and wb["flag"]:
        acc_s = float(np.linalg.norm([c[2] for c in wb["cs"]])) * S
        vs = lambda T: float(np.linalg.norm([c[1] + c[2] * (T - wb["t0"]) for c in wb["cs"]])) * S
        print(f"WOBBLE: a periodic swing (period {wb['period']:.3f}s, about {wb['amp']:.1f} px, {wb['explained']:.0%} of what a "
              f"smooth path leaves) rides on this track -- a flapping, walking or spinning body: short windows measure the "
              f"swing, not the path. The smooth path over the whole track: |acc| = {acc_s:.4g} {U}/s^2"
              + (f", speed at t={a.at}s = {vs(a.at):.4g} {U}/s" if a.at is not None else "")
              + ". Use these for the body's motion (and as a ruler), unless the motion is not constant-acceleration")
    print(f"{len(t)} samples, t {t.min():.3f}..{t.max():.3f}s; components {','.join(names)}; "
          + (f"scale {S:g} {a.unit} per px" if a.scale else f"units: {U}"))
    if a.at is not None:
        T = a.at
        if T < t.min() - 1e-9 or T > t.max() + 1e-9:
            print(f"WARNING: t={T} is outside the tracked span -- extrapolating")
        print(f"instantaneous at t={T}s (polynomial order {a.order} fitted to the 2k+1 samples nearest to t):")
        print(f"  {'k':>3} {'span (s)':>15} {'speed':>10} " + " ".join(f"{'v' + c:>9}" for c in names) + f" {'|acc|':>10} {'a_tan':>9} {'rms ' + rU:>8}")
        vs, accs = [], []
        for k in a.k:
            r = poly_at(t, P, T, k, a.order)
            if r is None:
                continue
            vs.append(r["v"] * S); accs.append(r["a"] * S)
            print(f"  {k:>3} {r['span'][0]:7.3f}-{r['span'][1]:<7.3f} {r['v'] * S:10.4g} " + " ".join(f"{x * S:9.3g}" for x in r["vel"])
                  + f" {r['a'] * S:10.4g} {r['a_tan'] * S:9.3g} {r['rms']:8.3g}")
        if vs:
            print(f"  median speed {np.median(vs):.4g} {U}/s; median |acc| {np.median(accs):.4g} {U}/s^2 "
                  f"(spread across windows: speed {min(vs):.3g}-{max(vs):.3g})")
            if np.median(vs) > 0 and (max(vs) - min(vs)) / np.median(vs) > 0.25:
                print("  WARNING: the speed depends strongly on the window -- look for a held/dropped frame, a bounce or "
                      "collision, or tracking noise before using it (especially as a ruler)")
            if np.median(accs) > 0 and (max(accs) - min(accs)) / np.median(accs) > 0.5:
                print("  WARNING: the acceleration depends strongly on the window -- do not use it as a ruler; fit only a "
                      "window of smooth, free motion")
            print("  |acc| = magnitude of the acceleration VECTOR (includes turning); a_tan = only the part along the "
                  "motion (rate of change of speed). \"Acceleration\" means |acc| unless the question says otherwise.")
    if a.avg:
        T0, T1 = a.avg
        sel = (t >= T0 - 1e-3) & (t <= T1 + 1e-3)
        if sel.sum() < 2:
            print("average: fewer than 2 samples in the window")
        else:
            Ps = np.vstack([pos_at(t, P, T0), P[sel], pos_at(t, P, T1)])
            disp = float(np.linalg.norm(Ps[-1] - Ps[0]))
            steps = np.linalg.norm(np.diff(Ps, axis=0), axis=1)
            raw = float(steps.sum())
            n = len(Ps); st = max(1, min(3, (n - 1) // 4))
            k = sorted(set(list(range(0, n, st)) + [n - 1]))
            coarse = float(np.linalg.norm(np.diff(Ps[k], axis=0), axis=1).sum())
            dt = T1 - T0
            print(f"average over [{T0}, {T1}]s ({sel.sum()} samples inside; end positions = measured samples, "
                  "linearly interpolated if the window ends fall between samples):")
            print(f"  straight-line displacement {disp * S:.4g} {U} -> {disp * S / dt:.4g} {U}/s")
            print(f"  path length, every sample {raw * S:.4g} {U} -> {raw * S / dt:.4g} {U}/s (position jitter inflates it)"
                  + (f";  every {st} samples {coarse * S:.4g} {U} -> {coarse * S / dt:.4g} {U}/s" if st > 1 else ""))
            tol = 0.5 * float(np.median(np.diff(t))) if len(t) > 1 else 1e-3
            if T0 < t[0] - tol or T1 > t[-1] + tol:
                print(f"  WARNING: the track only covers {t[0]:.3f}..{t[-1]:.3f}s; outside it the position is held at the "
                      "last sample, so these averages are wrong for the asked window -- extend the track or reason explicitly")
            if disp > 0 and raw / disp > 1.15:
                print(f"  NOTE: path is {raw / disp:.2f}x the straight line: either the motion curves or the positions are noisy "
                      "-- look at the track picture")
    if a.accel:
        T0, T1 = a.accel
        sel = (t >= T0 - 1e-3) & (t <= T1 + 1e-3)
        if sel.sum() < 4:
            print("accel: fewer than 4 samples in the window")
        else:
            tt = t[sel] - t[sel].mean()
            accs = []
            for j, name in enumerate(names):
                c = np.polyfit(tt, P[sel, j], 2)
                rms = np.sqrt(np.mean((np.polyval(c, tt) - P[sel, j]) ** 2))
                accs.append(2 * c[0])
                print(f"  {name}(t) quadratic over {t[sel][0]:.3f}..{t[sel][-1]:.3f}s: acc {2 * c[0] * S:.4g} {U}/s^2, "
                      f"velocity at window centre {c[1] * S:.4g} {U}/s, rms {rms:.3g} {rU}")
            print(f"  |acc| = {np.linalg.norm(accs) * S:.4g} {U}/s^2 (constant-acceleration fit)")


# ───────────────────────── depth: 3D positions from apparent size (pinhole camera) ─────────────────────────
def cmd_depth(a):
    """Z(t) = Z_ref * s_ref / s(t) for a rigid object whose apparent size s changes only with distance;
    X = (u - W/2) * Z / f, Y = (v - H/2) * Z / f with focal length f in px."""
    v = Video()
    rows = list(csv.DictReader(open(a.csv)))
    for col, val in (("track", a.track), ("obj_id", a.obj)):
        if val is not None and rows and col in rows[0]:
            rows = [r for r in rows if int(float(r[col])) == int(val)]
    if rows and "edge" in rows[0]:
        rows = [r for r in rows if r["edge"] in ("0", "")]
    if len(rows) < 2:
        sys.exit("qpt depth: fewer than 2 usable rows")
    t = np.array([float(r["t"]) for r in rows]); o = np.argsort(t); t = t[o]
    u = np.array([float(rows[i][a.ucol]) for i in o]); w = np.array([float(rows[i][a.vcol]) for i in o])
    s = np.array([float(rows[i][a.size_col]) for i in o])
    # smooth the apparent size: 1/s is proportional to distance, which is smooth in time
    k = max(1, a.smooth)
    inv = 1.0 / s
    inv_s = np.array([np.polyval(np.polyfit(t[max(0, i - k):i + k + 1] - t[i], inv[max(0, i - k):i + k + 1],
                                             min(2, len(t[max(0, i - k):i + k + 1]) - 1)), 0) for i in range(len(t))])
    T, Zref = a.anchor
    s_ref = 1.0 / float(np.interp(T, t, inv_s))
    Z = Zref * s_ref * inv_s
    if a.f:
        f = a.f; fsrc = "given"
    elif a.size:
        f = s_ref * Zref / a.size; fsrc = f"from true size {a.size:g} m: f = {s_ref:.2f} px * {Zref:g} m / {a.size:g} m"
    else:
        f = None
    print(f"apparent size '{a.size_col}' at t={T}s: {s_ref:.2f} px (smoothed); distance there {Zref:g} m (anchor)")
    print(f"distance range {Z.min():.3f}..{Z.max():.3f} m over t {t[0]:.3f}..{t[-1]:.3f}s")
    if f is None:
        print("no focal length: give --size (the object's true size in m) or --f; writing Z only")
    else:
        fov = 2 * math.degrees(math.atan(v.W / 2 / f))
        print(f"focal length {f:.1f} px ({fsrc}); principal point = image centre ({v.W / 2:.0f},{v.H / 2:.0f}); "
              f"horizontal field of view {fov:.0f} deg")
        if fov_note(f, v.W, v.source):
            print(fov_note(f, v.W, v.source))
    X = (u - v.W / 2) * Z / f if f else np.full_like(Z, np.nan)
    Y = (w - v.H / 2) * Z / f if f else np.full_like(Z, np.nan)
    OUT.mkdir(exist_ok=True)
    out = OUT / (a.name or (Path(a.csv).stem + "_3d")) ; out = out.with_suffix(".csv")
    with open(out, "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(["f", "t", "X", "Y", "Z", "size_px", "size_px_smoothed"])
        for i in range(len(t)):
            wr.writerow([int(round(t[i] * v.fps)), f"{t[i]:.4f}", f"{X[i]:.5f}", f"{Y[i]:.5f}", f"{Z[i]:.5f}", f"{s[i]:.2f}", f"{1 / inv_s[i]:.2f}"])
    print(f"-> {out}  (metres, camera frame: X right, Y down, Z away from the camera)")
    if f:
        print(f"use: qpt fit --csv {out} --at T   (3D speed; components X,Y,Z)")
    rel = np.std(s / (1 / inv_s) - 1)
    print(f"size noise: {rel:.1%} per frame around the smooth curve" + ("  -- noisy: check the masks / use a steadier size column" if rel > 0.05 else ""))


# ───────────────────────── cut: classical segmentation inside a box (CPU) ─────────────────────────
def cut_mask(v, f, box, method, thr):
    img = v.frame(f)
    x0, y0, x1, y1 = box
    if method == "motion":          # static camera: pixels that differ from the clip's median background
        bg = median_background(v)
        d = cv2.absdiff(img, bg).max(axis=2)
        m = np.zeros(d.shape, np.uint8); m[y0:y1, x0:x1] = (d[y0:y1, x0:x1] > thr)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    else:                            # GrabCut initialised with the box: object inside, background outside
        m = np.zeros(img.shape[:2], np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, m, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, 6, cv2.GC_INIT_WITH_RECT)
        m = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    return m.astype(bool)


def cmd_cut(a):
    v = Video()
    box = parse_box(a.box, v)
    if a.t_from is not None or a.t_to is not None:
        f0, f1 = v.span(a); idx = list(range(f0, f1 + 1, a.every))
    else:
        idx = [v.pick(a)]
    print(f"method {a.method}: " + ("pixels differing from the median background by > %d inside the box" % a.thr
                                    if a.method == "motion" else "GrabCut seeded by the box (object inside, background outside)")
          + "; largest connected piece kept")
    print("  f " + STAT_HDR[2:])
    rows, keep = [], []
    for f in idx:
        m = cut_mask(v, f, box, a.method, a.thr)
        s = mask_stats(m)
        if s is None:
            print(f"{f:>3} nothing found"); continue
        rows.append((f, s)); keep.append((f, m, s))
        touch = s["x0"] <= box[0] or s["y0"] <= box[1] or s["x1"] >= box[2] - 1 or s["y1"] >= box[3] - 1
        print(f"{f:>3}" + stat_line(0, 1.0, s)[2:] + ("  TOUCHES BOX EDGE: enlarge the box" if touch else ""))
    print(STAT_KEY)
    if len(rows) > 1:
        for k in ("bw", "bh", "feret", "rect_long", "rect_short"):
            vals = [s[k] for _, s in rows]
            print(f"  {k:>10}: max {max(vals):.1f} at f{rows[int(np.argmax(vals))][0]}, median {np.median(vals):.1f}, min {min(vals):.1f}")
    show = keep[:: max(1, len(keep) // 4)][:4]
    tiles, labs = [], []
    for f, m, s in show:
        pad = max(6, int(0.15 * max(box[2] - box[0], box[3] - box[1])))
        vb = parse_box([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], v)
        fa, fb = s["feret_a"], s["feret_b"]
        can, z, st = render_view(v, overlay(v.frame(f), [m]), vb, None, None, lines=[(fa[0], fa[1], fb[0], fb[1])],
                                 rects=[(s["x0"], s["y0"], s["x1"], s["y1"]), (box[0], box[1], box[2] - 1, box[3] - 1)])
        tiles.append(can); labs.append(f"{flabel(v, f)}  box {s['bw']}x{s['bh']}  feret {s['feret']:.0f}")
    if tiles:
        print(save(tile_sheet(tiles, labs, cols=min(2, len(tiles)), tile_w=700), f"cut_f{idx[0]}_{box[0]}_{box[1]}", jpg=True))


# ───────────────────────── xyz: back-project pixels to 3D (pinhole camera) ─────────────────────────
def cmd_xyz(a):
    """3D camera coordinates of image points with known distance, and the distances between them."""
    v = Video()
    cx, cy = (a.pp if a.pp else (v.W / 2, v.H / 2))
    f = a.f or v.lab_f
    if not f:
        sys.exit("qpt: give --f (focal length in px; qpt focal)")
    if not a.f and v.lab_f:
        print(f"lab recording: using the lab camera f = {f:.0f} px (0.58 x width, ~82 deg horizontal FOV)")
    fov = 2 * math.degrees(math.atan(v.W / 2 / f))
    print(f"focal length {f:g} px, principal point ({cx:g},{cy:g}), horizontal field of view {fov:.0f} deg")
    if fov_note(f, v.W, v.source):
        print(fov_note(f, v.W, v.source))
    print(f"distances are {'ranges (straight-line camera-to-point)' if not a.axial else 'axial depths (along the optical axis)'}")
    P = []
    for k, (u, w, d) in enumerate(a.pt):
        u, w, d = float(u), float(w), float(d)
        ray = np.array([(u - cx) / f, (w - cy) / f, 1.0])
        Z = d if a.axial else d / float(np.linalg.norm(ray))
        X = ray * Z
        P.append(X)
        print(f"  P{k + 1}: pixel ({u:g},{w:g}) dist {d:g} -> X {X[0]:+.4f}  Y {X[1]:+.4f}  Z {X[2]:.4f}  (range {np.linalg.norm(X):.4f})")
    if len(P) > 1:
        print("distances between points (same units as the given distances):")
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                print(f"  P{i + 1}-P{j + 1}: {np.linalg.norm(P[i] - P[j]):.4f}")
        if len(P) > 2:
            path = sum(np.linalg.norm(P[i + 1] - P[i]) for i in range(len(P) - 1))
            print(f"  path P1->P{len(P)} through all points: {path:.4f}")


def cmd_plane(a):
    """Ground plane from >= 3 image points with known camera distance; then any pixel ON the plane
    -> 3D (ray-plane intersection), distances along the plane, and heights above it."""
    v = Video()
    cx, cy = (a.pp if a.pp else (v.W / 2, v.H / 2))
    f = a.f or v.lab_f
    if not f:
        sys.exit("qpt: give --f (focal length in px; qpt focal)")
    if not a.f and v.lab_f:
        print(f"lab recording: using the lab camera f = {f:.0f} px (0.58 x width, ~82 deg horizontal FOV)")
    if fov_note(f, v.W, v.source):
        print(fov_note(f, v.W, v.source))
    def ray(u, w):
        return np.array([(u - cx) / f, (w - cy) / f, 1.0])
    P, Hs = [], []
    for spec in a.pt:
        if len(spec) not in (3, 4):
            sys.exit("qpt plane: --pt U V DIST [H] (H = the point's height above the floor, e.g. a ball's centre = its radius)")
        u, w, d = (float(x) for x in spec[:3]); Hs.append(float(spec[3]) if len(spec) == 4 else 0.0)
        r = ray(u, w); Z = d if a.axial else d / np.linalg.norm(r)
        P.append(r * Z)
    P, Hs = np.array(P), np.array(Hs)
    a.pt = [tuple(x[:3]) for x in a.pt]
    if a.horizon is not None:                # the floor's horizon row fixes the plane's tilt (no roll): any 1+ anchors
        n = np.array([0.0, 1.0, -(a.horizon - cy) / f]); n = -n / np.linalg.norm(n)
        c = n * float(np.mean([(p_ - h_ * n) @ n for p_, h_ in zip(P, Hs)]))   # anchors at their height above the floor
        print(f"plane from the horizon row {a.horizon:g} (camera pitch {math.degrees(math.atan((cy - a.horizon) / f)):+.1f} deg, "
              f"no roll) and {len(P)} anchor(s)")
    else:
        if len(P) < 3:
            sys.exit("qpt plane: give 3+ --pt anchors, or --horizon ROW with 1+")
        c = P.mean(0); n = np.linalg.svd(P - c)[2][-1]
        for _ in range(8):                   # anchors at a known height above the floor: fit the floor under them
            if not Hs.any():
                break
            if n[1] > 0:
                n = -n
            Q = P - Hs[:, None] * n; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][-1]
    if n[1] > 0:
        n = -n                               # normal points up (image y is down)
    if a.offset:                             # points at a known height above the floor (a ball's centre = its radius)
        c = c + a.offset * n
    resid = (P - c) @ n - Hs
    h_cam = float(-(c @ n))                  # camera height above the plane (signed)
    tilt = math.degrees(math.acos(min(1.0, abs(n[1]))))
    print(f"plane from {len(P)} points: camera {abs(h_cam):.3f} above it, plane tilted {tilt:.1f} deg from the camera's "
          f"horizontal; fit residuals {np.abs(resid).max():.3f} (same unit as the distances)")
    for k, ((u, w, d), r_) in enumerate(zip(a.pt, resid)):
        print(f"  anchor {k + 1} ({u},{w}) dist {d}: off the plane by {r_:+.3f}")
    # horizon: image row where rays become parallel to the plane, at the image centre column
    if abs(n[1]) > 1e-6:
        hrow = cy - f * (n[2] * 1.0 + n[0] * 0.0) / n[1] if abs(n[1]) > 1e-9 else float('nan')
        print(f"  implied horizon of the plane at image row ~{hrow:.0f} (centre column) -- compare with the visible "
              "horizon / vanishing lines; a mismatch means a bad anchor or focal length")
    rows = [float(x[1]) for x in a.pt]
    if len(P) > 3 and np.abs(resid).max() > 0.05 * np.linalg.norm(P - c, axis=1).max():
        print("  WARNING: the points are not on one plane -- a distance, a pixel or the focal length is off")
    G = []
    for k, (u, w) in enumerate(a.on or []):
        if a.horizon is None and not (min(rows) - 0.25 * (max(rows) - min(rows)) <= float(w) <= max(rows) + 0.25 * (max(rows) - min(rows))):
            print(f"  NOTE: pixel row {w} is outside the anchors' rows {min(rows):.0f}-{max(rows):.0f}: extrapolated, less reliable")
        r = ray(float(u), float(w)); den = r @ n
        if abs(den) < 1e-9:
            print(f"  G{k + 1}: pixel ({u},{w}) ray is parallel to the plane"); G.append(None); continue
        tt = (c @ n) / den
        if tt <= 0:
            print(f"  G{k + 1}: pixel ({u},{w}) is above the horizon -- not on the plane"); G.append(None); continue
        X = r * tt; G.append(X)
        print(f"  G{k + 1}: pixel ({u},{w}) -> X {X[0]:+.4f} Y {X[1]:+.4f} Z {X[2]:.4f} (range {np.linalg.norm(X):.4f})")
    def onp(u, w):                           # pixel -> point on the plane (None above the horizon)
        r = ray(float(u), float(w)); den = r @ n
        return None if abs(den) < 1e-9 or (c @ n) / den <= 0 else r * ((c @ n) / den)
    for k, (u, w) in enumerate(a.on or []):
        if G[k] is None:
            continue
        lo, hi = onp(u, float(w) - 1), onp(u, float(w) + 1)
        if lo is not None and hi is not None:
            dz = np.linalg.norm(lo - hi) / 2
            if dz > 0.02 * np.linalg.norm(G[k]):
                print(f"    G{k + 1}: a grazing view -- 1 px of row moves this point {dz:.3f} along the view "
                      f"({dz / np.linalg.norm(G[k]):.0%} of its range)")
    ok = [(k, g) for k, g in enumerate(G) if g is not None]
    for i in range(len(ok)):
        for j in range(i + 1, len(ok)):
            (ki, gi), (kj, gj) = ok[i], ok[j]
            d = gj - gi; L = float(np.linalg.norm(d))
            view = (gi + gj) / np.linalg.norm(gi + gj)
            along = abs(float(d @ view)); across = float(np.sqrt(max(L * L - along * along, 0.0)))
            (ui, wi), (uj, wj) = a.on[ki], a.on[kj]
            alt = [np.linalg.norm(q2 - q1) for s1 in (-1, 1) for s2 in (-1, 1)
                   for q1, q2 in [(onp(ui, float(wi) + s1), onp(uj, float(wj) + s2))] if q1 is not None and q2 is not None]
            msg = f"  G{ki + 1}-G{kj + 1}: {L:.4f} (across the view {across:.4f}, along it {along:.4f})"
            if alt and L > 0:
                spread = (max(alt + [L]) - min(alt + [L])) / L
                msg += f"; with each end 1 px higher/lower: {min(alt + [L]):.4f}-{max(alt + [L]):.4f}"
                if spread > 0.30:
                    msg += (" -- ILL-CONDITIONED: its part along the view rests on a pixel or two of row: do not use this length; "
                            "measure this dimension where it runs across the view (another edge, frame or instance) -- the "
                            f"across part alone ({across:.4f}) is only a lower bound")
                elif spread > 0.15:
                    msg += " -- loose: 1 px of row changes it by more than 15%"
            print(msg)
    if a.track:
        rows = list(csv.DictReader(open(a.track)))
        for col, val in (("track", a.track_id), ("obj_id", a.obj)):
            if val is not None and rows and col in rows[0]:
                rows = [r for r in rows if int(float(r[col])) == int(val)]
        if rows and "edge" in rows[0]:
            rows = [r for r in rows if r["edge"] in ("0", "")]
        out = OUT / (Path(a.track).stem + "_plane.csv"); OUT.mkdir(exist_ok=True)
        n_ok = 0
        with open(out, "w", newline="") as fh:
            wr = csv.writer(fh); wr.writerow(["f", "t", "X", "Y", "Z"])
            for r_ in rows:
                u, w = float(r_[a.ucol]), float(r_[a.vcol])
                rr = ray(u, w); den = rr @ n
                if abs(den) < 1e-9 or (c @ n) / den <= 0:
                    continue
                X = rr * ((c @ n) / den); n_ok += 1
                f_ = r_.get("f") or round(float(r_["t"]) * v.fps)
                wr.writerow([f_, r_.get("t") or f"{int(f_) / v.fps:.4f}", f"{X[0]:.5f}", f"{X[1]:.5f}", f"{X[2]:.5f}"])
        print(f"track: {n_ok} of {len(rows)} points placed on the plane{' (offset ' + str(a.offset) + ')' if a.offset else ''} "
              f"-> {out}  (use: qpt fit --csv {out} --at T)")
    for k, (fu, fw, tu, tw) in enumerate(a.height or []):
        r = ray(float(fu), float(fw)); den = r @ n
        if abs(den) < 1e-9 or (c @ n) / den <= 0:
            print(f"  H{k + 1}: foot pixel not on the plane"); continue
        Fp = r * ((c @ n) / den)            # foot on the plane; the top lies on the line Fp + s*n
        q = ray(float(tu), float(tw))
        A = np.array([[n @ n, -(n @ q)], [q @ n, -(q @ q)]]); b = np.array([-(Fp @ n), -(Fp @ q)])
        s_, _ = np.linalg.solve(A, b)
        rel = 0.0
        r2 = ray(float(fu), float(fw) + 1); d2 = r2 @ n
        if abs(d2) > 1e-9 and (c @ n) / d2 > 0:
            rel = abs(np.linalg.norm(r2 * ((c @ n) / d2)) / np.linalg.norm(Fp) - 1)
        print(f"  H{k + 1}: height above the plane at foot ({fu},{fw}) up to ({tu},{tw}): {s_:.4f}  (foot at range {np.linalg.norm(Fp):.4f})"
              + (f" -- grazing view: 1 px of foot row changes the foot's distance, and this height, by {rel:.0%}" if rel > 0.03 else ""))


def true_times(t, v):
    """Times on the true axis when qpt_out/gaps.txt lists dropped source frames (see fit --gaps)."""
    if GAPS_FILE.exists() and GAPS_FILE.read_text().strip():
        gaps = full_gaps(GAPS_FILE.read_text().strip().split(","), v.fps, v.n, v.fps_container)[0]
        return gap_time(np.round(np.asarray(t) * v.fps), gaps, v.fps, v.n - 1, gap_unit(gaps, v.fps, v.n - 1, v.fps_container)[0]), gaps
    return np.asarray(t, float), []


def cmd_flight(a):
    """Focal length from free flight: one 3D ballistic path per object (|g| = 9.8 m/s^2, one camera: shared f, roll,
    pitch) through the tracked pixels, each anchored by the object's stated camera distance at one time."""
    from scipy.optimize import least_squares
    v = Video(); cx, cy = (v.W - 1) / 2, (v.H - 1) / 2; g = a.g
    objs = []
    for spec in a.path:
        if len(spec) not in (3, 5):
            sys.exit("qpt: --path CSV[:ID] T DIST [FROM TO]")
        src, ident = (spec[0].rsplit(":", 1) + [None])[:2] if ":" in spec[0] else (spec[0], None)
        ns = argparse.Namespace(csv=src, track=None, obj=None, keep_edge=False, cols=None)
        if ident is not None:
            hdr = open(src).readline()
            ns.track, ns.obj = (int(ident), None) if "track" in hdr.split(",") else (None, int(ident))
        t, P, names = load_track(ns)
        if names[:2] not in (["cx", "cy"], ["u", "v"], ["x", "y"]):
            sys.exit(f"qpt: {src}: flight needs a pixel track (cx,cy) from motion / track / segtrack")
        T, D = float(spec[1]), float(spec[2])
        lo, hi = (float(spec[3]), float(spec[4])) if len(spec) == 5 else (-1e9, 1e9)
        k = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        if k.sum() < 6:
            sys.exit(f"qpt: {spec[0]}: fewer than 6 samples in the window")
        if not (t[k].min() - 1.5 / v.fps <= T <= t[k].max() + 1.5 / v.fps):
            sys.exit(f"qpt flight: {spec[0]}: the range is given at t = {T}s, outside this path's samples "
                     f"({t[k].min():.3f}-{t[k].max():.3f}s): the object must be in this free flight at T -- use a distance "
                     "at a time inside the flight (e.g. from qpt xyz / plane at a frame in flight)")
        tt, gaps = true_times(t[k], v); Tt = float(true_times([T], v)[0][0])
        objs.append(dict(name=spec[0], t=t[k], tt=tt, P=P[k, :2], Tt=Tt, D=D,
                         u0=[np.interp(Tt, tt, P[k, 0]), np.interp(Tt, tt, P[k, 1])]))

    def model(o, q, f, roll, pitch, dt):
        r = np.array([(q[0] - cx) / f, (q[1] - cy) / f, 1.0]); P0 = o["D"] * r / np.linalg.norm(r)
        gd = np.array([math.sin(roll) * math.cos(pitch), math.cos(roll) * math.cos(pitch), math.sin(pitch)]) * g
        X = P0[None] + np.outer(dt, q[2:5]) + 0.5 * np.outer(dt ** 2, gd)
        return np.c_[cx + f * X[:, 0] / X[:, 2], cy + f * X[:, 1] / X[:, 2]], X

    def solve(f_fixed, sel):
        """sel: per object a boolean mask of samples. Returns f, rms (px, per point), params."""
        best = None
        pit = None if a.pitch is None else math.radians(a.pitch)
        for f0 in ([f_fixed] if f_fixed else np.linspace(0.4, 2.5, 8) * v.W):
            x0, lb, ub = [0.0, pit or 0.0], [-0.8, -0.6 if pit is None else pit - 1e-6], [0.8, 0.6 if pit is None else pit + 1e-6]
            if not f_fixed:
                x0.append(f0); lb.append(0.2 * v.W); ub.append(8 * v.W)
            for o, m in zip(objs, sel):
                vel = np.polyfit(o["tt"][m] - o["Tt"], o["P"][m], 2)[1] * o["D"] / f0
                x0 += [*o["u0"], vel[0], vel[1], 0.0]; lb += [-np.inf] * 5; ub += [np.inf] * 5
            nb = 2 if f_fixed else 3
            def res(x):
                f = f_fixed or x[2]; out = []
                for j, (o, m) in enumerate(zip(objs, sel)):
                    out.append((model(o, x[nb + 5 * j: nb + 5 * j + 5], f, x[0], x[1], o["tt"][m] - o["Tt"])[0] - o["P"][m]).ravel())
                return np.concatenate(out)
            try:
                r = least_squares(res, x0, bounds=(lb, ub), x_scale="jac")
            except ValueError:
                continue
            if best is None or r.cost < best.cost:
                best = r
        return (f_fixed or best.x[2]), float(np.sqrt(np.mean(best.fun ** 2)) * math.sqrt(2)), best.x, (2 if f_fixed else 3)

    full = [np.ones(len(o["tt"]), bool) for o in objs]
    f, rms, x, nb = solve(None, full)
    gaps = true_times([0.0], v)[1]
    print(f"free-flight fit: {len(objs)} object(s), {sum(len(o['tt']) for o in objs)} samples"
          f"{' on the true time axis (dropped frames ' + ','.join(map(str, gaps)) + ')' if gaps else ''}; |g| = {g} m/s^2; "
          "principal point at the image centre; one camera (shared f, roll, pitch)")
    print(f"  f = {f:.1f} px (horizontal FOV {2 * math.degrees(math.atan(v.W / 2 / f)):.0f} deg), rms {rms:.2f} px; camera roll "
          f"{math.degrees(x[0]):+.1f} deg, pitch {math.degrees(x[1]):+.1f} deg{' (fixed)' if a.pitch is not None else ''}")
    for j, o in enumerate(objs):
        q = x[nb + 5 * j: nb + 5 * j + 5]
        e = model(o, q, f, x[0], x[1], o["tt"] - o["Tt"])[0] - o["P"]
        print(f"    {o['name']}: {len(o['tt'])} samples t {o['t'].min():.3f}-{o['t'].max():.3f}s, range {o['D']} m at "
              f"{o['Tt']:.3f}s (true time), rms {np.sqrt(np.mean(np.sum(e ** 2, 1))):.2f} px, 3D velocity there "
              f"({q[2]:.2f}, {q[3]:.2f}, {q[4]:.2f}) m/s")
    fov = 2 * math.degrees(math.atan(v.W / 2 / f))
    if v.source == "simulation":
        print("  simulation: rendered motion can be keyframed and need not follow 9.8 m/s^2 -- this f is a check only when "
              "the clip's own prior is gravity")
    elif v.source == "lab":
        print(f"  lab recording: the lab camera's f is ~{v.lab_f:.0f} px; use it -- a different f here means the clock, the "
              "track or the window is off")
    elif not 30 <= fov <= 90:
        print(f"  field of view {fov:.0f} deg is outside 30-90 for real footage: check the clock first (with no dropped "
              "frames given, one dropped per second makes f ~8% low; `qpt info`), then the distances")
    if rms > 2.0:
        print(f"  WARNING: rms {rms:.1f} px -- not one free flight per object (a throw or catch, a bounce, a tracking jump, "
              "or dropped frames not given to fit --gaps): narrow the windows before using f")
    halves = [[np.arange(len(o["tt"])) < (2 * len(o["tt"])) // 3 for o in objs],
              [np.arange(len(o["tt"])) >= len(o["tt"]) // 3 for o in objs]]
    fs = [solve(None, h)[0] for h in halves if all(m.sum() >= 5 for m in h)]
    if fs:
        print(f"  windows (first / last two thirds of each path): f = {' / '.join(f'{x_:.0f}' for x_ in fs)} px"
              + (" (less data each)  -- spread > 25%: the flight does not pin f down (short, or mostly toward the camera)"
                 if max(fs + [f]) > 1.25 * min(fs + [f]) else " (less data each; consistent)"))
    if a.f:
        _, rms2, _, _ = solve(a.f, full)
        print(f"  at f = {a.f:g} px: rms {rms2:.2f} px ({rms2 / max(rms, 1e-9):.1f}x the best fit) -- "
              + ("no verdict: the best fit itself is poor" if rms > 2.0 else
                 "consistent with free flight" if rms2 < 1.5 * rms + 0.3 else "free flight does NOT fit this focal length"))
    if a.pitch is None and abs(math.degrees(x[1])) > 25:
        print("  note: the fitted pitch is large -- if the camera is level, refit with --pitch 0")
    for j, o in enumerate(objs):
        _, X = model(o, x[nb + 5 * j: nb + 5 * j + 5], f, x[0], x[1], o["tt"] - o["Tt"])
        out = Path("qpt_out") / f"{Path(o['name'].split(':')[0]).stem}{'_' + o['name'].split(':')[1] if ':' in o['name'] else ''}_flight.csv"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["t", "X", "Y", "Z"])
            for ti, Xi in zip(o["tt"], X):
                w.writerow([f"{ti:.5f}", f"{Xi[0]:.5f}", f"{Xi[1]:.5f}", f"{Xi[2]:.5f}"])
        print(f"  fitted 3D path of {o['name']} (m, camera axes, true times) -> {out}")


def vanishing_point(lines):
    """Least-squares intersection (homogeneous) of 2+ image lines given as (x1, y1, x2, y2)."""
    L = []
    for x1, y1, x2, y2 in lines:
        l = np.cross([x1, y1, 1.0], [x2, y2, 1.0]); L.append(l / max(np.hypot(l[0], l[1]), 1e-12))
    w = np.linalg.svd(np.array(L))[2][-1]
    return w


def focal_vp(set1, set2, c):
    v1, v2 = vanishing_point(set1), vanishing_point(set2)
    if abs(v1[2]) < 1e-12 or abs(v2[2]) < 1e-12:
        return np.nan, v1, v2
    p1, p2 = v1[:2] / v1[2], v2[:2] / v2[2]
    f2 = -np.dot(p1 - c, p2 - c)
    return (math.sqrt(f2) if f2 > 0 else np.nan), p1, p2


def cmd_focal_vp(a, v):
    def lines(vals, name):
        vals = [float(x) for x in vals]
        if len(vals) < 8 or len(vals) % 4:
            sys.exit(f"qpt focal: {name} takes 2+ lines, 4 numbers each (X1 Y1 X2 Y2 ...)")
        return [vals[i:i + 4] for i in range(0, len(vals), 4)]
    s1, s2 = lines(a.vp1, "--vp1"), lines(a.vp2, "--vp2")
    c = np.array([(v.W - 1) / 2, (v.H - 1) / 2])
    f, p1, p2 = focal_vp(s1, s2, c)
    for name, p_ in (("vp1", p1), ("vp2", p2)):
        if np.ndim(p_) == 1 and len(p_) == 2:
            print(f"  {name} at ({p_[0]:.0f}, {p_[1]:.0f}): {np.hypot(*(p_ - c)) / v.W:.1f} image widths from the centre")
        else:
            print(f"  {name}: the lines are parallel in the image (vanishing point at infinity)")
    if not np.isfinite(f):
        print("no real focal length: the two sets are not perpendicular in 3D, or a vanishing point is wrong / at infinity")
        return
    rng = np.random.default_rng(0); fs = []
    for _ in range(300):    # endpoints +-0.5 px, principal point +-1.5% of the image size
        jit = lambda S: [[x + rng.uniform(-0.5, 0.5) for x in ln] for ln in S]
        cc = c + rng.uniform(-0.015, 0.015, 2) * [v.W, v.H]
        fj = focal_vp(jit(s1), jit(s2), cc)[0]
        if np.isfinite(fj):
            fs.append(fj)
    lo, hi = (np.percentile(fs, 5), np.percentile(fs, 95)) if len(fs) > 30 else (np.nan, np.nan)
    print(f"f = {f:.1f} px from two perpendicular sets of parallel lines (principal point at the image centre); horizontal "
          f"field of view {2 * math.degrees(math.atan(v.W / 2 / f)):.0f} deg")
    print(f"  conditioning: endpoints +-0.5 px and principal point +-1.5% give f {lo:.0f}-{hi:.0f} px (5-95%)"
          + ("" if len(fs) > 270 else f"; {300 - len(fs)} of 300 perturbations had no real f"))
    if not np.isfinite(lo) or (hi - lo) / f > 0.3 or len(fs) < 270:
        print("  ILL-CONDITIONED: a vanishing point is far away (lines nearly parallel in the image) or the sets are not "
              "perpendicular -- do not use this f; use objects of known size at stated distances (--obj) or free flight")
    elif (hi - lo) / f > 0.15:
        print("  loosely determined (about +-10%): use it only together with another kind of evidence (--obj, free flight)")


def cmd_focal(a):
    """Focal length from objects of known size at known distance; one line per object, so they can be compared."""
    v = Video()
    if a.vp1 or a.vp2:
        if not (a.vp1 and a.vp2):
            sys.exit("qpt focal: give both --vp1 and --vp2")
        cmd_focal_vp(a, v)
        if not a.obj:
            return
    if not a.obj:
        sys.exit("qpt focal: give --obj (known-size objects) or --vp1/--vp2 (two perpendicular sets of parallel lines)")
    fs = []
    cx0, cy0 = v.W / 2, v.H / 2
    for k, o in enumerate(a.obj):
        if len(o) not in (3, 5):
            sys.exit("qpt focal: --obj takes PIXEL_SIZE TRUE_SIZE DIST [U V]")
        px, size, d = float(o[0]), float(o[1]), float(o[2])
        f = px * d / size
        if len(o) == 5:                 # DIST is a camera-to-object range: convert to depth along the axis
            r2 = (float(o[3]) - cx0) ** 2 + (float(o[4]) - cy0) ** 2
            for _ in range(30):
                f = px * (d / math.sqrt(1 + r2 / f ** 2)) / size
        fs.append(f)
        fov = 2 * math.degrees(math.atan(v.W / 2 / f))
        print(f"  object {k + 1}: {px:g} px wide, true size {size:g}, distance {d:g} -> f = {f:.1f} px "
              f"(field of view {fov:.0f} deg)" + ("" if 30 <= fov <= 90 else "\n" + fov_note(f, v.W, v.source)))
    if len(fs) > 1:
        print(f"spread {min(fs):.0f}-{max(fs):.0f} px ({(max(fs) / min(fs) - 1):.0%}): a camera has ONE focal length -- "
              "if these disagree by more than a few %, one of the measurements (or sizes, or distances) is wrong")


# ───────────────────────── SAM 3 via the tool server ─────────────────────────
class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path, timeout=900):
        super().__init__("localhost", timeout=timeout); self.sock_path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout); self.sock.connect(self.sock_path)


def server(route, payload):
    sp = os.environ.get("QPT_SOCK", "/opt/qpt/qpt.sock")
    if not Path(sp).exists():
        sys.exit("qpt: the SAM tool server is not available in this session (no socket); use the CPU tools")
    c = UnixHTTP(sp)
    c.request("POST", route, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    r = c.getresponse(); body = r.read()
    if r.status != 200:
        sys.exit(f"qpt: tool server error {r.status}: {body[:400].decode(errors='replace')}")
    return json.loads(body)


def png_b64(img):
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


STAT_HDR = f"{'#':>2} {'score':>5} {'cx':>7} {'cy':>7} {'bbox x0,y0-x1,y1':>19} {'bw':>5} {'bh':>5} {'rect L':>6} {'rect S':>6} {'feret':>6} {'major':>6} {'equiv':>6} {'area':>7}"


def stat_line(k, sc, s):
    bb = "%d,%d-%d,%d" % (s["x0"], s["y0"], s["x1"], s["y1"])
    return (f"{k:>2} {sc:5.2f} {s['cx']:7.1f} {s['cy']:7.1f} {bb:>19} {s['bw']:5d} {s['bh']:5d} "
            f"{s['rect_long']:6.1f} {s['rect_short']:6.1f} {s['feret']:6.1f} {s['major']:6.1f} {s['equiv']:6.1f} {s['area']:7d}")


STAT_KEY = ("sizes in px: bw/bh = axis-aligned box width/height; rect L/S = long/short side of the tightest rotated box; "
            "feret = longest tip-to-tip distance; major = ellipse-equivalent length; equiv = equal-area circle diameter")


def overlay(img, masks, ids=None):
    im = img.copy()
    for k, m in enumerate(masks):
        c = COLORS[(ids[k] if ids else k) % len(COLORS)]
        im[m] = (0.55 * im[m] + 0.45 * np.array(c)).astype(np.uint8)
        cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(im, cs, -1, c, 1)
    return im


def cmd_seg(a):
    v = Video()
    i = v.pick(a)
    img = v.frame(i)
    pay = {"image": png_b64(img), "threshold": a.thr}
    if a.text:
        pay["text"] = a.text
    if a.box:
        pay["box"] = [float(s) for s in a.box]
    if a.point:
        pay["points"] = [[float(p[0]), float(p[1])] for p in a.point]
        pay["labels"] = [1] * len(a.point)
    if a.neg:
        pay.setdefault("points", []).extend([[float(p[0]), float(p[1])] for p in a.neg])
        pay.setdefault("labels", []).extend([0] * len(a.neg))
    if not (a.text or a.box or a.point):
        sys.exit("qpt seg: give --text, --box or --point")
    tag = a.name or (re.sub(r"\W+", "_", a.text)[:30] if a.text else "box" if a.box else "pt")
    r = server("/seg", pay)
    inst = [(d["score"], unpack_mask(d["mask"])) for d in r["instances"]]
    print(f"{flabel(v, i)}  mode {r['mode']}: {len(inst)} instance(s)" + (f" for text {a.text!r}" if a.text else ""))
    if not inst:
        return
    OUT.mkdir(exist_ok=True)
    print(STAT_HDR)
    keep = []
    for k, (sc, m) in enumerate(inst):
        s = mask_stats(m)
        if s is None:
            continue
        keep.append((k, sc, m, s))
        print(stat_line(k, sc, s))
        cv2.imwrite(str(OUT / f"mask_f{i:04d}_{tag}_{k}.png"), (m * 255).astype(np.uint8))
    print(STAT_KEY)
    im = overlay(img, [m for _, _, m, _ in keep], [k for k, *_ in keep])
    for k, sc, m, s in keep:
        cv2.putText(im, f"#{k}", (s["x0"], max(12, s["y0"] - 4)), FONT, 0.6, COLORS[k % len(COLORS)], 2, cv2.LINE_AA)
    tiles, labs = [im], [f"{flabel(v, i)}  {len(keep)} instance(s)"]
    for k, sc, m, s in keep[:3]:      # zoomed crop of each instance, with a pixel grid
        pad = max(6, int(0.25 * max(s["bw"], s["bh"])))
        box = parse_box([s["x0"] - pad, s["y0"] - pad, s["x1"] + pad, s["y1"] + pad], v)
        ov = overlay(img, [m], [k])
        fa, fb = s["feret_a"], s["feret_b"]
        can, z, st = render_view(v, ov, box, None, None, lines=[(fa[0], fa[1], fb[0], fb[1])],
                                 rects=[(s["x0"], s["y0"], s["x1"], s["y1"])])
        tiles.append(can); labs.append(f"#{k}: box {s['bw']}x{s['bh']}, feret {s['feret']:.0f} (line)")
    W0 = 1200
    top = np.vstack([label_band(labs[0], W0), cv2.resize(tiles[0], (W0, int(W0 * tiles[0].shape[0] / tiles[0].shape[1])))])
    parts = [top]
    if len(tiles) > 1:
        parts.append(tile_sheet(tiles[1:], labs[1:], cols=len(tiles) - 1, tile_w=(W0 - 4 * (len(tiles) - 2)) // (len(tiles) - 1)))
        parts[-1] = parts[-1][:, :W0] if parts[-1].shape[1] >= W0 else np.hstack([parts[-1], np.zeros((parts[-1].shape[0], W0 - parts[-1].shape[1], 3), np.uint8)])
    p = save(np.vstack(parts), f"seg_f{i:04d}_{tag}", jpg=True)
    print(p)
    print(f"masks saved as {OUT}/mask_f{i:04d}_{tag}_<#>.png")


def cmd_segtrack(a):
    v = Video()
    f0, f1 = v.span(a)
    pay = {"video": base64.b64encode(open(v.path, "rb").read()).decode(), "fps": v.fps,
           "f0": f0, "f1": f1, "stride": a.every, "threshold": a.thr, "show": 8}
    if a.text:
        pay["text"] = a.text
    else:
        fs = v.pick(a)
        pay["frame"] = fs
        if a.box:
            pay["box"] = [float(s) for s in a.box]
        if a.point:
            pay["points"] = [[float(p[0]), float(p[1])] for p in a.point]; pay["labels"] = [1] * len(a.point)
        if not (a.box or a.point):
            sys.exit("qpt segtrack: give --text, or --f/--t with --box or --point")
    r = server("/track", pay)
    rows = r["rows"]
    OUT.mkdir(exist_ok=True)
    name = a.name or ("segtrack_" + re.sub(r"\W+", "_", a.text or f"f{pay.get('frame')}")[:40])
    path = OUT / f"{name}.csv"
    cols = ["obj_id", "f", "t", "score", "cx", "cy", "x0", "y0", "x1", "y1", "bw", "bh", "rect_long", "rect_short", "feret", "major", "minor", "equiv", "area"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for d in rows:
            w.writerow([d["obj_id"], d["f"], f"{d['f'] / v.fps:.4f}", f"{d['score']:.3f}"] +
                       [f"{d[c]:.2f}" if isinstance(d[c], float) else d[c] for c in cols[4:]])
    print(f"mode {r['mode']}: {r['n_frames']} frames processed (f{f0}..f{f1} every {a.every}) -> {path}")
    ids = sorted({d["obj_id"] for d in rows})
    print(f"{'obj':>3} {'frames':>12} {'n':>4} {'score':>5} {'med bw':>6} {'med bh':>6} {'med feret':>9} {'med equiv':>9}  start (cx,cy) -> end (cx,cy)")
    for o in ids:
        rs = [d for d in rows if d["obj_id"] == o]
        a0, a1 = rs[0], rs[-1]
        fr_ = "f%d-f%d" % (a0["f"], a1["f"])
        print(f"{o:>3} {fr_:>12} {len(rs):>4} {np.median([d['score'] for d in rs]):5.2f} "
              f"{np.median([d['bw'] for d in rs]):6.0f} {np.median([d['bh'] for d in rs]):6.0f} "
              f"{np.median([d['feret'] for d in rs]):9.1f} {np.median([d['equiv'] for d in rs]):9.1f}  "
              f"({a0['cx']:.0f},{a0['cy']:.0f}) -> ({a1['cx']:.0f},{a1['cy']:.0f})")
    print("use: qpt fit --csv " + str(path) + " --obj <id> ...   (centre columns cx,cy)")
    tiles, labs = [], []
    for s in r["shown"]:
        f = s["f"]
        ms = [unpack_mask(m["mask"]) for m in s["masks"]]
        im = overlay(v.frame(f), ms, [m["obj_id"] for m in s["masks"]])
        for m, mk in zip(s["masks"], ms):
            st = mask_stats(mk)
            if st:
                cv2.putText(im, f"id{m['obj_id']}", (st["x0"], max(12, st["y0"] - 4)), FONT, 0.6, COLORS[m["obj_id"] % len(COLORS)], 2, cv2.LINE_AA)
        tiles.append(im); labs.append(flabel(v, f) + ("  NO MASK" if not ms else ""))
    if tiles:
        print(save(tile_sheet(tiles, labs, title=f"SAM 3 track: {a.text or 'prompt at f' + str(pay.get('frame'))}"), f"{name}_sheet", jpg=True))


# ───────────────────────── CLI ─────────────────────────
def main():
    P = argparse.ArgumentParser(prog="qpt", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = P.add_subparsers(dest="cmd", required=True)

    def when(p, span=False):
        p.add_argument("--f", type=int, help="frame index"); p.add_argument("--t", type=float, help="time in s (nearest frame)")
        if span:
            p.add_argument("--from", dest="t_from", type=float, help="start time (s)")
            p.add_argument("--to", dest="t_to", type=float, help="end time (s)")

    p = sub.add_parser("info", help="video facts, duplicate-frame check, every question with its times mapped to frames")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("frames", help="contact sheet of labelled frames (evenly spaced, or chosen ones)")
    p.add_argument("--n", type=int, default=12, help="number of evenly spaced frames (default 12)")
    p.add_argument("--every", type=int, help="every k-th frame instead of --n")
    p.add_argument("--f", nargs="+", help="these frame indices"); p.add_argument("--t", nargs="+", help="these times (s)")
    p.add_argument("--from", dest="t_from", type=float); p.add_argument("--to", dest="t_to", type=float)
    p.add_argument("--cols", type=int); p.add_argument("--save", action="store_true", help="also save each full-resolution frame")
    p.set_defaults(fn=cmd_frames)

    p = sub.add_parser("view", help="crop + zoom one frame with a pixel grid labelled in original coordinates; draw lines/points/boxes")
    when(p)
    p.add_argument("--box", nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="crop region (inclusive pixel coords)")
    p.add_argument("--zoom", type=float, help="magnification (default: fit ~1200 px)")
    p.add_argument("--grid", type=int, help="grid spacing in original px (default auto)")
    p.add_argument("--line", nargs=4, action="append", metavar=("X1", "Y1", "X2", "Y2"), help="draw a line and print its length (repeatable)")
    p.add_argument("--pt", nargs=2, action="append", metavar=("X", "Y"), help="mark a point (repeatable)")
    p.add_argument("--rect", nargs=4, action="append", metavar=("X0", "Y0", "X1", "Y1"), help="draw a box (repeatable)")
    p.add_argument("--enhance", action="store_true", help="local contrast enhancement (CLAHE)")
    p.add_argument("--diff", action="store_true", help="show the difference from the clip's median background (moving things light up)")
    p.add_argument("--smooth", action="store_true", help="smooth (Lanczos) upscale instead of nearest-neighbour")
    p.add_argument("--name", help="output file name")
    p.set_defaults(fn=cmd_view)

    p = sub.add_parser("edges", help="find where an object starts/ends along a line you place across it (sub-pixel extent)")
    when(p)
    p.add_argument("--line", nargs=4, required=True, metavar=("X1", "Y1", "X2", "Y2"), help="start and end OUTSIDE the object, crossing it")
    p.add_argument("--frac", type=float, default=0.5, help="threshold as a fraction of the max colour difference (default 0.5 = half-contrast edge)")
    p.add_argument("--ref", type=float, default=3, help="px at each end used as the background colour (default 3)")
    p.add_argument("--inside", action="store_true", help="reference colour = line centre (object) instead of the ends")
    p.add_argument("--sweep", type=float, metavar="W", help="also measure parallel lines offset up to +-W px and report the widest "
                   "(small objects: the full width, not a chord)")
    p.set_defaults(fn=cmd_edges)

    p = sub.add_parser("cut", help="segment the object inside a box without SAM (GrabCut, or motion vs background) -> sizes + picture; over a time range too")
    when(p, span=True)
    p.add_argument("--box", nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"), help="box a little larger than the object")
    p.add_argument("--method", choices=["grabcut", "motion"], default="grabcut")
    p.add_argument("--thr", type=int, default=25, help="motion method: colour change threshold")
    p.add_argument("--every", type=int, default=1, help="with --from/--to: every k-th frame")
    p.set_defaults(fn=cmd_cut)

    p = sub.add_parser("motion", help="static camera: moving blobs per frame (background subtraction) linked into tracks -> csv")
    p.add_argument("--from", dest="t_from", type=float); p.add_argument("--to", dest="t_to", type=float)
    p.add_argument("--every", type=int, default=1); p.add_argument("--thr", type=int, default=20, help="colour change threshold (default 20; lower it for faint or small objects)")
    p.add_argument("--box", nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="only look for moving things inside this region")
    p.add_argument("--min-area", type=int); p.add_argument("--max-jump", type=float, help="max px per frame for linking")
    p.add_argument("--top", type=int, default=6, help="keep the N largest tracks")
    p.set_defaults(fn=cmd_motion)

    p = sub.add_parser("track", help="follow a box through the clip by template matching -> csv of centre positions")
    when(p, span=True)
    p.add_argument("--box", nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"), help="object box at the start frame")
    p.add_argument("--scale-search", action="store_true", help="allow the object to grow/shrink")
    p.add_argument("--search", type=float, help="search radius in px per frame (default 1.5x box size)")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_track)

    p = sub.add_parser("fit", help="speed / acceleration from a position csv (from motion, track or segtrack)")
    p.add_argument("--csv", required=True); p.add_argument("--track", type=int, help="track id (motion.csv)")
    p.add_argument("--obj", type=int, help="obj_id (segtrack csv)"); p.add_argument("--cols", help="position columns, default cx,cy")
    p.add_argument("--at", type=float, help="instantaneous speed/acceleration at this time (s)")
    p.add_argument("--k", type=int, nargs="+", default=[2, 3, 5, 8], help="half-window sizes in samples (default 2 3 5 8)")
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--avg", type=float, nargs=2, metavar=("T0", "T1"), help="average speed over [T0, T1]")
    p.add_argument("--accel", type=float, nargs=2, metavar=("T0", "T1"), help="constant-acceleration fit over [T0, T1]")
    p.add_argument("--axis", default="all", help="components to use: all (default), x, y, xy, or column names like X,Z")
    p.add_argument("--keep-edge", action="store_true", help="keep samples cut by the frame border (edge=1)")
    p.add_argument("--gaps", help="steps k (k->k+1) where a source frame was dropped, e.g. 11,35 (saved to qpt_out/gaps.txt and "
                   "used by every later fit), 'auto' (doubled steps on this track, this fit only) or 'none'")
    p.add_argument("--scale", type=float, help="physical units per pixel"); p.add_argument("--unit", default="m")
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("flight", help="focal length from free flight: 3D ballistic fit (|g| = 9.8) through pixel tracks, each anchored by a stated range")
    p.add_argument("--path", nargs="+", action="append", required=True, metavar="ARG",
                   help="CSV[:ID] T DIST [FROM TO]: a pixel track (ID = track/obj id), its camera distance DIST (m) at time T, "
                        "and the free-flight window; repeat --path for several objects (one camera)")
    p.add_argument("--f", type=float, help="also test this focal length (px): does free flight fit it?")
    p.add_argument("--pitch", type=float, help="camera pitch in deg if known (default: fitted, within +-34)")
    p.add_argument("--g", type=float, default=9.8)
    p.set_defaults(fn=cmd_flight)

    p = sub.add_parser("depth", help="3D track from apparent size: distance(t) = d_ref * size_ref / size(t), plus X,Y via the focal length -> csv for fit")
    p.add_argument("--csv", required=True, help="per-frame track with a size column (segtrack, motion, cut)")
    p.add_argument("--track", type=int); p.add_argument("--obj", type=int)
    p.add_argument("--size-col", default="equiv", help="apparent size column (default equiv; e.g. bw, feret, w)")
    p.add_argument("--ucol", default="cx"); p.add_argument("--vcol", default="cy")
    p.add_argument("--anchor", nargs=2, type=float, required=True, metavar=("T", "DIST_M"), help="a known camera distance (m) at time T (s), as depth along the optical axis (convert a far off-centre range with qpt xyz)")
    p.add_argument("--size", type=float, help="the object's true size in m along the size column (gives the focal length)")
    p.add_argument("--f", type=float, help="focal length in px, if known from another object")
    p.add_argument("--smooth", type=int, default=3, help="half-window (samples) for smoothing the size (default 3)")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_depth)

    p = sub.add_parser("xyz", help="3D positions of pixel points with known camera distance (pinhole), and 3D distances between them")
    p.add_argument("--f", type=float, help="focal length in px (qpt focal); lab recordings default to the lab camera")
    p.add_argument("--pt", nargs=3, action="append", required=True, metavar=("U", "V", "DIST"), help="pixel point and its camera distance (repeatable)")
    p.add_argument("--axial", action="store_true", help="DIST is depth along the optical axis (default: straight-line range)")
    p.add_argument("--pp", nargs=2, type=float, metavar=("CX", "CY"), help="principal point (default image centre)")
    p.set_defaults(fn=cmd_xyz)

    p = sub.add_parser("plane", help="ground plane from >= 3 pixels with known camera distance; then pixels on it -> 3D, distances along it, heights above it")
    p.add_argument("--f", type=float, help="focal length in px (qpt focal); lab recordings default to the lab camera")
    p.add_argument("--pt", nargs="+", action="append", required=True, metavar="U V DIST [H]",
                   help="a pixel with its camera distance, on the floor or H above it (a ball's centre: its radius); 3+, or 1+ with --horizon")
    p.add_argument("--horizon", type=float, help="image row of the floor's horizon (vanishing line of the ground, or the row an "
                   "approaching object's track converges to): the plane then needs only 1 anchor")
    p.add_argument("--on", nargs=2, action="append", metavar=("U", "V"), help="a pixel on the plane to place in 3D (repeatable)")
    p.add_argument("--height", nargs=4, action="append", metavar=("FU", "FV", "TU", "TV"), help="foot pixel on the plane and top pixel straight above it -> height")
    p.add_argument("--axial", action="store_true"); p.add_argument("--pp", nargs=2, type=float, metavar=("CX", "CY"))
    p.add_argument("--offset", type=float, default=0.0, help="height above the plane of the --on / --track points (e.g. a rolling ball's centre = its radius)")
    p.add_argument("--track", help="csv track (motion/track/segtrack) -> 3D positions on the plane, for qpt fit")
    p.add_argument("--track-id", type=int); p.add_argument("--obj", type=int)
    p.add_argument("--ucol", default="cx"); p.add_argument("--vcol", default="cy")
    p.set_defaults(fn=cmd_plane)

    p = sub.add_parser("focal", help="focal length (px) from objects of known size at known distance; checks that they agree")
    p.add_argument("--obj", nargs="+", action="append", metavar="V",
                   help="PIXEL_SIZE TRUE_SIZE DIST [U V]: one object (same length unit for size and distance). With its pixel "
                        "position U V, DIST is the stated camera-to-object range (converted to depth along the axis); without "
                        "it, DIST is taken as depth along the axis")
    p.add_argument("--vp1", nargs="+", metavar="X", help="2+ image lines (X1 Y1 X2 Y2 each) along one set of parallel 3D lines")
    p.add_argument("--vp2", nargs="+", metavar="X", help="2+ lines along a second set, perpendicular in 3D to the first; "
                   "f from the two vanishing points, with how much it moves for small errors")
    p.set_defaults(fn=cmd_focal)

    p = sub.add_parser("seg", help="SAM 3 on one frame: by text (all instances) or by box/point (one object) -> sizes + mask picture")
    when(p)
    p.add_argument("--text", help="noun phrase, e.g. 'red ball'"); p.add_argument("--box", nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--point", nargs=2, action="append", metavar=("X", "Y"), help="a point ON the object (repeatable)")
    p.add_argument("--neg", nargs=2, action="append", metavar=("X", "Y"), help="a point NOT on the object (repeatable)")
    p.add_argument("--thr", type=float, default=0.4, help="text-mode score threshold (default 0.4)")
    p.add_argument("--name")
    p.set_defaults(fn=cmd_seg)

    p = sub.add_parser("segtrack", help="SAM 3 through the video: by text, or by box/point at one frame -> per-frame sizes/centres csv")
    when(p, span=True)
    p.add_argument("--text"); p.add_argument("--box", nargs=4, metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--point", nargs=2, action="append", metavar=("X", "Y"))
    p.add_argument("--every", type=int, default=1, help="process every k-th frame (long clips)")
    p.add_argument("--thr", type=float, default=0.4); p.add_argument("--name")
    p.set_defaults(fn=cmd_segtrack)

    a = P.parse_args()
    try:
        a.fn(a)
    except (FileNotFoundError, KeyError, ValueError, IndexError, ZeroDivisionError) as e:
        if os.environ.get("QPT_DEBUG"):
            raise
        sys.exit(f"qpt {a.cmd}: error: {type(e).__name__}: {e} (check the arguments; `qpt {a.cmd} -h`)")


if __name__ == "__main__":
    main()
