"""SAM 3 text-prompted video segmentation + measurement, with mandatory visual output.

Measurement rule of this project: never trust a number whose mask you have not seen.
Every tracking call writes a contact sheet so the mask can be eyeballed.
"""
from __future__ import annotations
import os
from pathlib import Path

import cv2, numpy as np, pandas as pd, torch

_MODEL = {}

# SAM 3's detection floor. The default 0.5/0.7 silently discards compositional
# phrases: "big bird" returns 0 instances at 0.5 and 2 at 0.10, on a clip that
# plainly contains two birds. But dropping it also INVENTS objects -- at 0.05,
# "person on the left" returns 2 instances on a video containing only birds, and
# a false positive reported to the VLM as a measured fact is worse than silence.
# So this is a precision/recall dial to be FITTED, not lowered on principle.
# Override with QP_SAM_SCORE_THR / QP_SAM_NEWDET_THR.
SCORE_THR = float(os.environ.get("QP_SAM_SCORE_THR", 0.5))
NEWDET_THR = float(os.environ.get("QP_SAM_NEWDET_THR", 0.7))


def _load(device="cuda"):
    key = ("m", SCORE_THR, NEWDET_THR)
    if key not in _MODEL:
        from transformers import Sam3VideoModel, Sam3VideoProcessor
        _MODEL.clear()
        _MODEL["p"] = Sam3VideoProcessor.from_pretrained("facebook/sam3")
        _MODEL[key] = Sam3VideoModel.from_pretrained(
            "facebook/sam3", dtype=torch.bfloat16,
            score_threshold_detection=SCORE_THR,
            new_det_thresh=NEWDET_THR).to(device).eval()
    return _MODEL["p"], _MODEL[key]


def read_video(path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path)); out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(fr)
    cap.release()
    return out


def implausible(st: dict, frame_hw) -> bool:
    """True if a mask cannot be a real compact object.

    SAM reports full-frame SLIVERS with confidence 1.0 — internet_0027 has an
    11 x 480 px vertical strip at the left edge that it calls a "billiard ball".
    Trackers latch onto these and teleport, which silently destroys derivative
    estimates (one such jump turned an acceleration of 0.05 into 1.76 m/s^2).
    Confidence does NOT catch this; shape does.
    """
    H, W = frame_hw
    if st is None:
        return True
    if st["minor"] > 0 and st["major"] / st["minor"] > 6:
        return True
    if st["bh"] > 0.9 * H or st["bw"] > 0.9 * W:
        return True
    return False


def _clean(mask: np.ndarray):
    """Largest connected component only — see note in _stats."""
    m = mask.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 2:
        k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lab == k).astype(np.uint8)
    return m.astype(bool)


def _stats(mask: np.ndarray) -> dict | None:
    """Sub-pixel-ish shape descriptors of a boolean mask.

    `area` and the principal-axis lengths from image moments are far more stable
    than a bounding box when the object rotates, which matters because we convert
    apparent size into depth.
    """
    m = mask.astype(np.uint8)
    if m.sum() < 4:
        return None
    # Keep ONLY the largest connected component. SAM sometimes returns the target
    # PLUS a distant speck (a star, a highlight). Area stays plausible but the
    # bounding box and shape moments explode -- e.g. an astronaut mask reading
    # 937 px "major axis" across a 480 px frame. Since apparent size is our ruler,
    # that silently corrupts the depth estimate.
    m = _clean(m).astype(np.uint8)
    if m.sum() < 4:
        return None
    ys, xs = np.nonzero(m)
    M = cv2.moments(m, binaryImage=True)
    cu, cv_ = M["m10"]/M["m00"], M["m01"]/M["m00"]
    mu20, mu02, mu11 = M["mu20"]/M["m00"], M["mu02"]/M["m00"], M["mu11"]/M["m00"]
    common = np.sqrt(max(0.0, 4*mu11**2 + (mu20-mu02)**2))
    lam1, lam2 = (mu20+mu02+common)/2, (mu20+mu02-common)/2
    return dict(u=cu, v=cv_, area=int(m.sum()),
                bw=int(xs.max()-xs.min()+1), bh=int(ys.max()-ys.min()+1),
                major=float(4*np.sqrt(max(lam1, 0))),      # ~length of the object
                minor=float(4*np.sqrt(max(lam2, 0))),
                equiv=float(2*np.sqrt(m.sum()/np.pi)))     # diameter of equal-area circle


def track_video(path, phrase, fps, out_png=None, device="cuda",
                keep_largest=True) -> tuple[pd.DataFrame, list]:
    """Track a text-named object through a clip. Returns (per-frame stats, masks)."""
    proc, model = _load(device)
    bgr = read_video(path)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr]
    sess = proc.init_video_session(video=rgb, inference_device=device, dtype=torch.bfloat16)
    proc.add_text_prompt(sess, phrase)

    rows, masks = [], []
    for i, out in enumerate(model.propagate_in_video_iterator(sess)):
        r = proc.postprocess_outputs(sess, out)
        ms = r.get("masks"); sc = r.get("scores")
        best, bscore = None, 0.0
        if ms is not None and len(ms):
            arr = [np.asarray(m.squeeze().cpu(), dtype=bool) for m in ms]
            if keep_largest:
                k = int(np.argmax([a.sum() for a in arr]))
            else:
                k = int(np.argmax([float(s) for s in sc]))
            best = arr[k]; bscore = float(sc[k]) if sc is not None and len(sc) > k else np.nan
        if best is not None:
            best = _clean(best)          # the picture must show the measured mask
        masks.append(best)
        st = _stats(best) if best is not None else None
        rows.append(dict(f=i, t=i/fps, score=bscore, found=st is not None,
                         **(st or dict(u=np.nan, v=np.nan, area=0, bw=0, bh=0,
                                       major=np.nan, minor=np.nan, equiv=np.nan))))
    d = pd.DataFrame(rows)
    if out_png:
        contact_sheet(bgr, masks, d, phrase, out_png)
    return d, masks


def contact_sheet(bgr, masks, d, title, out_png, n=8, width=340):
    """Overlay masks on evenly-spaced frames so a human can check them."""
    idx = np.linspace(0, len(bgr)-1, min(n, len(bgr))).astype(int)
    tiles = []
    for i in idx:
        vis = bgr[i].copy()
        m = masks[i]
        if m is not None and m.sum():
            vis[m] = (0.45*np.array([0, 0, 255]) + 0.55*vis[m]).astype(np.uint8)
            cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)
            r = d.iloc[i]
            cv2.putText(vis, f"t={r.t:.2f}s maj={r.major:.0f}px", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(vis, f"t={i/max(1,len(bgr)-1):.2f} NO MASK", (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        h = int(width*vis.shape[0]/vis.shape[1])
        tiles.append(cv2.resize(vis, (width, h)))
    rows_ = [np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]
    w = min(r.shape[1] for r in rows_)
    grid = np.vstack([r[:, :w] for r in rows_])
    band = np.zeros((26, grid.shape[1], 3), np.uint8)
    cv2.putText(band, title[:110], (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack([band, grid]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out_png


def track_multi(path, phrase, fps, device="cuda", min_area=30):
    """Track ALL instances of a phrase, keeping SAM's persistent object ids.

    Needed when a clip contains several of the same thing ("the LEFT tennis ball"
    vs "the RIGHT tennis ball"). Returns (DataFrame with an obj_id column,
    {frame_idx: {obj_id: mask}}).
    """
    proc, model = _load(device)
    bgr = read_video(path)
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in bgr]
    sess = proc.init_video_session(video=rgb, inference_device=device, dtype=torch.bfloat16)
    proc.add_text_prompt(sess, phrase)

    rows, per_frame = [], {}
    for i, out in enumerate(model.propagate_in_video_iterator(sess)):
        r = proc.postprocess_outputs(sess, out)
        ms, sc = r.get("masks"), r.get("scores")
        ids = r.get("object_ids")
        per_frame[i] = {}
        if ms is None or not len(ms):
            continue
        for j in range(len(ms)):
            m = np.asarray(ms[j].squeeze().cpu(), dtype=bool)
            if m.sum() < min_area:
                continue
            m = _clean(m)
            st = _stats(m)
            if st is None or implausible(st, bgr[0].shape[:2]):
                continue
            oid = int(ids[j]) if ids is not None and len(ids) > j else j
            per_frame[i][oid] = m
            rows.append(dict(f=i, t=i/fps, obj_id=oid,
                             score=float(sc[j]) if sc is not None and len(sc) > j else np.nan,
                             **st))
    return pd.DataFrame(rows), per_frame, bgr


def pick_object(df, rule):
    """Choose one persistent object id. rule: 'leftmost'|'rightmost'|'largest'|'longest'."""
    if not len(df):
        return None
    g = df.groupby("obj_id")
    if rule == "leftmost":   return int(g.u.median().idxmin())
    if rule == "rightmost":  return int(g.u.median().idxmax())
    if rule == "largest":    return int(g.area.median().idxmax())
    if rule == "longest":    return int(g.size().idxmax())
    raise ValueError(rule)


def sheet_multi(bgr, per_frame, df, title, out_png, obj_id=None, n=12, width=420, fps=24.0):
    """Contact sheet: raw frame on top, measured mask(s) below, stats between."""
    COL = [(0,0,255),(0,255,0),(255,128,0),(255,0,255),(0,255,255)]
    idx = np.linspace(0, len(bgr)-1, min(n, len(bgr))).astype(int)
    cols = []
    for i in idx:
        raw = bgr[i].copy(); vis = bgr[i].copy()
        got = per_frame.get(i, {})
        show = {obj_id: got[obj_id]} if (obj_id is not None and obj_id in got) else got
        txt = f"t={i/fps:.2f}s  " + ("NO MASK" if not show else "")
        for k, (oid, m) in enumerate(sorted(show.items())):
            c = COL[oid % len(COL)]
            vis[m] = (0.40*np.array(c) + 0.60*vis[m]).astype(np.uint8)
            cs = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(vis, cs, -1, c, 2)
            r = df[(df.f == i) & (df.obj_id == oid)]
            if len(r):
                txt += f"id{oid}:eq={r.equiv.iloc[0]:.1f}px "
        w = width; h = int(w*raw.shape[0]/raw.shape[1])
        raw = cv2.resize(raw, (w, h)); vis = cv2.resize(vis, (w, h))
        band = np.zeros((20, w, 3), np.uint8)
        cv2.putText(band, txt[:78], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,255,255), 1)
        cols.append(np.vstack([raw, band, vis]))
    rows_ = [np.hstack(cols[i:i+4]) for i in range(0, len(cols), 4)]
    W = min(r.shape[1] for r in rows_)
    grid = np.vstack([r[:, :W] for r in rows_])
    hdr = np.zeros((30, grid.shape[1], 3), np.uint8)
    cv2.putText(hdr, title[:150], (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), np.vstack([hdr, grid]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_png


# ─────────────────── fast path: one pass, many phrases ───────────────────
def track_phrases(path, phrases, fps, device="cuda", min_area=30,
                  stride=None, max_frames=96):
    """Segment SEVERAL phrases in ONE video pass, on a SUBSAMPLED set of frames.

    WHY THIS EXISTS, measured on a 428-frame clip:

        2 phrases x every frame (what we were doing)   88.9 s   15.4 GB peak
        2 phrases, one pass, every frame               60.5 s   15.4 GB
        2 phrases, one pass, stride 5 (86 frames)      10.1 s    4.6 GB   8.8x
        2 phrases, one pass, stride 9 (48 frames)       6.2 s    3.4 GB  14.4x

    Two independent wastes were being paid. First, the reference phrase and the
    target phrase were segmented in SEPARATE sessions, so the vision encoder ran
    over the whole video twice; `prompt_to_obj_ids` makes one session
    unambiguous, so it need only run once. Second, and much larger, we encoded
    every frame of the video when the measurements only need a few dozen
    samples: init_video_session normalises the entire clip to float32 at
    1024x1024 up front, which is where both the time and the 15 GB went.

    THE TRAP: frame indices must stay in the ORIGINAL numbering. Every
    downstream time conversion is t = f / fps, so recording the position within
    the subsampled list instead would scale every speed and acceleration by the
    stride -- silently, and in the direction that still looks plausible.
    """
    proc, model = _load(device)
    bgr = read_video(path)
    n_all = len(bgr)
    if not n_all:
        return {p: (pd.DataFrame(), {}) for p in phrases}, bgr, 0
    if stride is None:
        # max_frames None/0 means FULL RATE: encode every frame
        stride = max(1, int(np.ceil(n_all / max_frames))) if max_frames else 1
    keep = list(range(0, n_all, stride))
    rgb = [cv2.cvtColor(bgr[i], cv2.COLOR_BGR2RGB) for i in keep]

    phrases = [p for p in dict.fromkeys(phrases) if p and len(p) >= 3]
    if not phrases:
        return {}, bgr, len(keep)

    # ONE SESSION PER PHRASE, deliberately. Putting several text prompts in a
    # single session is 1.5x faster and WRONG: the prompts compete for objects.
    # Verified on internet_0004 -- asked for "ruler" and "wood block" together,
    # SAM assigned the long thin ruler object to "wood block" (major axis
    # 205.6 -> 558.4 px) and returned nothing at all for "ruler". Six other
    # phrases were bit-identical, which is exactly what makes it dangerous.
    # The frame subsampling is where the real 8.8x lives; this is not worth 1.5x.
    out = {}
    hw = bgr[0].shape[:2]
    for ph in phrases:
        # MEMORY FALLBACK, not a speed one. A handful of very long clips push
        # SAM's per-frame tracking state past 32 GB and the allocator then
        # thrashes indefinitely rather than failing fast. Subsampling costs
        # 0.0197 MRA and is refused globally for that reason -- but on a clip
        # that would otherwise return NOTHING, a strided pass is strictly
        # better. Full rate first; halve only on an actual OOM.
        sess = None
        use = rgb
        keep_now = keep
        for attempt in range(4):
            try:
                sess = proc.init_video_session(video=use, inference_device=device,
                                               dtype=torch.bfloat16)
                proc.add_text_prompt(sess, ph)
                break
            except torch.OutOfMemoryError:
                sess = None
                torch.cuda.empty_cache()
                st2 = 2 ** (attempt + 1)
                keep_now = list(range(0, n_all, stride * st2))
                use = [cv2.cvtColor(bgr[i], cv2.COLOR_BGR2RGB) for i in keep_now]
                print(f"    OOM on {len(rgb)} frames; retrying {ph!r} at "
                      f"{len(use)} frames", flush=True)
        if sess is None:
            out[ph] = (pd.DataFrame(), {})
            continue
        rows, per_frame = [], {}
        for k, o in enumerate(model.propagate_in_video_iterator(sess)):
            r = proc.postprocess_outputs(sess, o)
            fi = keep_now[k] if k < len(keep_now) else k * stride  # ORIGINAL index
            per_frame[fi] = {}
            ms, sc, ids = r.get("masks"), r.get("scores"), r.get("object_ids")
            if ms is None or not len(ms):
                continue
            for j in range(len(ms)):
                m = np.asarray(ms[j].squeeze().cpu(), dtype=bool)
                if m.sum() < min_area:
                    continue
                m = _clean(m)
                st = _stats(m)
                if st is None or implausible(st, hw):
                    continue
                oid = int(ids[j]) if ids is not None and len(ids) > j else j
                per_frame[fi][oid] = m
                rows.append(dict(f=fi, t=fi / fps, obj_id=oid,
                                 score=float(sc[j]) if sc is not None and len(sc) > j else np.nan,
                                 **st))
        del sess
        out[ph] = (pd.DataFrame(rows), per_frame)
    torch.cuda.empty_cache()
    return out, bgr, len(keep)
