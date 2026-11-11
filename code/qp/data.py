"""
Loading QuantiPhy splits and turning a question row into frames.

Two splits with DIFFERENT raw schemas, unified here:
  validation : data/QuantiPhy-validation/validation_dataset.csv  (has ground truth,
               prior column named `ground_truth_prior`, videos in validation_videos/,
               video_source misspelled 'segmentaion')
  test       : submission_template.csv (canonical: carries the scored `id` column and
               is row-for-row identical to test_dataset.parquet), videos alongside it.

FPS WARNING -------------------------------------------------------------------
91 of 568 test videos have a metadata `fps` that disagrees with the container fps,
by up to 1.5x (internet_0022: meta=25, actual=16.67). fps is the term converting
px/frame -> px/s -> m/s, so this scales every dynamic answer on those videos.
Which one the annotators used is UNRESOLVED. `fps_source` makes it switchable so
it can be settled empirically rather than assumed. Default 'meta' matches the
organizers' own baseline script (run_API_results.py uses fps from the CSV).
"""
from __future__ import annotations
import base64, re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VAL_CSV = ROOT / "data/QuantiPhy-validation/validation_dataset.csv"
VAL_VIDEOS = ROOT / "data/QuantiPhy-validation/validation_videos"
TEST_CSV = ROOT / "submission_template.csv"
TEST_VIDEOS = ROOT / "data/QuantiPhy-test"

# "at 1.5s", "when t=3s", "t = 0.20 s"
# Matches "at 1.5s", "at time 1.5s", "when t=3s", "t = 0.20 s".
# NOTE the optional "time" — 343 of 1000 timestamped test questions use
# "at time Xs" and were silently missed by an earlier version of this regex,
# so they got uniform sampling instead of a dense burst at the queried instant.
_TIME_RE = re.compile(
    r"(?:\bat\s+(?:time\s+)?|\bwhen\s+|\bt\s*=\s*)t?\s*=?\s*(\d+(?:\.\d+)?)\s*s\b", re.I)


def parse_time_refs(question: str) -> list[float]:
    """Timestamps (seconds) explicitly named in the question, if any."""
    return [float(x) for x in _TIME_RE.findall(str(question))]


def load_split(split: str, fps_source: str = "meta") -> pd.DataFrame:
    """Return a unified dataframe for 'validation' or 'test'.

    Columns: id, video_id, video_source, video_type, inference_type, question,
             prior, depth_info, video_path, fps_meta, fps_video, fps,
             category, target_kind, prior_kind, t_refs, wants_average
             (+ ground_truth_posterior for validation)
    """
    if split == "validation":
        df = pd.read_csv(VAL_CSV)
        df = df.loc[:, ~df.columns.str.match(r"Unnamed: (?!0)")]      # drop junk cols
        df = df.rename(columns={"Unnamed: 0": "id", "ground_truth_prior": "prior"})
        vdir = VAL_VIDEOS
    elif split == "test":
        df = pd.read_csv(TEST_CSV).rename(columns={"ground_truth_prior": "prior"})
        vdir = TEST_VIDEOS
    else:
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")

    df["video_path"] = df.video_id.map(lambda v: str(vdir / f"{v}.mp4"))
    missing = [p for p in df.video_path.unique() if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos missing, e.g. {missing[:3]}")

    df["fps_meta"] = pd.to_numeric(df["fps"], errors="coerce")
    df["fps_video"] = df.video_path.map(_container_fps)
    if fps_source == "meta":
        df["fps"] = df["fps_meta"]
    elif fps_source == "video":
        df["fps"] = df["fps_video"]
    else:
        raise ValueError("fps_source must be 'meta' or 'video'")

    vt, it = df.video_type.astype(str), df.inference_type.astype(str)
    df["category"] = it.str[0] + vt.str[1]        # S2 / D2 / S3 / D3 (the scored axis)
    df["prior_kind"] = it.str[0]                  # S=static prior, D=dynamic prior
    df["target_kind"] = it.str[1]                 # S=size/distance, D=velocity/accel
    df["t_refs"] = df.question.map(parse_time_refs)
    df["wants_average"] = df.question.str.contains(r"\baverage\b", case=False, na=False)
    return df


_FPS_CACHE: dict[str, float] = {}
def _container_fps(path: str) -> float:
    if path not in _FPS_CACHE:
        cap = cv2.VideoCapture(path)
        _FPS_CACHE[path] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
    return _FPS_CACHE[path]


# ─────────────────────────── frame sampling ───────────────────────────
@dataclass
class Frames:
    """Sampled frames plus the timing metadata needed to convert px -> px/s."""
    images: list[np.ndarray]
    times: list[float]          # seconds, using the CHOSEN fps
    indices: list[int]          # original frame numbers
    fps: float
    n_total: int

    def __len__(self): return len(self.images)


def read_frames(video_path: str, indices, fps: float) -> Frames:
    """Read specific frame indices. Sequential scan — far faster than seeking."""
    want = sorted(set(int(i) for i in indices))
    cap = cv2.VideoCapture(video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    imgs, got, i, wi = [], [], 0, 0
    while wi < len(want):
        ok, fr = cap.read()
        if not ok:
            break
        if i == want[wi]:
            imgs.append(fr); got.append(i); wi += 1
        i += 1
    cap.release()
    return Frames(imgs, [g / fps if fps else 0.0 for g in got], got, fps, n_total)


def plan_indices(n_total: int, fps: float, n: int = 8, t_refs=(),
                 window: float = 0.5, wants_average: bool = False) -> list[int]:
    """Choose WHICH frames to read, from the question's demands.

    Three regimes, because one strategy cannot serve all question types:

    1. No timestamp, or an 'average' question -> UNIFORM across the clip.
       Size questions don't care when you look; averages need the whole span.

    2. Timestamp(s) named -> DENSE burst around each t, plus a few uniform
       anchors for scene context. Instantaneous velocity needs closely-spaced
       frames near t; uniform sampling at 1.25s spacing cannot resolve it.

    3. Always include first and last frame (cheap, bounds the motion).
    """
    if n_total <= 0:
        return []
    if not t_refs or wants_average:
        return sorted(set(np.linspace(0, n_total - 1, min(n, n_total)).astype(int).tolist()))

    idx = set()
    anchors = max(2, n // 4)
    idx.update(np.linspace(0, n_total - 1, anchors).astype(int).tolist())
    per_t = max(3, (n - anchors) // max(1, len(t_refs)))
    for t in t_refs:
        c = int(round(t * fps))
        half = (per_t - 1) / 2
        lo, hi = c - half * (window * fps / max(half, 1)), c + half * (window * fps / max(half, 1))
        idx.update(np.clip(np.linspace(lo, hi, per_t).round().astype(int), 0, n_total - 1).tolist())
    return sorted(int(i) for i in idx)


# Frame plan v2 (from the Oct-14 input review). v1 above sends an EVEN burst
# that never contains the asked instant (919 test rows: nearest frame 2-3
# frames away), loses slots to clipping and de-duplication (432 rows get < 8
# frames; captured_0018 asks 'at 3.0s' of a 2.0 s clip and gets 2 frames), and
# reads 'from 0.5 s to 1.0 s' as no time at all (198 rows; 21 have no frame
# inside the interval).
_IV = re.compile(r"(?:from|between|over|in)\s+(?:time\s+)?(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)?\s*"
                 r"(?:to|and|-|until)\s*(?:time\s+)?(\d+(?:\.\d+)?|the end)\s*(?:s|sec|seconds?)?\b", re.I)
_IV2 = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s\b", re.I)
_FIRST = re.compile(r"\bfirst\s+(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?)\b", re.I)


def question_interval_frames(q: str):
    """(a, b) in seconds if the question names a span; b None = end of clip."""
    q = str(q)
    m = _IV.search(q) or _IV2.search(q)
    if m:
        a, b = float(m.group(1)), m.group(2)
        b = None if b.lower() == "the end" else float(b)
        return (a, b) if (b is None or b > a) else (b, a)
    m = _FIRST.search(q)
    return (0.0, float(m.group(1))) if m else None


def plan_indices_v2(n_total: int, fps: float, n: int = 8, t_refs=(), interval=None,
                    wants_average: bool = False, step_s: float = 0.1, burst: int = 5) -> list[int]:
    if n_total <= 0:
        return []
    if n_total <= n:
        return list(range(n_total))
    end = (n_total - 1) / fps
    pts = []
    if interval is not None:
        a, b = interval
        b = end if b is None else min(b, end)
        a = min(a, b)
        pts = list(np.linspace(a, b, max(3, n - 2)))
    elif t_refs and not wants_average:
        per = burst if len(t_refs) == 1 else 3
        st = max(1, int(round(step_s * fps)))
        half = per // 2
        for t in t_refs:
            c = int(round(min(t, end) * fps))
            lo = min(max(c - half * st, 0), max(0, n_total - 1 - 2 * half * st))
            pts += [(lo + k * st) / fps for k in range(per)]
    if not pts:      # no time named: exactly v1's uniform plan
        return sorted(set(np.linspace(0, n_total - 1, min(n, n_total)).astype(int).tolist()))
    idx = {int(np.clip(round(p * fps), 0, n_total - 1)) for p in pts}
    idx |= {0, n_total - 1}                      # clip anchors, as in v1
    while len(idx) < n:                          # fill the widest gaps first
        have = np.array(sorted(idx))
        cand = np.arange(n_total)
        gap = np.abs(cand[:, None] - have[None, :]).min(axis=1)
        idx.add(int(cand[gap.argmax()]))
    return sorted(idx)[:n] if len(idx) <= n else sorted(idx)


def sample_for_row(row, n: int = 8, window: float = 0.5) -> Frames:
    """Question-aware frame sampling for one dataframe row."""
    fps = float(row["fps"]) if pd.notna(row["fps"]) else _container_fps(row["video_path"])
    cap = cv2.VideoCapture(row["video_path"])
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idx = plan_indices(n_total, fps, n=n, t_refs=row["t_refs"],
                       window=window, wants_average=bool(row["wants_average"]))
    return read_frames(row["video_path"], idx, fps)


def encode(frames: Frames, max_side: int = 768, quality: int = 90) -> list[str]:
    """Base64 JPEGs, downscaled so the longest side is <= max_side.

    max_side trades tokens against measurement precision: at 768px a frame costs
    ~225 prompt tokens on Qwen3-VL. Raising it improves pixel-measurement accuracy
    (which is the whole task) at quadratic token cost. Tune on MRA.
    """
    out = []
    for img in frames.images:
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            out.append(base64.b64encode(buf).decode())
    return out


# ─────────────────────── batched extraction ───────────────────────
def sample_batched(df, n: int = 8, window: float = 0.5, max_side: int = 768,
                   quality: int = 90, workers: int = 8, with_times: bool = False,
                   plan: str = "v1") -> dict:
    """Frames for EVERY row, decoding each video exactly once.

    Per-row extraction reopens and rescans a video for each question. With 5.8
    questions per video that is 5.6x redundant decoding — measured at 594s for
    the test split, vs 395s of actual GPU inference. This groups rows by video,
    takes the UNION of frame indices needed, decodes in one pass, then slices
    per row. Returns {row_index: [b64 jpeg, ...]}, or with_times=True
    {row_index: [(b64 jpeg, t_seconds), ...]} with t = frame index / the row's fps
    (the same time base the questions and the evidence block use).
    """
    from concurrent.futures import ThreadPoolExecutor

    def one_video(item):
        vid, sub = item
        path = sub.iloc[0]["video_path"]
        cap = cv2.VideoCapture(path)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        plans, fpss = {}, {}
        for ridx, row in sub.iterrows():
            fps = float(row["fps"]) if pd.notna(row["fps"]) else _container_fps(path)
            fpss[ridx] = fps
            if plan == "v2":
                plans[ridx] = plan_indices_v2(
                    n_total, fps, n=n, t_refs=row["t_refs"],
                    interval=question_interval_frames(row["question"]),
                    wants_average=bool(row["wants_average"]))
            else:
                plans[ridx] = plan_indices(n_total, fps, n=n, t_refs=row["t_refs"],
                                           window=window, wants_average=bool(row["wants_average"]))
        need = sorted({i for p in plans.values() for i in p})
        fr = read_frames(path, need, 1.0)
        pos = {idx: k for k, idx in enumerate(fr.indices)}

        enc = {}
        for k, img in enumerate(fr.images):
            h, w = img.shape[:2]
            s = max_side / max(h, w)
            im = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                            interpolation=cv2.INTER_AREA) if s < 1 else img
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                enc[k] = base64.b64encode(buf).decode()
        if with_times:
            return {r: [(enc[pos[i]], i / fpss[r]) for i in p if i in pos and pos[i] in enc]
                    for r, p in plans.items()}
        return {r: [enc[pos[i]] for i in p if i in pos and pos[i] in enc]
                for r, p in plans.items()}

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(one_video, list(df.groupby("video_id"))):
            out.update(part)
    return out
