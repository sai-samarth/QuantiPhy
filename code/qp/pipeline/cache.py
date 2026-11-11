"""Disk cache for expensive model calls.

WHY: SAM runs at ~0.3 s/frame and the test set has 41,063 frames over 568 videos
with ~5.8 questions per video. Segmenting "the boat" once per question means
~6x redundant work. The cache key is (kind, video, phrase, params) so the same
object requested by different questions is computed once.

WHY DISK not memory: runs are long and get interrupted. A crashed run should not
throw away an hour of segmentation.

WHAT IS NOT CACHED: anything cheap or non-deterministic. Geometry is pure and
fast, so it is recomputed; that keeps the cache small and means a bug fix in the
arithmetic does not require invalidating segmentation.
"""
from __future__ import annotations
import hashlib, json, pickle
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2] / "cache"

# Keys embed absolute video paths. Swap the checkout's own prefix for a fixed one,
# so a cache stays valid wherever the repository is checked out.
_REPO = str(Path(__file__).resolve().parents[2])
_KEY_ROOT = "/quantiphy"


def _canon(v):
    return _KEY_ROOT + v[len(_REPO):] if isinstance(v, str) and v.startswith(_REPO + "/") else v


def key_of(kind: str, **parts) -> str:
    parts = {k: _canon(v) for k, v in parts.items()}
    blob = json.dumps({"kind": kind, **parts}, sort_keys=True, default=str)
    return f"{kind}_{hashlib.sha1(blob.encode()).hexdigest()[:20]}"


def get_or_compute(kind: str, compute: Callable[[], Any], **parts) -> Any:
    d = ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key_of(kind, **parts)}.pkl"
    if p.exists():
        try:
            with p.open("rb") as f:
                return pickle.load(f)
        except Exception:
            p.unlink(missing_ok=True)          # corrupt entry: recompute
    val = compute()
    tmp = p.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(val, f, protocol=4)
    tmp.replace(p)                             # atomic: no half-written entries
    return val


def stats() -> dict:
    if not ROOT.exists():
        return {}
    return {d.name: (len(list(d.glob("*.pkl"))),
                     round(sum(f.stat().st_size for f in d.glob("*.pkl"))/1e6, 1))
            for d in ROOT.iterdir() if d.is_dir()}


def get(kind: str, **parts):
    """Cached value or None. For callers that batch several keys into one
    expensive computation and so cannot use get_or_compute's single-key shape."""
    f = ROOT / kind / f"{key_of(kind, **parts)}.pkl"
    if not f.exists():
        return None
    try:
        with f.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        f.unlink(missing_ok=True)
        return None


def put(kind: str, value, **parts) -> None:
    d = ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{key_of(kind, **parts)}.pkl"
    tmp = f.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(value, fh, protocol=4)
    tmp.replace(f)
