#!/usr/bin/env python
"""Build the TEST evidence exactly as sub08 shipped it, in one reproducible command.

sub07/sub08 were built by a scratch script that never reached the repo. This is
that recipe, committed:

  * QP_INSTANT_RATE=1 (the instant-rate instruments; tags are unconditional)
  * the 18 videos in data/test_videos_cap192.txt segmented at a 192-frame cap
    (QP_EVIDENCE_MAX_FRAMES=192) -- long/crowded clips whose SAM tracking state
    passes 32 GB; every other video at full frame rate. MAX_FRAMES is read at
    import time, so the two groups run in separate worker processes.
  * fps_source = meta

    scripts/build_test_evidence.py --out runs/evidence_test_v9.csv --workers 8
    scripts/build_test_evidence.py --out /tmp/x.csv --cache-only   # no GPU
    CUDA_VISIBLE_DEVICES=0 scripts/build_test_evidence.py --out /tmp/pre.csv \
        --only simulation_0140,internet_0027 --workers 1             # warm cache

--cache-only never runs SAM: a cache miss is logged and treated as NOT FOUND, so
it lists exactly which (video, phrase) pairs need the segmenter. It can
over-report: a miss on the first phrase makes the head-noun fallback ask for a
second one that a real run would never request. Never ship its output.
"""
import argparse, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CAP_FILE = ROOT / "data/test_videos_cap192.txt"
COLS = ["split", "id", "video_id", "n_ref", "n_tgt", "ref_phrase", "tgt_phrase",
        "ref_kind", "evidence", "error"]


def worker(a):
    import cv2
    import pandas as pd
    if a.cache_only:
        import qp.segment
        def _refuse(path, phrases, fps, **kw):
            for ph in phrases:
                print(f"MISS\t{Path(path).stem}\t{ph}\t{kw.get('max_frames')}", flush=True)
            return {}, None, 0
        qp.segment.track_phrases = _refuse
    from qp.data import load_split
    from qp.evidence import gather, render

    cap = set(CAP_FILE.read_text().split())
    df = load_split("test")
    df = df[df.video_id.isin(cap)] if a.cap else df[~df.video_id.isin(cap)]
    if a.only:
        df = df[df.video_id.isin(a.only.split(","))]
    mine = sorted(df.video_id.unique())[a.shard::a.num_shards]
    df = df[df.video_id.isin(mine)]
    out = []
    for _, row in df.sort_values("video_id").iterrows():
        try:
            c = cv2.VideoCapture(row["video_path"])
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT))
            hw = (int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(c.get(cv2.CAP_PROP_FRAME_WIDTH)))
            c.release()
            ev = gather(row, n, hw)
            out.append(dict(split="test", id=row["id"], video_id=row["video_id"],
                            n_ref=len(ev["ref_inst"]), n_tgt=len(ev["tgt_inst"]),
                            ref_phrase=ev["ref_phrase"], tgt_phrase=ev["tgt_phrase"],
                            ref_kind=ev.get("ref_kind"), evidence=render(ev), error=""))
        except Exception as e:
            out.append(dict(split="test", id=row["id"], video_id=row["video_id"],
                            n_ref=0, n_tgt=0, ref_phrase="", tgt_phrase="", ref_kind="",
                            evidence="", error=f"{type(e).__name__}: {e}"))
            print(f"ERR\t{row['id']}\t{type(e).__name__}: {e}", flush=True)
    pd.DataFrame(out, columns=COLS).to_csv(a.out, index=False)
    print(f"DONE {len(out)} rows", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8, help="full-rate shards; cap group gets 2")
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated video_ids (partial build; no merge check)")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--cap", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    import pandas as pd
    out = Path(a.out)
    parts_dir = out.parent / (out.stem + "_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(0, k, a.workers) for k in range(a.workers)] + [(1, k, 2) for k in range(2)]
    procs = []
    for cap, k, n in jobs:
        env = dict(os.environ, QP_INSTANT_RATE="1")
        env.pop("QP_EVIDENCE_MAX_FRAMES", None)
        if cap:
            env["QP_EVIDENCE_MAX_FRAMES"] = "192"
        stem = parts_dir / f"{'cap' if cap else 'main'}_s{k}"
        cmd = [sys.executable, __file__, "--worker", "--out", f"{stem}.csv", "--cap", str(cap),
               "--shard", str(k), "--num-shards", str(n), "--only", a.only] + \
              (["--cache-only"] if a.cache_only else [])
        procs.append((stem, subprocess.Popen(cmd, env=env, stdout=open(f"{stem}.log", "w"),
                                             stderr=subprocess.STDOUT)))
    bad = [str(s) for s, p in procs if p.wait() != 0]
    if bad:
        raise SystemExit(f"workers failed: {bad}")

    d = pd.concat([pd.read_csv(f"{s}.csv") for s, _ in procs]).sort_values("id")
    if d.id.duplicated().any() or (len(d) != 3289 and not a.only):
        raise SystemExit(f"merge produced {len(d)} rows, {int(d.id.duplicated().sum())} duplicate ids")
    d.to_csv(out, index=False)
    logs = "".join(Path(f"{s}.log").read_text() for s, _ in procs)
    misses = sorted({l for l in logs.splitlines() if l.startswith("MISS")})
    errs = [l for l in logs.splitlines() if l.startswith("ERR")]
    print(f"wrote {out}: {len(d)} rows | ref found {(d.n_ref > 0).mean():.0%} | "
          f"both {((d.n_ref > 0) & (d.n_tgt > 0)).mean():.0%} | errors {len(errs)}")
    if a.cache_only:
        print(f"cache misses (video, phrase, max_frames): {len(misses)}")
        for l in misses:
            print("  " + l)


if __name__ == "__main__":
    main()
