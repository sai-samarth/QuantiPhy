#!/usr/bin/env python
"""Stage one agent task directory per video: video.mp4, questions.json (no answers),
measurements.json (our SAM 3 evidence block, optional hints) and, unless --no-guide,
GUIDE.md.

    scripts/agent_stage.py --split val  --out .work/val_skill --no-guide
    scripts/agent_stage.py --split test --out ... [--ids-file ids.txt] [--videos v1,v2]

Validation questions come from the organisers' CSV, test from the release CSV. The
ground-truth column never reaches a task file; the leak check at the end greps for
it.
"""
import argparse, json, shutil
from pathlib import Path

import cv2
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SPLITS = {
    "val": dict(csv="QuantiPhy/quantiphy_validation.csv", videos="data/QuantiPhy-validation/validation_videos",
                evidence="runs/evidence_val_conf.csv"),
}


SOURCES = {"simulation": "simulation (rendered scene)", "captured": "lab recording (real camera with a depth sensor)",
           "lab": "lab recording (real camera with a depth sensor)",
           "internet": "internet footage (real video)"}


def load_split(split):
    if split == "val":
        s = SPLITS["val"]
        q = pd.read_csv(REPO / s["csv"]).rename(columns={"Unnamed: 0": "id"})
        return q, REPO / s["videos"], REPO / s["evidence"]
    import sys
    sys.path.insert(0, str(REPO))
    from qp.data import load_split as ls   # test: the same loader the pipeline uses
    q = ls("test")
    return q, None, REPO / "runs/ev_v9/evidence_test_v9.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-guide", action="store_true", help="skill runs: the skill replaces GUIDE.md")
    ap.add_argument("--no-measurements", action="store_true")
    ap.add_argument("--videos", help="comma-separated video ids (default all)")
    ap.add_argument("--videos-json", nargs=2, metavar=("FILE", "KEY"), help="video ids from a json file {key: [ids]}")
    a = ap.parse_args()
    v, vdir, evp = load_split(a.split)
    ev = pd.read_csv(evp).drop_duplicates("id").set_index("id").evidence if evp.exists() else pd.Series(dtype=str)
    root = Path(a.out)
    vids = a.videos.split(",") if a.videos else list(v.video_id.unique())
    if a.videos_json:
        vids = json.load(open(a.videos_json[0]))[a.videos_json[1]]
    for vid in vids:
        d = root / vid
        d.mkdir(parents=True, exist_ok=True)
        src = vdir / f"{vid}.mp4" if vdir else Path(v[v.video_id == vid].video_path.iloc[0])
        shutil.copy(src, d / "video.mp4")
        c = cv2.VideoCapture(str(src)); n = int(c.get(7)); W = int(c.get(3)); H = int(c.get(4)); c.release()
        sub = v[v.video_id == vid]
        fps = float(sub.fps.iloc[0])
        qs = [dict(id=int(r.id), question=r.question.strip(),
                   prior=None if pd.isna(r.get("ground_truth_prior", r.get("prior"))) else str(r.get("ground_truth_prior", r.get("prior"))).strip(),
                   depth_info=None if pd.isna(r.depth_info) else str(r.depth_info).strip(),
                   scene=("3D: objects may be at different distances from the camera" if str(r.video_type)[1] == "3"
                          else "2D: the measured objects are treated as lying in one plane facing the camera"))
              for _, r in sub.iterrows()]
        srcname = str(sub.video_source.iloc[0]) if "video_source" in sub and pd.notna(sub.video_source.iloc[0]) else vid.split("_")[0]
        masked = srcname == "segmentation" or vid.endswith("_segmented") or (vid.startswith("captured") and vid[-1] in "sxX")
        if srcname == "segmentation":                   # masked copies: the underlying source is the clip's own kind
            srcname = vid.split("_")[0]
        source = SOURCES.get(srcname, srcname) + ("; masked copy (background removed or flattened)" if masked else "")
        meta = dict(video="video.mp4", source=source, fps=fps, n_frames=n, duration_s=round((n - 1) / fps, 3), width=W, height=H,
                    note="Times in questions use t = frame_index / fps with this fps (frame 0 is t = 0).")
        json.dump(dict(video=meta, questions=qs), open(d / "questions.json", "w"), indent=2)
        if not a.no_measurements:
            json.dump({str(q["id"]): ev.get(q["id"], "") for q in qs}, open(d / "measurements.json", "w"), indent=2)
        if not a.no_guide:
            shutil.copy(REPO / "scripts/codex_guide.md", d / "GUIDE.md")
    bad = [p for p in root.rglob("*.json") if "ground_truth_posterior" in p.read_text()]
    print(f"{len(vids)} task dirs in {root}; files mentioning ground_truth_posterior: {len(bad)}")


if __name__ == "__main__":
    main()
