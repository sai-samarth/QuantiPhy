#!/usr/bin/env python
"""Per-case-rule guard for skill edits (the project's hard rule: no per-case rules).

    scripts/opt_guard.py <old skill dir> <new skill dir>

Scans the ADDED lines of every changed text file (SKILL.md, qpt code) for things that
would make the skill answer specific videos rather than teach a general method:
  * video ids (simulation_0123, captured_0013x, internet_0005_segmented ...);
  * ground-truth answer values from the validation set (any added number with >= 3
    significant figures within 0.5% of a truth value);
  * distinctive object phrases: 2-3 word phrases from validation or test questions that
    occur in only one or two videos.
Size rules (user, Oct 19): the skill text (every .md in the skill) must stay under 10,000
words -- a hard failure; and one revision should optimise the existing text, not pile on
fixes for the latest round -- growth of more than 10% or 250 words is flagged.
Exit status 1 and a list of hits when anything is found; 0 when clean. Hits are for a
human/agent to judge (a phrase can be generic), but nothing goes in unexplained.
"""
import difflib, re, sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
VID = re.compile(r"\b(simulation|captured|internet)_\d{3,4}[a-z_]*\b", re.I)
NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def added_lines(old: Path, new: Path):
    for f in sorted(new.rglob("*")):
        if f.is_dir() or "__pycache__" in f.parts or f.suffix not in (".md", ".py", ".sh", ".txt", ""):
            continue
        rel = f.relative_to(new)
        o = (old / rel).read_text(errors="replace").splitlines() if (old / rel).exists() else []
        n = f.read_text(errors="replace").splitlines()
        for line in difflib.unified_diff(o, n, lineterm="", n=0):
            if line.startswith("+") and not line.startswith("+++"):
                yield str(rel), line[1:]


def rare_phrases():
    sys.path.insert(0, str(REPO))
    v = pd.read_csv(REPO / "QuantiPhy/quantiphy_validation.csv")
    try:
        from qp.data import load_split
        t = load_split("test")[["video_id", "question"]]
    except Exception:
        t = pd.DataFrame(columns=["video_id", "question"])
    q = pd.concat([v[["video_id", "question"]], t])
    where = defaultdict(set)
    stop = {"what", "is", "the", "of", "in", "at", "a", "an", "and", "to", "from", "between", "meters", "cm", "m/s",
            "cm/s", "s", "time", "speed", "velocity", "acceleration", "distance", "length", "width", "height",
            "diameter", "average", "total", "on", "by", "with", "its", "this", "that", "how", "far", "much",
            "do", "not", "does", "are", "was", "be", "for", "as", "or", "if", "then", "than", "each", "every",
            "one", "two", "first", "last", "longest", "shortest", "side", "top", "bottom", "left", "right",
            "front", "back", "edge", "part", "point", "center", "centre", "end", "start", "frame", "camera"}
    for vid, text in zip(q.video_id, q.question):
        w = re.findall(r"[a-z][a-z'-]+", str(text).lower())
        for k in (2, 3):
            for i in range(len(w) - k + 1):
                g = w[i:i + k]
                if g[0] in stop or g[-1] in stop:
                    continue
                where[" ".join(g)].add(vid)
    return {p for p, vids in where.items() if len(vids) <= 2}


def main():
    old, new = Path(sys.argv[1]), Path(sys.argv[2])
    truths = pd.read_csv(REPO / "QuantiPhy/quantiphy_validation.csv").ground_truth_posterior.dropna()
    sig = lambda txt: len(re.sub(r"[^0-9]", "", txt).lstrip("0"))     # significant digits
    truths = [float(x) for x in truths if sig(f"{float(x):g}") >= 3]   # 2-digit values (10, 12, 25) collide with ordinary code
    rare = rare_phrases()
    hits = []
    for rel, line in added_lines(old, new):
        low = line.lower()
        for m in VID.finditer(line):
            hits.append((rel, "video id", m.group(0), line.strip()))
        for m in (NUM.finditer(line) if rel.endswith(".md") else []):   # answer values only matter in the text
            x = float(m.group(0))
            if x != 0 and sig(m.group(0)) >= 3 and any(abs(x - t) <= 0.005 * abs(t) for t in truths):
                hits.append((rel, "truth value", m.group(0), line.strip()))
        for p in rare:
            if re.search(r"\b" + re.escape(p) + r"\b", low):
                hits.append((rel, "rare object phrase", p, line.strip()))
    words = lambda d: sum(len(f.read_text(errors="replace").split()) for f in d.rglob("*.md")
                          if "__pycache__" not in f.parts and ".tmp" not in str(f))
    w_old, w_new = words(old), words(new)
    print(f"skill text: {w_old} -> {w_new} words ({w_new - w_old:+d})")
    if w_new > 10000:
        hits.append(("SKILL", "size", f"{w_new} words", "over the 10,000-word limit -- consolidate before adding"))
    elif w_new - w_old > max(250, 0.10 * w_old):
        hits.append(("SKILL", "growth", f"{w_new - w_old:+d} words", "more than 10% / 250 words in one revision -- rewrite or merge existing lines instead"))
    for h in hits:
        print(f"{h[0]}: {h[1]} '{h[2]}' in: {h[3][:160]}")
    print(f"opt_guard: {len(hits)} hit(s)" + ("" if hits else " -- clean"))
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
