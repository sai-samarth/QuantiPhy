#!/usr/bin/env python
"""Build the Track A submission (sub21g): per question, the geometric mean of two independent GPT agent runs on the
question's own video (skill v20-full and v20-lean), written into the portal's submission template
(data/portal_submission_template.csv; the sessions read their questions from submission_template.csv).

    # from the sessions' answers.json files
    scripts/build_tracka.py --full-tasks RUN/full/*/* --lean-tasks RUN/lean/*/* --unscored answers/unscored_rows.csv \
        --out sub21g.csv
    # or from the answer tables in the release package
    scripts/build_tracka.py --full answers/answers_full.csv --lean answers/answers_lean.csv \
        --unscored answers/unscored_rows.csv --out sub21g.csv

Rows that both runs answered get exp((ln full + ln lean) / 2). The only other rows allowed are the ones listed in
--unscored (the validation-overlap videos the board does not score, which this run did not cover); anything else
missing, or an unscored row that the runs also answered, is an error.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def from_tasks(dirs):
    out = {}
    for d in dirs:
        p = Path(d) / "answers.json"
        if not p.exists():
            continue
        for k, v in json.load(open(p)).items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v > 0:
                out[int(k)] = v
    return pd.Series(out, dtype=float)


def from_table(path):
    t = pd.read_csv(path, float_precision="round_trip")
    return t.set_index("id").value.astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", help="answers_full.csv (id, video_id, value)")
    ap.add_argument("--lean", help="answers_lean.csv (id, video_id, value)")
    ap.add_argument("--full-tasks", nargs="*", default=[], help="task folders of the v20-full run")
    ap.add_argument("--lean-tasks", nargs="*", default=[], help="task folders of the v20-lean run")
    ap.add_argument("--unscored", required=True, help="unscored_rows.csv (id, video_id, parsed_value)")
    ap.add_argument("--template", default=str(REPO / "data/portal_submission_template.csv"),
                    help="the portal's submission template (the submitted file was written into this copy)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    F = from_table(a.full) if a.full else from_tasks(a.full_tasks)
    L = from_table(a.lean) if a.lean else from_tasks(a.lean_tasks)
    both = F.index.intersection(L.index)
    gm = np.exp((np.log(F[both]) + np.log(L[both])) / 2)

    sub = pd.read_csv(a.template)
    unscored = pd.read_csv(a.unscored, float_precision="round_trip").set_index("id").parsed_value
    if set(unscored.index) & set(both):
        raise SystemExit("unscored rows overlap rows answered by both runs")
    vals = pd.concat([gm, unscored])
    extra = sorted(set(vals.index) - set(sub.id))
    if extra:
        raise SystemExit(f"{len(extra)} answered ids are not in the template, e.g. {extra[:10]}")
    missing = sorted(set(sub.id) - set(vals.index))
    if missing:
        raise SystemExit(f"{len(missing)} rows without an answer, e.g. {missing[:10]}")
    sub["parsed_value"] = vals.reindex(sub.id).values
    if not (sub.parsed_value.notna().all() and (sub.parsed_value > 0).all()):
        raise SystemExit("blank or non-positive values")
    sub.to_csv(a.out, index=False)
    print(f"{len(both)} rows from the two runs (full {len(F)}, lean {len(L)}), {len(unscored)} unscored rows; wrote {a.out}")


if __name__ == "__main__":
    main()
