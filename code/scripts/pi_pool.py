#!/usr/bin/env python
"""Run Pi on every task directory under <root>, spread over several model servers.

    scripts/pi_pool.py --root DIR --prompt FILE --models local/qwen38-27b:4,local2/qwen38-27b:4 \
        [--thinking max] [--skill DIR] [--toolsrv DIR] [--order hard-first]

Each `model:slots` entry is one server (one Pi provider) with that many concurrent
sessions. A task directory is skipped when it already has answers.json or a pi_run.sh
process is already working on it (e.g. a session left running by an earlier launcher),
so the pool can be restarted at any time. Each session is scripts/pi_run.sh (sealed
container, audit trail, one rescue). Prints one line per start / finish.
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def busy_dirs():
    """Task dirs some pi_run.sh process is already working on (the argument after the script)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    busy = set()
    for line in out.splitlines():
        tok = line.split()
        for j, t in enumerate(tok[:-1]):
            if t.endswith("pi_run.sh"):
                busy.add(str(Path(tok[j + 1]).resolve()))
    return busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--prompt", required=True)
    ap.add_argument("--models", required=True, help="provider/model:slots,...")
    ap.add_argument("--thinking", default="max")
    ap.add_argument("--skill"); ap.add_argument("--toolsrv")
    ap.add_argument("--order", default="name", choices=["name", "hard-first"],
                    help="hard-first: longest videos first, so the slowest sessions do not finish last")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    slots = []
    for spec in a.models.split(","):
        m, n = spec.rsplit(":", 1)
        slots += [m] * int(n)
    env = dict(os.environ)
    if a.skill:
        env["PI_SKILL"] = a.skill
    if a.toolsrv:
        env["PI_TOOLSRV"] = a.toolsrv
    busy = busy_dirs()
    todo = [d for d in sorted(root.iterdir()) if d.is_dir() and not (d / "answers.json").exists()
            and str(d) not in busy]
    if a.order == "hard-first":
        todo.sort(key=lambda d: -(d / "video.mp4").stat().st_size if (d / "video.mp4").exists() else 0)
    print(f"pi_pool: {len(todo)} task(s) to run ({len(busy)} already running elsewhere), "
          f"{len(slots)} slots over {len(set(slots))} server(s)", flush=True)
    running = {}                                   # slot index -> (Popen, task)
    t0 = time.time()
    while todo or running:
        for i, m in enumerate(slots):
            if i not in running and todo:
                d = todo.pop(0)
                p = subprocess.Popen([str(HERE / "pi_run.sh"), str(d), a.prompt, a.thinking, m], env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                running[i] = (p, d, time.time())
                print(f"[{(time.time() - t0) / 60:6.1f} min] start  {d.name} on {m}", flush=True)
        for i, (p, d, ts) in list(running.items()):
            if p.poll() is not None:
                last = (p.stdout.read() or "").strip().splitlines()
                print(f"[{(time.time() - t0) / 60:6.1f} min] finish {d.name} ({(time.time() - ts) / 60:.0f} min): "
                      f"{last[-1] if last else ''}", flush=True)
                del running[i]
        time.sleep(10)
    print(f"pi_pool: all done in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
