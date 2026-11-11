#!/usr/bin/env python
"""Audit Codex task sessions for access outside their task directory.

    scripts/codex_audit.py <task_dir> [<task_dir> ...]

A session is VOID if any shell command it ran, or any file it left in its task
directory (scripts it wrote and then executed), mentions the network or an
absolute path outside its sandbox. Validation answers exist on Hugging Face and
the agent's commands can reach the network, so this check is what makes a
validation score from these sessions mean anything.
"""
import json, re, sys
from pathlib import Path

NET = re.compile(r"\b(?:(?:curl|wget|aria2c|httpx?|urllib|huggingface|hf_hub)\b"   # whole words ("curled" is not curl)
                 r"|requests\.(?:get|post|put|head|delete|patch|request|Session)\b|import requests|socket\.(?:socket|create_connection|getaddrinfo|gethostbyname|AF_)|import socket|datasets\.load|pip3? install|uv pip|git clone)|https?://"
                 # a dict named requests ("requests.items()") voided a clean GPT session (g2)
                 # netcat only in command position ("nc -l", "| nc host 80"), not a Python
                 # variable named nc ("nc = Rinv @ ...", which voided two Qwen sessions)
                 # nc / ssh / scp / rsync only in command position ("ssh host", "| nc host 80"), not
                 # Python variables ("nc = Rinv @ ...", "uv, S, ssh = project(c)")
                 r"|(?:^|[;&|\n]|\$\()\s*(?:nc|ssh|scp|rsync)\s+(?![=+\-*/]?=)", re.I)
# absolute paths the sandbox legitimately exposes
OK_ROOTS = ("/work", "/tmp", "/opt/venv", "/usr", "/dev/null", "/dev/stdout", "/dev/stderr",
            "/proc/self", "/bin/", "/etc/os-release",
            "/AGENTS.md")   # Codex's own startup lookup for instruction files; absent in the container
PATH = re.compile(r"(?<![\w.~/)\]-])(/[A-Za-z][\w.\-/]*)")   # not after ) or ]: Python division (root[f]/root[g])
# Only real host locations count. The first full run flagged Python divisions
# ('x /np.dot(..)', '/np.pi', '/w.sum') as paths and voided 9 clean sessions.
HOST_ROOTS = ("/mnt", "/home", "/root", "/var", "/srv", "/data", "/media", "/opt", "/run", "/boot")
HOST_OK = ("/opt/venv", "/opt/skill", "/opt/qpt")   # our read-only skill + the SAM tool-server socket (scripts/pi_run.sh)
# the uv-managed Python the venv links to (mounted read-only at its own path), and Codex's skill folder
USER_OK = re.compile(r"/home/[^/]+/\.(?:local/share/uv|codex/skills)(?:/|$)")
CSV_OUT = re.compile(r"(?<![\w.])/(?!work/|opt/venv/|tmp/)[\w./-]+\.csv\b")   # container /tmp is its own (no host mount)
SCAN = re.compile(r"\b(find|ls|du|tree|locate|grep\s+-\w*r\w*)\s+(-\S+\s+)*/(\s|$|\*)|(?<![\w.])~/|\$HOME|os\.walk\(\s*['\"]/['\"]")
# Allow-list of event item types. Probe 2 showed a web search arriving as an
# 'mcp_tool_call' (codex_apps search_service.web_run) that a 'web_search' check
# missed, so anything that is not plain shell / file editing / talking voids.
ALLOWED = {"agent_message", "reasoning", "command_execution", "file_change", "todo_list", "error"}
# built-in MCP listing calls; with apps off and the user config ignored there are
# no MCP servers, so these return nothing (probe 3) -- any other MCP call voids
HARMLESS_MCP = {"list_mcp_resources", "list_mcp_resource_templates"}
WORDS = re.compile(r"quantiphy|ground_truth|validation_dataset|known_test_answers", re.I)
# 'answer key' alone is only a NOTE: agents legitimately write it about their own
# answers.json (a DeepSeek session printed "answer keys :", list(a.keys())). The real
# guards are the path and network checks; a note gets read by a reviewer.
WEAK_WORDS = re.compile(r"answer.?key", re.I)
# in command OUTPUT, 'answer key' is excluded: every session prints GUIDE.md,
# which itself says not to look for answer keys
OUT_WORDS = re.compile(r"quantiphy|ground_truth|validation_dataset|known_test_answers", re.I)


def flags_in(text: str) -> list:
    out = []
    if NET.search(text):
        out.append(f"network: {NET.search(text).group(0)!r}")
    for p in PATH.findall(text):
        # a host path must start with a whole root component ("/opt/x", not "/opt.scale" -- a
        # Python division like "px/opt.scale.iloc[0]" voided a clean session)
        if any(p == r or p.startswith(r + "/") for r in HOST_ROOTS) and not (p.startswith(HOST_OK) or (USER_OK.match(p) and "/.." not in p)):
            out.append(f"path outside sandbox: {p}")
    if WORDS.search(text):
        out.append(f"answer-key word: {WORDS.search(text).group(0)!r}")
    return out


PI_TOOLS = {"read", "bash", "edit", "write", "grep", "find", "ls"}


def audit_pi(task: Path) -> tuple[bool, list]:
    """Same rules for a Pi session (scripts/pi_run.sh): built-in tools only; every
    bash command and file path checked; every tool result scanned."""
    import glob
    reasons, notes, n = [], [], 0
    for f in sorted(glob.glob(str(task / "pi_session" / "*.jsonl"))):
        for line in open(f, errors="replace"):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            m = e.get("message") or {}
            if m.get("role") == "assistant":
                for part in m.get("content") or []:
                    if part.get("type") != "toolCall":
                        continue
                    n += 1
                    name, args = part.get("name"), part.get("arguments") or {}
                    if name not in PI_TOOLS:
                        reasons.append(f"tool outside allow-list: {name}")
                    text = " ".join(str(v) for v in args.values()) if isinstance(args, dict) else str(args)
                    reasons += [f"{name}: {r}" for r in flags_in(text)]
                    if WEAK_WORDS.search(text):
                        notes.append(f"{name}: mentions {WEAK_WORDS.search(text).group(0)!r} (check context)")
                    if name == "bash" and SCAN.search(text):
                        notes.append(f"scan (harmless unless its output is flagged): {SCAN.search(text).group(0)!r}")
            elif m.get("role") == "toolResult":
                o = " ".join(str(c.get("text", "")) for c in (m.get("content") or []) if isinstance(c, dict))
                if OUT_WORDS.search(o):
                    reasons.append(f"output mentions {OUT_WORDS.search(o).group(0)!r}")
                if CSV_OUT.search(o):
                    reasons.append(f"output lists a csv outside the sandbox: {CSV_OUT.search(o).group(0)}")
    for f in task.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".sh") and f.stat().st_size < 2_000_000:
            reasons += [f"{f.name}: {r}" for r in flags_in(f.read_text(errors="replace"))]
    reasons = sorted(set(reasons))
    return (not reasons), [f"{n} tool calls (pi)"] + reasons + sorted(set(notes))


def audit(task: Path) -> tuple[bool, list]:
    if (task / "pi_session").exists() and not (task / "codex.jsonl").exists():
        return audit_pi(task)
    reasons, notes, n_cmd = [], [], 0
    log = task / "codex.jsonl"
    if not log.exists():
        return False, ["no codex.jsonl"]
    for line in log.read_text(errors="replace").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        it = d.get("item") or {}
        if it.get("type") == "mcp_tool_call" and it.get("tool") in HARMLESS_MCP:
            continue
        if it.get("type") and it["type"] not in ALLOWED:
            reasons.append(f"tool outside allow-list: {it['type']} "
                           f"{str(it.get('tool') or it.get('query') or '')[:60]}")
        cmd = it.get("command")
        if cmd is not None and d.get("type") == "item.started":
            n_cmd += 1
            c = cmd if isinstance(cmd, str) else " ".join(cmd)
            reasons += [f"cmd: {r}" for r in flags_in(c)]
            if WEAK_WORDS.search(c):
                notes.append(f"cmd mentions {WEAK_WORDS.search(c).group(0)!r} (check context)")
            if SCAN.search(c):
                notes.append(f"scan (harmless unless its output is flagged): {SCAN.search(c).group(0)!r}")
        if d.get("type") == "item.completed" and it.get("aggregated_output"):
            o = str(it["aggregated_output"])
            if OUT_WORDS.search(o):
                reasons.append(f"output mentions {OUT_WORDS.search(o).group(0)!r}")
            if CSV_OUT.search(o):
                reasons.append(f"output lists a csv outside the sandbox: {CSV_OUT.search(o).group(0)}")
        for ch in it.get("changes") or []:
            p = str(ch.get("path", ""))
            if p and not p.startswith(("/work", str(task))):
                reasons.append(f"file change outside: {p}")
    for f in task.rglob("*"):
        if f.is_file() and f.suffix in (".py", ".sh") and f.stat().st_size < 2_000_000:
            reasons += [f"{f.name}: {r}" for r in flags_in(f.read_text(errors="replace"))]
    reasons = sorted(set(reasons))
    return (not reasons), [f"{n_cmd} commands"] + reasons + sorted(set(notes))


if __name__ == "__main__":
    bad = 0
    for t in sys.argv[1:]:
        ok, why = audit(Path(t))
        bad += not ok
        print(f"{'OK  ' if ok else 'VOID'} {t}: " + "; ".join(why[:6]))
    sys.exit(1 if bad else 0)
