#!/usr/bin/env bash
# Run one Pi session (https://pi.dev, @earendil-works/pi-coding-agent) on one task
# directory, inside the same sealed container as scripts/codex_run.sh.
#
#   scripts/pi_run.sh <task_dir> <prompt_file> [thinking=max] [model=openai/gpt-6.1-sol]
#
# Pi is installed privately in $PI_HOME and mounted read-only with Node; the Pi
# login folder $PI_AGENT_DIR (ChatGPT login via /login) is mounted for auth.
# MCP, extensions, skills, prompt templates and AGENTS.md discovery are off, and
# startup network calls (update checks, telemetry) are disabled, leaving Pi's
# built-in tools (read, bash, edit, write). The full session -- every message,
# tool call, per-call token usage and cost, and any compaction -- is saved to
# <task_dir>/pi_session/ for scripts/agent_stats.py and scripts/codex_audit.py;
# the JSON event stream goes to <task_dir>/pi_events.jsonl.
#
# Optional, by environment:
#   PI_SKILL=<skill dir>   frozen into skill_snapshots/<content hash>/<name>, mounted read-only at
#                          /opt/skill/<name> and loaded with --skill (other skill discovery stays
#                          off); its bin/ goes on PATH; <task>/skill_version.txt records the hash
#   PI_NET=none (default)  the container has NO network: model servers and the allow-list
#                          API proxy come in as unix sockets (agent/sandbox/netrelay.sh on the
#                          host, agent/sandbox/entry.sh relays them to 127.0.0.1 inside);
#                          PI_NET=open is the old shared docker network (internet reachable)
#   PI_RESCUE=0            do not resume a session that ended without answers.json
#   PI_RESCUE_ONLY=1       only the rescue step, for a session that already ran
#   PI_TOOLSRV=<dir>       directory holding the SAM tool server's socket (qpt.sock,
#                          agent/toolserver/sam_server.py), mounted read-only at /opt/qpt
#
# Machine locations, by environment [default]:
#   QP_WORK          working folder [code/.work]
#   VENV             the qpt tools' Python virtual environment [code/.venv]
#   PI_HOME          the Pi install, the folder where `npm ci` was run [$QP_WORK/pi]
#   PI_AGENT_DIR     Pi's login and settings [~/.pi/agent]
#   UV_PYTHON_DIR    uv-managed Pythons the venv links to, mounted at the same path
#                    [uv's own: $UV_PYTHON_INSTALL_DIR, else ~/.local/share/uv/python]
#   NODE_BIN         Node.js [node on PATH]
#   NETSOCK_DIR      relay sockets from agent/sandbox/netrelay.sh [$QP_WORK/netsock]
#   SKILL_SNAPSHOTS  frozen skill copies [$QP_WORK/skill_snapshots]
set -euo pipefail
TASK=$(readlink -f "$1"); PROMPT=$(readlink -f "$2")
THINK=${3:-max}; MODEL=${4:-openai/gpt-6.1-sol}
REPO=$(readlink -f "$(dirname "$(readlink -f "$0")")/..")
WORK=$(readlink -m "${QP_WORK:-$REPO/.work}")
VENV=$(readlink -m "${VENV:-$REPO/.venv}")
NODE=$(command -v "${NODE_BIN:-node}") || { echo "pi_run: node not found (set NODE_BIN)" >&2; exit 2; }
NODE_HOME=$(readlink -f "$NODE" | xargs dirname | xargs dirname)   # needs a real Node binary, not a version-manager shim
PI_HOME=$(readlink -m "${PI_HOME:-$WORK/pi}")
PI_AGENT_DIR=$(readlink -m "${PI_AGENT_DIR:-$HOME/.pi/agent}")
UV_PY=$(realpath -ms "${UV_PYTHON_DIR:-${UV_PYTHON_INSTALL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/python}}")
# docker -v would create a missing folder as root: check the set-up first
for d in "$VENV" "$PI_HOME" "$PI_AGENT_DIR"; do
  [ -d "$d" ] || { echo "pi_run: missing $d (see README, Setup)" >&2; exit 2; }
done
CHOME=/home/agent   # HOME inside the container
HERE_SANDBOX=$REPO/agent/sandbox
# Several pools may share a task root: never start a second session in a directory that is answered or busy.
if [ -z "${PI_RESCUE_ONLY:-}" ] && [ "${PI_FORCE:-0}" != 1 ]; then
  if [ -s "$TASK/answers.json" ]; then echo "pi_run: already answered, skipped $TASK"; exit 0; fi
  OTHER=$(python3 - "$TASK" "$$" <<'PY'
import os, sys
task, me = os.path.realpath(sys.argv[1]), int(sys.argv[2])
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        args = open(f"/proc/{pid}/cmdline", "rb").read().decode(errors="ignore").split("\0")
        ppid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
    except Exception:
        continue
    if int(pid) == me or ppid == me:              # this script and its own subshells
        continue
    for i, x in enumerate(args[:-1]):
        if x.endswith("pi_run.sh") and os.path.realpath(args[i + 1]) == task:
            print(pid); sys.exit(0)
PY
)
  if [ -n "$OTHER" ]; then echo "pi_run: busy in another pool (pid $OTHER), skipped $TASK"; exit 0; fi
fi
mkdir -p "$TASK/pi_session"
# OpenRouter models: the key comes from $OPENROUTER_API_KEY; it reaches the container
# by name (-e OPENROUTER_API_KEY), never on a command line or in a log.
KEYENV=()
if [[ "$MODEL" == openrouter/* ]]; then
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "pi_run: set OPENROUTER_API_KEY" >&2; exit 2; }
  export OPENROUTER_API_KEY
  KEYENV=(-e OPENROUTER_API_KEY)
fi
EXTRA=(); PIARGS=(); XPATH=""
if [ -n "${PI_SKILL:-}" ]; then
  SK=$(readlink -f "$PI_SKILL"); SN=$(basename "$SK")
  # Freeze the skill: the session mounts a content-addressed snapshot, so edits to
  # the working copy never leak into a running session, and the hash names the version.
  HASH=$(cd "$SK" && find . -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-12)
  SNAP=$(readlink -m "${SKILL_SNAPSHOTS:-$WORK/skill_snapshots}")/$HASH/$SN
  if [ ! -d "$SNAP" ]; then
    # several sessions can start at once: rename atomically and never INTO an existing
    # snapshot (mv -T fails if another session won the race; then drop our copy)
    mkdir -p "$(dirname "$SNAP")" && cp -r "$SK" "$SNAP.tmp$$" && find "$SNAP.tmp$$" -name __pycache__ -prune -exec rm -rf {} + ; mv -T "$SNAP.tmp$$" "$SNAP" 2>/dev/null || rm -rf "$SNAP.tmp$$"
  fi
  echo "$SN $HASH" > "$TASK/skill_version.txt"
  EXTRA+=(-v "$SNAP":/opt/skill/"$SN":ro); PIARGS+=(--skill /opt/skill/"$SN"); XPATH=/opt/skill/"$SN"/bin:
fi
if [ -n "${PI_TOOLSRV:-}" ]; then
  EXTRA+=(-v "$(readlink -f "$PI_TOOLSRV")":/opt/qpt:ro -e QPT_SOCK=/opt/qpt/qpt.sock)
fi
# a uv-made venv links to its Python by absolute path: mount the uv Pythons at the same path
if [ -d "$UV_PY" ]; then
  EXTRA+=(-v "$UV_PY":"$UV_PY":ro)
fi
# one Pi run in the sealed container; $1 = log suffix, $2 = extra docker args (array name), rest = Pi args
pi_docker() {
  local sfx=$1; local -n more=$2; shift 2
  docker run --rm --user "$(id -u):$(id -g)" "${NETARGS[@]}" \
    -v /usr:/usr:ro -v /etc/ssl/certs:/etc/ssl/certs:ro \
    -v /etc/alternatives:/etc/alternatives:ro -v /etc/ld.so.cache:/etc/ld.so.cache:ro \
    -v "$VENV":/opt/venv:ro -v "$NODE_HOME":/opt/node:ro -v "$PI_HOME":/opt/pi:ro \
    -v "$PI_AGENT_DIR":$CHOME/.pi/agent "${more[@]}" \
    -v "$TASK":/work -w /work "${EXTRA[@]}" \
    -e HOME=$CHOME -e PATH=${XPATH}/opt/venv/bin:/opt/node/bin:/usr/bin:/bin \
    -e SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt -e NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt \
    -e PI_TELEMETRY=0 -e PI_SKIP_VERSION_CHECK=1 -e MPLCONFIGDIR=/tmp/mpl -e XDG_CACHE_HOME=/tmp/cache "${KEYENV[@]}" \
    ubuntu:22.04 \
    "${ENTRY[@]}" /opt/node/bin/node /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js \
      --mode json --model "$MODEL" --thinking "$THINK" \
      --no-mcp --no-extensions --no-skills --no-prompt-templates --no-context-files --offline "${PIARGS[@]}" \
      --session-dir /work/pi_session "$@" \
    < /dev/null > "$TASK/pi_events$sfx.jsonl" 2> "$TASK/pi$sfx.stderr"   # stdin closed: Pi waits on an open stdin
}
# network: sealed by default
NETSOCK=$(readlink -m "${NETSOCK_DIR:-$WORK/netsock}")
if [ "${PI_NET:-none}" = open ]; then
  NETARGS=(--network qp-codex); ENTRY=()
else
  ls "$NETSOCK"/*.sock >/dev/null 2>&1 || { echo "pi_run: no sockets in $NETSOCK (agent/sandbox/netrelay.sh start)" >&2; exit 3; }
  NETARGS=(--network none -v "$NETSOCK":/opt/net:ro -v "$HERE_SANDBOX":/opt/sandbox:ro
           -e HTTPS_PROXY=http://127.0.0.1:3128 -e HTTP_PROXY=http://127.0.0.1:3128
           -e NO_PROXY=127.0.0.1,localhost -e no_proxy=127.0.0.1,localhost)
  ENTRY=(/opt/sandbox/entry.sh)
fi
rc=0; NONE=()
# PI_RESCUE_ONLY=1: skip the main run and only rescue an existing session
[ "${PI_RESCUE_ONLY:-0}" = 1 ] || pi_docker "" NONE "$(cat "$PROMPT")" || rc=$?
# Rescue: a session that ended without answers.json (provider error, OpenRouter's 30 MB
# image cap) is resumed once with images blocked -- a settings file mounted over Pi's
# for this container only -- and asked to write its best answers from the work done.
if [ ! -s "$TASK/answers.json" ] && ls "$TASK"/pi_session/*.jsonl >/dev/null 2>&1 && [ "${PI_RESCUE:-1}" = 1 ]; then
  RS=$(mktemp); python3 -c "import json,sys; d=json.load(open(sys.argv[2])); d.setdefault('images',{})['blockImages']=True; json.dump(d,open(sys.argv[1],'w'))" "$RS" "$PI_AGENT_DIR/settings.json"
  chmod 644 "$RS"; RESC=(-v "$RS":$CHOME/.pi/agent/settings.json:ro)
  echo "pi_run: no answers.json -- rescuing (resume, images off) $TASK"
  pi_docker "_rescue" RESC --continue "Your previous step failed with a technical error from the model provider (not your fault). Images are now switched off. Using the measurements you have already made (your files, scripts and notes in this directory), write answers.json now with your best estimate for every question id, and notes.md, as the instructions specify. Re-run quick computations if needed, but do not start over." || rc=$?
  rm -f "$RS"
fi
# Completeness: every question id needs a positive number. A session that left ids
# missing, null, zero or non-numeric (one answered null when its ruler was masked out) is
# resumed once and told exactly which ids to fix.
bad_ids() {
  python3 - "$TASK" <<'PYEOF'
import json, sys
t = sys.argv[1]
try:
    ids = [str(q["id"]) for q in json.load(open(f"{t}/questions.json")).get("questions", [])]
    a = json.load(open(f"{t}/answers.json"))
except Exception:
    print("ALL"); sys.exit()
bad = []
for i in ids:
    try:
        ok = float(a.get(i)) > 0
    except (TypeError, ValueError):
        ok = False
    if not ok:
        bad.append(i)
print(" ".join(bad))
PYEOF
}
if [ -s "$TASK/answers.json" ] && [ "${PI_RESCUE:-1}" = 1 ] && [ -n "$(bad_ids)" ]; then
  B=$(bad_ids)
  echo "pi_run: answers.json incomplete ($B) -- asking the session to finish $TASK"
  pi_docker "_complete" NONE --continue "answers.json is incomplete: question id(s) $B have no positive number. Every question must get your best positive estimate (an answer always scores better than none). If the prior's object is not visible in this clip, take the scale from the best available evidence (another object of standard, well-known size, or the clip's own geometry) and say so in notes.md. Update answers.json and notes.md now." || rc=$?
fi
if [ -s "$TASK/answers.json" ]; then
  [ -n "$(bad_ids)" ] && echo "pi_run: WARNING still incomplete ($(bad_ids)) $TASK"
  echo "pi_run: DONE (exit $rc) $TASK"
elif ls "$TASK"/pi_session/*.jsonl >/dev/null 2>&1 && grep -q '"role":"assistant"' "$TASK"/pi_session/*.jsonl; then
  echo "pi_run: NO ANSWERS (exit $rc) $TASK"
else
  echo "pi_run: FAILED (exit $rc) $TASK -- $(tail -c 300 "$TASK/pi.stderr")"
fi
exit $rc
