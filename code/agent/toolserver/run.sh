#!/usr/bin/env bash
# Start / stop / check the SAM 3 tool server that qpt seg/segtrack talk to.
#
#   agent/toolserver/run.sh start <gpu> [sockdir]   # e.g. start 0
#   agent/toolserver/run.sh stop  [sockdir]
#   agent/toolserver/run.sh status [sockdir]
#
# sockdir (default $QP_WORK/toolsrv, i.e. code/.work/toolsrv) holds qpt.sock, the
# server's pid file and log; pass the same directory to Pi as PI_TOOLSRV.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")"); REPO=$(readlink -f "$HERE/../..")
cmd=${1:-status}
if [ "$cmd" = start ]; then GPU=${2:?gpu index}; DIR=${3:-${QP_WORK:-$REPO/.work}/toolsrv}
else DIR=${2:-${QP_WORK:-$REPO/.work}/toolsrv}; fi
mkdir -p "$DIR"
case $cmd in
  start)
    if [ -f "$DIR/server.pid" ] && kill -0 "$(cat "$DIR/server.pid")" 2>/dev/null; then
      echo "already running (pid $(cat "$DIR/server.pid"))"; exit 0; fi
    source "$REPO/env.sh"
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU nohup "$REPO/.venv-vllm/bin/python" "$HERE/sam_server.py" \
      --sock "$DIR/qpt.sock" --preload > "$DIR/server.log" 2>&1 &
    echo $! > "$DIR/server.pid"
    for _ in $(seq 120); do grep -q "tool server on" "$DIR/server.log" && break; kill -0 "$(cat "$DIR/server.pid")" 2>/dev/null || break; sleep 1; done
    tail -2 "$DIR/server.log" ;;
  stop)
    [ -f "$DIR/server.pid" ] && kill "$(cat "$DIR/server.pid")" 2>/dev/null && echo stopped; rm -f "$DIR/server.pid" "$DIR/qpt.sock" ;;
  status)
    if [ -f "$DIR/server.pid" ] && kill -0 "$(cat "$DIR/server.pid")" 2>/dev/null; then
      echo "running (pid $(cat "$DIR/server.pid"))"
      curl -s --unix-socket "$DIR/qpt.sock" http://localhost/health; echo
    else echo "not running"; fi ;;
esac
