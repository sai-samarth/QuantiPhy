#!/usr/bin/env bash
# Host side of the network-less sandbox: unix sockets the agent containers can use.
#
#   agent/sandbox/netrelay.sh start [ports...]   # default 8010 8011 8012 8013
#   agent/sandbox/netrelay.sh stop | status
#
# For each port: llm-<port>.sock -> a local vLLM server at $LLM_HOST:<port> (default 127.0.0.1).
# proxy.sock: allow-list HTTPS proxy (agent/sandbox/allow_proxy.py) for model APIs; its
# log (proxy.log) records every connection attempt, allowed or denied.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(readlink -f "$HERE/../..")
DIR=$(readlink -m "${NETSOCK_DIR:-${QP_WORK:-$REPO/.work}/netsock}")
GW=${LLM_HOST:-127.0.0.1}
cmd=${1:-status}; shift || true
mkdir -p "$DIR"
case $cmd in
  start)
    ports=${*:-8010 8011 8012 8013}
    for p in $ports; do
      rm -f "$DIR/llm-$p.sock"
      nohup socat UNIX-LISTEN:"$DIR/llm-$p.sock",fork,mode=666 TCP:"$GW":"$p" > /dev/null 2>&1 &
      echo $! > "$DIR/llm-$p.pid"
    done
    nohup python3 "$HERE/allow_proxy.py" --sock "$DIR/proxy.sock" --log "$DIR/proxy.log" > "$DIR/proxy.out" 2>&1 &
    echo $! > "$DIR/proxy.pid"
    sleep 1; ls -la "$DIR"/*.sock ;;
  stop)
    for f in "$DIR"/*.pid; do [ -f "$f" ] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
    rm -f "$DIR"/*.sock; echo stopped ;;
  status)
    for f in "$DIR"/*.pid; do [ -f "$f" ] || continue; n=$(basename "$f" .pid)
      if kill -0 "$(cat "$f")" 2>/dev/null; then echo "$n running"; else echo "$n DEAD"; fi; done ;;
esac
