#!/bin/sh
# Container entrypoint for the network-less agent sandbox (docker --network none).
# Every unix socket mounted in /opt/net becomes a loopback TCP port inside the container:
#   llm-<port>.sock -> 127.0.0.1:<port>   (a model server on the host, e.g. vLLM)
#   proxy.sock      -> 127.0.0.1:3128     (allow-list HTTPS proxy for model APIs)
# Then the real command runs. There is no other way out of the container.
for s in /opt/net/*.sock; do
  [ -S "$s" ] || continue
  case "$(basename "$s")" in
    llm-*.sock) p=$(basename "$s" .sock); p=${p#llm-} ;;
    proxy.sock) p=3128 ;;
    *) continue ;;
  esac
  socat TCP-LISTEN:"$p",bind=127.0.0.1,fork,reuseaddr UNIX-CONNECT:"$s" 2>/dev/null &
done
sleep 0.3
exec "$@"
