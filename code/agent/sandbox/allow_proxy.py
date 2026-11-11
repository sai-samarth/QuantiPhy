#!/usr/bin/env python
"""Allow-list HTTPS proxy for the agent sandbox, listening on a unix socket.

The agent container has NO network (docker --network none). Its only way out is this
proxy, reached through a mounted unix socket (agent/sandbox/entry.sh relays it to
127.0.0.1:3128 inside the container; Pi uses HTTPS_PROXY). Only CONNECT to the model
API hosts below is allowed; everything else gets 403. Every request is logged with its
verdict, so each run leaves a network record for the audit.

    allow_proxy.py --sock /path/proxy.sock --log /path/proxy.log [--allow host ...]
"""
import argparse, asyncio, os, time

DEFAULT_ALLOW = {
    "openrouter.ai",                                   # OpenRouter models (Track B API)
    "chatgpt.com", "auth.openai.com", "api.openai.com",  # GPT via ChatGPT login (Track A)
}


async def pipe(r, w):
    try:
        while True:
            b = await r.read(65536)
            if not b:
                break
            w.write(b)
            await w.drain()
    except (ConnectionError, asyncio.CancelledError):
        pass
    finally:
        try:
            w.close()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True); ap.add_argument("--log", required=True)
    ap.add_argument("--allow", nargs="*", default=sorted(DEFAULT_ALLOW))
    a = ap.parse_args()
    allow = set(a.allow)
    log = open(a.log, "a", buffering=1)

    async def handle(cr, cw):
        line = (await cr.readline()).decode(errors="replace").strip()
        while (await cr.readline()) not in (b"\r\n", b"\n", b""):
            pass                                        # drop the rest of the request header
        parts = line.split()
        host, port = None, None
        if len(parts) >= 2 and parts[0].upper() == "CONNECT" and ":" in parts[1]:
            host, port = parts[1].rsplit(":", 1)
        ok = host in allow and port == "443"
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {'ALLOW' if ok else 'DENY '} {line[:200]}\n")
        if not ok:
            cw.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n"); await cw.drain(); cw.close(); return
        try:
            ur, uw = await asyncio.wait_for(asyncio.open_connection(host, 443), 20)
        except Exception as e:
            log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} FAIL  {host}: {e}\n")
            cw.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n"); await cw.drain(); cw.close(); return
        cw.write(b"HTTP/1.1 200 Connection established\r\n\r\n"); await cw.drain()
        await asyncio.gather(pipe(cr, uw), pipe(ur, cw))

    async def run():
        if os.path.exists(a.sock):
            os.unlink(a.sock)
        srv = await asyncio.start_unix_server(handle, path=a.sock)
        os.chmod(a.sock, 0o666)
        log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} proxy up, allow={sorted(allow)}\n")
        async with srv:
            await srv.serve_forever()

    asyncio.run(run())


if __name__ == "__main__":
    main()
