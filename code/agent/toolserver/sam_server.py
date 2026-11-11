#!/usr/bin/env python
"""SAM 3 tool server for the sealed agent container: HTTP over a unix socket.

    CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python agent/toolserver/sam_server.py --sock .work/toolsrv/qpt.sock

The agent container gets the socket file mounted (scripts/pi_run.sh) and talks to
it through `qpt seg` / `qpt segtrack`. The server never opens host files: every
request carries its own frame (PNG) or video bytes, so the sandbox boundary is the
same as without it. Requests are served one at a time (one GPU).

  POST /seg    {image, text?, box?, points?, labels?, threshold}
               text only -> SAM 3 concept segmentation: every instance above threshold
               box/points -> SAM 3 tracker (promptable): the one object at the prompt
  POST /track  {video, fps, f0, f1, stride, show, text? | frame + box?/points?}
               text -> SAM 3 video (all instances, persistent ids)
               prompt at `frame` -> SAM 3 tracker video, propagated forward and backward
  GET  /health
Masks go back bit-packed (geom.pack_mask); per-frame shape statistics are computed
here with the same geom.mask_stats the client uses.
"""
import argparse, base64, json, os, socketserver, sys, tempfile, threading, time, traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "skill" / "video-physics" / "qpt"))
from geom import mask_stats, pack_mask, largest_component  # noqa: E402

CKPT = os.environ.get("QPT_SAM_CKPT", "facebook/sam3")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MAX_VIDEO_FRAMES = int(os.environ.get("QPT_MAX_VIDEO_FRAMES", 240))
_M = {}
LOCK = threading.Lock()


def model(kind):
    if kind not in _M:
        import transformers as T
        t0 = time.time()
        if kind == "image":
            _M[kind] = (T.Sam3Processor.from_pretrained(CKPT), T.Sam3Model.from_pretrained(CKPT, dtype=DT).to(DEV).eval())
        elif kind == "tracker":
            _M[kind] = (T.Sam3TrackerProcessor.from_pretrained(CKPT), T.Sam3TrackerModel.from_pretrained(CKPT, dtype=DT).to(DEV).eval())
        elif kind == "video":
            _M[kind] = (T.Sam3VideoProcessor.from_pretrained(CKPT), T.Sam3VideoModel.from_pretrained(CKPT, dtype=DT).to(DEV).eval())
        elif kind == "tracker_video":
            _M[kind] = (T.Sam3TrackerVideoProcessor.from_pretrained(CKPT),
                        T.Sam3TrackerVideoModel.from_pretrained(CKPT, dtype=DT).to(DEV).eval())
        print(f"loaded {kind} in {time.time() - t0:.1f}s", flush=True)
    return _M[kind]


def decode_png(b64):
    return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)


def to_rgb_pil(bgr):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


@torch.inference_mode()
def seg(req):
    img = decode_png(req["image"])
    H, W = img.shape[:2]
    if req.get("box") or req.get("points"):
        P, M = model("tracker")
        kw = {}
        if req.get("box"):
            kw["input_boxes"] = [[list(map(float, req["box"]))]]
        if req.get("points"):
            kw["input_points"] = [[[list(map(float, p)) for p in req["points"]]]]
            kw["input_labels"] = [[[int(l) for l in req["labels"]]]]
        inp = P(images=to_rgb_pil(img), return_tensors="pt", **kw).to(DEV)
        out = M(**inp, multimask_output=False)
        masks = P.post_process_masks(out.pred_masks.float().cpu(), inp["original_sizes"].cpu())[0]
        m = np.asarray(masks[0, 0] > 0.5) if masks.ndim == 4 else np.asarray(masks[0] > 0.5)
        sc = float(out.iou_scores.float().flatten()[0]) if getattr(out, "iou_scores", None) is not None else 1.0
        inst = [{"score": sc, "mask": pack_mask(largest_component(m))}] if m.sum() >= 3 else []
        return {"mode": "prompt (box/points) -> one object", "instances": inst}
    P, M = model("image")
    inp = P(images=to_rgb_pil(img), text=req["text"], return_tensors="pt").to(DEV)
    out = M(**inp)
    r = P.post_process_instance_segmentation(out, threshold=float(req.get("threshold", 0.4)), mask_threshold=0.5,
                                             target_sizes=inp["original_sizes"].tolist())[0]
    inst = []
    for k in np.argsort(-r["scores"].float().cpu().numpy()):
        m = np.asarray(r["masks"][k].cpu(), dtype=bool)
        if m.sum() >= 3:
            inst.append({"score": float(r["scores"][k]), "mask": pack_mask(m)})
    return {"mode": "text -> all instances", "instances": inst}


def read_video(b64, f0, f1, stride):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as fh:
        fh.write(base64.b64decode(b64)); fh.flush()
        cap = cv2.VideoCapture(fh.name); frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    f1 = min(f1, len(frames) - 1)
    keep = list(range(max(0, f0), f1 + 1, max(1, stride)))
    while len(keep) > MAX_VIDEO_FRAMES:
        stride = max(1, stride) * 2
        keep = list(range(max(0, f0), f1 + 1, stride))
    return frames, keep, stride


def row(fi, oid, score, m):
    st = mask_stats(m)
    if st is None:
        return None
    st.pop("feret_a"); st.pop("feret_b")
    return dict(f=int(fi), obj_id=int(oid), score=float(score), **st)


def track(req):
    """Video tracking with an out-of-memory fallback: on OOM, retry on every 2nd / 4th / 8th frame."""
    stride = int(req.get("stride", 1))
    for attempt in range(4):
        try:
            out = _track(dict(req, stride=stride))
            if attempt:
                out["mode"] += f" (GPU memory: fell back to every {stride}th frame)"
            return out
        except torch.OutOfMemoryError:
            if DEV == "cuda":
                torch.cuda.empty_cache()
            stride *= 2
            print(f"  OOM -> retry with stride {stride}", flush=True)
    raise RuntimeError("out of GPU memory even at stride %d; use --every, a shorter --from/--to, or qpt seg per frame" % stride)


@torch.inference_mode()
def _track(req):
    frames, keep, stride = read_video(req["video"], int(req.get("f0", 0)), int(req.get("f1", 10 ** 9)), int(req.get("stride", 1)))
    rgb = [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in keep]
    H, W = frames[0].shape[:2]
    per = {}                       # original frame -> {obj_id: mask}
    scores = {}
    if req.get("text"):
        mode = "text -> SAM 3 video, all instances"
        P, M = model("video")
        sess = P.init_video_session(video=rgb, inference_device=DEV, dtype=DT)
        P.add_text_prompt(sess, req["text"])
        for k, o in enumerate(M.propagate_in_video_iterator(sess)):
            r = P.postprocess_outputs(sess, o)
            fi = keep[getattr(o, "frame_idx", k)] if getattr(o, "frame_idx", k) < len(keep) else keep[min(k, len(keep) - 1)]
            per[fi] = {}
            ms, sc, ids = r.get("masks"), r.get("scores"), r.get("object_ids")
            if ms is None:
                continue
            for j in range(len(ms)):
                m = largest_component(np.asarray(ms[j].squeeze().cpu(), dtype=bool))
                if m.sum() < 3:
                    continue
                oid = int(ids[j]) if ids is not None and len(ids) > j else j
                per[fi][oid] = m; scores[(fi, oid)] = float(sc[j]) if sc is not None and len(sc) > j else float("nan")
        del sess
    else:
        mode = "prompt -> SAM 3 tracker video, one object (id 1)"
        P, M = model("tracker_video")
        fs = int(req["frame"])
        k0 = int(np.argmin([abs(i - fs) for i in keep]))
        sess = P.init_video_session(video=rgb, inference_device=DEV, dtype=DT)
        kw = {}
        if req.get("box"):
            kw["input_boxes"] = [[list(map(float, req["box"]))]]
        if req.get("points"):
            kw["input_points"] = [[[list(map(float, p)) for p in req["points"]]]]
            kw["input_labels"] = [[[int(l) for l in req["labels"]]]]
        P.add_inputs_to_inference_session(inference_session=sess, frame_idx=k0, obj_ids=1, **kw)
        for rev in (False, True):
            for o in M.propagate_in_video_iterator(sess, start_frame_idx=k0, reverse=rev):
                ms = P.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                fi = keep[o.frame_idx]
                m = largest_component(np.asarray(ms[0].squeeze().cpu(), dtype=bool))
                per.setdefault(fi, {})
                if m.sum() >= 3:
                    per[fi][1] = m; scores[(fi, 1)] = float("nan")
        del sess
    if DEV == "cuda":
        torch.cuda.empty_cache()
    rows = []
    for fi in sorted(per):
        for oid, m in sorted(per[fi].items()):
            r = row(fi, oid, scores.get((fi, oid), float("nan")), m)
            if r:
                rows.append(r)
    show_idx = sorted(set(np.linspace(0, len(keep) - 1, min(int(req.get("show", 8)), len(keep))).round().astype(int).tolist()))
    shown = [{"f": keep[k], "masks": [{"obj_id": oid, "mask": pack_mask(m)} for oid, m in sorted(per.get(keep[k], {}).items())]}
             for k in show_idx]
    return {"mode": mode + (f" (stride raised to {stride})" if stride != int(req.get("stride", 1)) else ""),
            "n_frames": len(keep), "rows": rows, "shown": shown}


class Handler(BaseHTTPRequestHandler):
    def address_string(self):
        return "unix"

    def log_message(self, fmt, *args):
        print(time.strftime("%H:%M:%S"), fmt % args, flush=True)

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        self._send(200, {"ok": True, "device": DEV, "loaded": sorted(_M)})

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            t0 = time.time()
            with LOCK:
                out = {"/seg": seg, "/track": track}[self.path](req)
            print(f"  {self.path} {out['mode']} in {time.time() - t0:.1f}s", flush=True)
            self._send(200, out)
        except KeyError as e:
            self._send(400, {"error": f"bad request: missing {e}"})
        except Exception as e:  # report, keep serving
            traceback.print_exc()
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--preload", action="store_true", help="load all four models at start")
    a = ap.parse_args()
    sp = Path(a.sock); sp.parent.mkdir(parents=True, exist_ok=True)
    if sp.exists():
        sp.unlink()
    if a.preload:
        for k in ("image", "tracker", "video", "tracker_video"):
            model(k)
    srv = Server(str(sp), Handler)
    os.chmod(sp, 0o666)
    print(f"SAM 3 tool server on {sp} ({DEV})", flush=True)
    srv.serve_forever()
