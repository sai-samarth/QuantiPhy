"""VLM-in-the-loop grounding: ask the model what to search for, and whether it worked.

WHY THIS EXISTS, measured. On the 12 worst validation failures, only 3 could be
answered from the evidence we produce. Of the other 9:
    4  the target was NEVER SEGMENTED -- no information at all
    5  segmented, but the semantically correct measurement is still wrong
So better reasoning over the current evidence cannot fix them. The information
has to get better, and grounding is where it is missing.

TWO PRIMITIVES, both cheap:

  REPHRASE  SAM finds nothing for "the vertical depth of the open roof frame".
            The VLM looks at the frames and answers "roof beam" -- a thing SAM
            can actually ground.

  VERIFY    SAM returns a mask. We draw it on the frame and ask the VLM whether
            that is the object the question means. This is the check that
            catches a CONFIDENT WRONG grounding, which no internal score does:
            SAM returned an 11x480 px strip at confidence 1.0 and called it a
            billiard ball, and segmented a power tool when asked for a ruler.

WHAT THE PROMPTS ENCODE. SAM 3 grounds simple concrete nouns well and
compositional phrases barely at all -- measured: "big bird" returns 0 instances
on a clip containing two birds, "bird" returns 2; likewise "black road sign" 0
vs "road sign" 3, "person on the left" 0 vs "person" 5. So the rephrase prompt
asks for a SIMPLE NOUN, not a better description. Telling the model to be more
specific would make things worse.
"""
from __future__ import annotations
import base64
import re

import cv2
import numpy as np

from .vlm import chat

REPHRASE_SYSTEM = (
    "You help a video segmentation model locate objects.\n"
    "It grounds SIMPLE CONCRETE NOUNS well: 'bird', 'car', 'ball', 'cup', 'boat'.\n"
    "It FAILS on compositional phrases ('big bird', 'the person on the left'), on "
    "relations ('the height of the pier above the water'), and on abstract or "
    "measurement words ('vertical depth', 'displacement').\n"
    "ITS VOCABULARY IS ALSO IDIOSYNCRATIC: it may know one word for a thing and "
    "not another. It finds 'jetty' but not 'pier'.\n"
    "So give SEVERAL different single-word or two-word names for the object the "
    "question is about -- synonyms, the material, the general category, the part "
    "and the whole.\n"
    "Reply with 3 to 5 candidates on ONE line, comma separated, lower case, "
    "most likely first. No explanation."
)

# NB: the region shown is the track's BOUNDING BOX, not its exact silhouette.
# Per-frame masks are deliberately not cached (they cost ~25 MB per entry, 500x
# the rest), and re-segmenting just to draw one would cost more than the check
# is worth. A box is enough to answer "WHICH object is this?", which is the
# failure being caught; it cannot answer "is the outline precise?", so the
# prompt does not ask.
VERIFY_SYSTEM = (
    "You verify object detection. The image shows a video frame with one region "
    "marked by a RED BOX and shading.\n"
    "Decide whether the marked region is ON the object the question asks about.\n"
    "The box is approximate -- judge WHICH object it marks, not how tightly.\n"
    "Reply on ONE line, exactly one of:\n"
    "  YES\n"
    "  NO <simple noun phrase to search for instead, 1-3 words>\n"
    "Answer NO only if it marks a DIFFERENT object than the question means."
)


def _b64(img, max_side=768, quality=88) -> str:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode() if ok else ""


def overlay(frame_bgr, mask) -> str:
    """The frame with the mask outlined in red and lightly shaded.

    Shading AND outline on purpose: an outline alone is easy to miss on a busy
    frame, and a fill alone hides the object being judged.
    """
    im = frame_bgr.copy()
    m = mask.astype(bool)
    im[m] = (0.55 * im[m] + 0.45 * np.array([60, 60, 255])).astype(im.dtype)
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
    return _b64(im)


_CLEAN = re.compile(r"[^a-z0-9 '\-]")


def _tidy(text: str, max_words: int = 3) -> str:
    t = _CLEAN.sub(" ", str(text).strip().lower())
    t = re.sub(r"^(the|a|an)\s+", "", t).strip()
    return " ".join(t.split()[:max_words])


def ask_rephrase(frames_b64, question: str, tried, n: int = 5, **kw) -> list[str]:
    """Several candidate nouns to try, best first, excluding what already failed.

    A LIST rather than one phrase, because the segmenter's vocabulary is
    idiosyncratic rather than wrong: on the lake clip it finds 'jetty' and not
    'pier', and the VLM -- reasonably -- says 'pier'. One suggestion means one
    roll of that dice; five costs the same single VLM call.
    """
    t = ", ".join(f"'{x}'" for x in tried if x)
    msg = (f"Question about this video: {question}\n"
           + (f"Already searched, found NOTHING: {t}\n" if t else "")
           + f"Give {n} different names for the object the question is about.")
    r = chat(frames_b64, msg, system=REPHRASE_SYSTEM, max_tokens=48, **kw)
    if str(r).startswith("__ERROR__"):
        return []
    seen = {str(x).lower() for x in tried}
    out = []
    for piece in str(r).split("\n")[0].split(","):
        c = _tidy(piece)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out[:n]


def ask_verify(overlay_b64: str, question: str, phrase: str, **kw):
    """(is_correct, suggested_phrase). A wrong mask is worse than no mask,
    because it is reported downstream as a measured fact."""
    msg = (f"Question about this video: {question}\n"
           f"The red region is what was found when searching for '{phrase}'.\n"
           f"Is the red region the object the question asks about?")
    r = str(chat([overlay_b64], msg, system=VERIFY_SYSTEM, max_tokens=24, **kw))
    if r.startswith("__ERROR__"):
        return True, ""                      # never reject on a transport error
    head = r.strip().split("\n")[0].strip()
    if head.upper().startswith("YES"):
        return True, ""
    if head.upper().startswith("NO"):
        sug = _tidy(head[2:])
        # A suggestion that shares a word with what we searched for is the model
        # being PEDANTIC about the same object, not naming a different one: it
        # rejected a perfectly good "boat" mask asking for "boat hull". Treat
        # that as agreement -- swapping to a narrower phrase for the same thing
        # risks a worse mask for no gain.
        a = set(phrase.lower().split()); b = set(sug.split())
        # SUBSET, not overlap. Overlap called 'soccer ball' and 'tennis ball'
        # the same object because both contain "ball" -- and that is precisely
        # the confusion this check exists to catch. A suggestion is the same
        # object only when one phrase is a refinement of the other.
        if sug and (a <= b or b <= a):
            return True, ""
        return False, sug
    return True, ""                          # unparseable: keep what we have


def ground_loop(video_path: str, question: str, phrase: str, fps: float,
                frames_b64, segment_fn, tried=None, max_candidates: int = 4,
                do_verify: bool = True, **kw):
    """Ask SAM; if it finds nothing or the wrong thing, ask the VLM and retry.

    Returns (phrase_used, df, trail). `segment_fn(phrase) -> df or None` is
    injected so this module never imports the segmenter and stays testable.

    DESIGN NOTES, each from a measurement:

    * The rephrase asks for SEVERAL names, not one. SAM's vocabulary is
      idiosyncratic rather than wrong -- on the lake clip it grounds 'jetty' and
      not 'pier', and the VLM sensibly says 'pier'. Five candidates cost the
      same single VLM call as one.

    * A NO from the verifier only takes effect IF ITS SUGGESTION ACTUALLY
      GROUNDS. The verifier is slightly over-strict: it rejected a perfectly
      good boat mask wanting 'boat hull'. Rejecting a usable mask in favour of
      nothing would be a regression, so the original is kept as a floor.

    * Verification is what catches a CONFIDENT wrong grounding, which no
      internal score does. On deliberate mismatches it was right 3 times out of
      3 and named the correct object each time.
    """
    trail = []
    tried = list(tried or [])
    fallback = None

    def attempt(ph):
        nonlocal fallback
        if not ph or len(ph) < 3 or ph in tried:
            return None
        tried.append(ph)
        df = segment_fn(ph)
        if df is None or not len(df):
            trail.append(f"{ph}: nothing")
            return None
        if not do_verify:
            trail.append(f"{ph}: found")
            return (ph, df)
        ok, sug = verify_track(video_path, ph, df, question, **kw)
        if ok:
            trail.append(f"{ph}: verified")
            return (ph, df)
        trail.append(f"{ph}: rejected -> {sug or 'no suggestion'}")
        if fallback is None:
            fallback = (ph, df)          # keep it; better than nothing
        return ("__retry__", sug)

    r = attempt(phrase)
    if r and r[0] != "__retry__":
        return r[0], r[1], trail
    suggested = r[1] if r else ""

    cands = []
    if suggested:
        cands.append(suggested)
    cands += ask_rephrase(frames_b64, question, tried, **kw)
    for c in cands[:max_candidates]:
        r = attempt(c)
        if r and r[0] != "__retry__":
            return r[0], r[1], trail
        if r and r[1] and r[1] not in cands:
            cands.append(r[1])

    if fallback:
        trail.append("kept the rejected mask (nothing better grounded)")
        return fallback[0], fallback[1], trail
    return phrase, None, trail


def verify_track(video_path: str, phrase: str, df, question: str, **kw):
    """Draw the largest track's mask on a representative frame and ask."""
    from qp.pipeline.cache import get, put
    # v2 in the key: the verdict depends on the PROMPT and the parsing rules, so
    # changing either must invalidate the cache. It did not, and a stale verdict
    # silently survived a logic fix -- the exact failure this project has hit
    # before with SAM thresholds.
    key = dict(video=video_path, phrase=phrase, q=question[:120], v="v3")
    hit = get("verify", **key)
    if hit is not None:
        return hit
    try:
        oid = int(df.groupby("obj_id").size().idxmax())
        g = df[df.obj_id == oid]
        fi = int(g["f"].iloc[len(g) // 2])
        # ONE frame, by seek. Decoding the whole clip here made verification
        # ~50x more expensive than the check itself and stalled the run.
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok_read, frame = cap.read()
        cap.release()
        if not ok_read or frame is None:
            return True, ""
        row = g.iloc[len(g) // 2]
        m = np.zeros(frame.shape[:2], bool)
        y0 = max(0, int(row["v"] - row["bh"] / 2)); y1 = min(m.shape[0], int(row["v"] + row["bh"] / 2))
        x0 = max(0, int(row["u"] - row["bw"] / 2)); x1 = min(m.shape[1], int(row["u"] + row["bw"] / 2))
        m[y0:y1, x0:x1] = True
        out = ask_verify(overlay(frame, m), question, phrase, **kw)
    except Exception:
        return True, ""
    put("verify", out, **key)
    return out
