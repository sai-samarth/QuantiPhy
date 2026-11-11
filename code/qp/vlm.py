"""VLM client + numeric answer parsing."""
from __future__ import annotations
import json, re, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Units we strip before hunting for a number. Order matters: longest first, so
# 'cm/s^2' is consumed before 'cm'.
_UNITS = (r"m/s\^?2|m/s²|cm/s\^?2|cm/s²|mm/s\^?2|ft/s\^?2|"
          r"km/h|kph|mph|m/s|cm/s|mm/s|ft/s|"
          r"meters?|metres?|centimet(?:er|re)s?|millimet(?:er|re)s?|kilomet(?:er|re)s?|"
          r"inch(?:es)?|feet|foot|yards?|seconds?|secs?|"
          r"\bm\b|\bcm\b|\bmm\b|\bkm\b|\bin\b|\bft\b|\byd\b|\bs\b")
_NUM = re.compile(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_MARKERS = ["final answer:", "the answer is:", "the answer is", "answer:", "=>", "="]


def strip_think(text: str) -> str:
    """Everything after the LAST </think>. A thinking reply is full of numbers;
    the last-number heuristic must only ever see the final response. If the
    block never closed (truncation), there is no final answer -- return empty so
    the caller records a failure instead of a number plucked from mid-reasoning."""
    t = str(text)
    # vLLM's template opens the block in the PROMPT, so a well-formed reply
    # contains </think> but never <think>. Therefore in a thinking arm the
    # closing tag is the only reliable marker: absent means the reasoning was
    # truncated and there IS no final answer.
    if "</think>" not in t:
        return ""
    return t.rsplit("</think>", 1)[-1]


def parse_number(text: str) -> float:
    """Extract the answer number from a model response.

    Strategy, in order:
      1. If the whole response is just '<number> [unit]', take it.
      2. Otherwise cut at the LAST answer marker ('=', 'Answer:', ...) and
         search only after it — models restate the question before answering.
      3. Strip units, then take the LAST number in that region.
    Returns NaN if nothing parses. Callers must NOT write NaN to a submission.
    """
    if text is None:
        return float("nan")
    t = str(text).strip()
    # Transport failures must NEVER become numbers. "HTTP Error 400" would
    # otherwise parse to 400.0 and produce a plausible-looking low score
    # instead of a loud failure. This cost us one corrupted sweep.
    if t.startswith("__ERROR__"):
        return float("nan")
    if not t:
        return float("nan")

    clean = re.sub(rf"\s*(?:{_UNITS})\s*$", "", t, flags=re.I).strip()
    m = _NUM.fullmatch(clean.replace(",", ""))
    if m:
        return float(m.group().replace(",", ""))

    low = t.lower()
    cut = max((low.rfind(k) + len(k) for k in _MARKERS if k in low), default=0)
    tail = t[cut:] if cut else t
    tail = re.sub(rf"(?:{_UNITS})", " ", tail, flags=re.I)
    nums = _NUM.findall(tail)
    if not nums:
        nums = _NUM.findall(re.sub(rf"(?:{_UNITS})", " ", t, flags=re.I))
    if not nums:
        return float("nan")
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return float("nan")


# ───────────── unit-aware final answers ─────────────
# parse_number strips units without reading them, so "6 cm" on an "in meters"
# question became 6 m (two test rows, ~10x high after pooling), and
# "1.2 × 10⁻⁵ m" would read as 10. parse_answer reads the unit next to the
# number, understands x10^n notation, prefers a number that carries a unit of
# the asked kind, and converts to the asked unit.
_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_EXP = re.compile(r"(\d)\s*(?:[x×*·]|\\times|\\cdot)\s*10\s*(?:\^|\*\*)?\s*\{?\s*([-+]?\d+)\s*\}?")
_PREFIX = {"k": 1e3, "c": 1e-2, "m": 1e-3, "µ": 1e-6, "μ": 1e-6, "u": 1e-6, "n": 1e-9, "": 1.0}
_WORD = {"kilomet": "k", "centimet": "c", "millimet": "m", "micromet": "µ", "nanomet": "n", "met": ""}
_UNIT_AFTER = re.compile(
    r"\s*(km/h|kph|(?:kilo|centi|milli|micro|nano)?met(?:er|re)s?|[kcmµμun]?m)"
    r"(\s*/\s*s(?:ec)?\s*(?:\^\s*2|²|2)?|\s+per\s+second(?:\s+squared)?)?(?![a-z])", re.I)


def _unit(s: str):
    """'cm/s^2' -> ('L/T2', 0.01); 'meters' -> ('L', 1); None if not a length unit."""
    m = _UNIT_AFTER.match(" " + str(s).strip())
    if not m:
        return None
    base, rate = m.group(1).lower(), (m.group(2) or "").lower()
    if base in ("km/h", "kph"):
        return "L/T", 1000 / 3600
    for w, p in _WORD.items():
        if base.startswith(w):
            f = _PREFIX[p]
            break
    else:
        f = _PREFIX.get(base[:-1], None) if base.endswith("m") else None
        if f is None:
            return None
    dim = "L"
    if rate:
        dim = "L/T2" if ("2" in rate or "²" in rate or "squared" in rate) else "L/T"
    return dim, f


def question_unit(question: str):
    m = re.search(r"\bin\s+((?:kilo|centi|milli|micro|nano)?met(?:er|re)s?|[kcmµun]?m"
                  r"(?:\s*/\s*s(?:\^?2|²)?)?|km/h)\s*[\?\.]?\s*$", str(question).strip(), re.I)
    return _unit(m.group(1)) if m else None


def parse_answer(text: str, question: str | None = None) -> float:
    """The answer in the QUESTION's unit. Falls back to parse_number."""
    if text is None or str(text).startswith("__ERROR__"):
        return float("nan")
    t = str(text).translate(_SUP)
    t = re.sub(r"\\(?:boxed|text|mathrm)\{([^}]*)\}", r"\1", t).replace("$", "")
    t = _EXP.sub(lambda m: f"{m.group(1)}e{m.group(2)}", t)
    want = question_unit(question) if question else None
    low = t.lower()
    cut = max((low.rfind(k) + len(k) for k in _MARKERS if k in low), default=0)
    region = t[cut:] if cut else t
    found = []
    for m in _NUM.finditer(region):
        s = m.group().replace(",", "")
        if s in ("", ".", "+", "-"):
            continue
        try:
            v = float(s)
        except ValueError:
            continue
        found.append((v, _unit(region[m.end():m.end() + 30].lstrip(" ]})*\"'"))))
    if want:
        same = [f for f in found if f[1] and f[1][0] == want[0]]
        if same:
            v, u = same[-1]
            return float(v if u[1] == want[1] else v * u[1] / want[1])
    # no number carries a unit of the asked kind: the old behaviour, except
    # that x10^n notation is now read correctly
    return parse_number(t)


def chat(images_b64, text, model="qwen3vl-8b", port=8001, max_tokens=96,
         temperature=0.0, system=None, timeout=300, top_p=None, top_k=None,
         presence_penalty=None, chat_template_kwargs=None, image_labels=None) -> str:
    """One OpenAI-compatible chat completion against a local vLLM server.

    The sampling extras exist for THINKING models: Qwen3.8's card specifies
    temperature=1.0, top_p=0.95, top_k=20 for thinking mode and warns against
    greedy decoding there. chat_template_kwargs passes enable_thinking /
    reasoning_effort through vLLM to the checkpoint's own template.

    image_labels, if given, is one string per image, sent as a text part
    immediately before that image (e.g. its timestamp)."""
    if image_labels is not None and len(image_labels) != len(images_b64):
        raise ValueError("image_labels must match images one-to-one")
    content = []
    for k, b in enumerate(images_b64):
        if image_labels is not None:
            content.append({"type": "text", "text": image_labels[k]})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    content.append({"type": "text", "text": text})
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": content}]
    payload = {"model": model, "messages": msgs,
               "max_tokens": max_tokens, "temperature": temperature}
    if top_p is not None: payload["top_p"] = top_p
    if top_k is not None: payload["top_k"] = top_k
    if presence_penalty is not None: payload["presence_penalty"] = presence_penalty
    if chat_template_kwargs: payload["chat_template_kwargs"] = chat_template_kwargs
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.load(urllib.request.urlopen(req, timeout=timeout))
        return r["choices"][0]["message"]["content"]
    except Exception as e:
        return f"__ERROR__ {type(e).__name__}: {e}"


def chat_many(jobs, workers=16, **kw):
    """Run many chat() calls concurrently. jobs = [(images_b64, text[, labels]), ...]

    Concurrency is the whole point of the batch tier: vLLM batches these
    server-side, which is why --max-num-seqs matters so much.
    """
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda j: chat(j[0], j[1], image_labels=(j[2] if len(j) > 2 else None),
                                           **kw), jobs))


def error_rate(responses) -> float:
    """Fraction of responses that were transport failures."""
    if not len(responses):
        return 0.0
    return sum(str(r).startswith("__ERROR__") for r in responses) / len(responses)


def assert_healthy(responses, max_rate: float = 0.02) -> None:
    """Raise if too many requests failed — a silent partial failure is worse
    than a crash, because it yields a number that looks real."""
    r = error_rate(responses)
    if r > max_rate:
        bad = next(x for x in responses if str(x).startswith("__ERROR__"))
        raise RuntimeError(f"{r:.1%} of requests failed (>{max_rate:.0%}); first: {bad[:200]}")
