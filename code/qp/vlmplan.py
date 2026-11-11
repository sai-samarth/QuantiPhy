"""PLAN by language model instead of by regex.

WHY THIS AND NOT MORE REGEX. The row-by-row audit found the planner grounding
the wrong noun on 293 test rows / 8.2 macro points, and the cause is structural
rather than a missing pattern: head_noun takes the TAIL of the phrase, which is
correct for a compound ("trash bags") and wrong for every post-modified one --
    "light right above the woman's head"      -> woman's head
    "pedestal of the central sculpture"       -> central sculpture
    "road that the bike is crossing through"  -> crossing through
Each of those is a different shape and patching them one at a time has not
converged over four rounds.

WHY THIS IS NOT THE VLM EXPERIMENT THAT FAILED. The rephrase loop and the mask
verifier (HANDOFF 6e) are IDENTITY tasks -- "is this a boat or a vessel", "is
this Callisto" -- and they failed on fine-grained recognition, shown a text label
reading Ganymede and asked about Callisto. This asks nothing about pixels. It is
sentence parsing, with no image in the request at all, so that failure mode is
not reachable from here.

TWO NOUNS PER OBJECT, DELIBERATELY. SAM 3 grounds simple concrete nouns and
barely grounds compositional phrases ("big bird" 0 instances, "bird" 2). But
HANDOFF 6d also measured that simplifying the phrase at the planner is WRONG --
the specificity is what lets the VLM pick the right instance later. So the model
returns both: the phrase as the question means it, and the plain noun to fall
back to. That fallback is precisely what head_noun was computing badly.
"""
from __future__ import annotations
import json
import os
import re

from .pipeline import cache
from .vlm import chat

# Off by default. Nothing about the shipped 0.545 path changes unless this is set.
USE = os.environ.get("QP_VLM_PLAN") == "1"
PORT = int(os.environ.get("QP_PLAN_PORT", "8004"))
MODEL = os.environ.get("QP_PLAN_MODEL", "qwen3vl-32b")

# Bumped whenever the prompt or the schema changes. The cached value depends on
# both, so leaving them out of the key means a re-run silently replays the old
# parse and the experiment measures nothing -- this has cost two A/Bs already.
VERSION = "plan-v3"

SYSTEM = (
    "You convert a physics measurement question into a structured plan. "
    "Reply with ONE JSON object and nothing else.\n"
    "\n"
    "Fields:\n"
    '  target        the object the QUESTION asks about, as a short phrase\n'
    '  target_noun   the same object as a PLAIN noun a detector can find, 1-2 words\n'
    '  target_axis   length|width|height|diameter|thickness|wingspan|none\n'
    '  quantity      size|distance|speed|acceleration\n'
    '  reference     the object the GIVEN PRIOR describes, as a short phrase\n'
    '  reference_noun  the same object as a plain noun, 1-2 words\n'
    '  reference_axis  length|width|height|diameter|thickness|speed|acceleration|none\n'
    "\n"
    "Rules:\n"
    "- The HEAD of a phrase comes FIRST when a modifier follows it. In 'the light "
    "right above the woman's head' the object is the LIGHT, not the head. In 'the "
    "pedestal of the central sculpture' it is the PEDESTAL.\n"
    "- target keeps the words that pick out WHICH one ('big bird', 'left tennis "
    "ball', 'the woman in yellow'). target_noun drops them ('bird', 'tennis ball', "
    "'woman').\n"
    "- If the prior names a whole-object dimension, reference_axis is that "
    "dimension. If it names a speed or an acceleration, use those.\n"
    "- If the prior describes no object (for example plain gravity), set "
    "reference and reference_noun to \"\".\n"
    "- quantity is what the ANSWER is: a size, a distance travelled, a speed, or "
    "an acceleration. Trust the unit over a misspelt word.\n"
    "- If the prior names an ACTIVITY rather than a thing ('walking velocity', "
    "'running speed'), the reference is whatever does it: a person.\n"
    "- target_noun and reference_noun must name a PHYSICAL THING you could point "
    "at. NEVER a measurement word. 'the velocity of the ball' -> ball. "
    "\"the eagle's wingspan\" -> eagle. 'the displacement of the puck' -> puck.\n"
    "- target and reference may be long; the two _noun fields must be short.\n"
    "- target names the OBJECT, never the measurement. For 'the length of the "
    "wood block' the target is 'the wood block'. For 'the width of the green "
    "car' it is 'the green car'. The dimension goes in target_axis and nowhere "
    "else."
)

# Words that name a QUANTITY, not a thing. A detector handed one of these
# grounds nothing at best; at worst it grounds a caption. The model puts them in
# the noun fields often enough that a deterministic check is cheaper than a
# better prompt: "the velocity of the ball" came back with target_noun
# "velocity", and "the eagle's wingspan" with "wingspan".
_NOT_A_THING = re.compile(
    r"^(velocity|speed|acceleration|displacement|distance|height|width|length|"
    r"depth|thickness|diameter|radius|wingspan|size|area|volume|time|"
    r"position|motion|direction)s?$", re.I)

_FIELDS = ("target", "target_noun", "target_axis", "quantity",
           "reference", "reference_noun", "reference_axis")

# "the length of the wood block" -> "the wood block". The model puts the
# dimension back into `target` often enough to matter: the phrase is what SAM is
# asked for first, so it wastes a pass on every such row and cost one row its
# grounding entirely ("the vertical height of the 'skynova' lettering"). Telling
# it not to did not fully take, and this is the case a regex IS good at --
# anchored at the start, closed vocabulary, no ambiguity about what to remove.
_LEADING_MEASURE = re.compile(
    r"^\s*(?:the\s+)?(?:average\s+|total\s+|maximum\s+|minimum\s+|vertical\s+|"
    r"horizontal\s+|initial\s+|final\s+|side\s+)*"
    r"(?:length|width|height|depth|thickness|diameter|radius|wingspan|breadth|"
    r"size|speed|velocity|acceleration|displacement|distance)\s+of\s+", re.I)


def strip_measure(phrase: str) -> str:
    out = _LEADING_MEASURE.sub("", str(phrase)).strip()
    return out or str(phrase).strip()


def _extract(text: str) -> dict | None:
    """The JSON object in a reply, tolerating fences and surrounding prose."""
    if not text or str(text).startswith("__ERROR__"):
        return None
    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "target_noun" not in d:
        return None
    out = {k: (str(d.get(k, "")).strip() if d.get(k) is not None else "") for k in _FIELDS}
    for k in ("target", "reference"):
        out[k] = strip_measure(out[k])
    # Only the NOUN fields go to a detector, so only they are length-checked.
    # An earlier version applied this to `target` too and rejected three good
    # parses out of six, because "the light right above the woman's head" is a
    # perfectly correct value for a field whose job is to be specific.
    for k in ("target_noun", "reference_noun"):
        if len(out[k].split()) > 3:
            return None
        if out[k] and _NOT_A_THING.match(out[k].strip()):
            return None
    if out["quantity"] not in ("size", "distance", "speed", "acceleration"):
        return None
    return out


def parse(question: str, prior: str, port: int | None = None) -> dict | None:
    """Structured plan for one question, or None if the model did not comply.

    None means CALL THE REGEX. A refusal to guess is the point: a plan invented
    from a malformed reply would be believed downstream exactly like a measured
    one, and that is the failure mode this project keeps paying for.
    """
    q, p = str(question).strip(), str(prior).strip()

    def compute():
        msg = f"Given that {p}.\n{q}" if p and p.lower() != "nan" else q
        return chat([], msg, model=MODEL, port=(port or PORT),
                    max_tokens=300, temperature=0.0, system=SYSTEM)

    raw = cache.get_or_compute("vlmplan", compute, q=q, p=p,
                               model=MODEL, version=VERSION)
    got = _extract(raw)
    if got is None:                      # do not cache a dud as if it were an answer
        cache.put("vlmplan", "", q=q, p=p, model=MODEL, version=VERSION + "-retry")
    return got
