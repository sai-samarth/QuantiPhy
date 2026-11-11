"""PLAN stage: question + prior -> a measurement recipe.

Two strategies, cheapest first:
  1. rules   — regex over the question and prior. Free, deterministic, debuggable.
  2. vlm     — ask a model. Costs a call; handles phrasing the rules miss.

We measure rule coverage before paying for the model, so the VLM only ever sees
the residual. Anything the rules handle can never be broken by a model update.
"""
from __future__ import annotations
import re

from .types import Plan, Result, Target

UNIT_SI = {"m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
           "cm": 0.01, "mm": 0.001, "km": 1000.0}

_AXIS = [
    (r"\bwingspan\b|\bbreadth\b", "width"),
    (r"\bdiameter\b|\bcalibre\b|\bcaliber\b", "diameter"),
    (r"\bheight\b|\btall\b|\bhow high\b|\bdepth \(thickness\)|\bthickness\b", "height"),
    (r"\bwidth\b|\bhow wide\b|\bwide\b", "width"),
    (r"\blength\b|\bhow long\b|\blong\b", "length"),
]

_QUANTITY = [
    (r"\bacceler", "acceleration"),
    (r"\bspeed\b|\bvelocit", "speed"),
    (r"\bdistance\b|\bdisplacement\b|\bbetween\b|\bfrom .* to \b", "distance"),
]

# Every way this benchmark writes a time, in one place.
# These cost 342 test rows (9.2 macro points) as PLAN failures because each new
# spelling was being patched into a different regex: "at time1.75 s" has no space
# after "time", "from 0.5 s to 1.0 s" puts a space before the unit, and
# "from 0s to 0.3s" has none. Written once, applied to every phrase.
_TIME = (
    r"(?:\bat\s*(?:time)?\s*t?\s*=?\s*\d+(?:\.\d+)?\s*s\b)"          # at 1.2s / at time1.75 s / at t=1.5s
    r"|(?:\bwhen\s*t?\s*=?\s*\d+(?:\.\d+)?\s*s\b)"                    # when t = 3 s
    r"|(?:\b(?:from|between|in|over|during)\s+\d+(?:\.\d+)?\s*s?\s*"
    r"(?:to|and|-|until)\s*\d+(?:\.\d+)?\s*s\b)"                       # from 0.5 s to 1.0 s
    r"|(?:\b(?:before|after|by|until|past|within)\s+\d+(?:\.\d+)?\s*s\b)"    # before 0.5s
    r"|(?:\bt\s*=\s*\d+(?:\.\d+)?\s*s?\b)"                             # t=1.5
    r"|(?:\b\d+(?:\.\d+)?\s*s\s*(?:to|and|-)\s*\d+(?:\.\d+)?\s*s\b)"  # 1.00s to 2.00s
)
_TIME_RE = re.compile(_TIME, re.I)


def strip_time(text: str) -> str:
    """Remove any time reference from a phrase, in every spelling used here."""
    out = _TIME_RE.sub(" ", str(text))
    # a trailing bare "1.75 s" the alternations above did not anchor
    out = re.sub(r"\b\d+(?:\.\d+)?\s*s\b", " ", out)
    return re.sub(r"\s+", " ", out).strip(" ,.-")


# phrases that are scene furniture, not the object being measured
_STRIP = re.compile(
    r"^(what is|what's|how (long|wide|tall|high|big)) |^the |\bin (meters?|metres?|cm|mm|km|m|m/s|cm/s|m/s\^?2|cm/s\^?2)\b.*$"
    r"|\bat \d+(\.\d+)?\s*s\b|\bat time \d+(\.\d+)?\s*s\b|\bwhen t\s*=\s*\d+(\.\d+)?\s*s\b|\?",
    re.I)


def _axis(q: str) -> str:
    for pat, a in _AXIS:
        if re.search(pat, q, re.I):
            return a
    return "any"


def _quantity(q: str) -> str:
    for pat, k in _QUANTITY:
        if re.search(pat, q, re.I):
            return k
    return "size"


def _unit(q: str) -> str:
    m = re.search(r"\bin\s+(cm/s\^?2|m/s\^?2|cm/s|m/s|cm|mm|km|meters?|metres?|m)\b", q, re.I)
    return (m.group(1).lower() if m else "m")


# gerunds / abstractions that are not groundable objects, mapped to what they imply
# NB: "gravity" is deliberately NOT mapped to "falling object". It was, and that
# phrase grounds NOTHING (0 instances on every clip), which pre-empted the
# reference := target fallback below and cost 285 test rows their scale. A
# gravity prior names no object because the object is whatever the question asks
# about; leaving the phrase empty is what triggers that fallback.
_IMPLIED_OBJECT = {
    "walking": "person walking", "pedestrian walking": "pedestrian",
}
_GRAVITY = re.compile(r"^\s*gravit\w*(\s+acc\w*)?\s*$", re.I)
# words that can never start a groundable noun phrase
_NOT_AN_OBJECT = re.compile(r"^(distance|displacement|speed|velocity|acceleration)\b", re.I)


def _object_phrase(q: str) -> str:
    """The noun phrase to hand a grounding model."""
    s = str(q).replace("\u2019", "'").replace("\u02bc", "'").strip()
    s = re.sub(r"^(what is|what's)\s+", "", s, flags=re.I)
    s = re.sub(r"\s*\?$", "", s)
    s = re.sub(r"\s*\([^)]*\)", " ", s)          # BEFORE the 'of' split, see below
    # 'the height of the bowling pin in meters' -> 'the bowling pin'
    m = re.search(r"\b(?:travell?ed|moved|covered)\s+by\s+(.+?)(?:\s+in\s+[a-z/^0-9]+)?$", s, re.I)
    if not m:
        m = re.search(r"\bof\s+(.+?)(?:\s+in\s+[a-z/^0-9]+)?$", s, re.I)
    if m:
        s = m.group(1)
    s = _STRIP.sub(" ", s)
    # "white ball's average velocity in 1.00s to 2.00s" -> "white ball"
    s = re.sub(r"'s\s+(average\s+)?(velocity|speed|acceleration|height|width|length|"
               r"displacement|distance|diameter|thickness|wingspan).*$", "", s, flags=re.I)
    s = re.sub(r"\bin\s+\d+(\.\d+)?\s*s\s+to\s+\d+(\.\d+)?\s*s\b.*$", "", s, flags=re.I)
    s = re.sub(r"\b(between|from)\s+the\s+beginning\s+to\s+the\s+end\b", "", s, flags=re.I)
    s = re.sub(r"\s*\([^)]*\)", " ", s)              # drop parenthetical definitions
    s = re.sub(r"\s+from\s+the\s+.*$", "", s, flags=re.I)   # "pier from the shore-side edge..."
    s = re.sub(r"\bbetween\s+\d.*$", "", s, flags=re.I)     # "between 1.0s and 1.4s"
    s = re.sub(r"[~]+", " ", s)
    s = strip_time(s)
    s = re.sub(r"\s*,?\s*in\s+(m|cm|mm|km|meters?|metres?)\b.*$", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip(" ,.")
    s = re.sub(r"^(the|a|an)\s+", "", s, flags=re.I)
    return _IMPLIED_OBJECT.get(s.lower(), s)


_TIMEY = re.compile(r"^\s*\d+(?:\.\d+)?\s*s?\s*$", re.I)


def is_time_phrase(x: str) -> bool:
    """True if this 'object' is really a timestamp.

    'between A and B' means two OBJECTS in 'the distance between the two lamps'
    and two TIMES in 'the displacement of the ball between 1.0s and 1.3s'. The
    same regex matched both, so 94 test rows ended up hunting the video for an
    object called '1.0s' -- and, worse, would have measured the separation
    between two timestamps had the phrase validator not caught it.
    """
    return bool(_TIMEY.match(str(x)))


def prior_kind(unit: str, text: str = "") -> str:
    """'length' | 'rate' | 'accel' from the unit token AND the wording.

    The '2' in 'm/s^2' is the whole difference between a speed prior and an
    acceleration prior, and losing it is a units error, not a rounding error.
    But the unit alone is not enough: the dataset contains priors written
    'acceleration of the box before 0.6s = 9.8m/s' -- the WORD says acceleration
    and the unit has a typo. Trust the word over the token, because 9.8 divided
    by a pixel SPEED instead of a pixel acceleration is a 3x scale error.
    """
    u = str(unit).lower().replace(" ", "")
    says_accel = bool(re.search(r"\bacceler|\bacc\b|\bdecel", str(text), re.I))
    if not u or "/s" not in u:
        return "accel" if (says_accel and "/s" in u) else "length"
    return "accel" if ("2" in u or says_accel) else "rate"


_INTERVAL = re.compile(
    r"(?:from|between|in|over|during)\s+(\d+(?:\.\d+)?)\s*s?\s*"
    r"(?:to|and|-|until)\s*(\d+(?:\.\d+)?)\s*s", re.I)


def question_interval(q: str):
    """(t_from, t_to) if the question names a span of time, else None."""
    m = _INTERVAL.search(str(q))
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    return (a, b) if b > a else (b, a)


def prior_time(prior: str) -> float | None:
    """The instant the PRIOR describes, if it names one.

    't=1.5, ball acceleration = 3.0 m/s^2' and 'velocity of the soccer ball at
    1.5s = 5.21 m/s' both pin their value to a moment. That moment is where the
    scale must be measured, and it is NOT the moment the question asks about.
    """
    m = re.search(r"(?:\bat\s*(?:time)?\s*|\bt\s*=\s*|\bbefore\s+|\bafter\s+)"
                  r"(\d+(?:\.\d+)?)\s*s?\b", str(prior), re.I)
    return float(m.group(1)) if m else None


def parse_prior(prior: str) -> tuple[str, float, str, bool, str]:
    """(object phrase, value in SI, axis, is_motion, raw unit)."""
    p = str(prior).strip()
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(cm/s\^?2|m/s\^?2|cm/s|m/s|mm|cm|km|m)\b", p, re.I)
    if not m:
        return (_object_phrase(p), 0.0, "any", False, "")
    val, unit = float(m.group(1)), m.group(2).lower()
    motion = "/s" in unit
    # abs(): "acceleration of the orange car = -2.86 m/s^2" is a DECELERATION,
    # and only the MAGNITUDE calibrates the scale -- pixel acceleration is
    # measured as a magnitude too. 47 test rows were rejected as "reference
    # value missing or non-positive" purely for having a sign.
    si = abs(val) * (UNIT_SI.get(unit.split("/")[0], 1.0) if not motion else
                     (0.01 if unit.startswith("cm") else 1.0))
    head = p[:m.start()]
    head = re.sub(r"\b(of|the|a|an)\b", " ", head, flags=re.I)
    head = re.sub(r"\b(speed|velocity|acceleration|acc|diameter|height|width|length|calibre|caliber)\b",
                  " ", head, flags=re.I)
    # NB: the leading-timestamp strip must REQUIRE an '=' or a digit after the 't'.
    # Without that it ate the first letter of any word starting with t:
    # "tennis ball" -> "ennis ball".
    head = re.sub(r"[=,]", " ", head)
    head = re.sub(r"^\s*t\s*(?==|\d)\s*[\d.]*\s*", " ", head)
    head = re.sub(r"\bat\s+(time\s+)?\d+(\.\d+)?\s*s\b", " ", head, flags=re.I)
    head = re.sub(r"\bt\s*=\s*\d+(\.\d+)?\b", " ", head, flags=re.I)
    phrase = re.sub(r"[~]+", " ", head)
    phrase = strip_time(phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip(" ,.-")
    phrase = _IMPLIED_OBJECT.get(phrase.lower(), phrase)
    if _GRAVITY.match(phrase):
        phrase = ""                      # -> reference := target, see plan_by_rules
    return (phrase, si, _axis(p), motion, unit)


def plan_by_rules(question: str, prior: str, depth_info=None) -> Result:
    q = str(question)
    quantity = _quantity(q)
    axis = _axis(q)
    unit = _unit(q)
    ref_phrase, ref_val, ref_axis, is_motion, raw_unit = parse_prior(prior)
    p_at = prior_time(prior)
    tgt = _object_phrase(q)

    tm = re.search(r"(?:\bat\s+(?:time\s+)?|\bwhen\s+t\s*=\s*)(\d+(?:\.\d+)?)\s*s\b", q, re.I)
    at_time = float(tm.group(1)) if tm else None
    iv = question_interval(q)
    t_from, t_to = iv if iv else (None, None)

    if not tgt:
        return Result.bad("could not extract a target phrase", "rules")
    if not ref_phrase or len(ref_phrase) < 3:
        # A bare "acceleration = 9.8 m/s^2" is GRAVITY: it names no object
        # because the object it acts on is the one the question asks about.
        # 40 test rows failed here for want of saying so.
        if is_motion:
            ref_phrase = tgt
            # (the reference and target are then the same object, which the
            #  segmenter deduplicates for free)
        else:
            return Result.bad("could not extract a reference phrase from the prior", "rules")

    target_b = None
    # "distance between the two black road signs" -> two instances of one phrase
    two = re.search(r"between\s+the\s+two\s+(.+?)(?:\s+at\b|\s+in\b|$)", q, re.I)
    if two:
        ph = re.sub(r"\s+", " ", two.group(1)).strip(" ,.")
        ph = re.sub(r"s$", "", ph)                      # plural -> singular
        return Result.good(Plan(
            quantity="distance", axis="separation", unit=unit,
            reference=Target(phrase=ref_phrase, role="reference"),
            target=Target(phrase=ph, role="target"),
            target_b=Target(phrase=ph, role="target_b"),
            at_time=at_time, reference_is_motion=is_motion,
            reference_value=ref_val, reference_axis=ref_axis,
            reference_kind=prior_kind(raw_unit, prior), prior_at_time=p_at,
            t_from=t_from, t_to=t_to,
            notes=["two instances of the same phrase"]), "rules")
    bm = re.search(r"between\s+(.+?)\s+and\s+(.+?)(?:\s+at\b|\s+in\b|$)", q, re.I)
    if bm:
        a = re.sub(r"^(the|a|an)\s+", "", bm.group(1).strip(), flags=re.I)
        b = re.sub(r"^(the|a|an)\s+", "", bm.group(2).strip(), flags=re.I)
        if is_time_phrase(a) or is_time_phrase(b):
            # a TIME interval, not two objects: the target stays the single
            # moving object and the interval is what we measure it over
            quantity = "distance"
            axis = "any"
        else:
            tgt, target_b = a, Target(phrase=b, role="target_b")
            axis = "separation"; quantity = "distance"

    return Result.good(Plan(
        quantity=quantity, axis=axis, unit=unit,
        reference=Target(phrase=ref_phrase, role="reference"),
        target=Target(phrase=tgt, role="target"),
        target_b=target_b, at_time=at_time,
        reference_is_motion=is_motion, reference_value=ref_val, reference_axis=ref_axis,
        reference_kind=prior_kind(raw_unit, prior), prior_at_time=p_at,
        t_from=t_from, t_to=t_to,
    ), "rules")


def validate_plan(p: Plan) -> str | None:
    """What the GROUND stage requires. A plan that fails this is worse than no
    plan, because it sends the segmenter hunting for something ungroundable."""
    for t in [p.reference, p.target] + ([p.target_b] if p.target_b else []):
        ph = (t.phrase or "").strip()
        if len(ph) < 3:
            return f"{t.role} phrase too short: {ph!r}"
        if _NOT_AN_OBJECT.match(ph):
            return f"{t.role} phrase is a quantity, not an object: {ph!r}"
        if len(ph.split()) > 8:
            return f"{t.role} phrase too long to ground: {ph!r}"
        if re.search(r"\d+(\.\d+)?\s*s\b", ph):
            return f"{t.role} phrase still contains a timestamp: {ph!r}"
    if p.reference_value <= 0:
        return "reference value missing or non-positive"
    return None
