# Pipeline architecture

    Question ──► PLAN ──► GROUND ──► MEASURE ──► SOLVE ──► GATE ──► Answer
                 what      which      how many    convert   do we
                 to        pixels     pixels      to        trust
                 measure   are it                 metres    it?

RUN composes the first four. GATE decides whether the result is used at all --
which is what makes a HYBRID possible: the pipeline does not have to work
everywhere, it has to work where it works and know when it does not.

## Principles (each earned from a real bug)

1. **Stages talk only through `types.py`.** No stage imports another. That is what
   lets SAM be swapped out and lets the geometry be unit-tested with no video files.

2. **Failure is a value, not an exception.** Every stage returns `Result` = ok(value)
   or fail(reason). A stage that cannot do its job says so; it never invents a number.
   MRA gives partial credit for a hedged guess and zero for confident nonsense.

3. **Fallbacks are declarative data, not nested try/except.** A stage declares an
   ordered list of `Strategy`. Because the chain is data we can log it, count which
   rungs actually fire, and delete the ones that never win.

4. **Validation is separate from strategy.** A strategy can SUCCEED and be WRONG — a
   full-frame sliver is a successful segmentation of the wrong thing. Every result is
   checked by an independent validator; failing it falls through like an error.

5. **A stage's success criterion is whether the NEXT stage can use its output.**
   `plan_by_rules` "succeeded" on 159/159 questions while emitting ungroundable
   phrases like `walking`. `validate_plan` encodes what GROUND actually needs.

6. **Cheap deterministic first, models only on the residual.** Rules plan 97% of
   validation questions for free; the VLM sees the other 3%. Anything rules handle
   can never be broken by a model update.

7. **Cache at the expensive boundary only.** SAM calls are cached to disk (atomic
   writes, survives crashes). Geometry is pure and fast, so it is recomputed — a
   fix to the arithmetic must not require invalidating segmentation.

## Files
    types.py     the contracts. Read this first.
    strategy.py  fallback chains + validation
    cache.py     disk cache for model calls
    plan.py      question -> measurement recipe (rules; VLM fallback pending)
    ground.py    recipe + video -> object masks (SAM3 chain)
    measure.py   masks -> pixel lengths  [DONE]
    geometry.py  the depth verdict: how many metres is one pixel, and where
    solve.py     pixel observables + prior -> a number in the asked-for unit
    run.py       orchestration, instance choice, provenance
    gate.py      the trust policy: which answers to use, which to hand back

### Lessons encoded in measure.py
* WHICH FRAMES matters more than which formula. Choose frames by SLOWEST motion,
  because blur scales with speed. ("longest stable run" picked the blurred window
  on the tennis ball purely because it was longer.)
* Frame choice cannot fix a BIASED ruler. Mask-minor undershot the tennis ball by
  7.5% in every frame; only the gradient method reaches the true edge.
* The gradient is the most accurate method where it applies and SILENTLY WRONG
  where it does not: patterned objects (soccer ball 48.8 vs true 95.4) and
  non-circular ones (pin 18.3 vs 55.0) — both with LOW spread, so a stability
  check cannot catch them.
* Therefore: accept the gradient only when an INDEPENDENT mask measurement
  corroborates it (within 0.8-1.25x). Two methods disagreeing is the evidence.

### Gates are PER-FAILURE-MODE, not global
The depth gate reports a DEPTH confidence. On the validation benchmark it separates
median error 5.4% (both depths known) from 53.9% (target depth unknown) — a 10x split.
But it correctly reports confidence 1.0 for 2D scenes that then fail for a completely
different reason (SAM segmented a power tool when asked for a "ruler"). One gate per
way of being wrong; the answer's confidence is their combination.

### The one formula in solve.py
A size prior and a motion prior calibrate the SAME unknown, so there is one code
path rather than four:

    m_per_px = (the prior, in SI) / (that same object, in pixels)
    answer   = (the target, in pixels) * m_per_px * (Z_target / Z_ref)

Whether the prior is `0.01 m` or `2 m/s` changes only which pixel number goes in
the denominator. fps then cancels for SS and DD and survives for SD and DS --
which is WHY only the two mixed types are exposed to the fps disagreement
between container and metadata. That falls out of the algebra; it is not
special-cased anywhere.

### Why gate.py is separate from run.py
RUN reports what happened. GATE decides whether to believe it. Keeping them
apart means the trust policy can be refit against validation without touching
perception, and every rule can be counted: a rule that fires on rows scoring the
same as everything else is a superstition, and gets deleted.

Gate rules are calibrated by `scripts/fit_gate.py`, which measures each rule's
lift and then sweeps the threshold against the LEADERBOARD number (VLM answering
wherever the gate rejects) -- not against the pipeline's own MRA. Those are
different objectives and only the second one pays.

## Status
PLAN   155/159 validation questions pass the validator (97%)
GROUND chain works; correctly EXHAUSTS and reports failure on "pencil bag"
       rather than measuring the wrong object. Border rejection is now
       AXIS-AWARE: a ruler running off the left and right edges is still a
       valid ruler to measure the calibre of.
MEASURE 4/4 verified cases route correctly: gradient where valid, mask where not,
       each with an auditable reason.
SOLVE  all four inference types verified exact on synthetic input, and every
       unit/kind mismatch refuses instead of returning a plausible wrong number.
RUN    end-to-end over the validation size questions; shardable across GPUs
       (scripts/run_pipeline_sharded.sh), cache keyed per video so a video is
       never segmented twice.
