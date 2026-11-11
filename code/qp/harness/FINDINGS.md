# Harness findings — synthetic shapes with exact ground truth

Method: render a shape at 16x, area-average down (true anti-aliasing), degrade
through realistic effects, measure with six estimators, score SIGNED relative error.

## 1. A normalisation bug that would have corrupted every small object
Estimating the foreground/background levels with global percentiles fails whenever
the object is small: a 12 px disc covers 0.17% of a 256x256 canvas, so even the
99th percentile is background. Reported +144% error before the fix. Use the two
histogram modes split at the midpoint of the global extremes instead — scale-free.

## 2. Thresholding in ENCODED light inflates; thresholding in LINEAR light shrinks
Blur is linear in RADIANCE, but 8-bit video stores gamma-encoded values, so
"halfway between object and background" in encoded units is NOT the 50%-coverage
point. Measured on a disc, 0.5 iso-contour + Taubin fit:

    d=12, sigma=3   encoded +12.45%   linear -16.78%   theory -12.5%
    d=36, sigma=1   encoded  +2.20%   linear  -0.86%   theory  -0.15%
    d=90, sigma=3   encoded  +3.08%   linear  -0.51%   theory  -0.22%

The sign REVERSES. The gamma error is larger than the physical curvature effect it
masks. This is why the brightness-based radial-gradient method read slightly large.
=> Linearise before any intensity threshold. Better: measure on an alpha/probability
field, where gamma does not apply at all.

## 3. Blur bias is a SMALL-OBJECT problem, and it is severe
Signed error under blur, all estimators within ~2 points of each other:
    d=200  sigma=3  ~ +1.4%
    d=90   sigma=3  ~ +3.1%
    d=36   sigma=3  ~ +8%
    d=20   sigma=3  ~ +19%
    d=12   sigma=3  ~ +38%
Below ~20 px, blur alone can exceed the entire 5% full-credit budget.

## 4. On CLEAN images all six estimators agree within 0.1%
bbox, area-equivalent, moments, iso-contour, iso-extent and coverage integral are
indistinguishable on an undegraded disc at every size. Estimator choice only starts
to matter under degradation — which is why comparing methods on clean data (as our
first three-object test effectively did) proves nothing.

## 5. JPEG is second-order
Partial-coverage band on a 36 px disc: clean 92 px, blur sigma=1 392 px, motion
smear 40 px 2408 px, JPEG q50 only 144 px. Codec ringing is not worth modelling
relative to blur and smear.
