---
name: video-physics
description: Measure physical sizes, distances, speeds and accelerations of objects in a short video from one known reference quantity, using pixel-accurate measuring tools (qpt). Use when a task gives video.mp4 plus questions.json asking for numeric physical quantities.
---

# Measuring physics from a video

You get one short video (`video.mp4`) and questions (`questions.json`). Each question
asks for ONE number (a size, distance, speed or acceleration of something in the
video) in a stated unit.

## Inputs
- `questions.json`: video metadata (fps, frame count, size, `source`: simulation, lab recording or internet
  footage, and whether it is a masked copy) and the questions. Each has
  - `prior`: one known physical quantity in the scene (e.g. "diameter of the ball =
    0.21 m", "speed of the car = 5 m/s", "gravity acc = 9.8 m/s^2"). It is the ruler
    that turns pixels into physical units.
  - `depth_info` (3D scenes only): stated camera-to-object distances, sometimes at
    several times.
  - `scene`: 2D (the measured objects as one plane facing the camera: one pixel scale; see
    "How the true answers were made") or 3D (objects can sit at different distances).
- `measurements.json` (may be absent): output of an automatic segmentation pipeline per
  question id. Often right, sometimes badly wrong (wrong object, instance or
  dimension, or a ruler measured on the wrong thing). A hint to check, never truth.

## Time
t = frame_index / fps with the fps in `questions.json` (frame 0 is t = 0). Some clips were
converted from a slightly higher frame rate by dropping a source frame now and then: an
object's step is then about 2x its neighbours at the same frame for every moving object, and
two frame intervals pass there, so EVERY interval is shorter than the clip's mean step. `qpt
info` and `fit` flag such steps (`fit` sees only its track's frames); when the same step
doubles on two or more movers, or on one while the whole frame jumps there, give those steps
to `fit --gaps` once (later fits and `flight` keep them) and use its true times and
asked-time frames in your own code: keeping windows off the step is not enough.

## Tools: `qpt` (run in the task directory; `qpt <cmd> -h` for options)
All coordinates are ORIGINAL video pixels (x right, y down, origin top-left), all
frame numbers are original indices, t = frame / fps. Images go to `qpt_out/`; open
them with the read tool. Every number a tool prints comes with a picture: look at it
before you trust the number.

| command | what it does |
|---|---|
| `qpt info` | video facts, held- and dropped-frame checks (whole-frame jumps; stated vs file clock), each question with its times mapped to frames |
| `qpt frames [--n 12] [--from T0 --to T1] [--f F ...]` | labelled contact sheet of frames |
| `qpt view --t T --box X0 Y0 X1 Y1 [--line X1 Y1 X2 Y2] [--pt X Y] [--diff] [--enhance]` | crop + zoom with a pixel grid labelled in original coordinates; draws your measuring lines and prints their lengths. `--diff` lights up what moves |
| `qpt edges --t T --line X1 Y1 X2 Y2 [--sweep W]` | sub-pixel start/end of the object along a line you place across it (start and end the line on the background just outside it; each end is half way up its first step from that background); `--sweep` also tries parallel lines within W px and reports the widest (along that direction) |
| `qpt cut --t T --box X0 Y0 X1 Y1 [--method motion] [--from T0 --to T1]` | segment the object inside a box without SAM (GrabCut, or change vs the background); same size table and picture as `seg`; over a time range it also gives max/median sizes |
| `qpt seg --t T --text "red ball"` | SAM 3: every instance of a phrase on one frame, with sizes (box w/h, rotated box, tip-to-tip, ellipse, equal-area diameter) and a mask picture |
| `qpt seg --t T --box X0 Y0 X1 Y1` or `--point X Y` | SAM 3: the one object at your box/point (use after locating it with `view`) |
| `qpt segtrack --text "phrase"` or `--t T --box ...` | SAM 3 through the video: per-frame centre and sizes -> csv |
| `qpt motion` | static camera: moving blobs per frame, linked into tracks -> `qpt_out/motion.csv` |
| `qpt track --t T --box X0 Y0 X1 Y1` | template tracking of a box through the clip -> csv |
| `qpt depth --csv FILE --obj N --anchor T DIST [--size TRUE_SIZE]` | 3D track of a rigid object from its apparent size: distance(t) = DIST x size(T) / size(t), and X, Y from the focal length -> csv for `fit` (3D speeds). Needs clean per-frame masks (`segtrack`); check its size-noise line |
| `qpt focal --obj PX SIZE DIST [U V] [--obj ...]` or `--vp1 LINES --vp2 LINES` | focal length in px from objects of known size at known distance (several must agree: one camera), or from the vanishing points of two perpendicular sets of parallel lines, with how far small errors move it |
| `qpt plane --f F --pt U V DIST [H] (x3+, or x1+ with --horizon ROW) [--on U V] [--height FU FV TU TV] [--track CSV --offset H]` | ground plane from 3+ pixels with known camera distance (H: a point that high above the floor, e.g. a ball's centre), or from the floor's horizon row and one such pixel; then any pixel on the floor -> 3D, distances along the floor, heights above it (foot + top pixel), or a whole track of an object moving on the floor (centre at height H) -> 3D csv for `fit` -- use it instead of writing your own plane geometry |
| `qpt flight --path CSV[:ID] T DIST [FROM TO] [--path ...] [--f F]` | focal length from free flight: a 3D path with \|g\| = 9.8 m/s^2 through each tracked object (one camera), anchored at its stated distance; `--f` tests a focal length you have |
| `qpt xyz --f F --pt U V DIST [--pt ...]` | 3D positions of image points whose camera distance you know (depth_info, or from `depth`), and the 3D distances between them |
| `qpt fit --csv FILE [--track N or --obj N] --at T / --avg T0 T1 / --accel T0 T1 [--scale U_PER_PX --unit m]` | speed / acceleration from positions, with several fit windows so you see how stable the number is |

SAM tools need the tool server; if `qpt seg` says it is unavailable, use `cut`,
`view`, `edges`, `motion` and `track`. Take speeds and accelerations, including those that
set the ruler or a focal length, from `qpt fit`, and 3D points from `qpt xyz` or `plane`:
they apply the asked-time convention, dropped frames and stated distances as ranges. Python
(numpy, scipy, opencv, pandas; save plots to files) and ffmpeg are for what no command covers.

## Method
1. `qpt info`, then `qpt frames` to see the whole clip. Name every object the questions and
   priors mention, and find each one (which instance, where, when).
2. The ruler. Measure the prior's object in pixels, the right instance, along the dimension
   the prior names, at a frame where that dimension is fully visible and square to the
   camera (prefer an instance lying in the targets' plane). Measure it two independent ways
   (a mask from `seg` or `cut` and `edges` endpoints, or two instances; a motion over a long
   and a short window; two masks, or two trackers on one window, share one error), in 3D
   also the focal length (two kinds of evidence, step 6), and start notes.md with `ruler:
   <object, dimension or motion, frames>: <value> / <value>` (3D: `f: <px> / <px>` with
   their sources). If they disagree by more than ~10%, find out why before going on: a wrong
   ruler makes every answer on the video wrong by the same factor. Scale = prior value /
   pixel length. The prior is the ONLY ruler. Never replace, average or "correct" the scale
   with typical real-world sizes of that kind of object, a recognised make or model, or the
   scale suggested in measurements.json.
3. Sizes: measure each target the same careful way and multiply by the scale. An edge is
   where the contrast is half way between object and background (`edges` reports the
   half-contrast extent). Never lower a threshold to make an object bigger: blur (a fast
   mover is smeared along its path: a mask includes the smear, `edges` half-contrast ends do
   not; else size it at rest if it ever rests, or where it moves slowly, else across the
   motion), glow, shadow, reflection and nearby objects are not the object, and a drawn
   outline counts to its middle line (ruler and targets alike). Do include both ends of the
   whole object (a hidden end does not stop at the cover: look beyond it, never measure
   another side instead) and thin parts that belong to it (a mast, a limb, a handle), and
   trim a mask's end only where a zoomed `view` shows it is not the object. Use one boundary
   per object for all its sizes, and place endpoints with `edges`, not by eye. Measure a
   rigid object between its own sides, not its slanted image box: an angled view adds depth,
   not size (a wheel's diameter lies across its face, not along the axle; a width runs side
   to side, not to a part nearer the camera; a top or bottom face in view adds no height).
   For small objects (under ~20 px) use `edges --sweep 4` along the asked dimension (a line
   off the centre measures a shorter chord) at several frames and take the median.
4. Motion: get the object's position in every frame of the asked window (`segtrack`,
   `motion` or `track`), check the overlay picture, then `qpt fit`. Instantaneous speed: a
   local fit around the asked time (`--at`), not a long average. Average speed: distance
   travelled / time over the asked window. Answer with the median `fit` prints across its
   windows, not one window you picked. Acceleration: a quadratic fit over a long window
   where it is plausibly constant (short windows mostly measure noise or a flapping, walking
   or spinning body's swing, which `fit` flags as a WOBBLE with the path's values; avoid
   fits that only reach one side of the asked time unless the track ends there). It is the
   magnitude of the acceleration vector (`fit` prints |acc|), which includes turning; the
   change of speed alone (a_tan) is only the part along the path. Two depth_info values at
   different times give an AVERAGE rate over that interval; use it as instantaneous only if
   the motion is smooth in between. A speed or acceleration used as a RULER must come from a
   stretch of smooth, free motion: read `fit`'s warnings (window-dependent values, irregular
   frame steps, a WOBBLE) and never fit across a bounce, a collision, a held frame, a
   dropped frame not given to `--gaps`, or the moment an object reappears from behind
   something.
5. Speed or acceleration priors: the scale comes from an object actually moving that way at
   that time: measure its pixel speed/acceleration from whole-object tracks (`segtrack` or
   `motion`) and check it with a second, independent track (within ~3%), scale = prior /
   pixel value: a speed with no time is the mean over its whole motion, an acceleration one
   quadratic over the whole accelerating stretch, a timed speed a local fit at that time.
   Only the sideways part of the motion shows as pixel speed: check the apparent-size trend
   `fit` prints (p %/s of size change = p % per second of distance change) and use sideways
   speed = sqrt(v^2 - v_depth^2). Lab recordings: never solve a focal length from motion;
   use the lab camera. Elsewhere a focal length solved from motion must be solved over 4+
   windows: write the spread. Spread under ~15%: use the median. Windows differing by more
   than ~1.5x mean the motion does not determine the focal length at all: do not use it (not
   even the median) -- use other evidence (objects of KNOWN size at stated distances, `qpt
   focal`, the floor, free flight).
6. 3D scenes: a pixel scale holds only at the ruler's distance. Apparent size is inversely
   proportional to distance: an object at distance d has scale s(d) = s_ruler * d / d_ruler.
   Motion toward or away from the camera is invisible in pixels; depth_info changing over
   time gives it (combine it with the in-image motion). `qpt depth` makes speeds and path
   lengths 3D from a rigid object's apparent size; use it only where the apparent size
   clearly changes (small changes are mostly noise); for an object rolling or sliding on a
   floor, `qpt plane --track --offset` (its centre's height) gives a steadier 3D track,
   above all for acceleration. Focal length in px = pixel size x depth along the camera axis
   / true size (give `qpt focal` the object's pixel U V and it converts a stated range); it
   is the same for every object in the clip, so a second object's estimate is a check. Lab
   recordings: one documented camera, f ~0.58 x width (`qpt` uses it by default). Other real
   cameras: horizontal field of view ~30-90 deg (a simulation's can be anything, ~8-103
   deg); outside it, check the clock (dropped frames), then the distances or sizes behind f
   (agreeing distances can come from another view): drop what does not fit; if nothing then
   gives f inside that range, use its nearest end. A focal length from motion (fragile when
   the motion is mostly toward the camera) must agree with one from the image (`qpt focal
   --obj`, or `--vp1 --vp2` on long parallel lines far apart, like a wall's top and bottom
   edges, never two close, nearly parallel lines; it says when f is ill-conditioned) or from
   free flight; else trust those. Lengths and distances in 3D: put each endpoint in 3D
   (pixel + its distance, `qpt xyz`) and take the 3D distance. Do not multiply a receding or
   oblique pixel length by one scale, nor use a length `plane` calls ILL-CONDITIONED (a
   grazing view): measure that dimension where it runs across the view. Check your camera
   model against EVERY stated depth_info distance (and the prior): if it cannot reproduce
   them within a few %, the model is wrong, never the prior. Check each stated distance
   against the image too, including the ruler object's own (where the object touches the
   ground, its apparent size, which object is in front of which). A distance the image
   contradicts is not used, nor the ruler's for a target without one: find its distance from
   the image (`qpt plane`, also `--horizon` from the vanishing point of long straight ground
   lines, two sets agreeing, or an approaching object's fixed row) and say so in notes.md.
   `qpt plane` anchors must be pixels whose OWN distance you know, on the floor or at a
   known height H -- a stated distance to an object's centre belongs to the centre pixel
   (`qpt xyz`, or `plane --pt U V DIST H` with its height above the floor), not to a floor
   pixel near it; read its anchor errors and horizon line before trusting it. In a
   simulation labelled 2D, objects visibly at different depths (on different floor rows, a
   base hidden behind nearer things, a top raised or leaning toward the camera -- looking
   down, anything above the floor) do not share one scale: measure a target at its own
   depth, or its part at the ruler's depth (seen from above: the edge at the ruler's height,
   not the longer upper one). In real footage free flight checks the camera and the clock (a
   simulation's motion can be keyframed and need not follow gravity unless the prior is
   gravity): a thrown, bouncing or falling object accelerates at 9.8 m/s^2 (`qpt flight`
   gives the focal length this implies, or tests yours with `--f`), and a drop from rest of
   height h takes t = sqrt(2h/g). A focal length or time axis that free flight does not fit
   is wrong, whatever it came from: the prior stays the ruler, the camera model changes.
7. One method per answer. Answer with the most direct measurement; other estimates are
   checks. If a check from a DIFFERENT method disagrees by more than ~10%, find which one is
   faulty and fix it -- do not average, weight or split the difference between methods.
   Repeats of the same method (several frames, several stable fit windows) may be combined
   by their median. Report at least 3 significant figures.
8. Plausibility: write one line per answer in notes.md -- "range check: <everyday range;
   agrees with the prior and the other answers, e.g. speed change = acceleration x time> ok
   / off". If an answer is far (more than ~1.5x) outside it, re-check the assumption behind
   it with DIFFERENT evidence (another object, a fall time, the floor geometry), not by
   re-measuring the same pixels. Several unrelated objects off by the same factor (even if
   each could be a mini version) means the shared ruler, its instance or the scale is wrong:
   re-check it with evidence that changes with it. Keep a value the independent check
   confirms; never move it toward typical. Simulations can be scaled arbitrarily (a giant
   cup, a tiny car): there the prior and depth_info set the scale and everyday ranges are
   only a hint.
9. If a named object is hard to find, search where and when it should be: `view --diff` /
   `--enhance` at several times, `motion --thr 10 --box ...` for faint movers. If it leaves
   the frame before the asked time, extrapolate its fitted motion to that time. If it truly
   cannot be measured, answer with the best estimate the clip supports -- the mover the
   question most plausibly means, or an object it moves with -- and say so in notes.md.
   Every question gets a positive number for the WHOLE quantity asked; never null, zero, a
   token value or one part of it (a range change alone is only the speed toward the camera).

## Dataset measurement conventions (from the dataset's README)
- Wingspan / width of flying animals: the fully spread, wing-tip-to-wing-tip extent,
  i.e. the maximum width attained during flight.
- Body length of animals (e.g. a prior "length of the small bird"): the object's
  axis-aligned bounding-box length in the simulator.
- Masked copies of a scene (background blacked out or flattened) keep the original's
  depth_info and prior, which may name objects masked out here.
- If the camera moves (a static object drifts in the image; masks can hide this), subtract
  the static object's track from every track, the ruler's included, before fitting.

## Default readings
- "Distance between A and B" (or from A to B) means between their centres, unless the
  question says gap, edge, surface or minimum distance.

## Reading the question
- The named object (or space: a road, a gap, a room) is the whole of it, never split
  into parts the image does not visibly divide; with several instances, the one the
  wording points to (position, role, colour or size against the others) at the asked time.
- A visible part boundary (a cap, stand, foot, lid) does not shorten the named object: its
  own stand or foot belongs to it. When an object and its support are both asked, every
  part goes to exactly one (a cap on the support, to the object above) so their sizes add
  up; one choice per object for all its questions.
- The time window is set by the words ("while moving", "during the jump", "from 1 s
  to 2 s"): leave out pauses the words exclude.
- Only if the WORDS still allow two readings that give different numbers, answer the
  smaller value and write both readings in notes.md. Never apply this to measurement
  uncertainty or to two ways of measuring one quantity (step 7), nor to parts, spaces or
  which instance a name means (rules above).

## How the true answers were made (the dataset's paper)
- Simulations (rendered): positions are each object's origin (a body's centre / pelvis, a vehicle's root); sizes are the
  object's box in the scene; depth_info is the straight-line range from the camera to that origin; a speed at t is the
  displacement over the frame ending at t; 2D clips keep the named objects at one distance from the camera. Renders need
  not obey real physics or real sizes.
- Lab recordings: one camera rig (~82 deg horizontal field of view), filmed at 25 fps and saved at 24 fps by dropping a
  frame once per second (after frames 11, 35, 59); sizes are tape-measured; depth_info is the depth sensor's distance at
  the object's centre; speeds come from the object's centre frame by frame. A question time past the clip's end refers to
  the original, longer recording: find the moment the question describes inside the clip; never extrapolate past it.
- Internet footage: labelled in the image plane with one scale from the prior (one metres-per-pixel factor for every
  object, no perspective correction); the camera may move.

## How the answer is scored
Relative error |answer - truth| / truth at 10 tolerance levels from 5% to 90%; within
5% scores 1.0, so precision pays. Give your best measured value; do not shift it up or down for scoring.

## Output (required)
Write `answers.json`: `{"<question id>": <number>, ...}` with every question id, each
a positive number in the unit the question asks for. Also write `notes.md` with 2-4
lines per question: what you measured, how, and the scale used.

Work only inside the task directory. Do not look for, open or download any other
files, datasets or answer keys, do not use the network, and do not spawn sub-agents.
