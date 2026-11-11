---
name: video-physics
description: Measure physical sizes, distances, speeds and accelerations of objects in a short video from one known reference quantity, using pixel-accurate measuring tools (qpt) and your own code. Use when a task gives video.mp4 plus questions.json asking for numeric physical quantities.
---

# Measuring physics from a video

You get one short video (`video.mp4`) and `questions.json`: video facts (fps, size, `source` -- simulation, lab
recording or internet footage, and whether it is a masked copy) and questions. Each question asks for ONE number in a
stated unit and gives a `prior` (one known quantity in the scene: the ruler that turns pixels into units), sometimes
`depth_info` (stated camera-to-object distances) and a `scene` (2D or 3D). `measurements.json`, if present, holds
automatic hints that are often wrong. Measure; you decide how. Look at the frames, zoom in, write code, check yourself.

## How the true answers were made (the dataset's paper)
- **Simulations** (rendered): positions are each object's origin (a body's centre / pelvis, a vehicle's root); sizes are
  the object's box in the scene; depth_info is the straight-line range from the camera to that origin; a speed at t is
  the displacement over the frame ending at t; "distance between A and B" is origin to origin. 2D clips keep the named
  objects at one distance from the camera, though an object visibly nearer or farther still has its own scale. Renders
  need not obey real physics or real sizes: motion can be keyframed (no 9.8 m/s^2 unless the prior says gravity), the
  camera's field of view can be anything (~8-103 deg), scenes can be scaled. The prior and depth_info are exact: they set
  the scale; everyday sizes are only a sanity hint.
- **Lab recordings**: one camera rig, ~82 deg horizontal field of view (f ~0.58 x image width; `qpt` uses it by default);
  filmed at 25 fps and saved at 24 fps by dropping a frame once per second (after frames 11, 35, 59: two intervals pass
  there and every interval is 40 ms; `qpt fit` applies this by default). Sizes are tape-measured; depth_info is the depth
  sensor's distance at the object's centre; speeds come from the object's centre frame by frame. Never solve the focal
  length from a motion prior or free flight here. A question time past the clip's end refers to the original, longer
  recording: find the moment the question describes inside the clip; never extrapolate past it.
- **Internet footage**: labelled in the image plane with one scale from the prior (one metres-per-pixel factor for every
  object, no perspective correction). The camera may move: check a static object before measuring motion.
- Masked copies (background removed or flattened) keep the original's prior and depth_info, which may name objects
  masked out here; a drawn outline counts to its middle line.
- Wingspan of a flying animal: its widest tip-to-tip extent during the flight; an animal's length: its box length.

## Tools: `qpt` (run in the task directory; `qpt <cmd> -h` for all options)
Coordinates are original video pixels (x right, y down); frames are original indices; t = frame / fps. Pictures go to
`qpt_out/`: open them. Python (numpy, scipy, opencv, pandas) and ffmpeg are there for anything else.

| command | use |
|---|---|
| `qpt info` | video facts, source notes, dropped/held-frame checks, each question's times as frames |
| `qpt frames [--n 12] [--from T0 --to T1]` / `qpt view --t T --box X0 Y0 X1 Y1 [--line ...] [--diff]` | contact sheet / zoomed crop with a pixel grid |
| `qpt edges --t T --line X1 Y1 X2 Y2 [--sweep W]` | sub-pixel ends of an object along a line placed across it (ends on the background just outside) |
| `qpt seg --t T --text "red ball"` or `--box ...` / `qpt cut ...` | object masks with sizes (SAM 3 / CPU) |
| `qpt segtrack --text "..."`, `qpt motion`, `qpt track --t T --box ...` | per-frame positions -> csv (whole-object tracks: segtrack, motion) |
| `qpt fit --csv F [--obj N] --at T / --avg T0 T1 / --accel T0 T1 [--gaps ...]` | speed / acceleration from positions over several windows; flags dropped frames and a WOBBLE (a flapping or walking body's swing) |
| `qpt focal --obj PX SIZE DIST [U V]` or `--vp1 ... --vp2 ...` | focal length from known sizes at known distances, or from vanishing points |
| `qpt xyz --pt U V DIST ...` / `qpt plane --pt U V DIST [H] ... [--horizon ROW] [--on U V] [--height ...] [--track CSV --offset H]` | 3D points from stated distances; a ground plane for floor points, heights and floor-bound tracks |
| `qpt flight --path CSV T DIST [FROM TO] [--f F]` | focal length from a free flight under gravity (real footage, or a gravity prior) |
| `qpt depth --csv F --obj N --anchor T DIST` | 3D track from a rigid object's changing apparent size |

## Essentials
1. **The ruler first.** Find the prior's object (right instance: position, role, colour, size against the others), along
   the dimension the prior names, where it is fully visible and square to the camera. Measure it two independent ways
   (a mask and `edges` endpoints, two instances, a long and a short motion window). A wrong ruler makes every answer
   wrong by one factor. The prior is the only scale: never replace it with typical sizes.
2. **Sizes:** the whole object between its own sides -- both ends (also an end hidden behind something), thin parts,
   its own stand or cap; not shadow, glow, reflection or motion smear (a mask includes the smear; `edges` half-contrast
   ends do not). An angled view adds depth, not size (a wheel's diameter lies across its face). Small objects: `edges
   --sweep`, several frames, median.
3. **Motion:** positions in every frame (whole-object tracks), then `qpt fit`. Instantaneous: a local fit at the asked
   time; average: distance / time over the asked window; acceleration: one quadratic over a long, smooth window (the
   magnitude of the acceleration vector). A motion prior: the prior object's pixel motion measured the same ways,
   checked by a second track within ~3%.
4. **3D:** scale is inversely proportional to distance. Put endpoints in 3D from their own distances (`xyz`, `plane`),
   never one pixel scale for a receding length. Your camera must reproduce every stated distance (and the prior).
   Distances to objects without one come from the image (`plane`, `--horizon`).
5. **Read the question:** the named object or space is whole (never split a road or a gap); the time window comes from
   the words; "distance between A and B" means centres unless the words say gap, edge or surface. Only when the WORDS
   allow two readings with different numbers, answer the smaller and note both -- never for measurement doubt, parts or
   which instance is meant.
6. **Check:** each answer against the prior and the other answers (speed change = acceleration x time; sizes against
   each other). Several unrelated objects off by one factor in real footage means the ruler is wrong. If a named object
   is hard to find, search frames, `view --diff`, `motion`; if it leaves the frame, extrapolate its fit; if it truly
   cannot be measured, give the best estimate from related evidence and say so.

## Output (required)
Write `answers.json`: `{"<question id>": <number>, ...}` with every question id, each a positive number in the unit
asked (at least 3 significant figures). Write `notes.md`: for each question, what you measured, how, and the scale.
Scoring: relative error at 10 levels from 5% to 90% (within 5% = full credit); give your best measured value.

Work only inside the task directory. Do not look for, open or download any other files, datasets or answer keys, do
not use the network, and do not spawn sub-agents.
