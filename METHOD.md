# Method (Code42, Track A): a frontier model as a measuring agent, two runs per video

**Idea.** Vision-language models guess physical quantities poorly from frames alone. They do much better when they act
like a careful analyst:
- look at the frames;
- measure pixels precisely with tools;
- turn pixels into units with the given prior and depth_info;
- check the physics.

We give GPT-6.1 Sol a sandbox, a written measuring method (a "skill") and a toolbox (`qpt`). One agent session answers
the questions of one video. We run each video twice, with two versions of the skill, and submit the per-question
geometric mean.

**Per video.**
- **Sandbox:** a fresh Docker container with no network, except an allow-list proxy to the model API and a socket to
  the local SAM 3 tool server.
- **Inputs:** that video, its questions (with priors, depth_info and scene type) and optional automatic hints (SAM 3
  measurements).
- **The agent:** Pi coding agent with reasoning effort max. It reads the skill, runs tools and its own Python, and looks
  at the pictures it makes.
- **Output:** `answers.json`, plus `notes.md` explaining every number.
- **Effort:** a median of 13 minutes and 63 tool calls per session (80% of sessions use 37-103 calls).
- **Audit:** afterwards, every session is checked for network use and paths outside the sandbox.

**The skill (v20).** Two versions with the same tools:
- **Full:** 3,643 words.
- **Lean:** 1,306 words; a condensed version of the same method.

Both state the following.
1. **How the dataset's true answers were made, per source.** Mainly from the QuantiPhy paper; we found the lab
   frame-drop pattern in the videos.
   - Simulations: object origins and world boxes; range to the origin; speed over the frame ending at t; scenes may be
     scaled.
   - Lab recordings: one camera rig with about 80° field of view; 25 fps captures saved at 24 fps by dropping a frame
     each second; tape-measured sizes; depth-sensor ranges.
   - Internet footage: one planar scale.
2. **A measuring method:**
   - Measure the ruler (the prior's object) two independent ways.
   - Sizes are edge to edge, without shadow or blur.
   - Motion comes from whole-object tracks and fitted speeds and accelerations, never one frame step.
   - 3D points come from stated distances.
   - Read the question's words exactly.
   - Check each answer against the prior and for physical plausibility.

**The tools** (`qpt`, about 1,900 lines of Python):
- **Frames:** contact sheets and zoomed views with pixel grids.
- **Edges and masks:** sub-pixel edge ends along a line; SAM 3 masks and video tracks from a GPU server.
- **Motion:** speed and acceleration fits over several windows. They apply the lab clock (dropped frames) and flag
  wobbling bodies.
- **Camera geometry:** focal length from known sizes, vanishing points or free flight (the lab camera's focal length is a
  default); 3D points from stated distances; ground planes for floor points and heights.

**Combining the two runs.** Both runs answer the same question from the same inputs. The final value is
`exp((ln a_full + ln a_lean) / 2)`. This step uses only the two runs' answers to that question, nothing from any other
question, video or masked variant. Within a session the agent sees all of its video's questions (see COMPLIANCE.md).

**Development.**
- **Method:** the skill was developed over about 20 versions.
  - Each version was checked on the 24 labelled validation videos and by reviewing transcripts of test sessions (with
    no labels).
  - Some skill choices were also compared on the board, through uploads that changed only a subset of videos.
  - Every skill change is general: there are no per-video rules, and a guard (`code/scripts/opt_guard.py`) checks each
    change for video ids or answer values.
- **Source of the conventions:** the QuantiPhy paper's description of how labels were made (Appendix C/D), the dataset
  README, and our own inspection of the videos.

**Results.**
- **Validation (24 videos, real labels):** lean 0.8165 and full 0.810 macro MRA.
- **Board:**

  | run | MRA |
  |---|---|
  | full | 0.824 |
  | lean | 0.826 |
  | per-question geometric mean (this submission) | 0.8285 |

  The geometric mean gives S2 0.852, D2 0.857, S3 0.797 and D3 0.808.
