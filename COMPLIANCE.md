# Independence of predictions (rules clarification of Oct 15, and the organisers' email of Oct 20)

**In short.**
- Every scored prediction was produced from its own video's inputs only, by two runs on that video.
- Nothing was combined across videos or masked variants.
- Our agent answers all questions of one video in a single session, so we cannot claim strict per-question independence.
  Below we disclose this, and the one video where another question's prior set an answer.

## What we confirm
- **Own video only.** Each of the 3,012 predictions outside the validation families (the board scores 3,010 of them)
  comes from two agent sessions on its own video. Those sessions saw only:
  - the video file;
  - that video's questions: text, prior, depth_info, fps and scene type;
  - automatic hints computed from that video and those questions (`measurements.json`);
  - the skill.

  Their only outside connections were the local SAM 3 tool server (a unix socket) and an allow-list proxy to OpenAI's
  hosts (the list also names openrouter.ai, which these runs did not use). They had:
  - no other test video;
  - no validation video or label, and no answer key;
  - no internet.

  An automatic audit of every session's transcript (`code/scripts/codex_audit.py`) passed for all 1,046 sessions.
- **Two runs of the same question.** The final value is the geometric mean of two independent runs' answers to the same
  question, from the same inputs. The two runs differ only in the skill text (full and lean).
- **No combining across items.** No answers were combined across videos, or across masked variants of the same recording
  (Oct 20 email). No post-processing touched the values beyond the geometric mean.
- **Unscored rows.** The 277 rows of the 45 validation-overlap videos are excluded from scoring and were not part of
  these runs. Their values are left over from our earlier development runs, also produced from each item's own inputs
  only. They are kept only so the file rebuilds byte for byte; they do not affect the score.

## What we disclose
**A video's questions were answered in one session.** One session received all of a video's questions and wrote all
their answers. Its instructions ask it to check each answer against the prior and against that video's other answers (for
example, speed change = acceleration x time).
- **Same prior (519 of the 523 videos in these runs).** All of the video's questions share one prior.
- **Same prior, reworded (2 videos).** In simulation_0263 and simulation_0274 the two priors state the same fact in
  other words.
- **Different priors (2 videos, 14 rows).**
  - **simulation_0084:** question 108's prior is the yellow car's length (5.67 m), and question 102 asks for that length.
    - The full run measured 5.75 m from question 102's own prior and only compared it with 5.67.
    - The lean run answered 5.67 m from question 108's prior.
    - The submitted value is their geometric mean, 5.711 m.
    - The other rows of this video (103-108) used their own priors.
  - **simulation_0087:** question 123's prior is the bike's speed. The other questions' prior is the car's acceleration,
    which question 123 asks for. Both runs measured question 123 with its own prior (4.036 and 4.044 against the stated
    4.03) and only noted the agreement. The other rows (124-129) used their own prior.

If useful, we can rerun the affected rows, or the whole test set, with one session per question using the same code, so
the organisers can measure the effect.
