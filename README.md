# Code42: QuantiPhy Challenge, Track A (main track). Reproduction package

- **Submission reproduced:** `predictions/sub21g_geomean.csv`, sha256
  `f8fc611cbb89284d148ca60fdcf524edd5b344412daed6bb1efe1a97e71e9cb6`. Uploaded to Track A as `sub21g_geomean.csv` on
  21 Oct 2026, between 10:17 and 10:55 UTC by our records; MRA 0.8285 (shown as 0.829 on the board).
- **Method (one page):** `METHOD.md`. **Independence statement:** `COMPLIANCE.md`.
- **Rebuild check:** `code/scripts/build_tracka.py` rebuilds the submitted file byte for byte. It can read the answer
  tables in `answers/`, or the kept sessions' `answers.json` files (the sessions are available on request).
- **This public copy** is the package we sent the organisers for verification, with machine-specific settings made
  portable and an MIT licence (`LICENSE`) added. Portable means: file locations are settings (see Setup); the
  containers run as your own user with their own home folder, and the audit allows the matching paths; the cache-key
  prefix is fixed; and the OpenRouter key is read only from `OPENROUTER_API_KEY`. The method, skills, prompt, inputs
  and answers are unchanged.

## What the file is
- **3,012 rows (523 videos) outside the 24 validation families.** The board scores 3,010 of them; it also excludes 2
  duplicate questions. Every one of these videos got two independent agent runs:
  - GPT-6.1 Sol with skill v20-full;
  - GPT-6.1 Sol with skill v20-lean.

  Each run is one session per video. The value of each row is the geometric mean of the two runs' answers to that
  question.
- **277 rows (45 videos)** belong to the validation-overlap families, which the organisers excluded from scoring.
  These runs did not cover them. The file holds values left over from our earlier development runs for these rows
  (`answers/unscored_rows.csv`), each also produced from its own item's inputs. They are kept only so the file rebuilds
  byte for byte; they do not affect the score.

## Contents
| path | what |
|---|---|
| `code/agent/skill/video-physics/` | skill v20-full: `SKILL.md` (3,643 words) and `qpt/` measuring tools. Snapshot hash `f03257ea46c2` |
| `code/agent/skill_lean/video-physics/` | skill v20-lean: `SKILL.md` (1,306 words), same `qpt/` tools. Snapshot hash `73f5591bc439` |
| `code/agent/toolserver/` | `sam_server.py`: the SAM 3 segmentation and tracking server the tools call. `run.sh` starts it |
| `code/agent/sandbox/` | the sealed container's network: allow-list HTTPS proxy (`allow_proxy.py`), `netrelay.sh`, `entry.sh` |
| `code/scripts/pi_run.sh` | runs ONE video: a fresh `ubuntu:22.04` container with `--network none`, holding Pi and the skill; GPT-6.1 Sol is reached through the allow-list proxy |
| `code/scripts/pi_pool.py` | runs many videos' sessions, N at a time |
| `code/scripts/agent_stage.py` | writes one task folder per video: `video.mp4`, `questions.json`, `measurements.json` |
| `code/scripts/build_test_evidence.py`, `code/qp/` | build the `measurements.json` hints: rule-based parsing plus SAM 3, no language model. Needed only to rebuild the hints, which the package ships in `inputs/` |
| `code/scripts/codex_audit.py` | audits each session's transcript: no network use, no paths outside the sandbox |
| `code/scripts/build_tracka.py` | per-question geometric mean of the two runs, written into the portal's submission template |
| `code/scripts/opt_guard.py` | the guard run on every skill change: flags video ids, validation answer values and video-specific phrases |
| `code/runs/agents/v20_test_sets.json` | the 523 scored videos, split into the batches that were run (`lab`, `sim60`, `inet20`, `rest`) |
| `code/submission_template.csv` | the test questions as our code reads them (the dataset's template, as in our repository) |
| `code/data/portal_submission_template.csv` | the portal's submission template; the submitted file was written into this copy |
| `config/` | the prompt (`prompt.md`) and Pi retry settings |
| `inputs/evidence_test_v9.csv` | the hints exactly as used, so staging does not need the SAM pipeline re-run |
| `answers/` | each run's answers (`answers_full.csv`, `answers_lean.csv`) and the values of the 277 unscored rows (`unscored_rows.csv`) |
| `predictions/` | the submitted CSV |
| `requirements/` | pinned Python packages (agent tools; SAM server) and Pi's `package-lock.json` |
| `LICENSE` | MIT licence |

The code is taken from our repository at commit `b6c01e0`, the state right after the run. `scripts/build_tracka.py` was
written afterwards to rebuild the file. The skills are the exact content-addressed snapshots the sessions mounted; each
session records the hash in its `skill_version.txt`.

## Models and settings
| role | model | settings |
|---|---|---|
| agent (all 3,012 scored answers) | OpenAI **GPT-6.1 Sol**, API name `gpt-6.1-sol` | Pi coding agent 1.2.0 (`@earendil-works/pi-coding-agent`), provider `openai`, signed in with a ChatGPT account (Pi `/login`); reasoning effort (`--thinking`) **max**; provider-default sampling, no temperature or seed set; built-in tools only (read, bash, edit, write); retries per `config/pi_settings.json` |
| segmentation and tracking (tool) | SAM 3, https://huggingface.co/facebook/sam3, revision `3c879f39826c281e95690f02c7821c4de09afae7` | local GPU, bfloat16 |

No other model was called by the v20 runs. Hints use rule-based question parsing plus SAM 3; `qp/vlmplan.py` and
`qp/vlmground.py` exist but were off when the hints were built.

## Hardware and runtime
- **Hardware:** NVIDIA RTX 5090 GPUs (32 GB). SAM 3 runs as a server process on a GPU. GPT-6.1 Sol ran remotely
  (OpenAI).
- **Runtime:** both runs went in parallel on Oct 20 2026, 16:57-22:56 UTC: about **6 hours** for 2 x 523 sessions, with
  2-30 sessions at a time per batch.
  - Sessions took a median of 13 minutes (90th percentile 25 minutes).
  - All 1,046 sessions answered every question and passed the audit.

## Setup
1. **Software:**
   - Docker; NVIDIA driver with CUDA 13; Node.js 24.21.0 (on `PATH`, or set `NODE_BIN`; a real binary, not a version-manager shim); `socat` and `python3` on
     the host. `pi_run.sh` mounts the host's `/usr` into each container, and the sandbox relay uses `socat`.
   - Python 3.11.16 for the tools: a virtual environment at `code/.venv` from `requirements/agent-venv.txt`, made
     with a uv-managed Python. The containers mount the uv Pythons at their own path (`UV_PYTHON_DIR`), so the
     environment's links still resolve.
   - Python 3.12 for the SAM server: a virtual environment at `code/.venv-vllm` from `requirements/sam-server-venv.txt`.
     That file pins torch 2.13.0+cu130 (install with `--extra-index-url https://download.pytorch.org/whl/cu130`) and
     transformers 5.15.1.
   - `agent/toolserver/run.sh` sources `code/env.sh`. Create it, even empty, or with `HF_HOME` if your Hugging Face
     cache is elsewhere.
   - Pi 1.2.0: copy `requirements/pi-package.json` and `requirements/pi-package-lock.json` into an empty folder as
     `package.json` and `package-lock.json` in `code/.work/pi` (the default `PI_HOME`), then run `npm ci` there.
   - SAM 3 weights: `huggingface-cli download facebook/sam3 --revision 3c879f39826c281e95690f02c7821c4de09afae7`. The
     server then runs offline (`HF_HUB_OFFLINE=1`).
2. **Sign Pi in to OpenAI:** start `pi`, run `/login` and choose OpenAI (ChatGPT account). Copy
   `config/pi_settings.json` into `~/.pi/agent/settings.json` (or the folder you set as `PI_AGENT_DIR`).
3. **Locations:** each machine-specific location is a setting with a default; override any of them with an
   environment variable.

   | setting | default |
   |---|---|
   | `QP_WORK`: working folder | `code/.work` |
   | `VENV`: the tools' virtual environment | `code/.venv` |
   | `PI_HOME`: the Pi install (the folder where you ran `npm ci`) | `$QP_WORK/pi` |
   | `PI_AGENT_DIR`: Pi's login and settings | `~/.pi/agent` |
   | `UV_PYTHON_DIR`: the uv-managed Pythons the virtual environment links to | uv's own: `$UV_PYTHON_INSTALL_DIR`, else `~/.local/share/uv/python` |
   | `NODE_BIN`: Node.js | `node` on `PATH` |
   | `NETSOCK_DIR`: relay sockets | `$QP_WORK/netsock` |
| `LLM_HOST`: a local model server for the relay (not used for Track A) | `127.0.0.1` |
   | `SKILL_SNAPSHOTS`: frozen skill copies | `$QP_WORK/skill_snapshots` |
   | tool-server socket folder | `$QP_WORK/toolsrv` for `run.sh`; pass the same folder to `pi_pool.py --toolsrv` |

   The agent containers run as your own user, with `HOME=/home/agent` inside.

   `code/` is the repository root. Put the test videos in `code/data/QuantiPhy-test/` (see `qp/data.py`).

## Run (from `code/`)
We ran the two skills and the four batches as parallel pools; the loop below runs them one after another.
```bash
agent/sandbox/netrelay.sh start                       # allow-list proxy + relay sockets for the sandbox
agent/toolserver/run.sh start 0                       # SAM 3 server on GPU 0; socket in .work/toolsrv
mkdir -p runs/ev_v9 && cp ../inputs/evidence_test_v9.csv runs/ev_v9/
for var in full lean; do
  SK=agent/skill/video-physics; [ $var = lean ] && SK=agent/skill_lean/video-physics
  for set in lab sim60 inet20 rest; do
    python scripts/agent_stage.py --split test --out RUN/$var/$set --no-guide --videos-json runs/agents/v20_test_sets.json $set
    python scripts/pi_pool.py --root RUN/$var/$set --prompt ../config/prompt.md --models openai/gpt-6.1-sol:30 \
        --thinking max --skill $SK --toolsrv .work/toolsrv --order hard-first
  done
done
python scripts/codex_audit.py RUN/*/*/*                # every session: OK
python scripts/build_tracka.py --full-tasks RUN/full/*/* --lean-tasks RUN/lean/*/* \
    --unscored ../answers/unscored_rows.csv --out sub21g.csv
```
**Rebuild the submitted file from the kept answers** (no model calls):
```bash
python scripts/build_tracka.py --full ../answers/answers_full.csv --lean ../answers/answers_lean.csv \
    --unscored ../answers/unscored_rows.csv --out sub21g.csv   # sha256 f8fc611c...
```

**Determinism.** GPT-6.1 Sol is sampled, so a fresh run gives slightly different numbers. On the 24 labelled validation
videos, the two skills scored 0.8165 (lean) and 0.810 (full) macro MRA. The submitted file is exactly the geometric mean
of the two kept runs. We keep every session's transcript, tool calls, pictures, scripts and notes (62 GB) and can share
them on request.
