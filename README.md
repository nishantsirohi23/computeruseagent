# macOS computer-use agent trained with GRPO (T4 trains, Mac runs)

A Qwen3.5-4B vision model that operates macOS through screenshots + mouse/keyboard, trained in two
GRPO stages on a free T4 (Colab or Kaggle) with rewards checked automatically on this Mac:

1. **Grounding**: "Click the "Save" button" → one `left_click` at relative (x, y). Reward: the click lands
   inside the target's Accessibility (AX) frame (GUI-G1 recipe adapted to points).
2. **Multi-step**: given task, action history and current screenshot, produce the next action. Reward:
   matches the verifiable next action recorded while an AX oracle solved random training tasks.

Done when, on the 50-task benchmark in `cua/tasks/registry.py`, the trained model beats the same
base model with prompting alone (`scripts/compare.py` prints both and a paired McNemar test).

```
 Mac (M-series)                                            T4 (Colab / Kaggle)
 ───────────────────────────────────────────────           ─────────────────────────────────
 cua/env.py      AX tree + screencapture + CGEvent          notebooks/01_grounding_grpo_t4
 cua/tasks/*     tasks: setup / check / oracle      ──HF──▶   Unsloth Qwen3.5-4B, 16-bit LoRA,
 scripts/collect_grounding.py  (stage-1 data)      dataset    TRL GRPO (GSPO, dr_grpo)
 scripts/collect_steps.py      (stage-2 data)                notebooks/02_multistep_grpo_t4
 scripts/merge_lora.py  ◀──── LoRA adapter (HF model repo) ── pushes adapter only
 scripts/run_bench.py   MLX inference, 50 tasks, checkers
```

## What the model sees and does

Same prompt at train and test time (`cua/prompts.py`): Qwen3.5's own `computer_use` tool schema
(rendered byte-identically to the chat template's `tools=` output, see `tests/test_shared.py`),
thinking off, one user turn = full main-display screenshot (resized to ~1000 visual tokens) +
task + numbered history of previous actions. Output: optional one-line thought + one tool call
in Qwen3.5's XML format. Coordinates are relative 0-1000. Actions: `left_click`, `double_click`,
`triple_click`, `right_click`, `left_click_drag`, `mouse_move`, `type`, `key`, `scroll`, `wait`,
`terminate`.

## The benchmark (50 tasks, 4 apps)

| App | Tasks | Checked by |
|---|---|---|
| SynthApp (native AppKit forms generated from a seed: text fields, checkboxes, radios, pop-ups, sliders, steppers, segmented controls, lists, tabs, menus) | 20 | app writes its exact widget state + submissions to JSON |
| Calculator (typed and button-only arithmetic, decimals, clear) | 10 | display value via AX |
| TextEdit (append, insert, replace word, delete line, clear, save, Transformations menu) | 10 | AX text value or the saved file |
| Finder (new folder, rename, duplicate, move, open, select, list view, folder-in-folder) | 10 | filesystem / Finder AppleScript |

Training tasks come from the same generators with different seeds; any instruction identical to a
benchmark instruction is rejected. Three templates (`textedit:uppercase`, `calc:clear`,
`finder:new_folder_inside`) never appear in training and are reported separately.

## Safety

Runs take over the mouse and keyboard of the **main display**, so do not use the Mac during a run.

- **Abort**: move the mouse (anything over 40 pt from where the agent left it), or push it to the
  top-left corner.
- **Confined to the task app**: every click is hit-tested via AX first and dropped unless it lands on
  the task app's own process. Finder tasks are further restricted to the task window and menus, so
  the Desktop and your other Finder windows are out of reach.
- **Keyboard**: keystrokes only go out while a task app is frontmost.
- **Blocked shortcuts**: Spotlight, app switcher, logout, force quit, screenshots, move-to-trash,
  hide, minimize and menu-bar keyboard access.
- **Apple menu**: every app's menu bar carries the Apple menu, with Restart, Shut Down and Log Out,
  so the Apple menu and those items are blocked explicitly.
- **Notifications**: turn on Do Not Disturb during runs. A notification banner can cover the
  top-right of the main display, and clicks on it are (correctly) blocked.
- **Your windows**: other apps are hidden during a run and unhidden afterwards. Your own TextEdit and
  Finder windows are minimized and restored.
- **Files**: everything is created under `~/CUABench`.

## Setup (Mac)

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -e . pytest
```

Grant **Accessibility** (System Settings > Privacy & Security > Accessibility) to the app that runs
the scripts (Terminal, iTerm, or Claude). Screen Recording is also required for `screencapture`.

## Pipeline

```bash
# 0. sanity: tasks are solvable and not pre-solved
python scripts/run_bench.py --oracle --out runs/oracle      # expect 50/50
python scripts/run_bench.py --noop   --out runs/noop        # expect 0/50

# 1. baseline: base model, prompting only (optionally also --thinking --out runs/base_thinking)
python scripts/run_bench.py --model Qwen/Qwen3.5-4B --out runs/base

# 2. stage-1 data -> HF (private)
python scripts/collect_grounding.py --synth 150 --real 36 --out data/grounding/run1
python scripts/export_dataset.py --kind grounding --src data/grounding --repo YOU/cua-macos-grounding --local_only
python scripts/eval_grounding.py --model Qwen/Qwen3.5-4B --split train --write data/base_train_hits.jsonl  # optional
python scripts/export_dataset.py --kind grounding --src data/grounding --repo YOU/cua-macos-grounding --easy_from data/base_train_hits.jsonl

# 3. T4: run notebooks/01_grounding_grpo_t4.ipynb (set HF_DATASET / HF_OUT, add HF_TOKEN secret)

# 4. stage-2 data -> HF
python scripts/collect_steps.py --episodes 300 --out data/steps/run1 --seed 1
python scripts/export_dataset.py --kind steps --src data/steps --repo YOU/cua-macos-steps

# 5. T4: run notebooks/02_multistep_grpo_t4.ipynb (starts from the stage-1 adapter)

# 6. merge the adapter into the bf16 base on the Mac, evaluate, compare
python scripts/merge_lora.py --adapter YOU/cua-qwen35-4b-agent --out models/agent
python scripts/run_bench.py --model models/agent --out runs/trained
python scripts/compare.py runs/base runs/trained
```

`python scripts/eval_grounding.py --model models/agent --split val` gives the offline click accuracy
on held-out screens without controlling the desktop. Benchmark runs resume if interrupted.

## Validation and baseline (this Mac, 2026-10-28)

| Run | Overall | SynthApp | Calculator | TextEdit | Finder | Held-out templates |
|---|---|---|---|---|---|---|
| AX oracle (`--oracle`) | 50/50 | 20/20 | 10/10 | 10/10 | 10/10 | 4/4 |
| No-op (`--noop`) | 0/50 | 0/20 | 0/10 | 0/10 | 0/10 | 0/4 |
| Qwen3.5-4B base, prompting only, thinking off (`runs/base`) | **11/50 (22%)** | 5/20 | 1/10 | 2/10 | 3/10 | 1/4 |

The base model grounds single targets well (it hit all 3 live click checks on SynthApp), but as an agent:

- **It never terminates**: 0/50 `terminate` calls, so every episode runs to `max_steps` (12 steps on average).
- **It loops**: on most tasks one identical action repeats 6-16 times. In Calculator it enters the first
  digits correctly, then clicks the same key until time runs out.
- **It writes no reasoning**: there is no text before its tool calls.

That is the target for stage 2.

MLX latency is about 4 s per step: ~2,000 prompt tokens and ~48 generated.

### Collected training data (2026-10-28)

| Dataset | Rows (train / val) | Groups | Size | Notes |
|---|---|---|---|---|
| Grounding (`data/export_grounding`) | 5,125 / 459 | 310 screens | 34 MB | 3,613 SynthApp, 720 Calculator, 720 Finder, 531 TextEdit; median target is 0.09% of the screen |
| Steps (`data/export_steps`) | 883 / 76 | 198 oracle episodes | 96 MB | 562 clicks, 94 type, 92 key, 198 terminate |

Collection covers the Desktop with a neutral backdrop (`cua/backdrop.py`). It only takes Finder targets
from the task window, and it drops any label containing the user's name, username or device names.
Screens are stored once under `images/`, and each parquet row holds a relative path.

## Training details

- Model/runtime: `unsloth/Qwen3.5-4B`, 16-bit LoRA r=16 on language layers, `fast_inference=False`
  (vLLM is not supported for Qwen3.5 RL), pins from Unsloth's `Qwen3_5_(4B)_Vision_GRPO` notebook
  (trl 0.22.2, transformers 5.2.0). TRL does not forward chat-template kwargs, so the notebooks wrap
  the processor to default `enable_thinking=False`.
- GRPO: 4 completions per prompt, GSPO sequence-level ratios, `loss_type="dr_grpo"` (no length
  normalization, as GUI-G1 recommends), beta 0 (no reference model), max 128-160 completion tokens.
- LoRA learning rate is 1e-4, with weight decay 0 and gradient clip 1.0. The first stage-1 run used
  1e-5 with weight decay 0.1. Its update was ||dW||/||W|| ~ 3e-4, too small to change a single greedy
  click: 62.7% before and after, after 250 steps and 3.15 h on a T4. Both notebooks now print the reward
  trend and the LoRA update size right after training, so a no-op run is caught immediately.
- Stage-1 reward (`cua/rewards.py`): `hit + 0.25 * gaussian proximity`, times a difficulty weight
  (1.0 for large targets up to 1.5 for tiny ones). `scale_rewards="none"` keeps that weight in the
  advantage, which is how GUI-G1's difficulty coefficient enters. Format reward +0.1 / -0.5.
- Stage-2 reward: 0.3 for the right action type + 0.7 for the right argument (point inside the AX box,
  exact text, same key set, terminate), max over listed alternatives.
- `tests/smoke_trl_grpo.py` runs one GRPO step of each stage on CPU with a tiny random Qwen3.5 and the
  notebook pins, checking that TRL renders exactly the agent's prompt.

## Layout

```
cua/        config, prompts/actions/rewards (shared with notebooks), ax, screen, input, apps, env,
            policy (MLX), runner, synthapp/ (AppKit form app + generators), tasks/ (4 families + registry)
scripts/    collect_grounding, collect_steps, export_dataset, eval_grounding, run_bench, compare, merge_lora
notebooks/  build_notebooks.py -> 01_grounding_grpo_t4.ipynb, 02_multistep_grpo_t4.ipynb
tests/      test_shared.py (pytest), smoke_trl_grpo.py
```

## Known limitations

- Stage 2 trains on states from clean oracle trajectories: it does not see recovery-from-mistake
  states. Fully online multi-turn RL would need the Mac to roll out the current policy each iteration
  (possible with `run_bench.py`-style rollouts + a custom update loop). That is the natural next step.
- One screen layout (this Mac's main display); the model is not expected to transfer to other
  resolutions or display orientations without new data.
