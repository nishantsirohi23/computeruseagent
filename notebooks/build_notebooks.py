"""Generates the two T4 training notebooks. Run: python notebooks/build_notebooks.py

Cells follow Unsloth's Qwen3_5_(4B)_Vision_GRPO.ipynb (install pins, FastVisionModel,
16-bit LoRA, no vLLM, GSPO + dr_grpo), with our data, prompts and rewards.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


INSTALL = r'''
%%capture
# Same install as Unsloth's Qwen3_5_(4B)_Vision_GRPO notebook (works on a free Colab T4 and on Kaggle T4).
import os, importlib.util
!pip install --upgrade -qqq uv
if importlib.util.find_spec("torch") is None or "COLAB_" in "".join(os.environ.keys()):
    try: import numpy, PIL; _numpy = f"numpy=={numpy.__version__}"; _pil = f"pillow=={PIL.__version__}"
    except: _numpy = "numpy"; _pil = "pillow"
    !uv pip install -qqq \
        "torch==2.8.0" "triton>=3.3.0" {_numpy} {_pil} torchvision "torchaudio==2.8.0" bitsandbytes xformers==0.0.32.post2 \
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
        "unsloth[base] @ git+https://github.com/unslothai/unsloth"
    !uv pip install -qqq --no-deps "torchcodec==0.7.0"
elif importlib.util.find_spec("unsloth") is None:
    !uv pip install -qqq unsloth
!uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
!unsloth install-kernels
!uv pip install --no-deps --upgrade "torchao>=0.16.0"
'''

LOGIN = r'''
# Hugging Face token with write access: Colab -> add a secret named HF_TOKEN; Kaggle -> Add-ons > Secrets > HF_TOKEN.
import os
def _hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from google.colab import userdata
        return userdata.get("HF_TOKEN")
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None
os.environ["HF_TOKEN"] = _hf_token() or ""
from huggingface_hub import login, snapshot_download, HfApi
login(token=os.environ["HF_TOKEN"])
'''

DATA = r'''
# The Mac-side exporter (scripts/export_dataset.py) uploads the parquet shards, meta.json, and the exact
# prompt/action/reward code the agent uses at inference (cua_shared/), so training and eval cannot drift apart.
import sys, json
local = snapshot_download(HF_DATASET, repo_type="dataset", local_dir="cua_data")
sys.path.insert(0, local)
from cua_shared import prompts, actions, rewards
meta = json.load(open(f"{local}/meta.json"))
print(meta)

import os
from datasets import load_dataset, Image
ds = load_dataset("parquet", data_files={
    "train": f"{local}/data/train-*.parquet",
    "val": f"{local}/data/val-*.parquet",
})
# Screens are stored once under images/; rows hold relative paths.
ds = ds.map(lambda ex: {"image": os.path.join(local, ex["image"])}).cast_column("image", Image())
print(ds)
SYSTEM = prompts.system_prompt()
'''

MODEL = r'''
from unsloth import FastVisionModel
import torch
max_seq_length = 16384 # Must be this long for VLMs
lora_rank = 16

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen3.5-4B",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # 16-bit LoRA: Unsloth advises against QLoRA for Qwen3.5
    fast_inference = False, # vLLM is not supported for Qwen3.5 RL
)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = False,
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = lora_rank,
    lora_alpha = lora_rank,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    use_gradient_checkpointing = "unsloth",
)

# Qwen3.5 thinks by default. The agent runs with thinking off (GUI-G1: long reasoning hurts grounding),
# and TRL 0.22.2 does not forward chat-template kwargs, so make enable_thinking=False the default here.
_apply_chat_template = tokenizer.apply_chat_template
def _apply_chat_template_no_think(*args, **kwargs):
    kwargs.setdefault("enable_thinking", False)
    return _apply_chat_template(*args, **kwargs)
tokenizer.apply_chat_template = _apply_chat_template_no_think
'''

GENERATE = r'''
import random
def generate(image, instruction, history, max_new_tokens=160):
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts.user_text(instruction, history)}]},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(images=[image], text=[text], add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
'''


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


GROUNDING = notebook([
    md("""
# Stage 1: GRPO for click grounding on macOS screenshots (Qwen3.5-4B, T4)

Data comes from the Mac (`scripts/collect_grounding.py` then `scripts/export_dataset.py`): full-screen macOS
screenshots, an instruction like *Click the "Save" button*, and the target element's box from the
Accessibility API. The model answers with the same `computer_use` tool call it uses as an agent, so this
stage trains the exact skill the agent needs.

Reward (GUI-G1 recipe, adapted to point clicks): hit reward + small proximity shaping, multiplied by a
difficulty weight for small targets, no length normalization (`loss_type="dr_grpo"`), no reward std
scaling (so the difficulty weight actually changes the advantage), fast-thinking (no `<think>`).

Run all on a T4. Set the two repo names in the config cell first.
"""),
    code(INSTALL),
    code(r'''
HF_DATASET = "nishantsirohi2307/cua-macos-grounding"   # created by scripts/export_dataset.py --kind grounding
HF_OUT     = "nishantsirohi2307/cua-qwen35-4b-grounding" # LoRA adapter goes here (private model repo)
MAX_STEPS  = 250
EVAL_N     = 150
'''),
    code(LOGIN),
    code(DATA),
    code(MODEL),
    md("### Dataset → GRPO prompts\nTRL adds the image placeholder in front of the user text, exactly as the Mac agent does."),
    code(r'''
def to_prompt(ex):
    return {"prompt": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompts.user_text(ex["instruction"], [])},
    ]}
train_dataset = ds["train"].map(to_prompt).shuffle(seed=0)
val_dataset = ds["val"]
print(train_dataset[0]["instruction"], train_dataset[0]["bbox"])
'''),
    code(GENERATE),
    code(r'''
def eval_grounding(rows, n):
    FastVisionModel.for_inference(model)
    rows = rows.select(range(min(n, len(rows))))
    hit, fmt = 0, 0
    for ex in rows:
        a = actions.parse_action(generate(ex["image"], ex["instruction"], []))
        fmt += a is not None
        hit += bool(a is not None and a.coordinate is not None and rewards.in_box(a.coordinate, ex["bbox"]))
    FastVisionModel.for_training(model)
    return {"click_acc": hit / len(rows), "valid_call": fmt / len(rows), "n": len(rows)}

print(generate(val_dataset[0]["image"], val_dataset[0]["instruction"], []))
before = eval_grounding(val_dataset, EVAL_N)
print("before GRPO:", before)
'''),
    md("### Train"),
    code(r'''
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 1e-4,             # LoRA from zero needs ~10x a full fine-tune LR; 1e-5 left the model unchanged
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.0,               # decay pulls the LoRA back to zero
    warmup_ratio = 0.05,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,   # 2 x 2 = one group of 4 completions per step
    num_generations = 4,               # lower to 2 if you hit OOM
    max_prompt_length = 4096,          # ~1000 image tokens + ~1000 system/tool tokens; never truncate
    max_completion_length = 128,       # one short sentence + one tool call
    temperature = 1.0,
    max_steps = MAX_STEPS,
    save_steps = 50,
    max_grad_norm = 1.0,
    report_to = "none",
    output_dir = "outputs_grounding",
    # GSPO sequence-level ratios as in the Unsloth notebook; GUI-G1: no length normalization, keep the
    # difficulty weight in the advantage.
    importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = "dr_grpo",
    scale_rewards = "none",
)
trainer = GRPOTrainer(
    model = model,
    args = training_args,
    processing_class = tokenizer,
    reward_funcs = [rewards.format_reward, rewards.grounding_reward],
    train_dataset = train_dataset,
)
# After a Colab disconnect, re-run all cells and use trainer.train(resume_from_checkpoint=True).
trainer.train()
'''),
    code(r"""
# Did training actually move the model? Reward trend, plus the size of the LoRA update.
import numpy as np
h = [x for x in trainer.state.log_history if "reward" in x]
def avg(rows, k):
    v = [x[k] for x in rows if k in x]
    return float(np.mean(v)) if v else float("nan")
for k in ["reward", "rewards/grounding_reward/mean", "frac_reward_zero_std", "completions/mean_length"]:
    print(f"{k:34s} first 25 steps = {avg(h[:25], k):.3f}   last 25 steps = {avg(h[-25:], k):.3f}")
rel = []
for mod in model.modules():
    if hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and "default" in mod.lora_A:
        d = (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()) * mod.scaling["default"]
        rel.append((d.norm() / mod.base_layer.weight.float().norm()).item())
print(f"LoRA update size ||dW||/||W||: median {np.median(rel):.1e} over {len(rel)} layers (stage-1 run 1 was 3e-4: no effect)")
"""),
    code(r'''
after = eval_grounding(val_dataset, EVAL_N)
print("before:", before)
print("after: ", after)
'''),
    md("### Save the LoRA adapter to the Hub\nThe Mac merges it into the original bf16 weights (`scripts/merge_lora.py`), so no 9 GB upload is needed."),
    code(r'''
model.save_pretrained("lora")
tokenizer.save_pretrained("lora")
from safetensors import safe_open
with safe_open("lora/adapter_model.safetensors", framework="pt") as f:  # LoRA B must be non-zero after training
    assert any((f.get_tensor(k) != 0).any() for k in f.keys() if "lora_B" in k)
json.dump({"before": before, "after": after, "steps": MAX_STEPS, "meta": meta}, open("lora/train_report.json", "w"))
api = HfApi()
api.create_repo(HF_OUT, private=True, exist_ok=True)
api.upload_folder(folder_path="lora", repo_id=HF_OUT, path_in_repo="lora")
print("uploaded to", HF_OUT)
'''),
])


MULTISTEP = notebook([
    md("""
# Stage 2: GRPO on multi-step macOS tasks (Qwen3.5-4B, T4)

Starts from the stage-1 LoRA. Data comes from the Mac (`scripts/collect_steps.py`): each row is one state
of a multi-step task, captured while an AX-driven oracle solved randomly generated *training* tasks. The
model sees the screenshot, the task and its action history and must produce the next action. The reward
is verifiable: right action type, right target element (point inside the AX box), right text or keys,
or `terminate` when the task is done. Benchmark tasks are never in this data.
"""),
    code(INSTALL),
    code(r'''
HF_DATASET = "nishantsirohi2307/cua-macos-steps"          # scripts/export_dataset.py --kind steps
HF_STAGE1  = "nishantsirohi2307/cua-qwen35-4b-grounding"  # stage-1 adapter (lora/ folder)
HF_OUT     = "nishantsirohi2307/cua-qwen35-4b-agent"      # stage-2 adapter goes here
MAX_STEPS  = 200   # ~50 s/step on a T4
EVAL_N     = 150
'''),
    code(LOGIN),
    code(DATA),
    code(MODEL),
    code(r'''
# Load the stage-1 adapter weights into the identically configured LoRA.
from safetensors.torch import load_file
from peft import set_peft_model_state_dict
s1 = snapshot_download(HF_STAGE1, allow_patterns=["lora/*"])
state = load_file(f"{s1}/lora/adapter_model.safetensors")
res = set_peft_model_state_dict(model, state)
assert not res.unexpected_keys, res.unexpected_keys[:5]
print("loaded", len(state), "stage-1 LoRA tensors")
'''),
    code(r'''
def to_prompt(ex):
    return {"prompt": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": prompts.user_text(ex["instruction"], ex["history"])},
    ]}
train_dataset = ds["train"].map(to_prompt).shuffle(seed=0)
val_dataset = ds["val"]
ex = train_dataset[0]
print(ex["instruction"], ex["history"], ex["gt"], sep="\n")
'''),
    code(GENERATE),
    code(r'''
def eval_steps(rows, n):
    FastVisionModel.for_inference(model)
    rows = rows.select(range(min(n, len(rows))))
    exact, score = 0, 0.0
    for ex in rows:
        a = actions.parse_action(generate(ex["image"], ex["instruction"], ex["history"]))
        s = rewards.step_score(a, json.loads(ex["gt"]))
        score += s
        exact += s >= 0.999
    FastVisionModel.for_training(model)
    return {"step_exact": exact / len(rows), "step_score": score / len(rows), "n": len(rows)}

before = eval_steps(val_dataset, EVAL_N)
print("before stage 2 (after stage 1):", before)
'''),
    code(r'''
from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 1e-4,             # LoRA from zero needs ~10x a full fine-tune LR; 1e-5 left the model unchanged
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.0,               # decay pulls the LoRA back to zero
    warmup_ratio = 0.05,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,
    num_generations = 4,
    max_prompt_length = 4096,
    max_completion_length = 160,
    temperature = 1.0,
    max_steps = MAX_STEPS,
    save_steps = 50,
    max_grad_norm = 1.0,
    report_to = "none",
    output_dir = "outputs_steps",
    importance_sampling_level = "sequence",
    mask_truncated_completions = False,
    loss_type = "dr_grpo",
    scale_rewards = "group",
)
trainer = GRPOTrainer(
    model = model,
    args = training_args,
    processing_class = tokenizer,
    reward_funcs = [rewards.format_reward, rewards.step_reward],
    train_dataset = train_dataset,
)
trainer.train()
'''),
    code(r"""
# Did training actually move the model? Reward trend, plus the size of the LoRA update.
import numpy as np
h = [x for x in trainer.state.log_history if "reward" in x]
def avg(rows, k):
    v = [x[k] for x in rows if k in x]
    return float(np.mean(v)) if v else float("nan")
for k in ["reward", "rewards/step_reward/mean", "frac_reward_zero_std", "completions/mean_length"]:
    print(f"{k:34s} first 25 steps = {avg(h[:25], k):.3f}   last 25 steps = {avg(h[-25:], k):.3f}")
rel = []
for mod in model.modules():
    if hasattr(mod, "lora_A") and hasattr(mod, "base_layer") and "default" in mod.lora_A:
        d = (mod.lora_B["default"].weight.float() @ mod.lora_A["default"].weight.float()) * mod.scaling["default"]
        rel.append((d.norm() / mod.base_layer.weight.float().norm()).item())
print(f"LoRA update size ||dW||/||W||: median {np.median(rel):.1e} over {len(rel)} layers (stage-1 run 1 was 3e-4: no effect)")
"""),
    code(r'''
after = eval_steps(val_dataset, EVAL_N)
print("before:", before)
print("after: ", after)
'''),
    code(r'''
model.save_pretrained("lora")
tokenizer.save_pretrained("lora")
json.dump({"before": before, "after": after, "steps": MAX_STEPS, "meta": meta}, open("lora/train_report.json", "w"))
api = HfApi()
api.create_repo(HF_OUT, private=True, exist_ok=True)
api.upload_folder(folder_path="lora", repo_id=HF_OUT, path_in_repo="lora")
print("uploaded to", HF_OUT)
'''),
])


if __name__ == "__main__":
    (HERE / "01_grounding_grpo_t4.ipynb").write_text(json.dumps(GROUNDING, indent=1))
    (HERE / "02_multistep_grpo_t4.ipynb").write_text(json.dumps(MULTISTEP, indent=1))
    print("wrote notebooks")
