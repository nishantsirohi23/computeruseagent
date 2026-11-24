"""CPU smoke test of the notebooks' data path with the notebooks' pins (trl==0.22.2, transformers==5.2.0).

Builds a tiny random-weight Qwen3.5 (real tokenizer/processor and chat template), feeds it rows in
the exported dataset format, and runs one GRPO step per stage with our reward functions. It cannot
test Unsloth (CUDA only) but it catches dataset/prompt/image/reward-signature mistakes before a T4 run.

    <venv with torch, torchvision, transformers==5.2.0, trl==0.22.2, datasets> python tests/smoke_trl_grpo.py
"""
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset, Image as HFImage
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cua import actions, prompts, rewards  # noqa: E402

path = snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.jinja", "*.txt"])
processor = AutoProcessor.from_pretrained(path)

cfg = AutoConfig.from_pretrained(path)
t, v = cfg.text_config, cfg.vision_config
t.hidden_size, t.intermediate_size, t.num_hidden_layers = 64, 128, 2
t.layer_types = ["linear_attention", "full_attention"]
t.full_attention_interval = 2
t.num_attention_heads, t.num_key_value_heads = 2, 1
t.linear_num_key_heads, t.linear_num_value_heads, t.linear_key_head_dim, t.linear_value_head_dim = 2, 2, 32, 32
t.mtp_num_hidden_layers = 0
v.depth, v.hidden_size, v.intermediate_size, v.num_heads, v.out_hidden_size = 1, 64, 128, 2, 64
torch.manual_seed(0)
model = Qwen3_5ForConditionalGeneration(cfg).float()
model.warnings_issued = {}  # vanilla TRL 0.22.2 expects this transformers-4 attribute; Unsloth patches it on T4
print("tiny model params:", sum(p.numel() for p in model.parameters()))

_apply = processor.apply_chat_template


def _no_think(*a, **k):
    k.setdefault("enable_thinking", False)
    return _apply(*a, **k)


processor.apply_chat_template = _no_think
SYSTEM = prompts.system_prompt()


def screen(i):
    img = Image.new("RGB", (768, 1344), (236, 236, 236))
    d = ImageDraw.Draw(img)
    d.rectangle([300 + i, 400, 420 + i, 440], fill=(30, 110, 250))
    d.text((320 + i, 410), "Save", fill="white")
    return img


tmp = Path(__file__).resolve().parent / "_smoke_imgs"
tmp.mkdir(exist_ok=True)
paths = []
for i in range(4):
    p = tmp / f"{i}.png"
    screen(i * 10).save(p)
    paths.append(str(p))

# ---- stage 1 format -----------------------------------------------------------
g = Dataset.from_dict({"image": paths, "instruction": ['Click the "Save" button.'] * 4,
                       "bbox": [[390.6, 297.6, 546.9, 327.4]] * 4, "scene": ["s"] * 4, "source": ["synth"] * 4,
                       "role": ["AXButton"] * 4, "element": ["Save"] * 4}).cast_column("image", HFImage())
g = g.map(lambda ex: {"prompt": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": prompts.user_text(ex["instruction"], [])}]})

seen = {}


def spy_grounding(completions, bbox, **kw):
    seen["bbox"], seen["kw"] = bbox, sorted(kw)
    return rewards.grounding_reward(completions, bbox, **kw)


def run(ds, funcs, loss, scale):
    args = GRPOConfig(
        output_dir=str(tmp / "out"), max_steps=1, per_device_train_batch_size=2, gradient_accumulation_steps=2,
        num_generations=4, max_prompt_length=4096, max_completion_length=12, learning_rate=1e-5, logging_steps=1,
        report_to="none", use_cpu=True, bf16=False, fp16=False, importance_sampling_level="sequence",
        loss_type=loss, scale_rewards=scale, mask_truncated_completions=False, save_strategy="no")
    tr = GRPOTrainer(model=model, args=args, processing_class=processor, reward_funcs=funcs, train_dataset=ds)
    out = tr.train()
    return tr, out


tr, out = run(g, [rewards.format_reward, spy_grounding], "dr_grpo", "none")
print("stage1 ok:", out.metrics.get("train_loss"), "| reward kwargs:", seen["kw"])
assert isinstance(seen["bbox"][0], list) and len(seen["bbox"][0]) == 4

# The prompt TRL actually rendered must equal what the Mac agent renders.
msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompts.user_text('Click the "Save" button.', [])}]}]
agent_text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
trl_msgs = json.loads(json.dumps(g[0]["prompt"]))
from trl.data_utils import maybe_apply_chat_template, prepare_multimodal_messages  # noqa: E402

prepare_multimodal_messages(trl_msgs, num_images=1)
trl_text = maybe_apply_chat_template({"prompt": trl_msgs}, processor)["prompt"]
assert trl_text == agent_text, "TRL prompt differs from agent prompt"
assert trl_text.endswith("<think>\n\n</think>\n\n")
print("TRL prompt == agent prompt")

# notebook's generate() path
inputs = processor(images=[screen(0)], text=[agent_text], add_special_tokens=False, return_tensors="pt")
print("image tokens:", int((inputs["input_ids"] == cfg.image_token_id).sum()), "pixel grid:", inputs["image_grid_thw"].tolist())
gen = model.generate(**inputs, max_new_tokens=5, do_sample=False)
print("generate ok:", processor.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:40])

# ---- stage 2 format -----------------------------------------------------------
gt = json.dumps({"action": "left_click", "coordinate": [468, 312], "bbox": [390.6, 297.6, 546.9, 327.4]})
s = Dataset.from_dict({"image": paths, "instruction": ["Set Name to Ana. Then click Save."] * 4,
                       "history": [["Click the Name field.", 'Type "Ana".']] * 4, "gt": [gt] * 4,
                       "episode": ["e"] * 4, "task_id": ["e"] * 4, "template": ["synth:text"] * 4,
                       "step": [2] * 4}).cast_column("image", HFImage())
s = s.map(lambda ex: {"prompt": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": prompts.user_text(ex["instruction"], ex["history"])}]})
tr, out = run(s, [rewards.format_reward, rewards.step_reward], "dr_grpo", "group")
print("stage2 ok:", out.metrics.get("train_loss"))
print("SMOKE TEST PASSED")
