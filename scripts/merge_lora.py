"""Merge a PEFT LoRA adapter (from the T4 notebooks) into the original bf16 Qwen3.5 checkpoint with MLX.

    python scripts/merge_lora.py --adapter YOUR_HF_USERNAME/cua-qwen35-4b-agent --out models/agent
    python scripts/merge_lora.py --adapter path/to/lora --out models/agent

The output is a normal HF-format checkpoint that `mlx_vlm.load` reads directly, merged in float32
and stored in the base model's bf16, so base and trained models run through the identical pipeline.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cua import config  # noqa: E402


def resolve(repo_or_path: str, sub: str | None = None, patterns=None) -> Path:
    p = Path(repo_or_path)
    if p.exists():
        return p / sub if sub and (p / sub).exists() else p
    from huggingface_hub import snapshot_download

    d = Path(snapshot_download(repo_or_path, allow_patterns=patterns))
    return d / sub if sub and (d / sub).exists() else d


def base_key_for(lora_key: str, base_keys: set[str]) -> str | None:
    stem = lora_key.split(".lora_A.")[0]
    if stem.startswith("base_model.model."):
        stem = stem[len("base_model.model."):]
    cands = [stem + ".weight"]
    # tolerate naming differences between PEFT wrappers and the checkpoint layout
    cands += [c.replace("model.language_model.model.", "model.language_model.") for c in cands]
    cands += ["model." + c for c in cands if not c.startswith("model.")]
    for c in cands:
        if c in base_keys:
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="HF repo (with lora/ folder) or local adapter dir")
    ap.add_argument("--base", default=config.BASE_MODEL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adapter = resolve(args.adapter, "lora", patterns=["lora/*"])
    base = resolve(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((adapter / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (r ** 0.5) if cfg.get("use_rslora") else alpha / r
    lora = mx.load(str(adapter / "adapter_model.safetensors"))
    print(f"adapter: {len(lora)} tensors, r={r}, alpha={alpha}, scale={scale}")

    shards = sorted(base.glob("model*.safetensors"))
    base_keys = set()
    for s in shards:
        base_keys.update(mx.load(str(s)).keys())

    deltas = {}
    for k in lora:
        if ".lora_A." not in k:
            continue
        bk = base_key_for(k, base_keys)
        if bk is None:
            raise SystemExit(f"cannot map LoRA key {k} to a base weight")
        A = lora[k].astype(mx.float32)
        B = lora[k.replace(".lora_A.", ".lora_B.")].astype(mx.float32)
        deltas[bk] = scale * (B @ A)
    print(f"mapped {len(deltas)} LoRA pairs onto base weights")

    merged = 0
    for s in shards:
        w = mx.load(str(s))
        for k in list(w):
            if k in deltas:
                w[k] = (w[k].astype(mx.float32) + deltas[k]).astype(w[k].dtype)
                merged += 1
        mx.save_safetensors(str(out / s.name), w, metadata={"format": "pt"})
        print("wrote", s.name)
    assert merged == len(deltas), (merged, len(deltas))
    for f in base.iterdir():
        if f.is_file() and not f.name.startswith("model") and f.suffix != ".safetensors" or f.name == "model.safetensors.index.json":
            shutil.copy(f, out / f.name)
    if (adapter / "train_report.json").exists():
        shutil.copy(adapter / "train_report.json", out / "train_report.json")
    print(f"merged {merged} weights -> {out}")


if __name__ == "__main__":
    main()
