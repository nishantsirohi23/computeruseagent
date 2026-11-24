"""Offline click-accuracy of a model on exported grounding rows (no desktop control needed).

    python scripts/eval_grounding.py --model Qwen/Qwen3.5-4B --split val
    python scripts/eval_grounding.py --model Qwen/Qwen3.5-4B --split train --limit 1500 --write data/base_train_hits.jsonl

With --write, per-row results are saved; export_dataset.py --easy_from uses them to down-sample rows
the base model already solves (all-correct GRPO groups carry no learning signal).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cua.policy import MLXPolicy  # noqa: E402
from cua.rewards import in_box  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "export_grounding" / "data"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write")
    args = ap.parse_args()
    from datasets import load_dataset

    ds = load_dataset("parquet", data_files={args.split: f"{args.data}/{args.split}-*.parquet"})[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    pol = MLXPolicy(args.model)
    hits = valid = 0
    out = open(args.write, "w") if args.write else None
    for i, ex in enumerate(ds):
        text, a, info = pol.act(ex["image"], ex["instruction"], [])
        ok = a is not None and a.coordinate is not None and in_box(a.coordinate, ex["bbox"])
        hits += ok
        valid += a is not None
        if out:
            out.write(json.dumps({"scene": ex["scene"], "instruction": ex["instruction"], "hit": bool(ok),
                                  "pred": a.args() if a else None}) + "\n")
        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(ds)} acc={hits / (i + 1):.3f} valid={valid / (i + 1):.3f} ({info.get('latency')}s/row)")
    print(json.dumps({"model": args.model, "split": args.split, "n": len(ds), "click_acc": round(hits / len(ds), 4),
                      "valid_call": round(valid / len(ds), 4)}))


if __name__ == "__main__":
    main()
