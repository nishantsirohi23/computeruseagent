"""Package collected data as a parquet dataset and push it (privately) to the Hugging Face Hub.

    python scripts/export_dataset.py --kind grounding --src data/grounding --repo YOUR_HF_USERNAME/cua-macos-grounding
    python scripts/export_dataset.py --kind steps --src data/steps --repo YOUR_HF_USERNAME/cua-macos-steps

Screenshots are copied once into images/ (many grounding rows share one screen) and the parquet
stores their relative paths; the notebooks resolve and decode them. Also uploads
cua/{prompts,actions,rewards}.py as cua_shared/ so the notebook uses the same code.
Split is by scene/episode (never by row) so validation screens are unseen.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cua import config  # noqa: E402


def load_rows(src: Path, kind: str) -> list[dict]:
    rows = []
    for f in sorted(src.rglob("rows.jsonl")):  # any depth: data/<kind>/<run>/<scene>/rows.jsonl
        for line in f.read_text().splitlines():
            r = json.loads(line)
            r["src_image"] = f.parent / r["image"]
            r["image"] = f"images/{f.parent.name}/{r['image']}"  # scene/episode dirs have unique names
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["grounding", "steps"], required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--val_frac", type=float, default=0.08)
    ap.add_argument("--local_only", action="store_true", help="write to data/export_<kind> without pushing")
    ap.add_argument("--easy_from", help="eval_grounding.py --write output; train rows the base model already hits ...")
    ap.add_argument("--easy_keep", type=float, default=0.3, help="... are kept with this probability")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict

    rows = load_rows(Path(args.src), args.kind)
    if not rows:
        raise SystemExit(f"no rows under {args.src}")
    group_key = "scene" if args.kind == "grounding" else "episode"
    groups = sorted({r[group_key] for r in rows})
    rng = random.Random(0)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * args.val_frac))
    val_groups = set(groups[:n_val])

    if args.kind == "grounding":
        cols = ["image", "instruction", "bbox", "scene", "source", "role", "element"]  # not "label": TRL reserves it
    else:
        cols = ["image", "instruction", "history", "gt", "episode", "task_id", "template", "step"]

    def build(sel):
        return Dataset.from_dict({c: [r[c] for r in sel] for c in cols})

    train = [r for r in rows if r[group_key] not in val_groups]
    if args.easy_from:
        easy = set()
        for line in Path(args.easy_from).read_text().splitlines():
            e = json.loads(line)
            if e["hit"]:
                easy.add((e["scene"], e["instruction"]))
        before = len(train)
        train = [r for r in train if (r[group_key], r["instruction"]) not in easy or rng.random() < args.easy_keep]
        print(f"difficulty filter: {before} -> {len(train)} train rows")
    val = [r for r in rows if r[group_key] in val_groups]
    dd = DatasetDict({"train": build(train), "val": build(val)})
    print(dd)

    out = ROOT / "data" / f"export_{args.kind}"
    if out.exists():
        shutil.rmtree(out)
    (out / "data").mkdir(parents=True)
    for r in rows:
        dst = out / r["image"]
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(r["src_image"], dst)
    for split, d in dd.items():
        d.to_parquet(str(out / "data" / f"{split}-00000-of-00001.parquet"))
    shared = out / "cua_shared"
    shared.mkdir()
    (shared / "__init__.py").write_text("")
    for m in ("prompts.py", "actions.py", "rewards.py"):
        shutil.copy(ROOT / "cua" / m, shared / m)
    meta = {"kind": args.kind, "rows": {k: len(v) for k, v in dd.items()}, "groups": len(groups),
            "image_max_pixels": config.IMAGE_MAX_PIXELS, "coord_scale": config.COORD_SCALE,
            "base_model": config.BASE_MODEL}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out)
    if args.local_only:
        return
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=args.repo, repo_type="dataset")
    print(f"pushed to https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()
