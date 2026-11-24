"""Collect stage-2 step data: run the AX oracle on random TRAINING tasks and record every state.

    python scripts/collect_steps.py --episodes 300 --out data/steps/run1 --seed 1

Each row: screenshot before the step, task instruction, history of previous actions (same text the
agent sees), and the oracle's action with its target box. Episodes whose checker fails are dropped.
Takes over mouse and keyboard; move the mouse to abort (re-run with a new --seed to continue).
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cua.actions import Action, describe  # noqa: E402
from cua.env import MacEnv, UserAbort  # noqa: E402
from cua.tasks.registry import sample_train  # noqa: E402


def run(env, task, ep_dir: Path) -> tuple[bool, list[dict]]:
    rows, history = [], []
    obs = env.reset(task)
    gen = task.oracle(env)
    for t in range(task.max_steps + 1):
        try:
            a = next(gen)
        except StopIteration:
            a = Action("terminate", status="success", thought="The task is complete.")
        img = f"s{t:02d}.png"
        obs.image.save(ep_dir / img)
        gt = a.args()
        if a.meta.get("bbox"):
            gt["bbox"] = a.meta["bbox"]
        if a.meta.get("alts"):
            gt["alts"] = a.meta["alts"]
        rows.append({"image": img, "instruction": task.instruction, "history": list(history), "gt": json.dumps(gt),
                     "episode": task.id, "task_id": task.id, "template": task.template, "step": t,
                     "thought": a.thought})
        if a.action == "terminate":
            break
        info = env.step(a)
        if info["blocked"]:
            print(f"  oracle action blocked: {info['blocked']}")
            return False, rows
        history.append(describe(a))
        obs = env.observe()
    return bool(task.check(env)), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    env = MacEnv(hide_others=True, backdrop=True)
    kept = failed = 0
    try:
        for i in range(args.episodes):
            task = sample_train(rng, f"train_s{args.seed}_{i:04d}")
            ep_dir = out / task.id
            if (ep_dir / "rows.jsonl").exists():
                kept += 1
                continue
            ep_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{i + 1}/{args.episodes}] {task.template}: {task.instruction}")
            try:
                ok, rows = run(env, task, ep_dir)
            except UserAbort:
                raise
            except Exception as e:
                ok, rows = False, []
                print(f"  error: {type(e).__name__}: {e}")
            finally:
                try:
                    env.teardown(task)
                except Exception as e:
                    print(f"  teardown error: {e}")
            if ok:
                (ep_dir / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                kept += 1
                print(f"  kept ({len(rows)} steps)")
            else:
                shutil.rmtree(ep_dir, ignore_errors=True)
                failed += 1
                print("  dropped (checker failed)")
    except UserAbort as e:
        print(f"\nABORTED by user: {e}")
    finally:
        env.close()
    print(f"kept {kept} episodes, dropped {failed}")


if __name__ == "__main__":
    main()
