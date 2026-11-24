"""Run the 50-task benchmark (or a subset) with a model on this Mac.

    python scripts/run_bench.py --model Qwen/Qwen3.5-4B --out runs/base
    python scripts/run_bench.py --model models/agent --out runs/trained
    python scripts/run_bench.py --oracle --out runs/oracle          # validates every task + checker

Takes over the mouse and keyboard of the main display. Move the mouse (or push it to the top-left
corner) to abort. Re-running with the same --out resumes where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cua.env import MacEnv, UserAbort  # noqa: E402
from cua.policy import MLXPolicy, OraclePolicy  # noqa: E402
from cua.runner import run_episode  # noqa: E402
from cua.tasks.registry import HELD_OUT, benchmark  # noqa: E402


def summarize(results: list[dict]) -> dict:
    def rate(rs):
        return {"n": len(rs), "success": sum(r["success"] for r in rs), "rate": round(sum(r["success"] for r in rs) / max(1, len(rs)), 3)}

    fams = sorted({r["family"] for r in results})
    return {
        "overall": rate(results),
        "by_family": {f: rate([r for r in results if r["family"] == f]) for f in fams},
        "seen_templates": rate([r for r in results if r["template"] not in HELD_OUT]),
        "held_out_templates": rate([r for r in results if r["template"] in HELD_OUT]),
        "avg_steps": round(sum(len(r["steps"]) for r in results) / max(1, len(results)), 2),
        "errors": sum(1 for r in results if r["error"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="HF repo id or local HF-format dir (base or merged)")
    ap.add_argument("--oracle", action="store_true", help="solve tasks with the AX oracle instead of a model")
    ap.add_argument("--noop", action="store_true", help="terminate immediately (checks tasks are not pre-solved)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="task ids or families to run")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true", help="let the model think before acting (extra baseline)")
    ap.add_argument("--no_hide", action="store_true", help="do not hide other apps during the run")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res_path = out / "results.jsonl"
    done = {}
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            r = json.loads(line)
            done[r["task_id"]] = r

    tasks = benchmark()
    if args.only:
        tasks = [t for t in tasks if t.id in args.only or t.family in args.only]

    if args.oracle:
        factory = lambda task, env: OraclePolicy(task, env)  # noqa: E731
        name = "oracle"
    elif args.noop:
        from cua.actions import Action

        class Noop:
            def act(self, *a):
                return "", Action("terminate", status="success"), {}

        factory = lambda task, env: Noop()  # noqa: E731
        name = "noop"
    else:
        policy = MLXPolicy(args.model, temperature=args.temperature, thinking=args.thinking)
        factory = lambda task, env: policy  # noqa: E731
        name = args.model
    (out / "config.json").write_text(json.dumps({"policy": name, "temperature": args.temperature, "thinking": args.thinking,
                                                 "started": time.ctime()}, indent=1))

    env = MacEnv(hide_others=not args.no_hide)
    try:
        for i, task in enumerate(tasks):
            if task.id in done:
                continue
            print(f"[{i + 1}/{len(tasks)}] {task.id}: {task.instruction}")
            r = run_episode(env, task, factory, out / "episodes" / task.id)
            print(f"  -> {'SUCCESS' if r['success'] else 'fail'} ({len(r['steps'])} steps, {r['seconds']}s)"
                  + (f" error={r['error']}" if r["error"] else ""))
            done[task.id] = r
            with res_path.open("a") as f:
                f.write(json.dumps(r) + "\n")
    except UserAbort as e:
        print(f"\nABORTED by user: {e}. Re-run the same command to resume.")
    finally:
        env.close()
    results = [done[t.id] for t in benchmark() if t.id in done]  # every completed task, not just --only
    s = summarize(results)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
