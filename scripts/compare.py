"""Compare two benchmark runs (e.g. base vs trained) task by task.

    python scripts/compare.py runs/base runs/trained

Reports success rates overall / per family / seen vs held-out templates, the tasks that flipped,
and an exact McNemar test on the paired outcomes.
"""
import json
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cua.tasks.registry import HELD_OUT  # noqa: E402


def load(d):
    rows = [json.loads(l) for l in (Path(d) / "results.jsonl").read_text().splitlines()]
    return {r["task_id"]: r for r in rows}


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main():
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    A, B = load(a_dir), load(b_dir)
    ids = sorted(set(A) & set(B))
    if len(ids) < max(len(A), len(B)):
        print(f"note: comparing the {len(ids)} tasks present in both runs")

    def rate(sel, R):
        return sum(R[i]["success"] for i in sel), len(sel)

    groups = {"overall": ids}
    for fam in sorted({A[i]["family"] for i in ids}):
        groups[fam] = [i for i in ids if A[i]["family"] == fam]
    groups["seen templates"] = [i for i in ids if A[i]["template"] not in HELD_OUT]
    groups["held-out templates"] = [i for i in ids if A[i]["template"] in HELD_OUT]

    print(f"{'':22s} {Path(a_dir).name:>14s} {Path(b_dir).name:>14s}")
    for g, sel in groups.items():
        if not sel:
            continue
        sa, n = rate(sel, A)
        sb, _ = rate(sel, B)
        print(f"{g:22s} {sa:>5d}/{n:<3d} {sa / n:5.0%}  {sb:>5d}/{n:<3d} {sb / n:5.0%}")
    only_b = [i for i in ids if B[i]["success"] and not A[i]["success"]]
    only_a = [i for i in ids if A[i]["success"] and not B[i]["success"]]
    print(f"\nsolved only by {Path(b_dir).name}: {len(only_b)}")
    for i in only_b:
        print("  +", i, "|", B[i]["instruction"])
    print(f"solved only by {Path(a_dir).name}: {len(only_a)}")
    for i in only_a:
        print("  -", i, "|", A[i]["instruction"])
    print(f"\nexact McNemar p = {mcnemar_exact(len(only_b), len(only_a)):.4f} (two-sided, on {len(only_a) + len(only_b)} discordant tasks)")


if __name__ == "__main__":
    main()
