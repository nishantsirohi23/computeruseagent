"""Step a benchmark task's oracle one action at a time, printing what happens (debugging aid).

    python tests/debug_task.py bench_13_synth
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cua.actions import Action, describe  # noqa: E402
from cua.env import MacEnv  # noqa: E402
from cua.tasks.registry import benchmark  # noqa: E402

task = next(t for t in benchmark() if t.id == sys.argv[1])
print(task.instruction)
env = MacEnv()
try:
    env.reset(task)
    for a in task.oracle(env):
        info = env.step(a)
        extra = ""
        if hasattr(task, "state"):
            extra = json.dumps(task.state().get("values"))
        print(describe(a), info, extra)
    print("check:", task.check(env))
    if len(sys.argv) > 2:
        env.observe().raw.save(sys.argv[2])
finally:
    env.teardown(task)
    env.close()
