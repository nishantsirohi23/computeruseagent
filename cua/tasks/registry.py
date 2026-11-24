"""The fixed 50-task benchmark and the training-task sampler.

Benchmark instances come from their own seeds and are excluded from training by
instruction text. Three templates never appear in training at all (HELD_OUT), so
the benchmark also measures generalization to unseen task types.
"""
from __future__ import annotations

import random

from .calculator import CalculatorTask
from .finder import FinderTask
from .synth import SynthTask
from .textedit import TextEditTask

HELD_OUT = {"textedit:uppercase", "calc:clear", "finder:new_folder_inside"}

BENCH_PLAN = [
    ("synth", None, 20),
    ("calculator", "arith2", 3), ("calculator", "arith2_buttons", 2), ("calculator", "arith3", 2),
    ("calculator", "decimal", 2), ("calculator", "clear", 1),
    ("textedit", "append_line", 2), ("textedit", "type_new", 1), ("textedit", "replace_word", 2),
    ("textedit", "delete_line", 1), ("textedit", "save_append", 1), ("textedit", "insert_top", 1),
    ("textedit", "clear_all", 1), ("textedit", "uppercase", 1),
    ("finder", "new_folder", 1), ("finder", "rename", 2), ("finder", "duplicate", 1), ("finder", "move_into", 1),
    ("finder", "open_folder", 1), ("finder", "select", 1), ("finder", "list_view", 1), ("finder", "new_folder_inside", 2),
]
FAMILY_CLS = {"synth": SynthTask, "calculator": CalculatorTask, "textedit": TextEditTask, "finder": FinderTask}
TRAIN_TEMPLATES = {
    "calculator": ["arith2", "arith2_buttons", "arith3", "decimal"],
    "textedit": ["append_line", "type_new", "replace_word", "delete_line", "save_append", "insert_top", "clear_all"],
    "finder": ["new_folder", "rename", "duplicate", "move_into", "open_folder", "select", "list_view"],
}


def benchmark() -> list:
    tasks = []
    for fam, tmpl, n in BENCH_PLAN:
        for j in range(n):
            rng = random.Random(f"bench/{fam}/{tmpl}/{j}")
            tid = f"bench_{len(tasks):02d}_{fam}" + (f"_{tmpl}" if tmpl else "")
            cls = FAMILY_CLS[fam]
            tasks.append(cls.sample(rng, tid) if tmpl is None else cls.sample(rng, tid, tmpl))
    assert len(tasks) == 50, len(tasks)
    return tasks


_BENCH_INSTR = None


def bench_instructions() -> set[str]:
    global _BENCH_INSTR
    if _BENCH_INSTR is None:
        _BENCH_INSTR = {t.instruction for t in benchmark()}
    return _BENCH_INSTR


def sample_train(rng: random.Random, task_id: str, weights: dict | None = None):
    weights = weights or {"synth": 0.55, "calculator": 0.15, "textedit": 0.15, "finder": 0.15}
    fams = list(weights)
    while True:
        fam = rng.choices(fams, weights=[weights[f] for f in fams])[0]
        if fam == "synth":
            t = SynthTask.sample(rng, task_id)
        else:
            t = FAMILY_CLS[fam].sample(rng, task_id, rng.choice(TRAIN_TEMPLATES[fam]))
        if t.template not in HELD_OUT and t.instruction not in bench_instructions():
            return t
