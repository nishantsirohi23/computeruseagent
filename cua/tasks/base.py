"""Task interface: setup, automatic check, teardown, and an oracle that solves the task via AX.

The oracle is a generator of Actions in the model's action space. Each yielded
action carries meta["bbox"] (relative 0..1000 target box for pointer actions) and
optionally meta["alts"] (other acceptable actions). The step-data collector
executes them one by one and records (screenshot, history, action) for stage-2
training; running it also proves each task is solvable and its checker works.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from .. import apps, ax
from ..actions import Action


class OracleError(RuntimeError):
    pass


@dataclass
class Task:
    id: str
    instruction: str
    family: str
    template: str
    max_steps: int = 15
    params: dict = field(default_factory=dict)

    # set during setup
    pid: int | None = field(default=None, repr=False)

    def setup(self, env):
        raise NotImplementedError

    def focus(self, env):
        if self.pid:
            apps.activate_pid(self.pid)
            w = apps.main_window(self.pid)
            if w is not None:
                ax.perform(w, "AXRaise")

    def check(self, env) -> bool:
        raise NotImplementedError

    def teardown(self, env):
        pass

    def oracle(self, env) -> Iterator[Action]:
        raise NotImplementedError

    # ---- helpers for oracles ------------------------------------------------
    def nodes(self, pid: int | None = None, root=None) -> list[ax.Node]:
        root = root if root is not None else ax.app(pid or self.pid)
        return ax.walk(root)

    def find_one(self, env, timeout: float = 3.0, pid: int | None = None, root=None, **crit) -> ax.Node:
        end = time.time() + timeout
        while True:
            hits = [n for n in ax.find(self.nodes(pid, root), **crit) if n.frame and n.frame[2] > 0 and n.frame[3] > 0
                    and env.bounds.contains(n.frame[0] + n.frame[2] / 2, n.frame[1] + n.frame[3] / 2)]
            if hits:
                return hits[0]
            if time.time() > end:
                raise OracleError(f"{self.id}: no element matching {crit}")
            time.sleep(0.2)

    @staticmethod
    def click(env, n: ax.Node, kind: str = "left_click", why: str = "", frame=None, alts=None) -> Action:
        fr = frame or n.frame
        a = Action(kind, coordinate=env.rel_center(fr), thought=why)
        a.meta = {"bbox": env.rel_box(fr), "label": n.label, "role": n.role}
        if alts:
            a.meta["alts"] = alts
        return a

    @staticmethod
    def key(keys: list[str], why: str = "", alts=None) -> Action:
        a = Action("key", keys=keys, thought=why)
        if alts:
            a.meta["alts"] = alts
        return a

    @staticmethod
    def type(text: str, why: str = "") -> Action:
        return Action("type", text=text, thought=why)
