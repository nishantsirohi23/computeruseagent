"""Tasks on SynthApp: fill a randomly generated native form, optionally submit / use a menu."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time

from .. import apps, ax, config
from ..actions import Action
from ..synthapp import spec as S
from .base import OracleError, Task

SLIDER_TOL = 7.0


def value_ok(w_type: str, got, want) -> bool:
    if w_type == "text":
        return isinstance(got, str) and got.strip() == str(want).strip()
    if w_type == "slider":
        return got is not None and abs(float(got) - float(want)) <= SLIDER_TOL
    return got == want


class SynthTask(Task):
    @classmethod
    def sample(cls, rng: random.Random, task_id: str) -> "SynthTask":
        spec = S.random_spec(rng, origin=config.WINDOW_ORIGIN)
        goal = S.random_goal(spec, rng)
        n_steps = sum({"text": 3, "popup": 2, "stepper": 4}.get(g["type"], 1) + (1 if g["tab"] else 0) for g in goal["goals"])
        n_steps += (1 if goal["submit"] else 0) + (2 if goal["menu"] else 0) + 1
        return cls(id=task_id, instruction=S.instruction(spec, goal, rng), family="synth",
                   template="synth:" + "+".join(sorted({g["type"] for g in goal["goals"]})),
                   max_steps=max(8, n_steps * 2), params={"spec": spec, "goal": goal})

    # ---- lifecycle ----------------------------------------------------------
    def setup(self, env):
        d = config.SANDBOX / "synth" / self.id
        d.mkdir(parents=True, exist_ok=True)
        self._spec_path, self._state_path = d / "spec.json", d / "state.json"
        self._spec_path.write_text(json.dumps(self.params["spec"]))
        self._state_path.unlink(missing_ok=True)
        self._log = open(d / "app.log", "w")
        self._proc = subprocess.Popen([sys.executable, "-m", "cua.synthapp.app", str(self._spec_path), str(self._state_path)],
                                      cwd=str(config.ROOT), stdout=self._log, stderr=subprocess.STDOUT)
        if not apps.wait_for(lambda: self._state_path.exists(), 15):
            raise RuntimeError(f"SynthApp did not start for {self.id}")
        self.pid = self._proc.pid
        env.allow(pid=self.pid)
        win = apps.wait_for(lambda: apps.main_window(self.pid), 10)
        if win is not None:
            apps.place_window(win, *config.WINDOW_ORIGIN)

    def state(self) -> dict:
        for _ in range(5):
            try:
                return json.loads(self._state_path.read_text())
            except (json.JSONDecodeError, FileNotFoundError):
                time.sleep(0.1)
        return {}

    def check(self, env) -> bool:
        time.sleep(0.3)
        st = self.state()
        if not st:
            return False
        goal = self.params["goal"]
        vals = st.get("values", {})

        def all_ok(v):
            return all(value_ok(g["type"], v.get(g["id"]), g["value"]) for g in goal["goals"])

        ok = all_ok(vals)
        if goal["submit"]:
            ok = ok and any(s["button"] == goal["submit"] and all_ok(s["values"]) for s in st.get("submissions", []))
        if goal["menu"]:
            ok = ok and goal["menu"][1] in st.get("menu", [])
        return bool(ok)

    def teardown(self, env):
        p = getattr(self, "_proc", None)
        if p is not None and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        if getattr(self, "_log", None):
            self._log.close()

    # ---- oracle -------------------------------------------------------------
    def oracle(self, env):
        spec, goal = self.params["spec"], self.params["goal"]
        wmap = {w["id"]: w for w, _ in S.all_widgets(spec)}
        for g in goal["goals"]:
            w = wmap[g["id"]]
            if g["tab"] and self.state().get("tab") != g["tab"]:
                tab = self.find_one(env, role="AXRadioButton", subrole="AXTabButton", title=g["tab"])
                yield self.click(env, tab, why=f'Open the {g["tab"]} tab.')
            yield from self._set(env, w, g["value"])
        if goal["menu"]:
            title, item = goal["menu"]
            bar = ax.get(ax.app(self.pid), "AXMenuBar")
            mb = self.find_one(env, root=bar, role="AXMenuBarItem", title=title)
            yield self.click(env, mb, why=f"Open the {title} menu.")
            mi = self.find_one(env, role="AXMenuItem", title=item)
            yield self.click(env, mi, why=f"Choose {item}.")
        if goal["submit"]:
            b = self.find_one(env, ident=goal["submit"])
            yield self.click(env, b, why=f'Click "{b.label}".')

    def _set(self, env, w, v):
        k, label = w["type"], w.get("label", "")
        if k == "text":
            n = self.find_one(env, ident=w["id"])
            yield self.click(env, n, why=f"Click the {label} field.")
            cur = self.state()["values"].get(w["id"], "")
            if cur:
                yield self.key(["cmd", "a"], why="Select the existing text.")
            yield self.type(v, why=f'Type "{v}".')
        elif k == "checkbox":
            n = self.find_one(env, ident=w["id"])
            yield self.click(env, n, why=f'{"Check" if v else "Uncheck"} "{label}".')
        elif k == "radio":
            n = self.find_one(env, ident=f'{w["id"]}::{v}')
            yield self.click(env, n, why=f'Select "{v}".')
        elif k == "popup":
            n = self.find_one(env, ident=w["id"])
            yield self.click(env, n, why=f"Open the {label} pop-up menu.")
            mi = self.find_one(env, pid=self.pid, role="AXMenuItem", title=v)
            yield self.click(env, mi, why=f"Choose {v}.")
        elif k == "segmented":
            n = self.find_one(env, ident=w["id"])
            segs = [c for c in ax.walk(n.el, max_depth=2)[1:] if c.label == v and c.frame]
            if segs:
                yield self.click(env, segs[0], why=f"Select the {v} segment.")
            else:
                i = w["options"].index(v)
                x, y, ww, hh = n.frame
                seg = (x + ww * i / len(w["options"]), y, ww / len(w["options"]), hh)
                yield self.click(env, n, why=f"Select the {v} segment.", frame=seg)
        elif k == "slider":
            n = self.find_one(env, ident=w["id"])
            x, y, ww, hh = n.frame
            knob = 20.0
            for c in ax.walk(n.el, max_depth=1)[1:]:
                if c.role == "AXValueIndicator" and c.frame:
                    knob = c.frame[2]
            lo, hi = w.get("min", 0), w.get("max", 100)

            def px(val):
                val = min(max(val, lo), hi)
                return x + knob / 2 + (ww - knob) * (val - lo) / (hi - lo)

            if v <= lo:
                tgt = (x, y, max(px(lo + SLIDER_TOL) - x, 4), hh)
            elif v >= hi:
                tgt = (px(hi - SLIDER_TOL), y, x + ww - px(hi - SLIDER_TOL), hh)
            else:
                tgt = (px(v - SLIDER_TOL), y, px(v + SLIDER_TOL) - px(v - SLIDER_TOL), hh)
            a = self.click(env, n, why=f"Click the {label} slider track at {v}%.", frame=tgt)
            a.coordinate = env.bounds.to_rel(px(v), y + hh / 2)
            yield a
        elif k == "stepper":
            n = self.find_one(env, ident=w["id"])
            cur = self.state()["values"][w["id"]]
            up = v > cur
            arrows = [c for c in ax.walk(n.el, max_depth=1)[1:] if c.frame and
                      c.subrole == ("AXIncrementArrow" if up else "AXDecrementArrow")]
            x, y, ww, hh = n.frame
            fr = arrows[0].frame if arrows else ((x, y, ww, hh / 2) if up else (x, y + hh / 2, ww, hh / 2))
            for _ in range(abs(v - cur)):
                yield self.click(env, n, why=f"Click the {'up' if up else 'down'} arrow of {label}.", frame=fr)
            if self.state()["values"][w["id"]] != v:
                raise OracleError(f"stepper {w['id']} did not reach {v}")
        elif k == "list":
            table = self.find_one(env, ident=w["id"])
            cell = self.find_one(env, root=table.el, value=v)
            yield self.click(env, cell, why=f"Select {v}.")
        else:
            raise OracleError(k)
