"""Calculator tasks. State is read from the display via AX."""
from __future__ import annotations

import random
import re
import time

from .. import apps, ax, config
from .base import OracleError, Task

OPS = {"+": ("+", "Add", "plus"), "-": ("−", "Subtract", "minus"), "*": ("×", "Multiply", "times"),
       "/": ("÷", "Divide", "divided by")}
# AX descriptions of Calculator keys (several spellings across macOS versions)
KEY_LABELS = {
    **{str(d): [str(d)] for d in range(10)},
    "+": ["Add", "Plus", "+"], "-": ["Subtract", "Minus", "−", "-"], "*": ["Multiply", "Times", "×"],
    "/": ["Divide", "÷"], "=": ["Equals", "="], ".": ["Point", "Decimal", "Decimal Point", "."],
    "C": ["All Clear", "Clear", "AC", "C"],
}


def _num(s: str) -> float | None:
    s = s.replace(",", "").replace("−", "-").replace("‎", "").replace(" ", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def evaluate_left_to_right(tokens: list) -> float:
    acc = float(tokens[0])
    for i in range(1, len(tokens), 2):
        op, b = tokens[i], float(tokens[i + 1])
        acc = {"+": acc + b, "-": acc - b, "*": acc * b, "/": acc / b}[op]
    return acc


def fmt(x) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


class CalculatorTask(Task):
    """params: tokens (e.g. [47, "+", 38]), expected (float) or mode_check, buttons (bool)."""

    @classmethod
    def sample(cls, rng: random.Random, task_id: str, template: str | None = None) -> "CalculatorTask":
        template = template or rng.choice(["arith2", "arith2_buttons", "arith3", "decimal", "clear"])
        if template in ("arith2", "arith2_buttons"):
            op = rng.choice("+-*/")
            if op == "/":
                b = rng.randint(2, 19)
                a = b * rng.randint(2, 40)
            elif op == "*":
                a, b = rng.randint(11, 99), rng.randint(3, 29)
            else:
                a, b = rng.randint(10, 999), rng.randint(10, 999)
            toks = [a, op, b]
            sym = OPS[op][0]
            if template == "arith2":
                instr = rng.choice([f"Use Calculator to compute {a} {sym} {b}.", f"In Calculator, work out {a} {OPS[op][2]} {b}.",
                                    f"Calculate {a} {sym} {b} in the Calculator app."])
            else:
                instr = f"Using only the on-screen buttons in Calculator (click them), compute {a} {sym} {b}."
        elif template == "arith3":
            a, b, c = rng.randint(2, 60), rng.randint(2, 60), rng.randint(2, 9)
            op1, op2 = rng.choice("+-"), rng.choice("*+")
            toks = [a, op1, b, op2, c]
            instr = f"In Calculator, compute {a} {OPS[op1][0]} {b}, then multiply the result by {c}." if op2 == "*" else \
                f"In Calculator, compute {a} {OPS[op1][0]} {b} {OPS[op2][0]} {c}."
        elif template == "decimal":
            a = rng.choice([1.5, 2.5, 3.5, 4.25, 7.5, 12.5, 0.75])
            b = rng.randint(2, 12)
            toks = [a, "*", b]
            instr = f"Use Calculator to multiply {a} by {b}."
        elif template == "clear":
            toks = None
            instr = "Clear the Calculator so the display shows 0."
        else:
            raise ValueError(template)
        exp = evaluate_left_to_right(toks) if toks else 0.0
        return cls(id=task_id, instruction=instr, family="calculator", template=f"calc:{template}", max_steps=12 if template != "arith2_buttons" else 16,
                   params={"tokens": toks, "expected": exp, "buttons": template == "arith2_buttons", "kind": template})

    # ---- lifecycle ----------------------------------------------------------
    def setup(self, env):
        self._was_running = apps.pid(apps.CALCULATOR) is not None
        self.pid = apps.launch(apps.CALCULATOR)
        env.allow(pid=self.pid)
        time.sleep(0.5)
        self.menu_press(["View", "Basic"])
        clear = self.button("C")
        if clear is not None:
            ax.perform(clear.el)
            ax.perform(clear.el)
        if self.params["kind"] == "clear":  # put a number on the display first
            for ch in "7*6=":
                b = self.button(ch)
                if b is not None:
                    ax.perform(b.el)
        win = apps.main_window(self.pid)
        if win is not None:
            apps.place_window(win, *config.WINDOW_ORIGIN)

    def teardown(self, env):
        if not self._was_running:
            apps.quit(apps.CALCULATOR)

    def menu_press(self, path: list[str]) -> bool:
        bar = ax.get(ax.app(self.pid), "AXMenuBar")
        el = bar
        for name in path:
            nxt = None
            for n in ax.walk(el, max_depth=3)[1:]:
                if n.title == name and n.role in ("AXMenuBarItem", "AXMenuItem"):
                    nxt = n.el
                    break
            if nxt is None:
                return False
            el = nxt
        return ax.perform(el)

    def button(self, ch: str):
        nodes = ax.walk(apps.main_window(self.pid) or ax.app(self.pid))
        for lbl in KEY_LABELS[ch]:
            for n in nodes:
                if n.role == "AXButton" and lbl in (n.desc, n.title) and n.frame:
                    return n
        return None

    def display(self) -> str | None:
        win = apps.main_window(self.pid)
        if win is None:
            return None
        nodes = ax.walk(win)
        for n in nodes:  # macOS 26: the entry/result line is the text inside the "StandardInputView" scroll area
            if n.ident == "StandardInputView":
                for c in ax.walk(n.el, max_depth=2)[1:]:
                    if c.role == "AXStaticText" and _num(c.value) is not None:
                        return c.value
        texts = [n for n in nodes if n.role in ("AXStaticText", "AXTextField") and n.frame
                 and _num(n.value or n.desc or "") is not None]
        if not texts:
            return None
        # The result is the biggest number on screen (history lines are smaller).
        best = max(texts, key=lambda n: (n.frame[3], n.frame[1]))
        return best.value or best.desc

    def check(self, env) -> bool:
        time.sleep(0.3)
        v = self.display()
        x = _num(v) if v is not None else None
        return x is not None and abs(x - self.params["expected"]) < 1e-6 * max(1.0, abs(self.params["expected"]))

    # ---- oracle -------------------------------------------------------------
    def oracle(self, env):
        toks = self.params["tokens"]
        if toks is None:
            b = self.button("C")
            yield self.click(env, b, why="Click the clear button.", alts=[{"action": "key", "keys": ["escape"]}])
            return
        expr = "".join(fmt(t) if not isinstance(t, str) else t for t in toks) + "="
        if not self.params["buttons"]:
            if len(toks) == 5 and toks[3] == "*":  # Calculator applies precedence; finish the first part first
                yield self.type(f"{fmt(toks[0])}{toks[1]}{fmt(toks[2])}=", why="Type the first calculation.")
                yield self.type(f"*{fmt(toks[4])}=", why="Multiply the result.")
            else:
                yield self.type(expr, why=f"Type the calculation {expr} on the keyboard.")
            return
        for ch in expr:
            b = self.button(ch)
            if b is None:
                raise OracleError(f"no Calculator button for {ch!r}")
            yield self.click(env, b, why=f"Click {b.label}.")
