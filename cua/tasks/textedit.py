"""TextEdit tasks on plain-text files in the sandbox. Checked via the AX text value or the file on disk."""
from __future__ import annotations

import random
import shutil
import time

import ApplicationServices as AS

from .. import apps, ax, config
from .base import OracleError, Task

LINES = ["Buy milk", "Call the bank", "Water the plants", "Book flights", "Pay rent", "Email Sara", "Fix the bike",
         "Order printer ink", "Renew passport", "Back up laptop", "Clean the garage", "Pick up dry cleaning"]
SENTENCES = ["The quick brown fox jumps over the lazy dog", "Meeting moved to Thursday at noon",
             "Remember to bring the blue folder", "The garden needs more sunlight", "Our team ships on Friday",
             "Please review the attached draft"]
SWAPS = [("cat", "dog"), ("red", "green"), ("Monday", "Friday"), ("small", "large"), ("north", "south"), ("tea", "coffee")]
FILES = ["notes.txt", "todo.txt", "draft.txt", "memo.txt", "list.txt", "ideas.txt"]


class TextEditTask(Task):
    @classmethod
    def sample(cls, rng: random.Random, task_id: str, template: str | None = None) -> "TextEditTask":
        template = template or rng.choice(["append_line", "type_new", "replace_word", "delete_line", "save_append",
                                           "insert_top", "clear_all"])
        fname = rng.choice(FILES)
        lines = rng.sample(LINES, rng.randint(3, 5))
        initial = "\n".join(lines)
        save = False
        if template == "append_line" or template == "save_append":
            new = rng.choice([l for l in LINES if l not in lines])
            expected = initial + "\n" + new
            save = template == "save_append"
            instr = (f'In the TextEdit document {fname}, add a new last line that says "{new}"' +
                     (", then save the document." if save else "."))
            p = {"new": new}
        elif template == "insert_top":
            new = rng.choice(["TODO", "Shopping list", "Errands", "This week", "Priorities"])
            expected = new + "\n" + initial
            instr = f'In {fname}, add a first line that says "{new}" above the existing text.'
            p = {"new": new}
        elif template == "type_new":
            initial = ""
            new = rng.choice(SENTENCES)
            expected = new
            instr = f'Type "{new}" into the empty TextEdit document {fname}.'
            p = {"new": new}
        elif template == "replace_word":
            old, new = rng.choice(SWAPS)
            s = rng.choice(["The %s sat by the door", "Wear the %s jacket today", "Ship it on %s", "Make the %s version",
                            "Drive %s for two hours", "A cup of %s please"]) % old
            lines[rng.randrange(len(lines))] = s
            initial = "\n".join(lines)
            expected = initial.replace(old, new)
            instr = f'In {fname}, replace the word "{old}" with "{new}".'
            p = {"old": old, "new": new}
        elif template == "delete_line":
            target = lines[rng.randrange(1, len(lines) - 1)]
            expected = "\n".join(l for l in lines if l != target)
            instr = f'In {fname}, delete the line "{target}" entirely.'
            p = {"target": target}
        elif template == "clear_all":
            expected = ""
            instr = f"Delete all the text in the TextEdit document {fname}."
            p = {}
        elif template == "uppercase":
            expected = initial.upper()
            instr = f"Make all the text in {fname} upper case using TextEdit's Transformations menu."
            p = {}
        else:
            raise ValueError(template)
        return cls(id=task_id, instruction=instr, family="textedit", template=f"textedit:{template}", max_steps=12,
                   params={"file": fname, "initial": initial, "expected": expected, "save": save, "kind": template, **p})

    # ---- lifecycle ----------------------------------------------------------
    def setup(self, env):
        d = config.SANDBOX / "textedit" / self.id
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        self.path = d / self.params["file"]
        self.path.write_text(self.params["initial"])
        self._was_running = apps.pid(apps.TEXTEDIT) is not None
        self._minimized = []
        if self._was_running:  # keep the user's own TextEdit windows out of reach
            for w in ax.windows(ax.app(apps.pid(apps.TEXTEDIT))):
                if not ax.get(w, "AXMinimized") and ax.set_attr(w, "AXMinimized", True):
                    self._minimized.append(w)
        for attempt in range(3):
            try:
                self.pid = apps.launch(apps.TEXTEDIT, str(self.path))
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        env.allow(pid=self.pid)
        win = apps.wait_for(self.window, 10)
        if win is None:
            raise RuntimeError("TextEdit window did not open")
        apps.place_window(win, *config.WINDOW_ORIGIN, 640, 460)

    def window(self):
        for w in ax.windows(ax.app(self.pid)):
            doc = ax.get(w, "AXDocument") or ""
            if str(doc).endswith(self.params["file"]) and str(self.path.parent.name) in str(doc):
                return w
        return None

    def focus(self, env):
        apps.activate_pid(self.pid)
        w = self.window()
        if w is not None:
            ax.perform(w, "AXRaise")
            ax.set_attr(w, "AXMain", True)

    def text_area(self):
        w = self.window()
        if w is None:
            return None
        for n in ax.walk(w):
            if n.role == "AXTextArea":
                return n
        return None

    def text(self) -> str | None:
        n = self.text_area()
        return None if n is None else str(ax.get(n.el, "AXValue") or "")

    def check(self, env) -> bool:
        time.sleep(0.3)
        want = self.params["expected"].strip()
        if self.params["save"]:
            return self.path.read_text().strip() == want
        got = self.text()
        return got is not None and got.strip() == want

    def teardown(self, env):
        try:
            apps.osascript(f'tell application "TextEdit" to close (every document whose path is "{self.path}") saving no')
        except Exception:
            pass
        for w in self._minimized:
            ax.set_attr(w, "AXMinimized", False)
        if not self._was_running:
            # Quit only a TextEdit we started and that holds no other documents (state restoration can
            # reopen the user's files on launch; those must never be closed unsaved).
            try:
                if apps.osascript('tell application "TextEdit" to count documents').strip() == "0":
                    apps.osascript('tell application "TextEdit" to quit')
            except Exception:
                pass
            # osascript returns before TextEdit exits; opening a file mid-quit fails with -600
            if not apps.wait_for(lambda: not apps.running(apps.TEXTEDIT), 10):
                apps.quit(apps.TEXTEDIT)

    # ---- oracle -------------------------------------------------------------
    def word_frame(self, word: str):
        n = self.text_area()
        s = self.text()
        i = s.find(word)
        if n is None or i < 0:
            raise OracleError(f"word {word!r} not found")
        rng = AS.AXValueCreate(AS.kAXValueCFRangeType, (i, len(word)))
        err, val = AS.AXUIElementCopyParameterizedAttributeValue(n.el, "AXBoundsForRange", rng, None)
        if err != 0:
            raise OracleError("AXBoundsForRange failed")
        ok, r = AS.AXValueGetValue(val, AS.kAXValueCGRectType, None)
        return (r.origin.x, r.origin.y, r.size.width, r.size.height)

    def oracle(self, env):
        k = self.params["kind"]
        area = self.text_area()
        if area is None:
            raise OracleError("no text area")
        if k in ("append_line", "save_append"):
            yield self.click(env, area, why="Click in the document.")
            yield self.key(["cmd", "down"], why="Move to the end of the document.")
            yield self.key(["return"], why="Start a new line.")
            yield self.type(self.params["new"], why="Type the new line.")
            if self.params["save"]:
                yield self.key(["cmd", "s"], why="Save the document.")
        elif k == "insert_top":
            yield self.click(env, area, why="Click in the document.")
            yield self.key(["cmd", "up"], why="Move to the start of the document.")
            yield self.type(self.params["new"], why="Type the new first line.")
            yield self.key(["return"], why="Break the line.")
        elif k == "type_new":
            yield self.click(env, area, why="Click in the empty document.")
            yield self.type(self.params["new"], why="Type the sentence.")
        elif k == "replace_word":
            fr = self.word_frame(self.params["old"])
            yield self.click(env, area, kind="double_click", why=f'Double-click "{self.params["old"]}" to select it.', frame=fr)
            yield self.type(self.params["new"], why=f'Type "{self.params["new"]}".')
        elif k == "delete_line":
            fr = self.word_frame(self.params["target"])
            yield self.click(env, area, kind="triple_click", why="Triple-click the line to select it.", frame=fr)
            yield self.key(["delete"], why="Delete the selected line.")
        elif k == "clear_all":
            yield self.click(env, area, why="Click in the document.")
            yield self.key(["cmd", "a"], why="Select all text.")
            yield self.key(["delete"], why="Delete it.")
        elif k == "uppercase":
            yield self.click(env, area, why="Click in the document.")
            yield self.key(["cmd", "a"], why="Select all text.")
            bar = ax.get(ax.app(self.pid), "AXMenuBar")
            edit = self.find_one(env, root=bar, role="AXMenuBarItem", title="Edit")
            yield self.click(env, edit, why="Open the Edit menu.")
            tr = self.find_one(env, root=bar, role="AXMenuItem", title="Transformations")
            yield self.click(env, tr, why="Open Transformations.")
            up = None
            for title in ("Make Uppercase", "Make Upper Case"):  # spelling differs across macOS versions
                try:
                    up = self.find_one(env, timeout=1.0, root=bar, role="AXMenuItem", title=title)
                    break
                except OracleError:
                    pass
            if up is None:
                raise OracleError("no Make Uppercase menu item")
            yield self.click(env, up, why=f"Choose {up.title}.")
        else:
            raise OracleError(k)
