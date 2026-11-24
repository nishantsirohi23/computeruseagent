"""Finder tasks inside a sandbox folder. Checked on the filesystem or via Finder's AppleScript state.

Clicks and keystrokes are confined to the task's own Finder window (plus menus) by env.guard, so the
agent cannot touch the Desktop or the user's other Finder windows.
"""
from __future__ import annotations

import random
import shutil
import time
from pathlib import Path

from .. import apps, ax, config
from .base import OracleError, Task

DOCS = ["report.txt", "budget.csv", "notes.md", "photo.png", "invoice.pdf", "draft.txt", "agenda.txt", "data.json",
        "summary.txt", "plan.md", "logo.png", "contacts.csv"]
DIRS = ["Archive", "Projects", "Images", "Receipts", "Drafts", "Old", "Shared", "Music"]
NEW = ["Reports", "2026", "Taxes", "Backup", "Clients", "Travel", "Ideas", "Scans"]
MENU_ROLES = ("AXMenuBarItem", "AXMenuItem", "AXMenu", "AXMenuBar")


def close_sandbox_windows():
    """Close every Finder window showing a folder inside the sandbox (ours, wherever the agent took it)."""
    try:
        apps.osascript(f'''tell application "Finder"
repeat with i from (count of Finder windows) to 1 by -1
    set w to Finder window i
    try
        if (POSIX path of (target of w as alias)) starts with "{config.SANDBOX}/" then close w
    end try
end repeat
end tell''')
    except Exception:
        pass


def _fake_png() -> bytes:
    return bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                         "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")


class FinderTask(Task):
    @classmethod
    def sample(cls, rng: random.Random, task_id: str, template: str | None = None) -> "FinderTask":
        template = template or rng.choice(["new_folder", "rename", "duplicate", "move_into", "open_folder", "select",
                                           "list_view", "new_folder_inside"])
        files = rng.sample(DOCS, rng.randint(3, 6))
        dirs = rng.sample(DIRS, rng.randint(1, 3))
        root = f"Workspace {rng.randint(10, 99)}"
        view = rng.choice(["icon", "list"]) if template != "list_view" else "icon"
        f, d = rng.choice(files), rng.choice(dirs)
        p = {"root": root, "files": files, "dirs": dirs, "view": view, "kind": template}
        if template == "new_folder":
            name = rng.choice([n for n in NEW if n not in dirs])
            instr, p["name"] = f'In the Finder window "{root}", create a new folder named "{name}".', name
        elif template == "rename":
            stem, ext = f.rsplit(".", 1)
            new = rng.choice(["final", "old", "v2", "backup", "shared"]) + "_" + stem + "." + ext
            instr, p["file"], p["new"] = f'Rename the file "{f}" to "{new}".', f, new
        elif template == "duplicate":
            instr, p["file"] = f'Duplicate the file "{f}" in the "{root}" window.', f
        elif template == "move_into":
            instr, p["file"], p["dir"] = f'Move the file "{f}" into the "{d}" folder.', f, d
        elif template == "open_folder":
            instr, p["dir"] = f'Open the folder "{d}" in this Finder window.', d
        elif template == "select":
            instr, p["file"] = f'Select the file "{f}" in Finder (and nothing else).', f
        elif template == "list_view":
            instr = f'Switch the "{root}" Finder window to list view.'
        elif template == "new_folder_inside":
            name = rng.choice([n for n in NEW if n not in dirs])
            instr, p["dir"], p["name"] = f'Inside the "{d}" folder, create a new folder named "{name}".', d, name
        else:
            raise ValueError(template)
        return cls(id=task_id, instruction=instr, family="finder", template=f"finder:{template}", max_steps=12, params=p)

    # ---- lifecycle ----------------------------------------------------------
    def setup(self, env):
        p = self.params
        base = config.SANDBOX / "finder" / self.id
        if base.exists():
            shutil.rmtree(base)
        self.dir = base / p["root"]
        self.dir.mkdir(parents=True)
        for f in p["files"]:
            (self.dir / f).write_bytes(_fake_png() if f.endswith(".png") else f"{f}\n".encode())
        for d in p["dirs"]:
            (self.dir / d).mkdir()
            (self.dir / d / "readme.txt").write_text("sandbox\n")
        self.pid = apps.pid(apps.FINDER)
        env.allow(pid=self.pid)
        close_sandbox_windows()
        self._pre_windows = list(ax.windows(ax.app(self.pid)))
        self._minimized = []
        for w in ax.windows(ax.app(self.pid)):
            if ax.get(w, "AXSubrole") == "AXStandardWindow" and not ax.get(w, "AXMinimized"):
                if ax.set_attr(w, "AXMinimized", True):
                    self._minimized.append(w)
        x, y = config.WINDOW_ORIGIN
        view = "list view" if p["view"] == "list" else "icon view"
        script = f'''
tell application "Finder"
    set w to make new Finder window to (POSIX file "{self.dir}" as alias)
    set toolbar visible of w to true
    set bounds of w to {{{x}, {y}, {x + 640}, {y + 520}}}
    set current view of w to {view}
    if "{view}" is "icon view" then
        set arrangement of icon view options of w to not arranged
        set arrangement of icon view options of w to arranged by name
    end if
end tell'''
        try:
            apps.osascript(script)
        except Exception:  # Finder occasionally stalls on its first scripting call
            time.sleep(2)
            apps.osascript(script)
        if apps.wait_for(self.window, 10) is None:
            raise RuntimeError("Finder window did not open")
        if view == "icon view" and not apps.wait_for(self._icons_inside, 3):
            for _ in range(3):  # Finder sometimes keeps a grid laid out for an older, wider window
                apps.osascript('''tell application "Finder"
    set arrangement of icon view options of front Finder window to not arranged
    set arrangement of icon view options of front Finder window to arranged by name
end tell''')
                if apps.wait_for(self._icons_inside, 2):
                    break
        env.guard = self._in_task_window

    def _icons_inside(self) -> bool:
        w = self.window()
        wf = ax.frame(w) if w is not None else None
        if wf is None:
            return False
        icons = [n for c in ax.walk(w) if c.ident == "IconView" for n in ax.walk(c.el) if n.role == "AXImage" and n.title]
        return len(icons) >= len(self.params["files"]) + len(self.params["dirs"]) and all(
            n.frame and n.frame[0] + n.frame[2] <= wf[0] + wf[2] for n in icons)

    def _names(self) -> set[str]:
        return {self.params["root"], *self.params["dirs"]}

    def _is_task_window(self, w) -> bool:
        # Finder replaces a window's AX element when it navigates, so identify ours by title (root or one
        # of its subfolders) and by the fixed position we gave it.
        if w is None or ax.get(w, "AXMinimized") or str(ax.get(w, "AXTitle") or "") not in self._names():
            return False
        fr = ax.frame(w)
        return fr is not None and abs(fr[0] - config.WINDOW_ORIGIN[0]) < 30 and abs(fr[1] - config.WINDOW_ORIGIN[1]) < 30

    def _in_task_window(self, el) -> bool:
        """True for elements of the task's own Finder window (or its menus), wherever it has navigated."""
        if ax.get(el, "AXRole") in MENU_ROLES:
            return True
        win = el if ax.get(el, "AXRole") == "AXWindow" else (ax.get(el, "AXWindow") or ax.get(el, "AXTopLevelUIElement"))
        if win is not None:
            return self._is_task_window(win)
        # Finder's inline rename field reports no window: accept it if it sits inside ours.
        fr, tw = ax.frame(el), self.window()
        twf = ax.frame(tw) if tw is not None else None
        if fr is None or twf is None:
            return False
        cx, cy = fr[0] + fr[2] / 2, fr[1] + fr[3] / 2
        return twf[0] <= cx <= twf[0] + twf[2] and twf[1] <= cy <= twf[1] + twf[3]

    def window(self):
        for w in ax.windows(ax.app(self.pid)):
            if self._is_task_window(w):
                return w
        return None

    def focus(self, env):
        apps.activate_pid(self.pid)
        w = self.window()
        if w is not None:
            ax.perform(w, "AXRaise")
            ax.set_attr(w, "AXMain", True)

    def finder(self, script: str) -> str:
        return apps.osascript(f'tell application "Finder" to {script}')

    def front_target(self) -> str:
        return self.finder("get POSIX path of (target of front Finder window as alias)")

    def check(self, env) -> bool:
        time.sleep(0.5)
        p, d = self.params, self.dir
        k = p["kind"]
        if k == "new_folder":
            return (d / p["name"]).is_dir()
        if k == "rename":
            return (d / p["new"]).exists() and not (d / p["file"]).exists()
        if k == "duplicate":
            stem, ext = p["file"].rsplit(".", 1)
            return (d / f"{stem} copy.{ext}").exists() and (d / p["file"]).exists()
        if k == "move_into":
            return (d / p["dir"] / p["file"]).exists() and not (d / p["file"]).exists()
        if k == "open_folder":
            return Path(self.front_target()).resolve() == (d / p["dir"]).resolve()
        if k == "select":
            sel = apps.osascript('''tell application "Finder"
set out to {}
repeat with i in (get selection)
    set end of out to name of i
end repeat
set AppleScript's text item delimiters to linefeed
return out as text
end tell''')
            return [s for s in sel.splitlines() if s.strip()] == [p["file"]]
        if k == "list_view":
            return self.finder("get current view of front Finder window").strip() == "list view"
        if k == "new_folder_inside":
            return (d / p["dir"] / p["name"]).is_dir()
        raise ValueError(k)

    def teardown(self, env):
        env.guard = None
        close_sandbox_windows()
        # Also close windows the agent opened elsewhere (e.g. the iCloud "Shared" view, which AppleScript
        # cannot close): anything that did not exist before setup.
        pre = getattr(self, "_pre_windows", [])
        for w in ax.windows(ax.app(self.pid)):
            if ax.get(w, "AXSubrole") == "AXStandardWindow" and not any(w == p for p in pre):
                btn = ax.get(w, "AXCloseButton")
                if btn is not None:
                    ax.perform(btn)
        for w in self._minimized:
            ax.set_attr(w, "AXMinimized", False)
        shutil.rmtree(self.dir.parent, ignore_errors=True)

    # ---- oracle -------------------------------------------------------------
    def item(self, env, name: str) -> ax.Node:
        """The clickable node for a file/folder name in the task window (icon or list view)."""
        end = time.time() + 4
        while time.time() < end:
            w = self.window()
            # Only the file area (icon / list / column view), never the sidebar, which has look-alike names.
            content = [n for n in ax.walk(w) if n.ident in ("IconView", "ListView", "ColumnView")] if w is not None else []
            for c in content:
                wf = ax.frame(w)
                for n in ax.walk(c.el):
                    if n.role in ("AXImage", "AXTextField") and name in (n.title, n.value) and n.frame and n.frame[2] > 4:
                        cx, cy = n.frame[0] + n.frame[2] / 2, n.frame[1] + n.frame[3] / 2
                        if wf and wf[0] <= cx <= wf[0] + wf[2] and wf[1] <= cy <= wf[1] + wf[3]:
                            return n
            time.sleep(0.2)
        raise OracleError(f"Finder item {name!r} not found")

    def oracle(self, env):
        p = self.params
        k = p["kind"]
        if k == "new_folder":
            yield self.key(["cmd", "shift", "n"], why="Create a new folder.")
            yield self.type(p["name"], why="Type the folder name.")
            yield self.key(["return"], why="Confirm the name.")
        elif k == "rename":
            yield self.click(env, self.item(env, p["file"]), why=f'Select "{p["file"]}".')
            yield self.key(["return"], why="Start renaming.")
            yield self.key(["cmd", "a"], why="Select the whole name.")
            yield self.type(p["new"], why="Type the new name.")
            yield self.key(["return"], why="Confirm the name.")
        elif k == "duplicate":
            yield self.click(env, self.item(env, p["file"]), why=f'Select "{p["file"]}".')
            yield self.key(["cmd", "d"], why="Duplicate it.")
        elif k == "move_into":
            yield self.click(env, self.item(env, p["file"]), why=f'Select "{p["file"]}".')
            yield self.key(["cmd", "c"], why="Copy it.")
            yield self.click(env, self.item(env, p["dir"]), kind="double_click", why=f'Open "{p["dir"]}".')
            yield self.key(["cmd", "alt", "v"], why="Move the copied file here.")
        elif k == "open_folder":
            yield self.click(env, self.item(env, p["dir"]), kind="double_click", why=f'Open "{p["dir"]}".')
        elif k == "select":
            yield self.click(env, self.item(env, p["file"]), why=f'Select "{p["file"]}".')
        elif k == "list_view":
            yield self.key(["cmd", "2"], why="Switch to list view.")
        elif k == "new_folder_inside":
            yield self.click(env, self.item(env, p["dir"]), kind="double_click", why=f'Open "{p["dir"]}".')
            yield self.key(["cmd", "shift", "n"], why="Create a new folder.")
            yield self.type(p["name"], why="Type the folder name.")
            yield self.key(["return"], why="Confirm the name.")
        else:
            raise OracleError(k)
