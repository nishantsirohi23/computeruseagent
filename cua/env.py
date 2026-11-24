"""MacEnv: observe / act on the real macOS desktop, confined to the current task's apps."""
from __future__ import annotations

import atexit
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass

from AppKit import NSRunningApplication
from PIL import Image

from . import apps, ax, config, input as inp, screen
from .actions import Action, POINTER_ACTIONS

# Shortcuts the agent may never send: system-wide UI, logout, force quit, screenshots, trash.
BLOCKED_COMBOS = [
    {"cmd", "space"}, {"cmd", "tab"}, {"cmd", "alt", "escape"}, {"cmd", "shift", "q"},
    {"cmd", "alt", "shift", "q"}, {"ctrl", "cmd", "q"}, {"cmd", "shift", "3"}, {"cmd", "shift", "4"},
    {"cmd", "shift", "5"}, {"cmd", "delete"}, {"cmd", "shift", "delete"}, {"cmd", "alt", "shift", "delete"},
    {"cmd", "alt", "delete"}, {"ctrl", "up"}, {"ctrl", "down"}, {"cmd", "alt", "d"}, {"cmd", "h"}, {"cmd", "alt", "h"},
    {"cmd", "m"}, {"ctrl", "f2"}, {"ctrl", "fn", "f2"},
]
# Every app's menu bar carries the Apple menu (restart, shut down, log out ...): never let the agent in.
UNSAFE_MENU_ITEM = re.compile(r"restart|shut ?down|log ?out|sleep|lock screen|force quit|system settings|app store|"
                              r"recent items|system information|about this mac", re.I)


class UserAbort(RuntimeError):
    """The user moved the mouse (or pushed it into the top-left corner): stop immediately."""


@dataclass
class Obs:
    image: Image.Image  # resized for the model
    raw: Image.Image  # full resolution
    t: float


class MacEnv:
    def __init__(self, hide_others: bool = True, settle: float = 0.8, failsafe: bool = True, log=print,
                 backdrop: bool = False):
        if not ax.trusted():
            raise RuntimeError(
                "Accessibility permission missing for this process. Enable the app that runs Python "
                "(Terminal, iTerm, or Claude) in System Settings > Privacy & Security > Accessibility."
            )
        self.bounds = screen.main_bounds()
        self.hide_others = hide_others
        self.settle = settle
        self.failsafe = failsafe
        self.log = log
        self.hider = apps.Hider()
        self.allowed_pids: set[int] = set()
        self.allowed_bundles: set[str] = {apps.SAVE_PANEL}
        self.allow_trash = False
        # Optional per-task predicate on the AX element under a click (e.g. "inside my Finder window").
        self.guard = None
        self._cursor: tuple[float, float] | None = None
        # Data collection covers the Desktop (the user's files, widgets, wallpaper) with a neutral backdrop.
        self._backdrop = None
        if backdrop:
            self._backdrop = subprocess.Popen([sys.executable, "-m", "cua.backdrop"], cwd=str(config.ROOT),
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            atexit.register(self._stop_backdrop)
            time.sleep(1.5)

    # ---- confinement -------------------------------------------------------
    def allow(self, pid: int | None = None, bundle: str | None = None):
        if pid:
            self.allowed_pids.add(int(pid))
        if bundle:
            self.allowed_bundles.add(bundle)

    def clear_allowed(self):
        self.allowed_pids.clear()
        self.allowed_bundles = {apps.SAVE_PANEL}
        self.allow_trash = False
        self.guard = None

    def pid_allowed(self, p: int | None) -> bool:
        if p is None:
            return False
        if p in self.allowed_pids:
            return True
        a = NSRunningApplication.runningApplicationWithProcessIdentifier_(p)
        return a is not None and a.bundleIdentifier() in self.allowed_bundles

    def _check_point(self, x: float, y: float) -> str | None:
        el = ax.element_at(x, y)
        if el is None:
            return "no accessible element at point"
        p = ax.pid_of(el)
        if not self.pid_allowed(p):
            a = NSRunningApplication.runningApplicationWithProcessIdentifier_(p) if p else None
            return f"point belongs to {a.localizedName() if a else p}, outside the task"
        role, title = ax.get(el, "AXRole"), str(ax.get(el, "AXTitle") or "")
        if role == "AXMenuBarItem" and title == "Apple" or role == "AXMenuItem" and UNSAFE_MENU_ITEM.search(title):
            return f"system menu item {title!r} is off limits"
        if self.guard is not None and p in self.allowed_pids and not self.guard(el):
            return "point is outside the task window"
        return None

    def _check_keys(self, keys: list[str] | None) -> str | None:
        front = apps.frontmost_pid()
        if not self.pid_allowed(front):
            return "frontmost app is outside the task"
        if self.guard is not None:
            foc = ax.get(ax.app(front), "AXFocusedUIElement")
            if foc is not None and not self.guard(foc):
                return "keyboard focus is outside the task window"
        if keys is not None:
            ks = set(keys)
            for combo in BLOCKED_COMBOS:
                if combo == ks and not (self.allow_trash and combo == {"cmd", "delete"}):
                    return f"blocked shortcut {'+'.join(sorted(ks))}"
            for k in keys:
                if k not in inp.KEYCODES:
                    return f"unknown key {k!r}"
        return None

    def _failsafe(self):
        if not self.failsafe:
            return
        cx, cy = inp.cursor()
        if cx <= self.bounds.x + 1 and cy <= self.bounds.y + 1:
            raise UserAbort("cursor in top-left corner")
        if self._cursor is not None and math.dist((cx, cy), self._cursor) > 40:
            raise UserAbort(f"mouse moved by someone else: expected {self._cursor}, at {(cx, cy)}")

    # ---- observe / act ----------------------------------------------------
    def observe(self) -> Obs:
        raw = screen.capture()
        return Obs(image=screen.model_image(raw), raw=raw, t=time.time())

    def step(self, a: Action) -> dict:
        """Execute one action. Returns {"ok": bool, "blocked": reason or None}."""
        self._failsafe()
        reason = None
        if a.action in POINTER_ACTIONS:
            x, y = self.bounds.to_abs(*a.coordinate)
            if a.action != "mouse_move":
                reason = self._check_point(x, y)
            if a.action == "left_click_drag" and reason is None and self._cursor is not None:
                reason = self._check_point(*self._cursor)
            if reason is None:
                if a.action == "mouse_move":
                    inp.move(x, y)
                elif a.action == "left_click":
                    inp.click(x, y)
                elif a.action == "right_click":
                    inp.click(x, y, button="right")
                elif a.action == "double_click":
                    inp.click(x, y, count=2)
                elif a.action == "triple_click":
                    inp.click(x, y, count=3)
                elif a.action == "left_click_drag":
                    sx, sy = self._cursor if self._cursor else inp.cursor()
                    inp.drag(sx, sy, x, y)
        elif a.action == "type":
            reason = self._check_keys(None)
            if reason is None:
                inp.type_text(a.text or "")
        elif a.action == "key":
            reason = self._check_keys(a.keys)
            if reason is None:
                inp.key_combo(a.keys)
        elif a.action == "scroll":
            if a.coordinate is not None:
                x, y = self.bounds.to_abs(*a.coordinate)
            else:
                x, y = inp.cursor()
            reason = self._check_point(x, y)
            if reason is None:
                lines = int(max(1, min(15, abs(a.pixels or 3)))) * (1 if (a.pixels or -1) > 0 else -1)
                inp.scroll(lines, x, y)
        elif a.action == "wait":
            time.sleep(min(max(a.time or 1.0, 0.0), 5.0))
        # terminate: nothing to do
        time.sleep(self.settle if a.action not in ("wait", "terminate") else 0)
        self._cursor = inp.cursor()
        if reason:
            self.log(f"  blocked: {reason}")
        return {"ok": reason is None, "blocked": reason}

    # ---- episode boundaries ----------------------------------------------
    def reset(self, task) -> Obs:
        self.clear_allowed()
        task.setup(self)
        if self.hide_others:
            self.hider.hide_others(self.allowed_pids)
        task.focus(self)
        time.sleep(self.settle)
        self._cursor = inp.cursor()
        return self.observe()

    def teardown(self, task):
        try:
            inp.release_modifiers()
            task.teardown(self)
        finally:
            self._cursor = None

    def close(self):
        inp.release_modifiers()
        self.hider.restore()
        self._stop_backdrop()

    def _stop_backdrop(self):
        if self._backdrop is not None and self._backdrop.poll() is None:
            self._backdrop.terminate()
        self._backdrop = None

    # ---- helpers for tasks/oracles ------------------------------------------
    def rel_box(self, frame) -> list[float]:
        return self.bounds.rect_to_rel(*frame)

    def rel_center(self, frame) -> tuple[float, float]:
        x, y, w, h = frame
        return self.bounds.to_rel(x + w / 2, y + h / 2)
