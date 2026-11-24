"""Launching, activating, placing and hiding applications."""
from __future__ import annotations

import subprocess
import time

from AppKit import NSDate, NSRunLoop, NSRunningApplication, NSWorkspace

from . import ax

FINDER = "com.apple.finder"
TEXTEDIT = "com.apple.TextEdit"
CALCULATOR = "com.apple.calculator"
# Hosts the open/save sheets of sandboxed apps (e.g. TextEdit's Save dialog).
SAVE_PANEL = "com.apple.appkit.xpc.openAndSavePanelService"


def osascript(script: str, timeout: float = 20) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"osascript failed: {r.stderr.strip()}")
    return r.stdout.strip()


def _pump():
    NSRunLoop.currentRunLoop().runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.01))


def running(bundle_id: str) -> list:
    _pump()  # the running-apps list is also notification-driven
    return list(NSRunningApplication.runningApplicationsWithBundleIdentifier_(bundle_id))


def pid(bundle_id: str) -> int | None:
    apps = running(bundle_id)
    return int(apps[0].processIdentifier()) if apps else None


def frontmost():
    # NSWorkspace updates frontmostApplication from notifications, which only arrive while a run loop
    # spins; scripts have none, so pump it briefly or the answer is stale.
    _pump()
    return NSWorkspace.sharedWorkspace().frontmostApplication()


def frontmost_pid() -> int | None:
    a = frontmost()
    return int(a.processIdentifier()) if a else None


def open_with(bundle_id: str, *paths: str, background: bool = False):
    cmd = ["open", "-b", bundle_id] + (["-g"] if background else []) + list(paths)
    subprocess.run(cmd, check=True, timeout=30)


def wait_for(cond, timeout: float = 10.0, interval: float = 0.15):
    end = time.time() + timeout
    while time.time() < end:
        v = cond()
        if v:
            return v
        time.sleep(interval)
    return None


def launch(bundle_id: str, *paths: str, timeout: float = 15.0) -> int:
    open_with(bundle_id, *paths)
    p = wait_for(lambda: pid(bundle_id), timeout)
    if p is None:
        raise RuntimeError(f"{bundle_id} did not start")
    wait_for(lambda: ax.windows(ax.app(p)), timeout)
    return p


def activate_pid(p: int) -> bool:
    ax.set_attr(ax.app(p), "AXFrontmost", True)
    if wait_for(lambda: frontmost_pid() == p, 2.0):
        return True
    a = NSRunningApplication.runningApplicationWithProcessIdentifier_(p)
    if a is not None:
        a.activateWithOptions_(0)
    return bool(wait_for(lambda: frontmost_pid() == p, 2.0))


def quit(bundle_id: str, timeout: float = 5.0):
    for a in running(bundle_id):
        a.terminate()
    if not wait_for(lambda: not running(bundle_id), timeout):
        for a in running(bundle_id):
            a.forceTerminate()


def main_window(p: int):
    app_el = ax.app(p)
    w = ax.get(app_el, "AXMainWindow") or ax.get(app_el, "AXFocusedWindow")
    if w is None:
        ws = ax.windows(app_el)
        w = ws[0] if ws else None
    return w


def place_window(win, x: float, y: float, w: float | None = None, h: float | None = None):
    ax.set_frame(win, x, y, w, h)
    ax.perform(win, "AXRaise")


class Hider:
    """Hides every other regular app while tasks run; restores them afterwards."""

    def __init__(self):
        self.hidden = []

    def hide_others(self, keep_pids: set[int]):
        me = NSRunningApplication.currentApplication().processIdentifier()
        for a in NSWorkspace.sharedWorkspace().runningApplications():
            p = int(a.processIdentifier())
            if a.activationPolicy() != 0 or p in keep_pids or p == me:  # 0 = regular app
                continue
            if not a.isHidden() and a.hide():
                self.hidden.append(a)

    def restore(self):
        for a in self.hidden:
            try:
                a.unhide()
            except Exception:
                pass
        self.hidden = []
