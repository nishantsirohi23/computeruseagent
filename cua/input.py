"""Synthetic mouse and keyboard events via Quartz CGEvent (needs Accessibility permission)."""
from __future__ import annotations

import time

import Quartz as Q

# Carbon virtual key codes (US ANSI layout).
KEYCODES = {
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12,
    "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23,
    "=": 24, "9": 25, "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34,
    "p": 35, "return": 36, "l": 37, "j": 38, "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44,
    "n": 45, "m": 46, ".": 47, "tab": 48, "space": 49, "`": 50, "delete": 51, "escape": 53,
    "cmd": 55, "shift": 56, "capslock": 57, "alt": 58, "ctrl": 59, "fn": 63,
    "f5": 96, "f6": 97, "f7": 98, "f3": 99, "f8": 100, "f9": 101, "f11": 103, "f10": 109, "f12": 111,
    "home": 115, "pageup": 116, "forwarddelete": 117, "f4": 118, "end": 119, "f2": 120, "pagedown": 121,
    "f1": 122, "left": 123, "right": 124, "down": 125, "up": 126,
}
FLAGS = {
    "cmd": Q.kCGEventFlagMaskCommand,
    "shift": Q.kCGEventFlagMaskShift,
    "alt": Q.kCGEventFlagMaskAlternate,
    "ctrl": Q.kCGEventFlagMaskControl,
    "fn": Q.kCGEventFlagMaskSecondaryFn,
}

_SRC = None


def _source():
    global _SRC
    if _SRC is None:
        _SRC = Q.CGEventSourceCreate(Q.kCGEventSourceStateHIDSystemState)
    return _SRC


def _post(ev):
    Q.CGEventPost(Q.kCGHIDEventTap, ev)


def cursor() -> tuple[float, float]:
    p = Q.CGEventGetLocation(Q.CGEventCreate(None))
    return (float(p.x), float(p.y))


def _mouse(kind, x: float, y: float, button=Q.kCGMouseButtonLeft, click_state: int | None = None):
    ev = Q.CGEventCreateMouseEvent(_source(), kind, (x, y), button)
    Q.CGEventSetFlags(ev, 0)  # never inherit modifier state
    if click_state is not None:
        Q.CGEventSetIntegerValueField(ev, Q.kCGMouseEventClickState, click_state)
    _post(ev)


def move(x: float, y: float):
    _mouse(Q.kCGEventMouseMoved, x, y)
    time.sleep(0.02)


def click(x: float, y: float, button: str = "left", count: int = 1):
    if button == "right":
        down, up, btn = Q.kCGEventRightMouseDown, Q.kCGEventRightMouseUp, Q.kCGMouseButtonRight
    else:
        down, up, btn = Q.kCGEventLeftMouseDown, Q.kCGEventLeftMouseUp, Q.kCGMouseButtonLeft
    move(x, y)
    time.sleep(0.05)
    for i in range(1, count + 1):
        for kind in (down, up):
            _mouse(kind, x, y, btn, i)
            time.sleep(0.02)
        time.sleep(0.06)


def drag(x1: float, y1: float, x2: float, y2: float, steps: int = 20):
    move(x1, y1)
    _mouse(Q.kCGEventLeftMouseDown, x1, y1)
    time.sleep(0.1)
    for i in range(1, steps + 1):
        _mouse(Q.kCGEventLeftMouseDragged, x1 + (x2 - x1) * i / steps, y1 + (y2 - y1) * i / steps)
        time.sleep(0.015)
    time.sleep(0.1)
    _mouse(Q.kCGEventLeftMouseUp, x2, y2)


def scroll(lines: int, x: float | None = None, y: float | None = None):
    """Positive = scroll up (content moves down), negative = down."""
    if x is not None and y is not None:
        move(x, y)
    step = 1 if lines > 0 else -1
    for _ in range(abs(int(lines))):
        ev = Q.CGEventCreateScrollWheelEvent(_source(), Q.kCGScrollEventUnitLine, 1, step)
        Q.CGEventSetFlags(ev, 0)
        _post(ev)
        time.sleep(0.02)


def _key_event(code: int, down: bool, flags: int):
    ev = Q.CGEventCreateKeyboardEvent(_source(), code, down)
    Q.CGEventSetFlags(ev, flags)
    _post(ev)
    time.sleep(0.015)


def key_combo(keys: list[str]):
    """Press modifiers + one or more keys, e.g. ["cmd", "s"]. Unknown keys raise KeyError.

    Modifier keys get real down/up events with cumulative flags, ending at flags 0. Posting only
    flagged key events leaves the modifier "held" in the HID state, which turns later clicks into
    cmd-clicks.
    """
    mods = [k for k in keys if k in FLAGS]
    rest = [k for k in keys if k not in FLAGS]
    if not rest:  # modifier-only press
        rest, mods = mods[-1:], mods[:-1]
    flags = 0
    for m in mods:
        flags |= FLAGS[m]
        _key_event(KEYCODES[m], True, flags)
    for k in rest:
        code = KEYCODES[k]
        down_flags = flags | (FLAGS[k] if k in FLAGS else 0)
        _key_event(code, True, down_flags)
        _key_event(code, False, flags)
        time.sleep(0.02)
    for m in reversed(mods):
        flags &= ~FLAGS[m]
        _key_event(KEYCODES[m], False, flags)


def release_modifiers():
    """Safety net: release every modifier in the HID state."""
    for code in (56, 60, 55, 54, 58, 61, 59, 62, 63):
        _key_event(code, False, 0)


def type_text(text: str, interval: float = 0.012):
    for ch in text:
        if ch == "\n":
            key_combo(["return"])
            continue
        if ch == "\t":
            key_combo(["tab"])
            continue
        for is_down in (True, False):
            ev = Q.CGEventCreateKeyboardEvent(_source(), 0, is_down)
            Q.CGEventKeyboardSetUnicodeString(ev, len(ch.encode("utf-16-le")) // 2, ch)
            Q.CGEventSetFlags(ev, 0)
            _post(ev)
        time.sleep(interval)
