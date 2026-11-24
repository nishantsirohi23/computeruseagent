"""Thin wrapper over the macOS Accessibility (AX) API via PyObjC."""
from __future__ import annotations

from dataclasses import dataclass, field

import ApplicationServices as AS
from CoreFoundation import CGPoint, CGSize

_POINT = getattr(AS, "kAXValueCGPointType", getattr(AS, "kAXValueTypeCGPoint", 1))
_SIZE = getattr(AS, "kAXValueCGSizeType", getattr(AS, "kAXValueTypeCGSize", 2))

TEXT_ATTRS = ("AXTitle", "AXDescription", "AXValue", "AXHelp", "AXPlaceholderValue", "AXIdentifier")


class AXError(RuntimeError):
    pass


def trusted(prompt: bool = False) -> bool:
    if prompt:
        return bool(AS.AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True}))
    return bool(AS.AXIsProcessTrusted())


def app(pid: int):
    el = AS.AXUIElementCreateApplication(pid)
    AS.AXUIElementSetMessagingTimeout(el, 2.0)
    return el


def system_wide():
    return AS.AXUIElementCreateSystemWide()


def get(el, name, default=None):
    err, val = AS.AXUIElementCopyAttributeValue(el, name, None)
    if err != 0:
        return default
    return val


def names(el) -> list[str]:
    err, val = AS.AXUIElementCopyAttributeNames(el, None)
    return list(val) if err == 0 and val else []


def actions(el) -> list[str]:
    err, val = AS.AXUIElementCopyActionNames(el, None)
    return list(val) if err == 0 and val else []


def set_attr(el, name, value) -> bool:
    return AS.AXUIElementSetAttributeValue(el, name, value) == 0


def perform(el, action: str = "AXPress") -> bool:
    return AS.AXUIElementPerformAction(el, action) == 0


def pid_of(el) -> int | None:
    err, pid = AS.AXUIElementGetPid(el, None)
    return pid if err == 0 else None


def frame(el) -> tuple[float, float, float, float] | None:
    pos, size = get(el, "AXPosition"), get(el, "AXSize")
    if pos is None or size is None:
        return None
    ok1, p = AS.AXValueGetValue(pos, _POINT, None)
    ok2, s = AS.AXValueGetValue(size, _SIZE, None)
    if not (ok1 and ok2):
        return None
    return (float(p.x), float(p.y), float(s.width), float(s.height))


def set_frame(el, x: float, y: float, w: float | None = None, h: float | None = None) -> bool:
    ok = set_attr(el, "AXPosition", AS.AXValueCreate(_POINT, CGPoint(x, y)))
    if w is not None and h is not None:
        ok = set_attr(el, "AXSize", AS.AXValueCreate(_SIZE, CGSize(w, h))) and ok
    return ok


def element_at(x: float, y: float):
    err, el = AS.AXUIElementCopyElementAtPosition(system_wide(), float(x), float(y), None)
    return el if err == 0 else None


def _s(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    return ""  # AXValueRefs, elements, arrays: not text


@dataclass
class Node:
    el: object = field(repr=False)
    role: str
    subrole: str
    title: str
    desc: str
    value: str
    help: str
    placeholder: str
    ident: str
    frame: tuple | None
    enabled: bool
    depth: int
    path: str
    title_ui: str = ""  # text of a linked label (AXTitleUIElement), for text fields

    @property
    def label(self) -> str:
        for t in (self.title, self.desc, self.title_ui, self.placeholder, self.help):
            if t and t.strip():
                return t.strip()
        if self.role in ("AXStaticText", "AXButton", "AXMenuItem", "AXCell", "AXRadioButton") and self.value.strip():
            return self.value.strip()
        return ""


def node(el, depth: int = 0, path: str = "") -> Node:
    title_ui = ""
    tui = get(el, "AXTitleUIElement")
    if tui is not None:
        title_ui = _s(get(tui, "AXValue")) or _s(get(tui, "AXTitle"))
    en = get(el, "AXEnabled")
    return Node(
        el=el,
        role=_s(get(el, "AXRole")),
        subrole=_s(get(el, "AXSubrole")),
        title=_s(get(el, "AXTitle")),
        desc=_s(get(el, "AXDescription")),
        value=_s(get(el, "AXValue"))[:200],
        help=_s(get(el, "AXHelp")),
        placeholder=_s(get(el, "AXPlaceholderValue")),
        ident=_s(get(el, "AXIdentifier")),
        frame=frame(el),
        enabled=True if en is None else bool(en),
        depth=depth,
        path=path,
        title_ui=title_ui,
    )


def children(el) -> list:
    kids = get(el, "AXChildren")
    return list(kids) if kids else []


def walk(root, max_nodes: int = 4000, max_depth: int = 40, max_children: int = 300) -> list[Node]:
    out: list[Node] = []
    stack = [(root, 0, "0")]
    while stack and len(out) < max_nodes:
        el, d, p = stack.pop()
        n = node(el, d, p)
        out.append(n)
        if d >= max_depth:
            continue
        kids = children(el)[:max_children]
        for i in range(len(kids) - 1, -1, -1):
            stack.append((kids[i], d + 1, f"{p}.{i}"))
    return out


def windows(app_el) -> list:
    w = get(app_el, "AXWindows")
    return list(w) if w else []


def find(nodes: list[Node], **crit) -> list[Node]:
    """Filter nodes by exact field values; `label=` matches the computed label, `contains=` is a label substring."""
    res = []
    for n in nodes:
        ok = True
        for k, v in crit.items():
            if k == "contains":
                ok = v.lower() in n.label.lower()
            elif k == "label":
                ok = n.label == v
            else:
                ok = getattr(n, k) == v
            if not ok:
                break
        if ok:
            res.append(n)
    return res


def is_ancestor_or_self(anc, el, max_up: int = 30) -> bool:
    cur = el
    for _ in range(max_up):
        if cur is None:
            return False
        if cur == anc:
            return True
        cur = get(cur, "AXParent")
    return False
