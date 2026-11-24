"""Action schema, model-output parsing and formatting. Pure Python (shared with notebooks)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

POINTER_ACTIONS = {"left_click", "right_click", "double_click", "triple_click", "mouse_move", "left_click_drag"}
ALL_ACTIONS = POINTER_ACTIONS | {"key", "type", "scroll", "wait", "terminate"}

KEY_ALIASES = {
    "command": "cmd", "meta": "cmd", "super": "cmd", "win": "cmd", "⌘": "cmd",
    "control": "ctrl", "ctl": "ctrl",
    "option": "alt", "opt": "alt", "⌥": "alt",
    "enter": "return", "ret": "return",
    "esc": "escape",
    "backspace": "delete", "back_space": "delete", "del": "forwarddelete", "forward_delete": "forwarddelete",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "page_up": "pageup", "page_down": "pagedown", "spacebar": "space", " ": "space",
}
MODIFIERS = ("cmd", "ctrl", "alt", "shift", "fn")


def normalize_keys(keys) -> list[str]:
    if isinstance(keys, str):
        keys = re.split(r"\s*\+\s*|\s+", keys.strip()) if len(keys.strip()) > 1 else [keys]
    out = []
    for k in keys:
        k = str(k).strip()
        if not k:
            continue
        low = k.lower()
        out.append(KEY_ALIASES.get(low, low))
    # modifiers first, then the rest in given order
    return [k for k in out if k in MODIFIERS] + [k for k in out if k not in MODIFIERS]


@dataclass
class Action:
    action: str
    coordinate: tuple[float, float] | None = None  # relative 0..1000
    text: str | None = None
    keys: list[str] | None = None
    pixels: float | None = None
    time: float | None = None
    status: str | None = None
    # Not part of the tool call: free text the model wrote before it, and oracle metadata.
    thought: str = ""
    meta: dict = field(default_factory=dict)

    def args(self) -> dict:
        d = {"action": self.action}
        if self.coordinate is not None:
            d["coordinate"] = [int(round(self.coordinate[0])), int(round(self.coordinate[1]))]
        if self.keys is not None:
            d["keys"] = list(self.keys)
        if self.text is not None:
            d["text"] = self.text
        if self.pixels is not None:
            d["pixels"] = int(self.pixels)
        if self.time is not None:
            d["time"] = self.time
        if self.status is not None:
            d["status"] = self.status
        return d

    def to_dict(self) -> dict:
        return asdict(self)


def format_tool_call(action: Action) -> str:
    """Render in Qwen3.5's native XML tool-call format (as its chat template does)."""
    parts = ["<tool_call>\n<function=computer_use>\n"]
    for k, v in action.args().items():
        val = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
        parts.append(f"<parameter={k}>\n{val}\n</parameter>\n")
    parts.append("</function>\n</tool_call>")
    return "".join(parts)


def format_completion(action: Action) -> str:
    return (action.thought.strip() + "\n\n" if action.thought.strip() else "") + format_tool_call(action)


_TC_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FN_RE = re.compile(r"<function=([^>\n]+)>(.*?)(?:</function>|$)", re.S)
_PARAM_RE = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def _coerce(v):
    if isinstance(v, str):
        s = v.strip()
        try:
            return json.loads(s)
        except Exception:
            return s
    return v


def _parse_args_block(block: str) -> dict | None:
    fn = _FN_RE.search(block)
    if fn:
        return {k.strip(): _coerce(v) for k, v in _PARAM_RE.findall(fn.group(2))}
    # JSON style: {"name": "computer_use", "arguments": {...}}
    m = re.search(r"\{.*\}", block, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
        if isinstance(obj, dict):
            args = obj.get("arguments", obj)
            if isinstance(args, str):
                args = _coerce(args)
            return args if isinstance(args, dict) else None
    return None


def _coord(v):
    if isinstance(v, str):
        nums = re.findall(r"-?\d+(?:\.\d+)?", v)
        v = [float(n) for n in nums]
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        try:
            return (float(v[0]), float(v[1]))
        except (TypeError, ValueError):
            return None
    return None


def parse_action(text: str) -> Action | None:
    """Parse the first computer_use call in a model completion. Returns None if malformed."""
    if text is None:
        return None
    if "</think>" in text:
        text = text.split("</think>")[-1]
    m = _TC_RE.search(text)
    if not m:
        return None
    thought = text[: m.start()].strip()
    args = _parse_args_block(m.group(1))
    if not args or "action" not in args:
        return None
    kind = str(args["action"]).strip().lower()
    if kind not in ALL_ACTIONS:
        return None
    a = Action(action=kind, thought=thought)
    if kind in POINTER_ACTIONS:
        a.coordinate = _coord(args.get("coordinate"))
        if a.coordinate is None:
            return None
    if kind == "type":
        if args.get("text") is None:
            return None
        a.text = str(args["text"])
    if kind == "key":
        keys = args.get("keys", args.get("text"))
        if keys is None:
            return None
        a.keys = normalize_keys(keys)
        if not a.keys:
            return None
    if kind == "scroll":
        try:
            a.pixels = float(args.get("pixels", -3))
        except (TypeError, ValueError):
            return None
        a.coordinate = _coord(args.get("coordinate"))
    if kind == "wait":
        try:
            a.time = float(args.get("time", 1))
        except (TypeError, ValueError):
            a.time = 1.0
    if kind == "terminate":
        a.status = str(args.get("status", "success")).strip().lower()
    return a


def describe(action: Action) -> str:
    """One-line history entry fed back to the model on later steps."""
    lead = action.thought.strip().splitlines()[0][:120] if action.thought.strip() else ""
    args = json.dumps(action.args(), ensure_ascii=False)
    return f"{lead} {args}".strip() if lead else args
