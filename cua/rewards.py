"""Verifiable rewards for GRPO. Pure Python (shared with notebooks).

Grounding (stage 1) follows the GUI-G1 recipe adapted to point outputs:
  - hit reward: click point inside the target element's box,
  - dense proximity term so near-misses rank above far misses,
  - difficulty weight: small targets get larger reward. With TRL's
    scale_rewards="none" this scales the group's advantages, which is how
    GUI-G1's per-query difficulty coefficient enters the GRPO update,
  - no length normalization (TRL loss_type="dr_grpo").

Step reward (stage 2) scores one action of a multi-step task against the
oracle action recorded in that state (plus any listed equivalent alternatives).
"""
from __future__ import annotations

import difflib
import json
import math

from .actions import Action, parse_action, POINTER_ACTIONS, normalize_keys


def _text(completion) -> str:
    if isinstance(completion, list):  # conversational: [{"role": "assistant", "content": ...}]
        return completion[0]["content"] if completion else ""
    return completion


def in_box(pt, box) -> bool:
    x, y = pt
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def proximity(pt, box) -> float:
    """Gaussian centred on the box, sigma = half the box size (min 8 units of 1000)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    sx = max((box[2] - box[0]) / 2, 8.0)
    sy = max((box[3] - box[1]) / 2, 8.0)
    d2 = ((pt[0] - cx) / sx) ** 2 + ((pt[1] - cy) / sy) ** 2
    return math.exp(-0.5 * d2)


def difficulty_weight(box) -> float:
    """1.0 for large targets up to 1.5 for tiny ones (area as a fraction of the screen)."""
    area = max((box[2] - box[0]) * (box[3] - box[1]) / 1e6, 1e-6)
    # area 1% of screen -> 1.0, 0.05% -> ~1.5
    w = 1.0 + 0.5 * (math.log10(0.01) - math.log10(area)) / (math.log10(0.01) - math.log10(0.0005))
    return float(min(max(w, 1.0), 1.5))


def grounding_score(action: Action | None, box) -> float:
    if action is None or action.action not in POINTER_ACTIONS or action.coordinate is None:
        return 0.0
    hit = 1.0 if in_box(action.coordinate, box) else 0.0
    return difficulty_weight(box) * (hit + 0.25 * proximity(action.coordinate, box))


def action_match(action: Action | None, gt: dict) -> float:
    """Score in [0, 1]: 0.3 for the right action type, 0.7 for the right arguments."""
    if action is None:
        return 0.0
    kind = gt["action"]
    if action.action != kind:
        # A different pointer click on the right element still beats nothing.
        if kind in POINTER_ACTIONS and action.action in POINTER_ACTIONS and gt.get("bbox") and action.coordinate:
            return 0.15 if in_box(action.coordinate, gt["bbox"]) else 0.0
        return 0.0
    arg = 0.0
    if kind in POINTER_ACTIONS:
        box = gt.get("bbox")
        if box and action.coordinate is not None:
            arg = 1.0 if in_box(action.coordinate, box) else 0.3 * proximity(action.coordinate, box)
    elif kind == "type":
        want, got = (gt.get("text") or "").strip(), (action.text or "").strip()
        if got == want:
            arg = 1.0
        else:
            r = difflib.SequenceMatcher(None, got, want).ratio()
            arg = 0.5 * r if r > 0.8 else 0.0
    elif kind == "key":
        arg = 1.0 if set(normalize_keys(action.keys or [])) == set(normalize_keys(gt.get("keys") or [])) else 0.0
    elif kind == "terminate":
        arg = 1.0 if (action.status or "success") == gt.get("status", "success") else 0.0
    elif kind == "scroll":
        arg = 1.0 if (action.pixels or 0) * (gt.get("pixels") or -1) > 0 else 0.0
    else:  # wait
        arg = 1.0
    return 0.3 + 0.7 * arg


def step_score(action: Action | None, gt: dict) -> float:
    cands = [gt] + list(gt.get("alts") or [])
    return max(action_match(action, c) for c in cands)


# ---- TRL reward functions: f(prompts, completions, **dataset_columns) -> list[float] ----

def format_reward(completions, **kwargs) -> list[float]:
    """+0.1 for one well-formed computer_use call, -0.5 for no parsable call or rambling text."""
    out = []
    for c in completions:
        t = _text(c)
        a = parse_action(t)
        if a is None:
            out.append(-0.5)
            continue
        r = 0.1
        if t.count("<tool_call>") > 1 or len(a.thought) > 400:
            r -= 0.2
        out.append(r)
    return out


def grounding_reward(completions, bbox, **kwargs) -> list[float]:
    return [grounding_score(parse_action(_text(c)), b) for c, b in zip(completions, bbox)]


def step_reward(completions, gt, **kwargs) -> list[float]:
    out = []
    for c, g in zip(completions, gt):
        g = json.loads(g) if isinstance(g, str) else g
        out.append(step_score(parse_action(_text(c)), g))
    return out
