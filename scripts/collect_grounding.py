"""Collect stage-1 grounding data: screenshots + "click X" instructions + AX target boxes.

    python scripts/collect_grounding.py --synth 150 --real 36 --out data/grounding/run1

Scenes: random SynthApp forms (sometimes with a pop-up or menu open), Calculator, TextEdit and a
sandbox Finder window, each also with menu-bar menus opened. Only elements that are on the main
display, labelled, unique on screen and actually hit at their centre (AX hit-test) are used.
Never captures apps other than these; everything else is hidden during collection.
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cua import apps, ax, input as inp  # noqa: E402
from cua.env import MacEnv, UserAbort  # noqa: E402
from cua.synthapp import spec as S  # noqa: E402
from cua.tasks.calculator import CalculatorTask  # noqa: E402
from cua.tasks.finder import FinderTask  # noqa: E402
from cua.tasks.synth import SynthTask  # noqa: E402
from cua.tasks.textedit import TextEditTask  # noqa: E402

CLICKABLE = {"AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXTextField", "AXSearchField",
             "AXMenuBarItem", "AXMenuItem", "AXSlider", "AXIncrementor", "AXImage", "AXCell", "AXRow", "AXStaticText",
             "AXTextArea", "AXComboBox", "AXDisclosureTriangle", "AXTab"}
WINDOW_BUTTONS = {"AXCloseButton": "close", "AXMinimizeButton": "minimize", "AXZoomButton": "zoom (full screen)"}


def phrasing(role, subrole, label, rng, ctx=""):
    q = f'"{label}"'
    if subrole in WINDOW_BUTTONS:
        return rng.choice([f"Click the window's {WINDOW_BUTTONS[subrole]} button.",
                           f"Click the {WINDOW_BUTTONS[subrole]} button in the title bar."])
    table = {
        "AXButton": [f"Click the {q} button.", f"Press {q}.", f"Click {q}."],
        "AXCheckBox": [f"Click the {q} checkbox.", f"Toggle {q}."],
        "AXRadioButton": [f"Click the {q} option.", f"Select {q}."],
        "AXPopUpButton": [f"Open the {label} pop-up menu.", f"Click the {label} pop-up button."],
        "AXMenuButton": [f"Click the {label} menu button."],
        "AXTextField": [f"Click the {label} text field.", f"Click into the {label} field."],
        "AXSearchField": [f"Click the {label} search field."],
        "AXMenuBarItem": [f"Open the {q} menu in the menu bar.", f"Click {q} in the menu bar."],
        "AXMenuItem": [f"Click {q} in the open menu.", f"Choose {q} from the menu."],
        "AXSlider": [f"Click the {label} slider."],
        "AXImage": [f"Click the {q} icon.", f"Click on {q}."],
        "AXCell": [f"Click {q}.", f"Select {q}."],
        "AXRow": [f"Click the row {q}.", f"Select {q}."],
        "AXStaticText": [f"Click the text {q}.", f"Click on {q}."],
        "AXTextArea": [f"Click in the {label} text area."],
        "AXDisclosureTriangle": [f"Click the disclosure triangle for {q}."],
    }
    s = rng.choice(table.get(role, [f"Click {q}."]))
    return s if not ctx else s[:-1] + f" {ctx}."


def synth_labels(spec) -> dict:
    """identifier -> human label from the spec (form controls are not AX-linked to their label text)."""
    m = {}
    for w, _ in S.all_widgets(spec):
        m[w["id"]] = w.get("label", "")
        for o in w.get("options", []):
            m[f'{w["id"]}::{o}'] = f'{o}" option under "{w.get("label", "")}'
    return m


def _private_re():
    """Labels that would leak the user's identity or devices into an uploaded dataset."""
    from Foundation import NSFullUserName, NSUserName

    toks = {t for t in str(NSFullUserName()).split() if len(t) >= 3} | {str(NSUserName()), "iPhone", "iPad"}
    return re.compile("|".join(re.escape(t) for t in sorted(toks)), re.I)


PRIVATE = _private_re()


def candidates(env, pids, label_map=None):
    out = []
    for p in pids:
        for n in ax.walk(ax.app(p)):
            # Finder's process also owns the Desktop: keep only the task window and menus.
            if env.guard is not None and not env.guard(n.el):
                continue
            if n.role not in CLICKABLE and n.subrole not in WINDOW_BUTTONS:
                continue
            if not n.frame or n.frame[2] < 6 or n.frame[3] < 6 or not n.enabled:
                continue
            x, y, w, h = n.frame
            if not (env.bounds.contains(x, y) and env.bounds.contains(x + w - 1, y + h - 1)):
                continue
            if w * h > 0.2 * env.bounds.w * env.bounds.h:
                continue
            label = (label_map or {}).get(n.ident) or n.label
            if n.subrole not in WINDOW_BUTTONS and (not label or len(label) > 60):
                continue
            if label and PRIVATE.search(label):
                continue
            if n.role == "AXMenuBarItem" and n.title == "Apple" or label.lower().startswith("this button"):
                continue  # the agent may never open the Apple menu; AX "this button ..." blurbs are not labels
            hit = ax.element_at(x + w / 2, y + h / 2)
            if hit is None or not ax.is_ancestor_or_self(n.el, hit, 4):
                continue
            out.append((n, label))
    # drop ambiguous labels (same text twice on screen)
    counts = Counter(l for n, l in out if l)
    return [(n, l) for n, l in out if not l or counts[l] == 1]


def save_scene(env, scene_dir: Path, scene: str, source: str, pids, rng, max_samples=20, label_map=None, ctx=None) -> int:
    obs = env.observe()
    cands = candidates(env, pids, label_map)
    rng.shuffle(cands)
    cands = cands[:max_samples]
    if not cands:
        return 0
    scene_dir.mkdir(parents=True, exist_ok=True)
    obs.image.save(scene_dir / "screen.png")
    rows = []
    for n, label in cands:
        role = n.role
        if label_map and n.ident and "::" in n.ident and role == "AXRadioButton":
            instr = f'Select the "{label}".'
        else:
            instr = phrasing(role, n.subrole, label, rng, (ctx or {}).get(role, ""))
        rows.append({"image": "screen.png", "instruction": instr, "bbox": env.rel_box(n.frame), "scene": scene,
                     "source": source, "role": role, "element": label})
    with (scene_dir / "rows.jsonl").open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def open_menu(pid, title) -> bool:
    bar = ax.get(ax.app(pid), "AXMenuBar")
    for n in ax.walk(bar, max_depth=1)[1:]:
        if n.title == title and n.frame:
            x, y, w, h = n.frame
            inp.click(x + w / 2, y + h / 2)
            time.sleep(0.6)
            return True
    return False


def close_menu():
    inp.key_combo(["escape"])
    time.sleep(0.3)


def menu_titles(pid) -> list[str]:
    bar = ax.get(ax.app(pid), "AXMenuBar")
    return [n.title for n in ax.walk(bar, max_depth=1)[1:] if n.title and n.title != "Apple"][1:]


def check_abort(env):
    x, y = inp.cursor()
    if x <= env.bounds.x + 1 and y <= env.bounds.y + 1:
        raise UserAbort("cursor in top-left corner")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", type=int, default=150)
    ap.add_argument("--real", type=int, default=36)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    rng = random.Random(args.seed)
    env = MacEnv(hide_others=True, failsafe=True, backdrop=True)
    total = 0
    try:
        for i in range(args.synth):
            check_abort(env)
            task = SynthTask.sample(rng, f"g{args.seed}_synth_{i:04d}")
            if (out / task.id / "rows.jsonl").exists():  # resuming an aborted run
                continue
            env.reset(task)
            try:
                lm = synth_labels(task.params["spec"])
                total += save_scene(env, out / task.id, task.id, "synth", [task.pid], rng, label_map=lm)
                popups = [w for w, tab in S.all_widgets(task.params["spec"]) if w["type"] == "popup" and tab is None]
                if popups and rng.random() < 0.5:  # a scene with an open pop-up menu
                    w = rng.choice(popups)
                    n = task.find_one(env, ident=w["id"])
                    x, y, ww, hh = n.frame
                    inp.click(x + ww / 2, y + hh / 2)
                    time.sleep(0.5)
                    total += save_scene(env, out / (task.id + "_popup"), task.id + "_popup", "synth", [task.pid], rng,
                                        label_map=lm)
                    close_menu()
                elif task.params["spec"].get("menus") and rng.random() < 0.4:
                    if open_menu(task.pid, task.params["spec"]["menus"][0]["title"]):
                        total += save_scene(env, out / (task.id + "_menu"), task.id + "_menu", "synth", [task.pid], rng,
                                            label_map=lm)
                        close_menu()
            finally:
                env.teardown(task)
            print(f"synth {i + 1}/{args.synth}: {total} samples")
        makers = [("calculator", CalculatorTask, "arith2"), ("textedit", TextEditTask, "append_line"),
                  ("finder", FinderTask, "select")]
        for i in range(args.real):
            fam, cls, tmpl = makers[i % len(makers)]
            check_abort(env)
            task = cls.sample(rng, f"g{args.seed}_{fam}_{i:04d}", tmpl)
            if (out / task.id / "rows.jsonl").exists():
                continue
            env.reset(task)
            try:
                if fam == "calculator" and rng.random() < 0.5:
                    task.menu_press(["View", rng.choice(["Scientific", "Programmer", "Basic"])])
                    time.sleep(0.6)
                    w = apps.main_window(task.pid)
                    if w is not None:
                        apps.place_window(w, 40, 60)
                if fam == "finder" and rng.random() < 0.4:
                    task.finder(f"set current view of front Finder window to {rng.choice(['list view', 'column view', 'icon view'])}")
                    time.sleep(0.5)
                total += save_scene(env, out / task.id, task.id, fam, [task.pid], rng)
                titles = menu_titles(task.pid)
                for t in rng.sample(titles, min(2, len(titles))):
                    if open_menu(task.pid, t):
                        total += save_scene(env, out / f"{task.id}_{t}", f"{task.id}_{t}", fam, [task.pid], rng,
                                            ctx={"AXMenuItem": ""})
                        close_menu()
            finally:
                env.teardown(task)
            print(f"real {i + 1}/{args.real} ({fam}): {total} samples")
    except UserAbort as e:
        print(f"\nABORTED by user: {e}")
    finally:
        env.close()
    print(f"total samples: {total}")


if __name__ == "__main__":
    main()
