"""Random SynthApp specs and goals (what the agent must set) for training and benchmark tasks."""
from __future__ import annotations

import random

NAMES = ["Ana Ruiz", "Ken Ito", "Priya Shah", "Tom Becker", "Lena Novak", "Omar Haddad", "Mia Chen", "Luis Ortega",
         "Sara Lind", "Ravi Kumar", "Zoe Martin", "Ivan Petrov", "Nora Quinn", "Yuki Sato", "Ella Brown"]
WORDS = ["alpha", "harbor", "maple", "orbit", "cedar", "nova", "pixel", "summit", "delta", "willow", "ember", "quartz"]
COUNTRIES = ["Canada", "India", "Japan", "Spain", "Brazil", "Kenya", "Norway", "Mexico", "Italy", "Chile", "Egypt"]
CITIES = ["Lisbon", "Oslo", "Kyoto", "Lima", "Cairo", "Perth", "Quito", "Riga", "Hanoi", "Dakar", "Bergen", "Porto"]
LANGS = ["English", "Hindi", "Japanese", "Spanish", "French", "German", "Korean", "Swahili"]
COLORS = ["Red", "Green", "Blue", "Orange", "Purple", "Graphite", "Teal"]

DOMAINS = {
    "Account Settings": {
        "text": [("name", "Full name", NAMES), ("user", "Username", WORDS), ("company", "Company", WORDS),
                 ("email", "Email", [f"{w}@example.com" for w in WORDS])],
        "checkbox": [("news", "Subscribe to newsletter"), ("tfa", "Enable two-factor authentication"),
                     ("remember", "Remember me"), ("terms", "Accept the terms")],
        "radio": [("plan", "Plan", ["Free", "Pro", "Team"]), ("billing", "Billing", ["Monthly", "Yearly"])],
        "popup": [("country", "Country", COUNTRIES), ("lang", "Language", LANGS)],
        "segmented": [("theme", "Theme", ["Light", "Dark", "Auto"])],
        "buttons": [["Cancel", "Save"], ["Reset", "Apply"], ["Cancel", "Update"]],
        "menus": [("Account", ["Sign Out", "Verify Email", "Export Data"])],
    },
    "Shipping": {
        "text": [("recipient", "Recipient", NAMES), ("street", "Street", [f"{n} {w.title()} Rd" for n, w in zip(range(10, 99, 7), WORDS)]),
                 ("city", "City", CITIES), ("zip", "ZIP code", ["10115", "94107", "560001", "75001", "2000", "0150"])],
        "checkbox": [("gift", "Gift wrap"), ("sig", "Signature required"), ("insure", "Add insurance")],
        "radio": [("speed", "Delivery", ["Standard", "Express", "Overnight"])],
        "popup": [("carrier", "Carrier", ["PostNord", "DHL", "UPS", "FedEx", "USPS"]), ("country", "Country", COUNTRIES)],
        "stepper": [("qty", "Quantity", 1, 9), ("boxes", "Boxes", 1, 6)],
        "buttons": [["Back", "Place Order"], ["Cancel", "Continue"]],
        "menus": [("Order", ["Track Package", "Print Label", "Duplicate Order"])],
    },
    "Preferences": {
        "checkbox": [("hidden", "Show hidden files"), ("sounds", "Play sounds"), ("autoupd", "Check for updates automatically"),
                     ("login", "Launch at login"), ("badge", "Show badge on icon")],
        "slider": [("vol", "Volume"), ("bright", "Brightness"), ("speed", "Scroll speed")],
        "popup": [("browser", "Default browser", ["Safari", "Firefox", "Chrome", "Arc", "Edge"]),
                  ("accent", "Accent color", COLORS)],
        "segmented": [("sidebar", "Sidebar", ["Left", "Right", "Hidden"]), ("size", "Icon size", ["Small", "Medium", "Large"])],
        "stepper": [("history", "Keep history (days)", 1, 30)],
        "buttons": [["Revert", "Done"], ["Defaults", "OK"]],
        "menus": [("Tools", ["Clear Cache", "Rebuild Index", "Reset Warnings"])],
    },
    "Booking": {
        "text": [("guest", "Guest name", NAMES), ("code", "Promo code", [w.upper() + "10" for w in WORDS])],
        "popup": [("room", "Room type", ["Single", "Double", "Suite", "Family"])],
        "stepper": [("guests", "Guests", 1, 8), ("nights", "Nights", 1, 14)],
        "radio": [("breakfast", "Breakfast", ["Included", "Not included"])],
        "list": [("dest", "Destination", CITIES)],
        "checkbox": [("pets", "Travelling with pets"), ("late", "Late check-in")],
        "buttons": [["Clear", "Book Now"], ["Cancel", "Reserve"]],
        "menus": [("Booking", ["Check Availability", "Show Map", "Email Itinerary"])],
    },
    "Survey": {
        "radio": [("sat", "Satisfaction", ["Poor", "Okay", "Good", "Great"]), ("again", "Would buy again", ["Yes", "No", "Maybe"])],
        "slider": [("rating", "Rating"), ("likely", "Likelihood to recommend")],
        "checkbox": [("contact", "You may contact me"), ("anon", "Submit anonymously")],
        "text": [("comment", "Comments", ["fast shipping", "great support", "too pricey", "works well"]),
                 ("nick", "Nickname", WORDS)],
        "list": [("product", "Product", ["Notebook", "Backpack", "Headphones", "Desk Lamp", "Keyboard", "Monitor", "Webcam"])],
        "buttons": [["Skip", "Submit"], ["Back", "Send"]],
    },
    "Editor Settings": {
        "popup": [("font", "Font", ["Menlo", "Monaco", "SF Mono", "Courier", "Fira Code"]),
                  ("enc", "Encoding", ["UTF-8", "UTF-16", "Latin-1", "ASCII"])],
        "stepper": [("tab", "Tab width", 1, 8), ("ruler", "Ruler column", 60, 120)],
        "checkbox": [("wrap", "Wrap lines"), ("nums", "Show line numbers"), ("trim", "Trim trailing spaces"),
                     ("minimap", "Show minimap")],
        "segmented": [("eol", "Line endings", ["LF", "CRLF", "CR"])],
        "slider": [("zoom", "Zoom")],
        "buttons": [["Cancel", "Save"], ["Reset", "Apply"]],
        "menus": [("Format", ["Indent Selection", "Convert to Spaces", "Sort Lines"])],
    },
}


def _widget(kind, entry, rng: random.Random):
    if kind == "text":
        wid, label, pool = entry
        w = {"type": "text", "id": wid, "label": label, "pool": pool}
        if rng.random() < 0.3:
            w["value"] = rng.choice(pool)
        elif rng.random() < 0.5:
            w["placeholder"] = rng.choice(pool)
        return w
    if kind == "checkbox":
        wid, label = entry
        return {"type": "checkbox", "id": wid, "label": label, "value": rng.random() < 0.4}
    if kind == "radio":
        wid, label, opts = entry
        return {"type": "radio", "id": wid, "label": label, "options": opts, "value": rng.choice(opts),
                "horizontal": rng.random() < 0.6}
    if kind == "popup":
        wid, label, opts = entry
        opts = rng.sample(opts, k=min(len(opts), rng.randint(3, 6)))
        return {"type": "popup", "id": wid, "label": label, "options": opts, "value": rng.choice(opts)}
    if kind == "segmented":
        wid, label, opts = entry
        return {"type": "segmented", "id": wid, "label": label, "options": opts, "value": rng.choice(opts)}
    if kind == "slider":
        wid, label = entry
        return {"type": "slider", "id": wid, "label": label, "min": 0, "max": 100, "value": rng.choice([10, 25, 50, 60, 75])}
    if kind == "stepper":
        wid, label, lo, hi = entry
        return {"type": "stepper", "id": wid, "label": label, "min": lo, "max": hi, "value": rng.randint(lo, min(hi, lo + 3))}
    if kind == "list":
        wid, label, opts = entry
        opts = rng.sample(opts, k=min(len(opts), rng.randint(4, 6)))
        return {"type": "list", "id": wid, "label": label, "options": opts, "visible_rows": len(opts)}
    raise ValueError(kind)


def random_spec(rng: random.Random, origin=(40, 60)) -> dict:
    title = rng.choice(list(DOMAINS))
    dom = DOMAINS[title]
    kinds = [k for k in dom if k not in ("buttons", "menus")]
    pool = [(k, e) for k in kinds for e in dom[k]]
    rng.shuffle(pool)
    n = rng.randint(3, min(7, len(pool)))
    used, widgets = set(), []
    for k, e in pool:
        if len(widgets) >= n:
            break
        if e[0] in used:
            continue
        used.add(e[0])
        widgets.append(_widget(k, e, rng))
    size = rng.choices(["small", "regular", "large"], weights=[2, 6, 2])[0]
    spec = {
        "title": title if rng.random() < 0.7 else f"{title} — {rng.choice(WORDS).title()}",
        "heading": title if rng.random() < 0.5 else "",
        "app_name": "Form",
        "width": rng.choice([420, 460, 500, 540, 600]),
        "label_w": rng.choice([110, 130, 150, 170]),
        "control_w": rng.choice([180, 220, 260]),
        "gap": rng.choice([8, 10, 12, 16]),
        "pad": rng.choice([16, 20, 24]),
        "control_size": size,
        "appearance": "dark" if rng.random() < 0.3 else "light",
        "origin": list(origin),
        "widgets": [],
        "tabs": [],
        "footer": [],
    }
    if len(widgets) >= 5 and rng.random() < 0.35:
        cut = rng.randint(2, len(widgets) - 2)
        spec["widgets"] = widgets[:cut]
        rest = widgets[cut:]
        mid = max(1, len(rest) // 2)
        names = rng.choice([("General", "Advanced"), ("Basic", "More"), ("Details", "Options")])
        spec["tabs"] = [{"title": names[0], "widgets": rest[:mid]}, {"title": names[1], "widgets": rest[mid:]}]
        spec["tab_h"] = 60 + 40 * max(len(rest[:mid]), len(rest[mid:])) + 40 * any(w["type"] in ("list", "radio") for w in rest) * 3
    else:
        spec["widgets"] = widgets
    labels = rng.choice(dom["buttons"])
    spec["footer"] = [{"type": "buttons", "items": [{"id": l.lower().replace(" ", "_"), "label": l} for l in labels]}]
    if dom.get("menus") and rng.random() < 0.5:
        t, items = rng.choice(dom["menus"])
        spec["menus"] = [{"title": t, "items": items}]
    # SynthApp ignores "pool"; it is only used to pick goal values.
    return spec


def all_widgets(spec) -> list[tuple[dict, str | None]]:
    """(widget, tab title or None) for every input widget."""
    out = [(w, None) for w in spec["widgets"]]
    for t in spec.get("tabs", []):
        out += [(w, t["title"]) for w in t["widgets"]]
    return out


def buttons(spec) -> list[dict]:
    return [b for row in spec["footer"] if row["type"] == "buttons" for b in row["items"]]


def random_goal_value(w, rng: random.Random):
    k = w["type"]
    if k == "text":
        choices = [v for v in w["pool"] if v != w.get("value", "")]
        return rng.choice(choices)
    if k == "checkbox":
        return not w["value"]
    if k in ("radio", "popup", "segmented"):
        return rng.choice([o for o in w["options"] if o != w.get("value")])
    if k == "slider":
        return rng.choice([v for v in (0, 20, 40, 80, 100) if abs(v - w["value"]) >= 15])
    if k == "stepper":
        lo, hi = w["min"], w["max"]
        return rng.choice([v for v in range(lo, min(hi, w["value"] + 4) + 1) if v != w["value"]])
    if k == "list":
        return rng.choice(w["options"])
    raise ValueError(k)


def phrase(w, v, rng: random.Random) -> str:
    k, label = w["type"], w.get("label", "")
    if k == "text":
        return rng.choice([f'type "{v}" in the {label} field', f'enter "{v}" as the {label}', f'set {label} to "{v}"'])
    if k == "checkbox":
        if v:
            return rng.choice([f'check "{label}"', f'turn on "{label}"', f'enable "{label}"'])
        return rng.choice([f'uncheck "{label}"', f'turn off "{label}"', f'disable "{label}"'])
    if k == "radio":
        return rng.choice([f'select "{v}" for {label}', f'choose the "{v}" option under {label}', f'set {label} to {v}'])
    if k == "popup":
        return rng.choice([f"choose {v} from the {label} pop-up menu", f"set {label} to {v}", f"pick {v} as the {label}"])
    if k == "segmented":
        return rng.choice([f"switch {label} to {v}", f"select {v} in the {label} control", f"set {label} to {v}"])
    if k == "slider":
        return rng.choice([f"set the {label} slider to about {v}%", f"drag {label} to roughly {v} percent"]) if v not in (0, 100) else \
            rng.choice([f"set {label} to the {'maximum' if v == 100 else 'minimum'}",
                        f"move the {label} slider all the way to the {'right' if v == 100 else 'left'}"])
    if k == "stepper":
        return rng.choice([f"set {label} to {v}", f"change {label} to {v} using the stepper arrows"])
    if k == "list":
        return rng.choice([f"select {v} in the {label} list", f"pick {v} from the {label} list"])
    raise ValueError(k)


def random_goal(spec, rng: random.Random, max_fields: int = 3) -> dict:
    ws = all_widgets(spec)
    n = rng.randint(1, min(max_fields, len(ws)))
    chosen = rng.sample(ws, n)
    goals = []
    for w, tab in chosen:
        goals.append({"id": w["id"], "type": w["type"], "value": random_goal_value(w, rng), "tab": tab})
    submit = None
    if rng.random() < 0.65:
        btns = buttons(spec)
        submit = btns[-1]["id"]  # the affirmative button is last
    menu = None
    if spec.get("menus") and rng.random() < 0.2:
        m = spec["menus"][0]
        menu = [m["title"], rng.choice(m["items"])]
    return {"goals": goals, "submit": submit, "menu": menu}


def instruction(spec, goal, rng: random.Random) -> str:
    wmap = {w["id"]: (w, tab) for w, tab in all_widgets(spec)}
    parts, said_tab = [], None
    for g in goal["goals"]:
        w, tab = wmap[g["id"]]
        p = phrase(w, g["value"], rng)
        if tab and tab != said_tab:
            p = f"on the {tab} tab, {p}"
        said_tab = tab or said_tab
        parts.append(p)
    if len(parts) == 1:
        text = parts[0]
    else:
        text = ", ".join(parts[:-1]) + ", and " + parts[-1]
    text = text[0].upper() + text[1:]
    if goal["menu"]:
        text += f'. Then choose {goal["menu"][0]} > {goal["menu"][1]} from the menu bar'
    if goal["submit"]:
        label = next(b["label"] for b in buttons(spec) if b["id"] == goal["submit"])
        text += rng.choice([f'. Then click "{label}".', f', then press the {label} button.', f'. Finish by clicking {label}.'])
    else:
        text += "."
    return text.replace("..", ".")
