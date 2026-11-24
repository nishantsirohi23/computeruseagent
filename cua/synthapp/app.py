"""SynthApp: a native AppKit form window built from a JSON spec.

    python -m cua.synthapp.app SPEC.json STATE.json

Every control is a stock Cocoa control, so it looks and exposes accessibility like
any macOS app. After every user interaction the full widget state is written
atomically to STATE.json, which task checkers read.
"""
import json
import os
import sys
import time

import objc
from AppKit import (
    NSApplication, NSApplicationActivationPolicyRegular, NSAppearance, NSBackingStoreBuffered, NSBezelStyleRounded,
    NSButton, NSButtonTypeRadio, NSButtonTypeSwitch, NSFont, NSMenu, NSMenuItem, NSPopUpButton, NSScrollView,
    NSSegmentedControl, NSSlider, NSStepper, NSTableColumn, NSTableView, NSTabView, NSTabViewItem, NSTextField,
    NSView, NSWindow, NSWindowStyleMaskClosable, NSWindowStyleMaskMiniaturizable, NSWindowStyleMaskTitled,
    NSScreen,
)
from Foundation import NSMakeRect, NSObject

CONTROL_SIZES = {"small": 1, "regular": 0, "large": 3}  # NSControlSize values
FONT_SIZES = {"small": 11.0, "regular": 13.0, "large": 15.0}
ROW_H = {"text": 24, "checkbox": 22, "popup": 26, "slider": 24, "stepper": 26, "segmented": 26, "static": 20}


class FlippedView(NSView):
    def isFlipped(self):
        return True


class Controller(NSObject):
    def initWithSpec_path_(self, spec, path):
        self = objc.super(Controller, self).init()
        if self is None:
            return None
        self.spec = spec
        self.path = path
        self.widgets = {}  # id -> (type, control or dict)
        self.tables = {}  # table identifier -> options
        self.clicks = {}
        self.submissions = []
        self.menu_hits = []
        self.events = []
        self.tabview = None
        self.closed = False
        return self

    # ---- state -------------------------------------------------------------
    def values(self):
        out = {}
        for wid, (kind, c) in self.widgets.items():
            if kind == "text":
                out[wid] = str(c.stringValue())
            elif kind == "checkbox":
                out[wid] = bool(c.state() == 1)
            elif kind == "radio":
                sel = [lbl for lbl, b in c.items() if b.state() == 1]
                out[wid] = sel[0] if sel else None
            elif kind == "popup":
                t = c.titleOfSelectedItem()
                out[wid] = str(t) if t is not None else None
            elif kind == "slider":
                out[wid] = round(float(c.doubleValue()), 2)
            elif kind == "stepper":
                out[wid] = int(c["stepper"].integerValue())
            elif kind == "segmented":
                i = c["control"].selectedSegment()
                out[wid] = c["options"][i] if 0 <= i < len(c["options"]) else None
            elif kind == "list":
                r = c.selectedRow()
                opts = self.tables[str(c.identifier())]
                out[wid] = opts[r] if 0 <= r < len(opts) else None
        return out

    def dump(self, event=None):
        if event:
            self.events.append(event)
        tab = None
        if self.tabview is not None:
            item = self.tabview.selectedTabViewItem()
            tab = str(item.label()) if item is not None else None
        st = {
            "pid": os.getpid(), "ts": time.time(), "values": self.values(), "clicks": self.clicks,
            "submissions": self.submissions, "menu": self.menu_hits, "tab": tab, "closed": self.closed,
            "events": self.events[-50:],
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, self.path)

    # ---- actions -----------------------------------------------------------
    def changed_(self, sender):
        if isinstance(sender, NSStepper):
            wid = str(sender.identifier())
            self.widgets[wid][1]["label"].setStringValue_(str(int(sender.integerValue())))
        self.dump({"change": str(sender.identifier())})

    def buttonClicked_(self, sender):
        wid = str(sender.identifier())
        self.clicks[wid] = self.clicks.get(wid, 0) + 1
        self.submissions.append({"button": wid, "values": self.values(), "ts": time.time()})
        self.dump({"click": wid})

    def menuHit_(self, sender):
        self.menu_hits.append(str(sender.title()))
        self.dump({"menu": str(sender.title())})

    def controlTextDidChange_(self, note):
        self.dump({"text": str(note.object().identifier())})

    def numberOfRowsInTableView_(self, tv):
        return len(self.tables.get(str(tv.identifier()), []))

    def tableView_objectValueForTableColumn_row_(self, tv, col, row):
        return self.tables[str(tv.identifier())][row]

    def tableViewSelectionDidChange_(self, note):
        self.dump({"select": str(note.object().identifier())})

    def tabView_didSelectTabViewItem_(self, tv, item):
        self.dump({"tab": str(item.label())})

    def windowWillClose_(self, note):
        self.closed = True
        self.dump({"close": True})
        NSApplication.sharedApplication().terminate_(None)


def _label(text, x, y, w, size, font):
    t = NSTextField.labelWithString_(text)
    t.setFrame_(NSMakeRect(x, y, w, 20))
    t.setFont_(font)
    return t


def _button(title, ident, ctrl, size, font):
    b = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 28))
    b.setTitle_(title)
    b.setBezelStyle_(NSBezelStyleRounded)
    b.setIdentifier_(ident)
    b.setTarget_(ctrl)
    b.setAction_("buttonClicked:")
    b.setControlSize_(size)
    b.setFont_(font)
    b.sizeToFit()
    return b


def build_rows(parent, widgets, ctrl, spec, top, width):
    """Lay out widgets top-to-bottom inside a flipped view. Returns the y after the last row."""
    size = CONTROL_SIZES[spec.get("control_size", "regular")]
    font = NSFont.systemFontOfSize_(FONT_SIZES[spec.get("control_size", "regular")])
    lw = spec.get("label_w", 130)
    pad = spec.get("pad", 20)
    gap = spec.get("gap", 12)
    cw = min(spec.get("control_w", 220), width - lw - 2 * pad - 10)
    x0 = pad + lw + 10
    y = top
    for w in widgets:
        kind, wid = w["type"], w.get("id", "")
        label = w.get("label", "")
        if kind == "buttons":
            x = width - pad
            row = []
            for b in reversed(w["items"]):
                btn = _button(b["label"], b["id"], ctrl, size, font)
                bw = btn.frame().size.width + 8
                x -= bw
                btn.setFrame_(NSMakeRect(x, y, bw, 30))
                parent.addSubview_(btn)
                x -= 8
                row.append(btn)
            y += 30 + gap
            continue
        if kind == "static":
            parent.addSubview_(_label(w["text"], pad, y, width - 2 * pad, size, font))
            y += 20 + gap
            continue
        if label and kind not in ("checkbox",):
            parent.addSubview_(_label(label + ":", pad, y + 3, lw, size, font))
        if kind == "text":
            c = NSTextField.alloc().initWithFrame_(NSMakeRect(x0, y, cw, 24))
            c.setStringValue_(w.get("value", ""))
            if w.get("placeholder"):
                c.setPlaceholderString_(w["placeholder"])
            c.setDelegate_(ctrl)
            h = 24
        elif kind == "checkbox":
            c = NSButton.alloc().initWithFrame_(NSMakeRect(pad if w.get("full") else x0, y, cw + (lw if w.get("full") else 0), 22))
            c.setButtonType_(NSButtonTypeSwitch)
            c.setTitle_(label)
            c.setState_(1 if w.get("value") else 0)
            c.setTarget_(ctrl)
            c.setAction_("changed:")
            # AppKit only toggles on the box + title, so shrink the frame (and the AX frame) to that.
            c.setControlSize_(size)
            c.setFont_(font)
            c.sizeToFit()
            fr = c.frame()
            c.setFrame_(NSMakeRect(fr.origin.x, y, fr.size.width + 4, 22))
            h = 22
        elif kind == "radio":
            group = {}
            box = FlippedView.alloc().initWithFrame_(NSMakeRect(x0, y, cw, 22 * len(w["options"])))
            horizontal = w.get("horizontal", False)
            bx = 0
            for i, opt in enumerate(w["options"]):
                r = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 100, 20))
                r.setButtonType_(NSButtonTypeRadio)
                r.setTitle_(opt)
                r.setIdentifier_(f"{wid}::{opt}")
                r.setControlSize_(size)
                r.setFont_(font)
                r.sizeToFit()
                rw = r.frame().size.width + 6
                if horizontal:
                    r.setFrame_(NSMakeRect(bx, 0, rw, 20))
                    bx += rw + 10
                else:
                    r.setFrame_(NSMakeRect(0, 22 * i, max(rw, 60), 20))
                r.setState_(1 if w.get("value") == opt else 0)
                r.setTarget_(ctrl)
                r.setAction_("changed:")
                box.addSubview_(r)
                group[opt] = r
            if horizontal:
                box.setFrame_(NSMakeRect(x0, y, max(bx, 10), 22))
            parent.addSubview_(box)
            ctrl.widgets[wid] = ("radio", group)
            y += box.frame().size.height + gap
            continue
        elif kind == "popup":
            c = NSPopUpButton.alloc().initWithFrame_pullsDown_(NSMakeRect(x0, y - 1, cw, 26), False)
            c.addItemsWithTitles_(w["options"])
            if w.get("value") in w["options"]:
                c.selectItemWithTitle_(w["value"])
            c.setTarget_(ctrl)
            c.setAction_("changed:")
            h = 26
        elif kind == "slider":
            c = NSSlider.alloc().initWithFrame_(NSMakeRect(x0, y, cw, 24))
            c.setMinValue_(w.get("min", 0))
            c.setMaxValue_(w.get("max", 100))
            c.setDoubleValue_(w.get("value", 50))
            c.setContinuous_(True)
            c.setTarget_(ctrl)
            c.setAction_("changed:")
            h = 24
        elif kind == "stepper":
            val = NSTextField.labelWithString_(str(w.get("value", 0)))
            val.setFrame_(NSMakeRect(x0, y + 3, 40, 20))
            val.setFont_(font)
            val.setIdentifier_(f"{wid}::value")
            parent.addSubview_(val)
            c = NSStepper.alloc().initWithFrame_(NSMakeRect(x0 + 44, y, 19, 26))
            c.setMinValue_(w.get("min", 0))
            c.setMaxValue_(w.get("max", 10))
            c.setIntegerValue_(w.get("value", 0))
            c.setValueWraps_(False)
            c.setTarget_(ctrl)
            c.setAction_("changed:")
            c.setIdentifier_(wid)
            c.setControlSize_(size)
            parent.addSubview_(c)
            ctrl.widgets[wid] = ("stepper", {"stepper": c, "label": val})
            y += 26 + gap
            continue
        elif kind == "segmented":
            c = NSSegmentedControl.alloc().initWithFrame_(NSMakeRect(x0, y, cw, 26))
            c.setSegmentCount_(len(w["options"]))
            for i, opt in enumerate(w["options"]):
                c.setLabel_forSegment_(opt, i)
            if w.get("value") in w["options"]:
                c.setSelectedSegment_(w["options"].index(w["value"]))
            c.setTarget_(ctrl)
            c.setAction_("changed:")
            c.setIdentifier_(wid)
            c.setControlSize_(size)
            c.setFont_(font)
            c.sizeToFit()
            parent.addSubview_(c)
            ctrl.widgets[wid] = ("segmented", {"control": c, "options": list(w["options"])})
            y += 26 + gap
            continue
        elif kind == "list":
            rows = len(w["options"])
            hh = min(rows, w.get("visible_rows", 6)) * 20 + 4
            sv = NSScrollView.alloc().initWithFrame_(NSMakeRect(x0, y, cw, hh))
            tv = NSTableView.alloc().initWithFrame_(NSMakeRect(0, 0, cw, hh))
            col = NSTableColumn.alloc().initWithIdentifier_("c")
            col.setWidth_(cw - 4)
            tv.addTableColumn_(col)
            tv.setHeaderView_(None)
            tv.setIdentifier_(wid)
            tv.setRowHeight_(18)
            ctrl.tables[wid] = list(w["options"])
            tv.setDataSource_(ctrl)
            tv.setDelegate_(ctrl)
            sv.setDocumentView_(tv)
            sv.setHasVerticalScroller_(True)
            parent.addSubview_(sv)
            ctrl.widgets[wid] = ("list", tv)
            y += hh + gap
            continue
        else:
            raise ValueError(f"unknown widget type {kind}")
        c.setIdentifier_(wid)
        if kind != "slider":
            c.setFont_(font)
        c.setControlSize_(size)
        parent.addSubview_(c)
        ctrl.widgets[wid] = (kind, c)
        y += h + gap
    return y


def build_menu(app, spec, ctrl):
    bar = NSMenu.alloc().init()
    app_item = NSMenuItem.alloc().init()
    bar.addItem_(app_item)
    app_menu = NSMenu.alloc().init()
    app_menu.addItemWithTitle_action_keyEquivalent_("Quit " + spec.get("app_name", "Form"), "terminate:", "q")
    app_item.setSubmenu_(app_menu)

    file_item = NSMenuItem.alloc().init()
    bar.addItem_(file_item)
    file_menu = NSMenu.alloc().initWithTitle_("File")
    file_menu.addItemWithTitle_action_keyEquivalent_("Close Window", "performClose:", "w")
    file_item.setSubmenu_(file_menu)

    edit_item = NSMenuItem.alloc().init()
    bar.addItem_(edit_item)
    edit = NSMenu.alloc().initWithTitle_("Edit")
    for title, sel, key in [("Undo", "undo:", "z"), ("Cut", "cut:", "x"), ("Copy", "copy:", "c"),
                            ("Paste", "paste:", "v"), ("Select All", "selectAll:", "a")]:
        edit.addItemWithTitle_action_keyEquivalent_(title, sel, key)
    edit_item.setSubmenu_(edit)

    for m in spec.get("menus", []):
        item = NSMenuItem.alloc().init()
        bar.addItem_(item)
        menu = NSMenu.alloc().initWithTitle_(m["title"])
        for t in m["items"]:
            mi = menu.addItemWithTitle_action_keyEquivalent_(t, "menuHit:", "")
            mi.setTarget_(ctrl)
        item.setSubmenu_(menu)
    app.setMainMenu_(bar)


def main():
    spec_path, state_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    if spec.get("appearance") == "dark":
        app.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameDarkAqua"))
    else:
        app.setAppearance_(NSAppearance.appearanceNamed_("NSAppearanceNameAqua"))
    ctrl = Controller.alloc().initWithSpec_path_(spec, state_path)
    build_menu(app, spec, ctrl)

    width = spec.get("width", 480)
    content = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, width, 1000))
    y = spec.get("pad", 20)
    if spec.get("heading"):
        h = NSTextField.labelWithString_(spec["heading"])
        h.setFont_(NSFont.boldSystemFontOfSize_(FONT_SIZES[spec.get("control_size", "regular")] + 3))
        h.setFrame_(NSMakeRect(spec.get("pad", 20), y, width - 40, 24))
        content.addSubview_(h)
        y += 34
    y = build_rows(content, spec.get("widgets", []), ctrl, spec, y, width)
    if spec.get("tabs"):
        tab_h = spec.get("tab_h", 260)
        tv = NSTabView.alloc().initWithFrame_(NSMakeRect(12, y, width - 24, tab_h))
        for t in spec["tabs"]:
            item = NSTabViewItem.alloc().initWithIdentifier_(t["title"])
            item.setLabel_(t["title"])
            v = FlippedView.alloc().initWithFrame_(NSMakeRect(0, 0, width - 40, tab_h - 40))
            build_rows(v, t["widgets"], ctrl, spec, 12, width - 40)
            item.setView_(v)
            tv.addTabViewItem_(item)
        tv.setDelegate_(ctrl)
        content.addSubview_(tv)
        ctrl.tabview = tv
        y += tab_h + 12
    y = build_rows(content, spec.get("footer", []), ctrl, spec, y, width)
    height = y + spec.get("pad", 20) - spec.get("gap", 12)
    content.setFrame_(NSMakeRect(0, 0, width, height))

    style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable
    scr = NSScreen.screens()[0].frame()  # main display, Cocoa coords (bottom-left origin)
    x, top = spec.get("origin", [40, 60])
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, scr.size.height - top - height - 28, width, height), style, NSBackingStoreBuffered, False)
    win.setTitle_(spec.get("title", "Form"))
    win.setContentView_(content)
    win.setDelegate_(ctrl)
    win.setReleasedWhenClosed_(False)
    win.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(True)
    ctrl.dump({"start": True})
    app.run()


if __name__ == "__main__":
    main()
