"""A plain full-screen backdrop over the main display's Desktop.

    python -m cua.backdrop

Sits above the Desktop icons/widgets and just below every normal window, so screenshots collected for
training show task windows on a neutral background instead of the user's files and wallpaper. It
accepts mouse events (so nothing reaches the Desktop through it), and env.MacEnv blocks clicks on it
because it belongs to no task.
"""
import Quartz
from AppKit import (
    NSApplication, NSApplicationActivationPolicyAccessory, NSBackingStoreBuffered, NSColor, NSScreen, NSWindow,
    NSWindowCollectionBehaviorCanJoinAllSpaces, NSWindowCollectionBehaviorStationary, NSWindowStyleMaskBorderless,
)


def main():
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon, no menu bar
    frame = NSScreen.screens()[0].frame()  # main display
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame, NSWindowStyleMaskBorderless, NSBackingStoreBuffered, False)
    win.setLevel_(Quartz.CGWindowLevelForKey(Quartz.kCGNormalWindowLevelKey) - 1)  # above Desktop icons and widgets
    win.setBackgroundColor_(NSColor.colorWithCalibratedRed_green_blue_alpha_(0.22, 0.27, 0.36, 1.0))
    win.setCollectionBehavior_(NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorStationary)
    win.setTitle_("CUA backdrop")
    win.orderFrontRegardless()
    app.run()


if __name__ == "__main__":
    main()
