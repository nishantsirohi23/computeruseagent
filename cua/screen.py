"""Main-display screenshots and coordinate mapping.

Global coordinates are Quartz points: origin at the top-left of the main display,
y down. AX frames and CGEvents both use them. Model coordinates are relative
0..1000 over the main display.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass

import Quartz
from PIL import Image

from . import config
from .prompts import smart_resize


@dataclass(frozen=True)
class Bounds:
    x: float
    y: float
    w: float
    h: float

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def to_rel(self, px: float, py: float) -> tuple[float, float]:
        return ((px - self.x) / self.w * config.COORD_SCALE, (py - self.y) / self.h * config.COORD_SCALE)

    def to_abs(self, rx: float, ry: float) -> tuple[float, float]:
        rx = min(max(rx, 0.0), config.COORD_SCALE - 1e-3)
        ry = min(max(ry, 0.0), config.COORD_SCALE - 1e-3)
        return (self.x + rx / config.COORD_SCALE * self.w, self.y + ry / config.COORD_SCALE * self.h)

    def rect_to_rel(self, x: float, y: float, w: float, h: float) -> list[float]:
        x1, y1 = self.to_rel(x, y)
        x2, y2 = self.to_rel(x + w, y + h)
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def main_bounds() -> Bounds:
    r = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    return Bounds(r.origin.x, r.origin.y, r.size.width, r.size.height)


def capture() -> Image.Image:
    """Full-resolution screenshot of the main display (cursor not included)."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="cua_")
    os.close(fd)
    try:
        subprocess.run(["screencapture", "-x", "-m", "-t", "png", path], check=True, timeout=15)
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    finally:
        os.unlink(path)


def model_image(img: Image.Image) -> Image.Image:
    """Resize to the exact size the Qwen processor would pick, so it does no further resizing."""
    h, w = smart_resize(img.height, img.width, config.IMAGE_FACTOR, config.IMAGE_MIN_PIXELS, config.IMAGE_MAX_PIXELS)
    if (w, h) == img.size:
        return img
    return img.resize((w, h), Image.LANCZOS)
