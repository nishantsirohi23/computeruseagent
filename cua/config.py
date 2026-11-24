"""Settings shared by the Mac side and the T4 notebooks.

Anything that changes what the model sees (image budget, prompt text) must be
identical at train and eval time, so it lives here and is copied into dataset
metadata.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"

# All files tasks create or modify live under this folder.
SANDBOX = Path.home() / "CUABench"

# Qwen3.5 vision: 16px patches merged 2x2 -> one visual token per 32x32 block.
IMAGE_FACTOR = 32
# ~1000 visual tokens. A 1080x1920 portrait display becomes 768x1344.
IMAGE_MAX_PIXELS = 1024 * IMAGE_FACTOR * IMAGE_FACTOR
IMAGE_MIN_PIXELS = 64 * IMAGE_FACTOR * IMAGE_FACTOR

# Model coordinates are relative, 0..1000 on both axes (Qwen computer_use convention).
COORD_SCALE = 1000

BASE_MODEL = "Qwen/Qwen3.5-4B"

# Where task windows are placed on the main display (points, top-left origin).
WINDOW_ORIGIN = (40, 60)
