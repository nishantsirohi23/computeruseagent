"""Policies that map (screenshot, task, history) -> model text + parsed Action."""
from __future__ import annotations

import os
import tempfile
import time

from PIL import Image

from .actions import Action, parse_action
from .prompts import build_messages


class MLXPolicy:
    """Qwen3.5 (base or GRPO-merged HF checkpoint) running locally with mlx-vlm."""

    def __init__(self, model_path: str, temperature: float = 0.0, max_tokens: int | None = None, seed: int | None = 0,
                 thinking: bool = False):
        from mlx_vlm import load

        self.name = model_path
        self.model, self.processor = load(model_path)
        self.temperature = temperature
        self.thinking = thinking  # extra baseline: base model allowed to reason before acting
        self.max_tokens = max_tokens or (2048 if thinking else 256)
        self.seed = seed

    def render(self, instruction: str, history: list[str]) -> str:
        msgs = build_messages(instruction, history, {"type": "image"})
        return self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=self.thinking)

    def act(self, image: Image.Image, instruction: str, history: list[str]) -> tuple[str, Action | None, dict]:
        from mlx_vlm import generate

        prompt = self.render(instruction, history)
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.save(path)
            t0 = time.time()
            kw = {"max_tokens": self.max_tokens, "temperature": self.temperature}
            if self.seed is not None:
                kw["seed"] = self.seed
            res = generate(self.model, self.processor, prompt, image=[path], verbose=False, **kw)
        finally:
            os.unlink(path)
        text = res.text
        info = {"latency": round(time.time() - t0, 2), "prompt_tokens": res.prompt_tokens,
                "generation_tokens": res.generation_tokens}
        return text, parse_action(text), info


class OraclePolicy:
    """Replays a task's oracle; used to validate tasks end to end."""

    name = "oracle"

    def __init__(self, task, env):
        self.gen = task.oracle(env)

    def act(self, image, instruction, history):
        try:
            a = next(self.gen)
        except StopIteration:
            a = Action("terminate", status="success")
        return "", a, {}
