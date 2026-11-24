"""Prompt construction. Pure Python: imported by the Mac agent and by the T4 notebooks.

The system prompt reproduces exactly what Qwen3.5's chat template renders when it
is given `tools=[COMPUTER_USE_TOOL]`. We inline it into the system message because
TRL's GRPOTrainer (0.22.2) does not forward `tools` to apply_chat_template.
`tests/test_prompts.py` checks the two renderings are byte-identical.
"""
import json

COMPUTER_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_use",
        "description": (
            "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
            "* This is an interface to a macOS desktop GUI. You do not have access to a terminal.\n"
            "* Some applications may take time to start or process actions, so you may need to wait and take "
            "successive screenshots to see the results of your actions.\n"
            "* The screen's resolution is 1000x1000.\n"
            "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
            "screenshot to determine the coordinates of the element before moving the cursor.\n"
            "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
            "Don't click boxes on their edges."
        ),
        "parameters": {
            "properties": {
                "action": {
                    "description": (
                        "The action to perform. The available actions are:\n"
                        "* `key`: Performs key down presses on the arguments passed in order, then performs key "
                        "releases in reverse order. Use \"cmd\" for the Command key.\n"
                        "* `type`: Type a string of text on the keyboard.\n"
                        "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
                        "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
                        "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
                        "* `wait`: Wait specified seconds for the change to happen.\n"
                        "* `terminate`: Terminate the current task and report its completion status."
                    ),
                    "enum": [
                        "key", "type", "mouse_move", "left_click", "left_click_drag", "right_click",
                        "double_click", "triple_click", "scroll", "wait", "terminate",
                    ],
                    "type": "string",
                },
                "keys": {"description": "Required only by `action=key`.", "type": "array"},
                "text": {"description": "Required only by `action=type`.", "type": "string"},
                "coordinate": {
                    "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                                   "coordinates to move the mouse to.",
                    "type": "array",
                },
                "pixels": {
                    "description": "The amount of scrolling to perform. Positive values scroll up, negative values "
                                   "scroll down. Required only by `action=scroll`.",
                    "type": "number",
                },
                "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"},
                "status": {
                    "description": "The status of the task. Required only by `action=terminate`.",
                    "type": "string",
                    "enum": ["success", "failure"],
                },
            },
            "required": ["action"],
            "type": "object",
        },
    },
}

# Verbatim from Qwen3.5's chat_template.jinja (tools branch).
_TOOLS_HEADER = "# Tools\n\nYou have access to the following functions:\n\n<tools>"
_TOOLS_FOOTER = (
    "\n</tools>"
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in "
    "natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer "
    "the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>"
)

AGENT_INSTRUCTIONS = (
    "You are a GUI agent operating a macOS computer. You are given a task, the actions you already took, "
    "and a screenshot of the current screen. Decide the single next action and call computer_use exactly once. "
    "Coordinates are relative: x and y range from 0 to 1000 across the screenshot. "
    "When the task is complete, call computer_use with action=terminate and status=success."
)


def _tojson(obj) -> str:
    # transformers' chat-template `tojson` filter: json.dumps(ensure_ascii=False), default separators.
    return json.dumps(obj, ensure_ascii=False)


def system_prompt() -> str:
    return _TOOLS_HEADER + "\n" + _tojson(COMPUTER_USE_TOOL) + _TOOLS_FOOTER + "\n\n" + AGENT_INSTRUCTIONS


def user_text(instruction: str, history: list[str]) -> str:
    if history:
        hist = "\n".join(f"{i + 1}. {h}" for i, h in enumerate(history[-12:]))
    else:
        hist = "None"
    return (
        f"Task: {instruction}\n\n"
        f"Previous actions:\n{hist}\n\n"
        "Output the next action as a single computer_use tool call."
    )


def build_messages(instruction: str, history: list[str], image_placeholder=None) -> list[dict]:
    """Messages in HF multimodal format. `image_placeholder` is the content item for the image:
    {"type": "image"} for TRL datasets (image passed separately), or {"type": "image", "image": path}."""
    img = image_placeholder if image_placeholder is not None else {"type": "image"}
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt()}]},
        {"role": "user", "content": [img, {"type": "text", "text": user_text(instruction, history)}]},
    ]


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Same rounding rule as transformers' Qwen2-VL image processor (returns h, w)."""
    import math

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar
