"""Tests for the pure-Python code shared by the Mac agent and the T4 notebooks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cua import prompts, rewards  # noqa: E402
from cua.actions import Action, format_completion, format_tool_call, normalize_keys, parse_action  # noqa: E402


def test_system_prompt_matches_native_tools_rendering():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    path = snapshot_download("Qwen/Qwen3.5-4B", allow_patterns=["*.json", "*.jinja", "*.txt"])
    tok = AutoTokenizer.from_pretrained(path)
    user = prompts.user_text("Click the Save button.", [])
    native = tok.apply_chat_template(
        [{"role": "system", "content": prompts.AGENT_INSTRUCTIONS}, {"role": "user", "content": user}],
        tools=[prompts.COMPUTER_USE_TOOL], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ours = tok.apply_chat_template(
        [{"role": "system", "content": prompts.system_prompt()}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert ours == native
    assert ours.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_roundtrip_xml():
    for a in [Action("left_click", coordinate=(412, 87)), Action("type", text='He said "hi"\nok'),
              Action("key", keys=["cmd", "s"]), Action("scroll", pixels=-5, coordinate=(500, 500)),
              Action("terminate", status="success"), Action("wait", time=2)]:
        b = parse_action(format_completion(a))
        assert b is not None and b.args() == a.args(), (a.args(), b and b.args())


def test_parse_variants():
    a = parse_action('I will click Save.\n\n<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n'
                     '</parameter>\n<parameter=coordinate>\n[512, 300]\n</parameter>\n</function>\n</tool_call>')
    assert a.action == "left_click" and a.coordinate == (512, 300) and a.thought == "I will click Save."
    j = parse_action('<tool_call>\n{"name": "computer_use", "arguments": {"action": "key", "keys": ["Command", "A"]}}\n</tool_call>')
    assert j.keys == ["cmd", "a"]
    assert parse_action("<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n</parameter>") is None
    assert parse_action("no call here") is None
    assert parse_action('<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":"(10, 20)"}}</tool_call>').coordinate == (10, 20)
    assert normalize_keys("ctrl+shift+Tab") == ["ctrl", "shift", "tab"]


def test_rewards():
    box = [400, 100, 440, 120]
    hit = format_tool_call(Action("left_click", coordinate=(420, 110)))
    near = format_tool_call(Action("left_click", coordinate=(450, 110)))
    far = format_tool_call(Action("left_click", coordinate=(900, 900)))
    g = rewards.grounding_reward([hit, near, far, "garbage"], bbox=[box] * 4)
    assert g[0] > 1.0 > g[1] > g[2] >= 0 and g[3] == 0
    f = rewards.format_reward([hit, "garbage"])
    assert f[0] > 0 > f[1]
    assert rewards.difficulty_weight([0, 0, 1000, 1000]) == 1.0
    assert rewards.difficulty_weight([0, 0, 10, 10]) == 1.5

    gt = json.dumps({"action": "type", "text": "Ana Ruiz"})
    s = rewards.step_reward([format_tool_call(Action("type", text="Ana Ruiz")),
                             format_tool_call(Action("type", text="Ana Ruis")),
                             format_tool_call(Action("left_click", coordinate=(1, 1)))], gt=[gt] * 3)
    assert s[0] == 1.0 and 0.3 < s[1] < 1.0 and s[2] == 0.0
    gt = {"action": "key", "keys": ["cmd", "a"], "alts": [{"action": "key", "keys": ["cmd", "down"]}]}
    assert rewards.step_score(Action("key", keys=["cmd", "down"]), gt) == 1.0
    gt = {"action": "left_click", "coordinate": [420, 110], "bbox": box}
    assert rewards.step_score(Action("double_click", coordinate=(420, 110)), gt) == 0.15


def test_smart_resize_portrait():
    from cua import config

    h, w = prompts.smart_resize(1920, 1080, config.IMAGE_FACTOR, config.IMAGE_MIN_PIXELS, config.IMAGE_MAX_PIXELS)
    assert (h, w) == (1344, 768)
