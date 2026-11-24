"""Run one task episode with a policy in the live environment and log everything."""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

from .actions import Action, describe
from .env import MacEnv, UserAbort


def run_episode(env: MacEnv, task, policy_factory, out_dir: Path | None = None, log=print) -> dict:
    """policy_factory(task, env) -> object with .act(image, instruction, history)."""
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    result = {"task_id": task.id, "family": task.family, "template": task.template, "instruction": task.instruction,
              "steps": [], "success": False, "terminated": False, "error": None}
    t0 = time.time()
    history: list[str] = []
    invalid_streak = 0
    try:
        obs = env.reset(task)
        policy = policy_factory(task, env)
        for t in range(task.max_steps):
            text, action, info = policy.act(obs.image, task.instruction, history)
            step = {"t": t, "output": text, "action": action.args() if action else None, **info}
            if out_dir:
                obs.image.save(out_dir / f"step{t:02d}.png")
            if action is None:
                invalid_streak += 1
                history.append("(invalid output, no action taken)")
                step["exec"] = {"ok": False, "blocked": "unparsable"}
                result["steps"].append(step)
                log(f"  [{t}] invalid output: {text[:120]!r}")
                if invalid_streak >= 3:
                    break
                continue
            invalid_streak = 0
            log(f"  [{t}] {describe(action)}")
            if action.action == "terminate":
                result["terminated"] = True
                result["claimed"] = action.status
                result["steps"].append(step)
                break
            step["exec"] = env.step(action)
            entry = describe(action) + (" (blocked: outside the task)" if step["exec"]["blocked"] else "")
            history.append(entry)
            result["steps"].append(step)
            obs = env.observe()
        result["success"] = bool(task.check(env))
    except UserAbort as e:
        result["error"] = f"UserAbort: {e}"
        raise
    except Exception as e:  # task setup or app failures count as failures, but are logged
        result["error"] = f"{type(e).__name__}: {e}"
        log(traceback.format_exc())
    finally:
        try:
            env.teardown(task)
        except Exception as e:
            log(f"  teardown error: {e}")
        result["seconds"] = round(time.time() - t0, 1)
        if out_dir:
            (out_dir / "episode.json").write_text(json.dumps(result, indent=1))
    return result
