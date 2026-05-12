import re
from typing import List, Any

from math_verify import parse, verify

import re


# ----------------------------
# Helpers used by rewards
# ----------------------------

_BOX_RE = re.compile(r"\\boxed\s*{[^}]*}\s*$", re.DOTALL)

def _get_content(completion: Any) -> str:
    """
    TRL/GRPO completions can be:
      - str
      - {"content": "..."}
      - [{"content": "..."}]
      - [[{"content": "..."}]]
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict) and "content" in completion:
        return completion["content"]
    if isinstance(completion, list) and completion:
        if isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0]["content"]
        if isinstance(completion[0], list) and completion[0]:
            if isinstance(completion[0][0], dict) and "content" in completion[0][0]:
                return completion[0][0]["content"]
    return str(completion)

def _unwrap_completions(completions: Any) -> List[str]:
    if not completions:
        return []
    return [_get_content(c) for c in completions]

# ----------------------------
# Main builder
# ----------------------------

def get_big_math_gsm8k_reward_funcs():
    """
    Reward funcs for LIMR-googletrans RL/GRPO.
    Returns: list of callables (prompts, completions, **kwargs) -> list[float]
    """


    # ---------- answer reward (math_verify relaxed) ----------
    def answer_reward_func(prompts, completions, **kwargs):
        """
        Binary reward:
        - 1.0 if verify(gold, pred) succeeds
        - 0.0 otherwise

        Works even when model doesn't output \\boxed{...}.
        """
        targets: List[str] = kwargs["answer"]
        contents = _unwrap_completions(completions)

        rewards: List[float] = []
        for i, (content, target) in enumerate(zip(contents, targets)):
            gold_parsed = parse(target, extraction_mode="first_match")
            if not gold_parsed:
                rewards.append(0.0)
                continue

            pred_parsed = parse(
                content,
                extraction_mode="first_match",
            )
            if not pred_parsed:
                rewards.append(0.0)
                continue

            try:
                reward = float(verify(gold_parsed, pred_parsed))
                rewards.append(reward)
            except Exception:
                reward = 0.0
                rewards.append(reward)

            if i == 0:
                print("prompt:", prompts[i], "answer:", target, "gold_parsed:", gold_parsed,"completion:", content, "pred_parsed:", pred_parsed, "reward:", reward)
        return rewards
    


    def ends_with_boxed_reward(
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """
        Dense shaping: reward if completion ends with a \\boxed{...}.
        - Returns 0.05 if ends with boxed, else 0.0
        """
        out = []
        for c in completions:
            if c is None:
                out.append(0.0)
                continue
            s = c.strip()
            out.append(0.05 if _BOX_RE.search(s) else 0.0)
        return out
        
    return [answer_reward_func, ends_with_boxed_reward]
