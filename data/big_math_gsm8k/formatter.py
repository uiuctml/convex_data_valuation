from __future__ import annotations
from typing import Any, Dict, Optional

from transformers import PreTrainedTokenizerBase

from .prompts.system import QWEN_SYSTEM_PROMPT
from ..base_formatter import BaseFormatter

def normalize_gsm8k_answer(ans: str) -> str:
    """
    Normalize GSM8K final answers by:
      - stripping whitespace
      - removing surrounding '$' (LaTeX math mode or currency)
      - dropping units / trailing words (keep first token)

    Examples:
      "$900$"            -> "900"
      "18 \\text{ km}"   -> "18"
      "1,200 dollars"    -> "1,200"
      "3.5 cm"           -> "3.5"
    """
    if ans is None:
        return ans

    s = ans.strip()

    # remove surrounding dollar signs (can be one or both sides)
    # do this BEFORE splitting on whitespace
    if s.startswith("$") and s.endswith("$") and len(s) > 1:
        s = s[1:-1]
    else:
        s = s.lstrip("$").rstrip("$")

    # drop units / trailing text
    return s.split()[0]


class BigMathRLGSM8KGoogleTransRLFormatter(BaseFormatter):
    """
    Formatter for Big-Math-RL (GSM8K source slice) after googletrans multilingual expansion.

    Expects:
        - "prompt":   str (translated GSM8K-style problem text)
        - "solution": str (ground-truth final answer string)
        - "lang":     str (language code, e.g., "en", "zh", "ja")

    Produces:
        - "prompt":  str (chat-templated with system prompt + user message)
        - "answer":  str (normalized ground truth via normalize_gsm8k_answer)
        - "problem": str (raw problem text before chat templating)
        - "lang":    str
        - "task":    str (partition name from DataModule)
        - passthrough metadata fields (source, language, domain, etc.)

    Note: Requires tokenizer for apply_chat_template().
    """

    # Optionally keep a small allowlist of passthrough fields.
    # This prevents accidental huge columns (e.g., tokenized fields) from being carried around.
    _PASSTHROUGH_KEYS = {
        "source",
        "language",
        "domain",
        "llama8b_solve_rate",
        "dataset",
        "split",
        "difficulty",
        "subject",
        "category",
    }

    def format(
        self,
        example: Dict[str, Any],
        task: str,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> Optional[Dict[str, Any]]:
        
        # Required fields
        if "prompt" not in example or "solution" not in example:
            return {
                "prompt": None,
                "answer": None,
                "problem": None,
                "lang": example.get("lang"),
                "task": task,
            }

        problem = (example.get("prompt") or "").strip()
        gt_solution = normalize_gsm8k_answer((example.get("solution") or ""))

        if not problem or not gt_solution:
            return {
                "prompt": None,
                "answer": None,
                "problem": None,
                "lang": example.get("lang"),
                "task": task,
            }
        
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        out = {
            "prompt": prompt,
            "answer": gt_solution,
            "problem": problem,
            "lang": example.get("lang"),
            "task": task,
        }

        # Safe passthrough fields (optional)
        for k in self._PASSTHROUGH_KEYS:
            if k in example:
                out[k] = example[k]

        return out


BIG_MATH_RL_GSM8K_GOOGLETRANS_RL_FORMATTER = BigMathRLGSM8KGoogleTransRLFormatter()