from __future__ import annotations
from typing import Any, Dict
from transformers import PreTrainedTokenizerBase

from ..base_formatter import BaseFormatter


class AyaSFTFormatter(BaseFormatter):
    """
    QA-style formatter (no chat template) with completion_mask for completion-only loss.
    """

    def __init__(self, add_eos: bool = True):
        self.add_eos = add_eos

    def format(
        self,
        example: Dict[str, Any],
        task: str,
        tokenizer: PreTrainedTokenizerBase,
    ) -> Dict[str, Any]:
        if tokenizer is None:
            raise ValueError("AyaSFTFormatter requires a tokenizer.")

        q: str = example["inputs"]
        a: str = example["targets"]

        # Prompt and full text
        prompt = q.strip()
        full_text = prompt + "\n\n" + a.strip()

        # Add EOS so the model learns to stop
        if self.add_eos and tokenizer.eos_token:
            full_text += tokenizer.eos_token

        max_seq_length = getattr(tokenizer, "model_max_length", 2048)

        # Tokenize prompt and full text separately to build completion_mask
        prompt_tok = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
        )
        full_tok = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            add_special_tokens=False,
            return_attention_mask=True,
        )

        input_ids = full_tok["input_ids"]
        attention_mask = full_tok["attention_mask"]

        # completion_mask: 0 for prompt tokens, 1 for completion tokens
        prompt_len = min(len(prompt_tok["input_ids"]), len(input_ids))
        completion_mask = [0] * prompt_len + [1] * (len(input_ids) - prompt_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "task_name": task,
            "language": example.get("language"),
            "language_code": example.get("language_code"),
            "annotation_type": example.get("annotation_type"),
            "user_id": example.get("user_id"),
        }


AYA_SFT_FORMATTER = AyaSFTFormatter(add_eos=True)