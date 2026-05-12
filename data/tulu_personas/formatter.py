from __future__ import annotations
from typing import Any, Dict
from transformers import PreTrainedTokenizerBase

from ..base_formatter import BaseFormatter

# Fallback chat template for models without one (e.g., Llama base)
FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}### User:\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'assistant' %}### Assistant:\n{{ message['content'] }}\n\n"
    "{% elif message['role'] == 'system' %}### System:\n{{ message['content'] }}\n\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}### Assistant:\n{% endif %}"
)


class TuluPersonasSFTFormatter(BaseFormatter):
    """
    Chat-template formatter for Tulu-3 personas SFT datasets.

    Input: example with `messages` field (list of {role, content} dicts).
    Output: pre-tokenized input_ids, attention_mask, completion_mask.
    """

    def __init__(self, add_eos: bool = True):
        self.add_eos = add_eos

    def format(
        self,
        example: Dict[str, Any],
        task: str,
        tokenizer: PreTrainedTokenizerBase = None,
    ) -> Dict[str, Any]:
        if tokenizer is None:
            raise ValueError("TuluPersonasSFTFormatter requires a tokenizer.")

        messages = example["messages"]
        if not messages:
            return {"input_ids": None, "attention_mask": None, "completion_mask": None, "task_name": task}

        max_seq_length = getattr(tokenizer, "model_max_length", 2048)

        # Use model's chat template if available, otherwise fallback
        chat_template = tokenizer.chat_template if tokenizer.chat_template else FALLBACK_CHAT_TEMPLATE

        # Tokenize full conversation (prompt + completion)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=chat_template,
        )

        # Tokenize prompt only (all messages except the last assistant turn)
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=chat_template,
        )

        if self.add_eos and tokenizer.eos_token and not full_text.endswith(tokenizer.eos_token):
            full_text += tokenizer.eos_token

        # Tokenize
        prompt_tok = tokenizer(
            prompt_text,
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
        }


TULU_PERSONAS_SFT_FORMATTER = TuluPersonasSFTFormatter(add_eos=True)
