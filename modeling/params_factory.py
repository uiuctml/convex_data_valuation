# modeling/params.py
from __future__ import annotations
from typing import Any

from .params_base import ParamStrategy
from .grpo.params_strategy import RLParamStrategy
from .sft.params_strategy import SFTParamStrategy


def get_param_strategy(task_type: str, cfg: Any) -> ParamStrategy:
    """
    Factory to choose RL vs SFT param strategy based on task_type / cfg.
    """
    t = task_type.lower()
    if t in {"grpo"}:
        return RLParamStrategy(
            lora_only=getattr(cfg, "lora_only", True),
        )
    if t == "sft":
        return SFTParamStrategy(
            lora_only=getattr(cfg, "lora_only", False),
        )
    raise ValueError(f"Unknown task_type '{task_type}' for parameter strategy.")