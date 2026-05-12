from __future__ import annotations
from typing import Dict, Callable
import torch
import torch.nn as nn

from ..params_base import ParamStrategy 


def _is_lora_param(name: str, param: torch.nn.Parameter) -> bool:
    if not param.requires_grad:
        return False
    lname = name.lower()
    return ("lora" in lname) or ("adapter" in lname)


class SFTParamStrategy(ParamStrategy):
    """
    Parameter-selection logic for *SFT* training.

    Behavior:
      - If lora_only=False: return trainable params excluding embeddings/heads/norms
      - If lora_only=True: return ONLY LoRA adapter params (PEFT-style SFT)
    """

    def __init__(self, lora_only: bool = False):
        self.lora_only = lora_only
        self.removing_keys = [
            "shared", "lm_head", "wte", "wpe", "ln", 
            "embed_tokens", "norm", "word_embeddings"
        ]

    def set_trainable(self, model: nn.Module) -> None:
        """
        Set trainable parameters for SFT:
          - If lora_only=True: no-op (PEFT already controls trainability)
          - If lora_only=False: freeze all, then unfreeze params except embeddings/heads/norms
        """
        if self.lora_only:
            # LoRA/PEFT already marks correct params as trainable
            return
        
        # Freeze all parameters first
        for p in model.parameters():
            p.requires_grad = False
        
        # Unfreeze parameters except those in removing_keys
        for name, param in model.named_parameters():
            if not any(key in name for key in self.removing_keys):
                param.requires_grad = True

    def select_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        param_filter = self.make_filter()
        return {
            name: param.data.detach().cpu()
            for name, param in model.named_parameters()
            if param_filter(name, param)
        }

    def make_filter(self) -> Callable[[str, torch.nn.Parameter], bool]:
        """
        Returns a param_filter callable for ParamVectorizer or model introspection.
        """
        def _filter(name: str, param: torch.nn.Parameter) -> bool:
            if not param.requires_grad:
                return False
            if self.lora_only:
                return _is_lora_param(name, param)
            # Exclude embedding/head/norm layers for full SFT
            if any(key in name for key in self.removing_keys):
                return False
            return True   # trainable set excluding removed keys
        return _filter