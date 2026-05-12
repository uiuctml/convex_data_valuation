from __future__ import annotations
from typing import Dict, Callable, Protocol
import torch
import torch.nn as nn


class ParamStrategy(Protocol):
    """
    Interface for parameter selection / trainability control.

    Subclasses (or concrete implementations) must provide:
      - set_trainable(model): possibly mutate requires_grad flags
      - select_params(model): return {name: tensor} of interesting params
      - make_filter(): (name, param) -> bool for attribution code
    """

    def set_trainable(self, model: nn.Module) -> None:
        ...

    def select_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        ...

    def make_filter(self) -> Callable[[str, torch.nn.Parameter], bool]:
        ...